"""Asyncio read-ready guard (CPython gh-115514), merged into model_fallback.

Why it lives here
-----------------
It used to be a standalone ``_asyncio_guard`` plugin that applied its patch
at *import time of its own hooks.py*. That never ran: the only importer of
a plugin ``hooks.py`` is ``helpers.plugins.call_plugin_hook``, which is
lazy and on-demand, and nothing calls ``install()`` for enabled plugins at
boot. The guard was therefore dead code on every restart.

The fix is the install site, not the patch. This module is pure logic with
no import-time side effects; ``extensions/python/startup_migration/
_00_install_early_guards.py`` calls :func:`install` at the earliest
reliable plugin hook, so the guard is in place before the server binds and
before any litellm HTTPS call.

The bug
-------
``_SelectorSocketTransport.close()`` clears ``_read_ready_cb`` but the fd
can stay registered for one more selector tick. ``_read_ready`` was a bare
``self._read_ready_cb()``, so that stray tick raised
``TypeError: 'NoneType' object is not callable`` -- and because the fd stays
registered it re-raised on *every* tick, wedging the event loop. In Agent
Zero the teardown is triggered by ``asyncio.wait_for`` cancelling in-flight
litellm HTTPS calls (the utility timeout guard, litellm's own retries), so
one cancelled request could take the whole process down for the night.

The fix mirrors the upstream backport (gh-129582 / 3.12.9, gh-129581 /
3.13.2): return early when ``_read_ready_cb`` is None. It does not touch
the selector, so the connection-setup window between ``__init__`` and
``_call_connection_made`` is unaffected.

Detection is behavioural, not structural: :func:`_needs_guard` asks
whether calling the real ``_read_ready`` with a torn-down transport
raises. The previous ``co_names``-length heuristic could not work -- an
already-fixed body has the same ``co_names`` as an unguarded one -- and a
jump-opcode allowlist fails too, because CPython 3.12 emits
``POP_JUMP_IF_NOT_NONE`` and leaves ``dis.hasjabs`` empty. See
:func:`_needs_guard` for the full reasoning.
"""

from __future__ import annotations

import asyncio.selector_events as _selector_events
import logging
from typing import Any

_log = logging.getLogger("model_fallback.asyncio_guard")

PATCHED_FLAG = "_a0_asyncio_guard_patched"


def _needs_guard(cls: Any) -> bool:
    """True when ``cls._read_ready`` still calls a None callback.

    Behavioural probe, not a bytecode/opcode heuristic. Earlier
    revisions tried to infer the answer from the code shape -- first
    ``len(code.co_names) > 1``, then a list of jump opcodes -- and both
    were wrong:

    * a *fixed* body compiles to ``co_names == ('_read_ready_cb',)``,
      identical to the unguarded one, so ``co_names`` cannot tell them
      apart;
    * CPython 3.12 also *inverts* the guard, emitting
      ``POP_JUMP_IF_NOT_NONE`` rather than ``POP_JUMP_IF_FALSE``, and
      ``dis.hasjabs`` / ``dis.hasjrel`` are empty on 3.12, so an opcode
      allowlist silently misses a fixed interpreter.

    Instead we ask the only question that matters -- does calling
    ``_read_ready`` with a torn-down transport raise? -- by invoking the
    real method on a bare instance. A guarded body returns; an unguarded
    one raises ``TypeError: 'NoneType' object is not callable``. This is
    correct on every CPython, patched or not.

    Fails SAFE: any unexpected error is treated as "needs the guard".
    """
    try:
        inst = cls.__new__(cls)
        inst._read_ready_cb = None
        cls._read_ready(inst)
    except Exception:  # noqa: BLE001
        return True
    return False


def is_applied() -> bool:
    """True when the guard is in place (patched now or by an earlier boot)."""
    cls = getattr(_selector_events, "_SelectorSocketTransport", None)
    if cls is None:
        return False
    return bool(getattr(cls, PATCHED_FLAG, False))


def install() -> bool:
    """Patch ``_read_ready`` to no-op on a torn-down transport. Idempotent.

    Returns True when the guard is in place after the call, False when the
    target class/method is missing.
    """
    cls = getattr(_selector_events, "_SelectorSocketTransport", None)
    if cls is None:
        _log.debug("asyncio guard: _SelectorSocketTransport not available")
        return False
    if getattr(cls, PATCHED_FLAG, False):
        return True

    fn = cls.__dict__.get("_read_ready")
    if fn is None or not callable(fn):
        _log.debug("asyncio guard: _read_ready not found on the transport")
        return False

    # Already fixed upstream -> mark done and leave the interpreter alone.
    if not _needs_guard(cls):
        setattr(cls, PATCHED_FLAG, True)
        _log.debug(
            "asyncio guard: upstream _read_ready already guards a torn-down "
            "transport; no patch needed"
        )
        return True

    def _read_ready(self):  # type: ignore[no-redef]
        if self._read_ready_cb is None:
            # Torn down while its fd is still in the selector. The normal
            # close() path removes the reader; we only need to survive the
            # stray tick(s) until it does.
            return
        self._read_ready_cb()

    _read_ready.__doc__ = getattr(fn, "__doc__", None)
    cls._read_ready = _read_ready
    setattr(cls, PATCHED_FLAG, True)
    _log.info("asyncio read-ready guard installed (gh-115514 doom-loop fix)")
    return True


def uninstall() -> bool:
    """Intentionally a no-op.

    Removing a safety monkeypatch from a live process is riskier than
    leaving it in place, and the class is shared with every other socket
    user. The patched flag stays set so a reinstall stays a clean no-op.
    """
    return False


def status() -> dict:
    """Introspection snapshot for the stats endpoint / diagnostics."""
    return {
        "installed": is_applied(),
        "transport_present": hasattr(_selector_events, "_SelectorSocketTransport"),
    }
