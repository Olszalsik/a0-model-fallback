"""Extension import guard, merged into model_fallback.

Problem
-------
``helpers.modules.load_classes_from_folder`` calls ``import_module`` on
every ``*.py`` file in an extension folder with **no try/except**. One
broken file -- a stale ``from ... import <missing-symbol>`` after a
sibling module changed, a syntax error, a missing dependency -- therefore
propagates out of ``_get_extension_classes`` -> ``call_extensions_*`` and
aborts the whole extension point.

``chat_model_call_before`` and ``util_model_call_before`` fire on EVERY
model call, so one stale file bricks every chat and utility call. That is
exactly what happened when ``model_fallback/models_ext.py`` was briefly out
of sync with its ``_01_force_chat_completions.py`` importer:
``ImportError: cannot import name 'should_force_chat_completions'``.

How this differs from the standalone plugin it replaces
-------------------------------------------------------
The standalone ``_extension_import_guard`` re-implemented
``load_classes_from_folder`` line-for-line as a "mirror" and shipped an
``inspect.getsource`` marker check to warn about drift. A mirror must be
re-synced by hand on every upstream change, and the marker check only
notices the day the function stops mentioning ``fnmatch``.

This implementation keeps the upstream function authoritative and shims
only the one call that can raise: ``modules.import_module``. For the
duration of the upstream call we swap in a shim that turns an import
failure into ``None``. ``inspect.getmembers(None, inspect.isclass)`` is
simply empty, so upstream contributes no classes for that file and moves
on to the next one -- per-file isolation with no duplicated logic, and
every future upstream change to ordering / ``one_per_file`` / subclass
filtering is inherited for free.

The swap is re-entrancy safe. Extension loading happens from several
threads (concurrent agent creation, watchdog cache clears), and a naive
save/restore lets one thread restore another's shim, pinning the process
to a permanently patched ``import_module``. A depth counter under a lock
plus a thread-local error sink removes that leak.

Only ``Exception`` is swallowed: ``KeyboardInterrupt`` / ``SystemExit``
(and ``asyncio.CancelledError``, a BaseException on 3.8+) still propagate.
"""

from __future__ import annotations

import sys
import threading
import types
from typing import Any, Callable, List, Optional

from helpers import modules

# Sentinel returned to upstream in place of a module that failed to import.
# Empty on purpose: ``inspect.getmembers(_EMPTY_MODULE, isclass) == []``.
_EMPTY_MODULE = types.ModuleType("_mfb_import_guard_skipped")

# abs file path -> "ExcClassName: message" for the process lifetime.
# Also republished on ``helpers.modules._import_guard_skipped``.
SKIPPED: dict = {}

_local = threading.local()
_swap_lock = threading.RLock()
_depth = 0
_original_import: Optional[Callable[[str], Any]] = None

# Marker the upstream loader is still expected to have. The guard no longer
# mirrors upstream, it only needs ``import_module`` to be called per file.
_UPSTREAM_MARKERS = ("import_module",)


def _stderr(msg: str) -> None:
    # stderr -> supervisor/docker logs; flush so a hard crash can't lose it.
    try:
        sys.stderr.write(msg + "\n")
        sys.stderr.flush()
    except Exception:  # noqa: BLE001
        pass


def _publish() -> None:
    try:
        modules._import_guard_skipped = SKIPPED
    except Exception:  # noqa: BLE001
        pass


def _record_skip(file_path: str, exc: BaseException) -> None:
    if file_path in SKIPPED:
        return
    SKIPPED[file_path] = f"{type(exc).__name__}: {exc}"
    _publish()
    _stderr(
        "[import_guard] SKIPPED broken extension file "
        f"(agent continues without it): {file_path} -- "
        f"{type(exc).__name__}: {exc}"
    )


def _record_recovery(file_path: str) -> None:
    if SKIPPED.pop(file_path, None) is not None:
        _publish()
        _stderr(f"[import_guard] RECOVERED extension file: {file_path}")



def _shimmed_import(file_path: str):
    """Stand-in for ``modules.import_module`` used inside the upstream call."""
    original = _original_import
    if original is None:  # pragma: no cover - defensive
        return None
    try:
        module = original(file_path)
    except Exception as exc:  # noqa: BLE001
        bucket = getattr(_local, "errors", None)
        if bucket is None:
            bucket = _local.errors = []
        bucket.append((file_path, exc))
        # An EMPTY module, not None: upstream does
        # ``inspect.getmembers(module, inspect.isclass)`` and None still
        # yields ``__class__ -> NoneType``, which is a class and would leak
        # into the result for a permissive base_class. An empty module
        # yields [] so the file contributes nothing, exactly like an
        # upstream file that defines no Extension subclass.
        return _EMPTY_MODULE
    _record_recovery(file_path)
    return module


def is_installed() -> bool:
    return bool(
        getattr(modules.load_classes_from_folder, "_import_guard_patched", False)
    )


def check_upstream_drift() -> List[str]:
    """Return upstream markers this guard expects but no longer finds.

    Non-fatal and informational. Drift is no longer a correctness risk --
    the guard does not mirror upstream -- but a loader that stopped
    calling ``import_module`` per file would silently become a no-op, so
    it is still worth reporting.
    """
    try:
        import inspect

        src = inspect.getsource(modules.load_classes_from_folder)
    except Exception:  # noqa: BLE001
        return []
    return [m for m in _UPSTREAM_MARKERS if m not in src]


def install() -> bool:
    """Install the per-file import guard. Idempotent."""
    if is_installed():
        return True

    missing = check_upstream_drift()
    if missing:
        _stderr(
            "[import_guard] DRIFT WARNING: upstream load_classes_from_folder "
            f"no longer contains {missing}; the guard may be a no-op."
        )

    original = modules.load_classes_from_folder

    def guarded(folder, name_pattern, base_class, one_per_file=True):
        global _depth, _original_import

        with _swap_lock:
            if _depth == 0:
                # Save the REAL import function (not the loader) -- the
                # shim calls it with a single file path.
                _original_import = modules.import_module
            _depth += 1
            modules.import_module = _shimmed_import
        _local.errors = []
        try:
            return original(folder, name_pattern, base_class, one_per_file)
        finally:
            with _swap_lock:
                _depth -= 1
                if _depth == 0:
                    modules.import_module = _original_import
                    _original_import = None
            for file_path, exc in _local.errors:
                _record_skip(file_path, exc)
            _local.errors = []

    guarded._import_guard_patched = True  # type: ignore[attr-defined]
    guarded._import_guard_original = original  # type: ignore[attr-defined]
    modules.load_classes_from_folder = guarded
    _publish()
    return True


def uninstall() -> bool:
    """Restore the upstream loader. Used by the plugin's uninstall hook."""
    global _original_import, _depth

    current = modules.load_classes_from_folder
    original = getattr(current, "_import_guard_original", None)
    if original is None:
        return False
    with _swap_lock:
        modules.load_classes_from_folder = original
        if _original_import is not None:
            modules.import_module = _original_import
            _original_import = None
        _depth = 0
    return True


def skipped() -> dict:
    """Copy of the skipped-file map, for the stats endpoint / diagnostics."""
    return dict(SKIPPED)
