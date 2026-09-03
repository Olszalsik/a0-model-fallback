"""Regression test for the v2.6.1 chat-cascade warm-read fix (2026-08-07).

The chat cascade (``_patched_call_chat_model``) references ``warm_timeout_s``
and ``warm_window_s`` at its per-candidate timeout resolution (mirroring the
utility cascade). v2.6 shipped the chat call-site usage but forgot the
matching config reads, so the names were unbound -- a ``NameError`` on chat
candidates with no user ``TIMEOUT`` kwarg, or a silently-dead warm/cold path
when one was set. This test guards the wiring at the compile-time level: the
chat cascade function MUST bind ``warm_timeout_s``/``warm_window_s`` as
locals (i.e. the config reads exist), and MUST still call
``_resolve_per_call_timeout`` on its per-candidate path.

A behavior test would need a full Agent mock (get_chat_model,
_build_candidates, _get_plugin_cfg, the loop, unified_call); the
compile-time check is deterministic, framework-free, and catches the exact
regression (missing assignment) that slipped through v2.6's replica-based
Phase-1 tests (those exercise ``_resolve_per_call_timeout`` directly, not the
chat cascade's call site).

To run:
    cd /a0
    REPO_ROOT_OVERRIDE="$(pwd)" pytest usr/plugins/_model_fallback/tests/test_chat_warm_wiring_v26.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(os.environ.get("REPO_ROOT_OVERRIDE") or "/a0")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from usr.plugins._model_fallback.fallback import (  # noqa: E402
    _patched_call_chat_model,
    _patched_call_utility_model,
    _run_rotation_cascade,
)

# v3.0.0: the warm reads and the per-candidate resolution call site moved
# into the shared engine; the two adapters delegate to it.
ENGINE = _run_rotation_cascade


def test_engine_binds_warm_vars():
    """The unified rotation engine must assign ``warm_timeout_s`` and
    ``warm_window_s`` (read them from plugin config) before the per-candidate
    warm-resolution call site. Before the v2.6.1 fix the chat cascade only
    ever referenced them, never assigned them -> treated as module globals
    -> ``NameError`` at runtime (no such global exists). The v3.0.0 engine
    owns the single copy of those reads, so the locals appear in its
    ``co_varnames``."""
    engine_varnames = ENGINE.__code__.co_varnames
    assert "warm_timeout_s" in engine_varnames, (
        "_run_rotation_cascade must read cascade_warm_timeout_s from plugin "
        "config. Without this read the warm path references an unbound name."
    )
    assert "warm_window_s" in engine_varnames, (
        "_run_rotation_cascade must read cascade_warm_window_s from plugin "
        "config."
    )


def test_engine_calls_resolve_per_call_timeout():
    """The engine must actually invoke ``_resolve_per_call_timeout`` (the
    warm/cold resolution) on its per-candidate path. The call site lives in
    the loop body of ``_run_rotation_cascade`` (not inside the nested
    ``_call_inner`` closure), so the global lookup lands in ``co_names``.
    Guards against the warm reads existing but the call site being dropped."""
    assert "_resolve_per_call_timeout" in ENGINE.__code__.co_names, (
        "_run_rotation_cascade must call _resolve_per_call_timeout on its "
        "per-candidate path to apply warm/cold timeouts."
    )


def test_adapters_delegate_to_engine():
    """v3.0.0: the utility and chat cascades are thin adapters. Their
    code objects must NOT carry the warm-resolution loop (that lives in
    the engine now) -- but they must not have regressed into something
    unrelated either."""
    # The engine owns the loop body.
    assert "_resolve_per_call_timeout" not in _patched_call_utility_model.__code__.co_names
    assert "_resolve_per_call_timeout" not in _patched_call_chat_model.__code__.co_names
    # ...and the adapters still reference it (they call the engine by name,
    # which the engine's closure then uses), so the delegation is real.
    assert "_run_rotation_cascade" in _patched_call_utility_model.__code__.co_names
    assert "_run_rotation_cascade" in _patched_call_chat_model.__code__.co_names