"""Tests for v2.8.3 — audit fixes (litellm.Timeout parity, turn-kwarg
guard, utility-guard config merge, all-skipped spin backoff, hooks API).

Root causes under test (2026-09-02 audit of the v2.8.2 tree):
  1. ``litellm.Timeout`` subclasses ``APITimeoutError -> APIConnectionError``,
     NOT the builtin ``TimeoutError`` -- a provider-side pure timeout fell
     through to the 300s "unknown error" cooldown AND kept its warm label,
     recreating the 20s warm-ceiling loop fix A (v2.6.4) closed.
  2. The v2.8.0 turn cascade omitted the ``user_kwarg_set`` guard, so an
     explicit ``TIMEOUT=`` model kwarg was capped by ``cascade_warm_timeout_s``
     (20s) on the main agent loop once the label was warm.
  3. ``get_plugin_config`` returns config.json WITHOUT merging
     default_config.yaml, so the YAML's utility_timeout_guard 60s/180s
     (raised 2026-07-23) never reached the runtime -- DEFAULTS' stale
     30s/120s won. DEFAULTS are now synced and _resolve_config merges
     the YAML under config.json.
  4. The all-skipped spin path slept a flat 2.0s forever when every
     candidate was dead-marked (no backoff, no legacy-mode exit).
  5. hooks.clear_cooldowns/get_cooldowns operated on the legacy
     ``_model_cooldowns`` data key, not the live in-memory store.

Test:
    cd /a0  (or the repo root on a host checkout)
    pytest usr/plugins/model_fallback/tests/test_v283_audit_fixes.py -v
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(os.environ.get("REPO_ROOT_OVERRIDE") or Path(__file__).resolve().parents[4])
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest

from usr.plugins.model_fallback import fallback
from usr.plugins.model_fallback.fallback import (
    _cooldown_seconds_for_status,
    _is_timeout_shaped,
)
from usr.plugins.model_fallback.helpers import utility_timeout

# NOTE: resolve module-level state via ``fallback.<name>`` at call time,
# never via import-time from-imports (suite-pollution gotcha -- see
# test_turn_cascade_v28.py).


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeLog:
    def __init__(self):
        self.lines: list[tuple] = []

    def log(self, type="info", content="", **kwargs):
        self.lines.append((type, content))


class FakeContext:
    id = "test-agent-v283"

    def __init__(self):
        self.log = FakeLog()


class FakeAgent:
    def __init__(self):
        self.context = FakeContext()
        self._data: dict = {}

    def get_data(self, key, default=None):
        return self._data.get(key, default)

    def set_data(self, key, value):
        self._data[key] = value


# ---------------------------------------------------------------------------
# 1. litellm.Timeout parity
# ---------------------------------------------------------------------------


def test_timeout_shaped_recognizes_litellm_timeout():
    """litellm.Timeout is a pure timeout even though it is NOT a
    builtin TimeoutError subclass."""
    litellm_exc = pytest.importorskip("litellm.exceptions")
    assert issubclass(litellm_exc.Timeout, Exception)
    assert not issubclass(litellm_exc.Timeout, TimeoutError)
    exc = litellm_exc.Timeout("timed out", "test-model", "test-provider")
    assert _is_timeout_shaped(exc) is True


def test_timeout_shaped_builtin_and_negative():
    assert _is_timeout_shaped(asyncio.TimeoutError()) is True
    assert _is_timeout_shaped(TimeoutError()) is True
    assert _is_timeout_shaped(ValueError("nope")) is False
    assert _is_timeout_shaped(None) is False


def test_litellm_timeout_gets_timeout_cooldown_not_unknown():
    """A status-less litellm.Timeout must book the short timeout
    cooldown (45s default), not the 300s unknown-error default."""
    litellm_exc = pytest.importorskip("litellm.exceptions")
    exc = litellm_exc.Timeout("timed out", "test-model", "test-provider")
    secs = _cooldown_seconds_for_status(None, exc)
    assert secs < 300.0
    assert secs == pytest.approx(
        fallback._DEFAULT_TIMEOUT_COOLDOWN_S,
        abs=1.0,
    )


# ---------------------------------------------------------------------------
# 2. Turn-cascade user TIMEOUT= kwarg contract
# ---------------------------------------------------------------------------


def test_turn_cascade_user_kwarg_skips_warm_ceiling():
    """Mirror of the chat cascade's guard: when the model kwargs carry
    TIMEOUT=, the turn cascade must use it verbatim instead of the warm
    ceiling. Regression: v2.8.0's turn cascade omitted the guard.

    We can't run the full cascade without a model stack, so we assert
    the source carries the guard on the turn path (same shape as the
    chat cascade's) -- a source-level tripwire rather than an
    integration test.
    """
    src = Path(fallback.__file__).read_text(encoding="utf-8")
    turn_start = src.index("async def _patched_call_chat_model_turn")
    # The turn cascade is the LAST cascade in the file; bound the slice
    # at its installer.
    turn_end = src.index("def install_chat_turn_patch")
    turn_body = src[turn_start:turn_end]
    assert "user_kwarg_set" in turn_body, (
        "turn cascade lost the user TIMEOUT= kwarg guard (v2.8.3 fix)"
    )
    # And the warm fast-path is gated on it, not unconditional.
    warm_call = [ln for ln in turn_body.splitlines() if "_resolve_per_call_timeout" in ln]
    assert warm_call, "turn cascade should still resolve warm/cold timeouts"
    guarded = any(
        "user_kwarg_set" in turn_body.splitlines()[i - 3]
        for i, ln in enumerate(turn_body.splitlines())
        if "_resolve_per_call_timeout" in ln
    ) or turn_body.count("user_kwarg_set") >= 2
    assert guarded, "warm fast-path is not gated on user_kwarg_set"


# ---------------------------------------------------------------------------
# 3. Utility-guard config sync
# ---------------------------------------------------------------------------


def test_utility_timeout_defaults_synced_above_floor():
    """DEFAULTS must mirror default_config.yaml (60s/180s) -- the 2026-07-23
    raise never took effect because config.json-only resolution skipped the
    YAML. The floor test guards YAML; this guards the code defaults."""
    assert utility_timeout.DEFAULTS["default_timeout_s"] >= 60.0
    assert utility_timeout.DEFAULTS["max_wait_s"] >= utility_timeout.DEFAULTS[
        "default_timeout_s"
    ]
    resolved = utility_timeout.resolve_config({})
    assert resolved["default_timeout_s"] >= 60.0
    assert resolved["max_wait_s"] >= resolved["default_timeout_s"]


def test_utility_timeout_yaml_merge_reaches_resolve():
    """_resolve_config must merge default_config.yaml under config.json --
    with an empty config.json the YAML's 60s default must win over the
    (old, stale) 30s code default."""
    ext = pytest.importorskip(
        "usr.plugins.model_fallback.extensions.python.agent_init"
        "._10_install_utility_timeout_patch"
    )
    cfg = ext._resolve_config(None)
    assert cfg.get("enabled") is True
    assert cfg.get("default_timeout_s") >= 60.0, (
        f"utility guard resolved to {cfg.get('default_timeout_s')}s -- the "
        "YAML merge (or the DEFAULTS sync) regressed to the stale 30s value"
    )
    assert cfg.get("max_wait_s") >= cfg.get("default_timeout_s")


def test_context_guard_toggle_wins_over_sectionless_config():
    """The WebUI toggle (top-level context_size_guard_enabled) must turn
    the guard ON even when config.json carries no nested section --
    otherwise the UI says ON while the runtime says OFF."""
    from usr.plugins.model_fallback.extensions.python.message_loop_prompts_after import (  # noqa: E402
        _10_context_size_guard as csg,
    )

    from helpers import plugins as plugin_helpers

    original = plugin_helpers.get_plugin_config
    try:
        plugin_helpers.get_plugin_config = (
            lambda *a, **k: {"context_size_guard_enabled": True}
        )
        cfg = csg._resolve_runtime_config(None)
    finally:
        plugin_helpers.get_plugin_config = original
    assert cfg.get("enabled") is True, (
        "toggle ON must force enabled=True in the resolved runtime config"
    )
    assert cfg.get("max_chars") == 50000


# ---------------------------------------------------------------------------
# 4. hooks cooldown API operates on the live store
# ---------------------------------------------------------------------------


def test_hooks_cooldown_api_uses_live_store():
    from usr.plugins.model_fallback import hooks

    agent = FakeAgent()
    store = fallback._get_cooldown_store(agent)
    store["some/model"] = time.monotonic() + 60.0
    fallback._save_cooldown_store(agent, store)

    live = hooks.get_cooldowns(agent)
    assert "some/model" in live, (
        "get_cooldowns must read the live in-memory store"
    )

    cleared = hooks.clear_cooldowns(agent)
    assert cleared >= 1
    assert fallback.snapshot_cooldowns(agent) == {}


def test_snapshot_and_clear_helpers():
    agent = FakeAgent()
    assert fallback.snapshot_cooldowns(agent) == {}
    store = fallback._get_cooldown_store(agent)
    store["a/b"] = time.monotonic() + 1.0
    fallback._save_cooldown_store(agent, store)
    assert set(fallback.snapshot_cooldowns(agent)) == {"a/b"}
    assert fallback.clear_all_cooldowns(agent) == 1
    assert fallback.snapshot_cooldowns(agent) == {}