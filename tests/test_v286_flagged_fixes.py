"""v2.8.6 round-3 flagged-fix tests.

Covers the five items the v2.8.5 audit flagged but did not fix:

  T1  (X) _is_permanent_for_rotation: a 429 is cooldown-permanent but NOT
          rotation-permanent -- the cycle_permanent_count sites and the
          turn n<=1 fail-fast gate use the new helper, so a rate-limited
          error falls through to the structured RetryAfterHours raise
          instead of re-raising the raw exception.
  T2  (Z) primary-strike counters decay: new primary_strike_decay_s knob
          (default 300s, 0 = off) in both cascades + the turn path
          (persisted timestamp DATA_KEY_TURN_PRIMARY_FAILS_AT).
  T3  (V) install_chat_turn_patch is version-aware: a stale turn cascade
          from an older plugin version is re-assigned, and the captured
          original is never overwritten by a re-assign.
  T4  (N) _00_install_fallback_patches.uninstall() restores the TRUE
          originals (utility/chat/turn), not just the top wrapper.
  T5  (wiring#11) clear_all_cooldowns(cross_context=True) clears every
          context's in-memory store, not just the caller's.

Run from repo root:  python -m pytest usr/plugins/model_fallback/tests/test_v286_flagged_fixes.py -q
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(
    os.environ.get("REPO_ROOT_OVERRIDE")
    or Path(__file__).resolve().parents[4]
)
sys.path.insert(0, str(REPO_ROOT))

from usr.plugins.model_fallback import fallback as fb  # noqa: E402


class FakeAgent:
    """Minimal agent stand-in: get_data/set_data + context.id pattern."""

    def __init__(self, ctx_id: str = "test-agent-v286"):
        self._data = {}
        self.context = type("Ctx", (), {"id": ctx_id})()

    def get_data(self, key):
        return self._data.get(key)

    def set_data(self, key, value):
        self._data[key] = value


# ---------------------------------------------------------------------------
# T1 (X) — rotation-permanent vs cooldown-permanent
# ---------------------------------------------------------------------------


def test_rotation_permanent_excludes_rate_limited():
    # "HTTP 429" in the message -> _is_rate_limited_error True ->
    # _is_permanently_failed_model True, but rotation must treat it as
    # transient so the RetryAfterHours path stays reachable.
    e429 = Exception("HTTP 429 too many requests")
    assert fb._is_rate_limited_error(e429)
    assert fb._is_permanently_failed_model(e429)
    assert not fb._is_permanent_for_rotation(e429)


def test_rotation_permanent_keeps_genuinely_permanent():
    # 404 gone / invalid key shapes are permanent for BOTH purposes.
    e404 = Exception("model not found")
    e404.status_code = 404
    assert fb._is_permanent_for_rotation(e404)
    # Plain unknown errors are not permanent at all.
    assert not fb._is_permanent_for_rotation(RuntimeError("boom"))


def test_rotation_sites_use_new_helper():
    src = (Path(__file__).resolve().parent.parent / "fallback.py").read_text(
        encoding="utf-8"
    )
    assert src.count("if _is_permanent_for_rotation(e):") == 1, (
        "the unified rotation engine's cycle_permanent_count site must use "
        "the rotation-permanent helper (v3.0.0: utility+chat share ONE "
        "engine; the turn cascade's gate is a different expression and "
        "is asserted below)"
    )
    turn_start = src.index("async def _patched_call_chat_model_turn")
    turn_src = src[turn_start:]
    assert "_is_permanent_for_rotation(e) and _classify_capacity(" in turn_src, (
        "the turn n<=1 fail-fast gate must exclude rate-limited errors"
    )


# ---------------------------------------------------------------------------
# T2 (Z) — strike decay
# ---------------------------------------------------------------------------


def test_strike_decay_knob_and_persisted_timestamp():
    root = Path(__file__).resolve().parent.parent
    yaml_src = (root / "default_config.yaml").read_text(encoding="utf-8")
    assert "primary_strike_decay_s: 300" in yaml_src
    py_src = (root / "fallback.py").read_text(encoding="utf-8")
    # Knob resolved in the unified engine + read in the turn path + decay
    # guard (v3.0.0: the two cascades merged, so the site count dropped).
    assert py_src.count("primary_strike_decay_s") >= 3
    assert 'DATA_KEY_TURN_PRIMARY_FAILS_AT = "mfb_turn_primary_fails_at"' in py_src
    turn_start = py_src.index("async def _patched_call_chat_model_turn")
    turn_src = py_src[turn_start:]
    assert "DATA_KEY_TURN_PRIMARY_FAILS_AT" in turn_src, (
        "the turn increment must persist the decay timestamp"
    )


# ---------------------------------------------------------------------------
# T3 (V) — version-aware turn patch install
# ---------------------------------------------------------------------------


def test_install_turn_patch_version_stamp(monkeypatch):
    saved = fb._ORIGINAL_CALL_CHAT_MODEL_TURN
    try:

        class _Cls:
            pass

        def _orig_turn(self, *a, **k):
            return "orig"

        _Cls.call_chat_model_turn = _orig_turn

        # First install captures the original.
        assert fb.install_chat_turn_patch(_Cls, version="2.8.6") is True
        assert fb._ORIGINAL_CALL_CHAT_MODEL_TURN is _orig_turn
        assert _Cls.call_chat_model_turn is fb._patched_call_chat_model_turn

        # Same version -> idempotent no-op.
        assert fb.install_chat_turn_patch(_Cls, version="2.8.6") is False
        assert fb._ORIGINAL_CALL_CHAT_MODEL_TURN is _orig_turn

        # Version bump -> re-assign WITHOUT overwriting the original.
        assert fb.install_chat_turn_patch(_Cls, version="2.8.7") is True
        assert fb._ORIGINAL_CALL_CHAT_MODEL_TURN is _orig_turn
        assert getattr(
            _Cls.call_chat_model_turn, "_fallback_turn_version", ""
        ) == "2.8.7"

        # No version (legacy callers) -> still idempotent.
        assert fb.install_chat_turn_patch(_Cls) is False
    finally:
        fb._ORIGINAL_CALL_CHAT_MODEL_TURN = saved


# ---------------------------------------------------------------------------
# T4 (N) — full uninstall chain
# ---------------------------------------------------------------------------


def test_agent_init_uninstall_restores_originals(monkeypatch):
    import importlib.util

    ext_path = (
        Path(__file__).resolve().parent.parent
        / "extensions"
        / "python"
        / "agent_init"
        / "_00_install_fallback_patches.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_00_install_fallback_patches_v286_test", ext_path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    class _DummyAgent:
        pass

    def _orig_u(self, *a, **k):
        return "u"

    def _orig_c(self, *a, **k):
        return "c"

    _DummyAgent.call_utility_model = _orig_u
    _DummyAgent.call_chat_model = _orig_c
    monkeypatch.setattr(mod, "Agent", _DummyAgent)

    # Simulate what execute() does: capture originals, then install ours.
    _DummyAgent._mfb_original_call_utility_model = _orig_u
    _DummyAgent._mfb_original_call_chat_model = _orig_u  # any non-None original
    _DummyAgent.call_utility_model = lambda self, *a, **k: "patched"

    assert mod.uninstall() is True
    assert _DummyAgent.call_utility_model is _orig_u
    assert _DummyAgent.call_chat_model is _orig_u
    assert not hasattr(_DummyAgent, "_mfb_original_call_utility_model")
    assert not hasattr(_DummyAgent, "_mfb_original_call_chat_model")


def test_hooks_uninstall_prefers_full_chain():
    src = (
        Path(__file__).resolve().parent.parent / "hooks.py"
    ).read_text(encoding="utf-8")
    assert "_00_install_fallback_patches" in src
    assert "_00_install_fallback_patches.uninstall()" in src


# ---------------------------------------------------------------------------
# T5 (wiring#11) — cross-context cooldown clear
# ---------------------------------------------------------------------------


def test_clear_all_cooldowns_cross_context():
    key_a, key_b = ("ctx-v286-a",), ("ctx-v286-b",)
    fb._INMEM_COOLDOWNS[key_a] = {"m1": 1.0, "m2": 2.0}
    fb._INMEM_COOLDOWNS[key_b] = {"m3": 3.0}
    try:
        # cross_context=True wipes EVERY store (other test files may have
        # seeded their own entries; `before` is computed after seeding and
        # the definitive assertions are the empty stores below).
        before = sum(
            len(s) for s in fb._INMEM_COOLDOWNS.values() if isinstance(s, dict)
        )
        cleared = fb.clear_all_cooldowns(FakeAgent(key_a[0]), cross_context=True)
        assert cleared == before
        assert fb._INMEM_COOLDOWNS[key_a] == {}
        assert fb._INMEM_COOLDOWNS[key_b] == {}

        # Default (per-context) clears only the caller's store.
        fb._INMEM_COOLDOWNS[key_a] = {"m1": 1.0}
        fb._INMEM_COOLDOWNS[key_b] = {"m3": 3.0}
        cleared = fb.clear_all_cooldowns(FakeAgent(key_a[0]))
        assert cleared == 1
        assert fb._INMEM_COOLDOWNS[key_a] == {}
        assert fb._INMEM_COOLDOWNS[key_b] == {"m3": 3.0}
    finally:
        fb._INMEM_COOLDOWNS.pop(key_a, None)
        fb._INMEM_COOLDOWNS.pop(key_b, None)


def test_hooks_clear_cooldowns_passes_cross_context():
    src = (
        Path(__file__).resolve().parent.parent / "hooks.py"
    ).read_text(encoding="utf-8")
    assert "cross_context=bool(cross_context)" in src
    assert "def clear_cooldowns(agent, cross_context: bool = False):" in src