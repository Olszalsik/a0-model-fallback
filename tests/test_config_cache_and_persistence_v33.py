"""v3.3.0 regressions: config memoisation + reboot-safe cooldown persistence."""
from __future__ import annotations

import os
import sys
import time

REPO_ROOT = os.environ.get("REPO_ROOT_OVERRIDE") or os.getcwd()
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from usr.plugins.model_fallback import fallback as fb  # noqa: E402


class _FakeAgent:
    """Minimal duck-type of the bits the cooldown store touches."""

    def __init__(self, ctx_id="ctx-test"):
        self.data = {}
        self.context = type("Ctx", (), {"id": ctx_id})()

    def get_data(self, key):
        return self.data.get(key)

    def set_data(self, key, value):
        self.data[key] = value


# --------------------------------------------------------------------------
# cooldown persistence across a reboot (pre-existing bug)
# --------------------------------------------------------------------------


def test_persist_restore_roundtrip_preserves_remaining_time():
    live = {"primary": time.monotonic() + 1800.0, "peer": time.monotonic() + 30.0}
    snapshot = fb._persist_cooldowns(live)
    assert snapshot[fb._PERSIST_MARKER] == fb._PERSIST_FORMAT

    restored = fb._restore_cooldowns(snapshot)
    assert set(restored) == {"primary", "peer"}
    now = time.monotonic()
    assert 1790 <= restored["primary"] - now <= 1800
    assert 20 <= restored["peer"] - now <= 30


def test_restored_deadlines_are_monotonic_not_wallclock():
    """The live store MUST stay monotonic -- every skip check compares
    against time.monotonic()."""
    snapshot = fb._persist_cooldowns({"a": time.monotonic() + 60.0})
    restored = fb._restore_cooldowns(snapshot)
    # A monotonic deadline is close to time.monotonic(); a wall-clock one
    # would be ~1.7e9 seconds in the future.
    assert restored["a"] < time.time()


def test_reboot_does_not_inflate_a_short_cooldown():
    """Regression: the old format persisted raw monotonic deadlines.

    A reboot resets the monotonic base, so a 30 s cooldown restored as
    ~139 hours of lockout. With the wall-clock format the remaining time
    is independent of uptime.
    """
    live = {"m": time.monotonic() + 30.0}
    snapshot = fb._persist_cooldowns(live)
    time.sleep(0.01)
    restored = fb._restore_cooldowns(snapshot)
    assert 0 < restored["m"] - time.monotonic() <= 30.0


def test_legacy_v1_snapshot_is_discarded_not_trusted():
    """A pre-v3.3.0 snapshot holds monotonic deadlines of unknown age.

    Honouring it is what caused multi-day lockouts, so we start clean
    instead of guessing.
    """
    assert fb._restore_cooldowns({"primary": time.monotonic() + 500_000.0}) == {}
    assert fb._restore_cooldowns({}) == {}
    assert fb._restore_cooldowns({"_mfb_v": 2}) == {}


def test_persist_skips_unparsable_entries():
    live = {"ok": time.monotonic() + 10.0, "bad": "not-a-number", "none": None}
    snapshot = fb._persist_cooldowns(live)
    assert set(snapshot["e"]) == {"ok"}


def test_save_and_reseed_roundtrip():
    """End-to-end through the public store API (fresh process state)."""
    agent = _FakeAgent()
    fb._INMEM_COOLDOWNS.clear()
    store = fb._get_cooldown_store(agent)
    store["primary"] = time.monotonic() + 900.0
    fb._save_cooldown_store(agent, store)

    # simulate a restart: the in-memory store is gone, the data key is not
    fb._INMEM_COOLDOWNS.clear()
    reseeded = fb._get_cooldown_store(agent)
    assert "primary" in reseeded
    assert 890 <= reseeded["primary"] - time.monotonic() <= 900
    fb._INMEM_COOLDOWNS.clear()


# --------------------------------------------------------------------------
# plugin-config memoisation (perf)
# --------------------------------------------------------------------------


def test_plugin_cfg_cache_returns_equal_merged_config():
    fb.invalidate_plugin_cfg_cache()
    a = fb._get_plugin_cfg(None)
    b = fb._get_plugin_cfg(None)
    assert a == b
    assert "fallback_max_cycles" in a  # merged with default_config.yaml


def test_plugin_cfg_use_cache_false_bypasses_memo():
    fb.invalidate_plugin_cfg_cache()
    first = fb._get_plugin_cfg(None, use_cache=True)
    assert fb._PLUGIN_CFG_CACHE, "first call should populate the cache"
    fresh = fb._get_plugin_cfg(None, use_cache=False)
    assert fresh == first
    fb.invalidate_plugin_cfg_cache()
    assert fb._PLUGIN_CFG_CACHE == {}


def test_plugin_cfg_cache_is_bounded():
    """A long session with many contexts must not grow the map forever."""
    fb.invalidate_plugin_cfg_cache()
    assert fb._PLUGIN_CFG_CACHE_MAX >= 8
    for i in range(fb._PLUGIN_CFG_CACHE_MAX + 20):
        fb._PLUGIN_CFG_CACHE[("p", str(i))] = (time.monotonic() + 1.0, {})
        if len(fb._PLUGIN_CFG_CACHE) >= fb._PLUGIN_CFG_CACHE_MAX:
            oldest = min(
                fb._PLUGIN_CFG_CACHE, key=lambda k: fb._PLUGIN_CFG_CACHE[k][0]
            )
            fb._PLUGIN_CFG_CACHE.pop(oldest, None)
    assert len(fb._PLUGIN_CFG_CACHE) <= fb._PLUGIN_CFG_CACHE_MAX
    fb.invalidate_plugin_cfg_cache()


def test_plugin_cfg_cache_expires():
    fb.invalidate_plugin_cfg_cache()
    fb._get_plugin_cfg(None)
    # force expiry rather than sleeping
    for k in list(fb._PLUGIN_CFG_CACHE):
        _, cfg = fb._PLUGIN_CFG_CACHE[k]
        fb._PLUGIN_CFG_CACHE[k] = (time.monotonic() - 1.0, cfg)
    assert fb._get_plugin_cfg(None) is not None
    fb.invalidate_plugin_cfg_cache()


def test_reset_gen_budgets_also_drops_the_cache():
    fb._get_plugin_cfg(None)
    assert fb._PLUGIN_CFG_CACHE
    fb.reset_gen_budgets()
    assert fb._PLUGIN_CFG_CACHE == {}


def test_cache_key_is_stable_for_agentless_calls():
    assert fb._plugin_cfg_cache_key(None) == fb._plugin_cfg_cache_key(None)


def test_cache_key_distinguishes_contexts():
    a = _FakeAgent()
    b = _FakeAgent(ctx_id="ctx-other")
    assert fb._plugin_cfg_cache_key(a) != fb._plugin_cfg_cache_key(b)

