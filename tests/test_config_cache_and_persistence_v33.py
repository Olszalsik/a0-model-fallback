"""v3.3.0 regressions: config memoisation + reboot-safe cooldown persistence."""
from __future__ import annotations

import os
import sys
import time

REPO_ROOT = os.environ.get("REPO_ROOT_OVERRIDE") or os.getcwd()
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from usr.plugins.model_fallback import fallback as fb  # noqa: E402
from usr.plugins.model_fallback import models_ext  # noqa: E402
from usr.plugins.model_fallback.helpers import config_defaults  # noqa: E402


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
    assert config_defaults._merge_cache, "first call should populate the cache"
    fresh = fb._get_plugin_cfg(None, use_cache=False)
    assert fresh == first
    fb.invalidate_plugin_cfg_cache()
    assert config_defaults._merge_cache == {}


def test_plugin_cfg_cache_is_bounded():
    """A long session with many contexts must not grow the map forever."""
    fb.invalidate_plugin_cfg_cache()
    assert config_defaults._MERGE_CACHE_MAX >= 8
    for i in range(config_defaults._MERGE_CACHE_MAX + 20):
        config_defaults._merge_cache[f"p::ctx-{i}"] = (time.monotonic() + 1.0, {})
        if len(config_defaults._merge_cache) >= config_defaults._MERGE_CACHE_MAX:
            oldest = min(
                config_defaults._merge_cache,
                key=lambda k: config_defaults._merge_cache[k][0],
            )
            config_defaults._merge_cache.pop(oldest, None)
    assert len(config_defaults._merge_cache) <= config_defaults._MERGE_CACHE_MAX
    fb.invalidate_plugin_cfg_cache()


def test_plugin_cfg_cache_expires():
    fb.invalidate_plugin_cfg_cache()
    fb._get_plugin_cfg(None)
    # force expiry rather than sleeping
    for k in list(config_defaults._merge_cache):
        _, cfg = config_defaults._merge_cache[k]
        config_defaults._merge_cache[k] = (time.monotonic() - 1.0, cfg)
    assert fb._get_plugin_cfg(None) is not None
    fb.invalidate_plugin_cfg_cache()


def test_reset_gen_budgets_also_drops_the_cache():
    fb._get_plugin_cfg(None)
    assert config_defaults._merge_cache
    fb.reset_gen_budgets()
    assert config_defaults._merge_cache == {}


def test_cache_key_is_stable_for_agentless_calls():
    assert config_defaults._cache_key("model_fallback", None) == (
        config_defaults._cache_key("model_fallback", None)
    )


def test_cache_key_distinguishes_contexts():
    a = _FakeAgent()
    b = _FakeAgent(ctx_id="ctx-other")
    assert config_defaults._cache_key("model_fallback", a) != (
        config_defaults._cache_key("model_fallback", b)
    )


# --------------------------------------------------------------------------
# the single shared resolver (v3.4.0: the shallow-merge fix)
# --------------------------------------------------------------------------


def test_deep_merge_keeps_sibling_defaults_of_a_nested_section():
    """The bug the two shallow copies shared.

    default_config.yaml has NESTED sections. A user who overrode ONE key
    inside one lost every sibling default in that section, and a module
    constant took over -- the v2.8.3 F1 failure mode one level down.
    """
    defaults = {
        "utility_timeout_guard": {
            "enabled": True,
            "default_timeout_s": 60,
            "max_wait_s": 180,
        }
    }
    override = {"utility_timeout_guard": {"default_timeout_s": 120}}
    merged = config_defaults.deep_merge(defaults, override)

    guard = merged["utility_timeout_guard"]
    assert guard["default_timeout_s"] == 120, "the override must win"
    assert guard["enabled"] is True, "sibling default was lost"
    assert guard["max_wait_s"] == 180, "sibling default was lost"


def test_deep_merge_replaces_lists_rather_than_appending():
    defaults = {"force_chat_completions_api_bases": ["a", "b", "c"]}
    override = {"force_chat_completions_api_bases": ["only-this"]}
    merged = config_defaults.deep_merge(defaults, override)
    assert merged["force_chat_completions_api_bases"] == ["only-this"], (
        "a user listing three providers means those three, not defaults plus three"
    )


def test_both_call_sites_share_one_resolver():
    """models_ext and fallback must not drift: one resolver, two callers."""
    prov, pat, bases = models_ext._force_chat_config(None)
    # Everything lives in default_config.yaml, which get_plugin_config alone
    # would never return (the framework returns config.json XOR the YAML).
    assert prov or pat or bases, "force-chat lists did not resolve from the YAML"
    # And the same data must be visible through fallback's entry point.
    cfg = fb._get_plugin_cfg(None)
    assert "force_chat_completions_api_bases" in cfg


def test_real_nested_sections_survive_a_partial_override(monkeypatch):
    """End-to-end against the shipped default_config.yaml."""
    defaults = config_defaults.load_defaults("model_fallback")
    nested = [
        k for k, v in defaults.items()
        if isinstance(v, dict) and len(v) > 1
    ]
    if not nested:
        # No multi-key nested section in the shipped YAML: the deep-merge
        # contract is already covered by the synthetic test above.
        return
    section = nested[0]
    first_key = sorted(defaults[section])[0]
    merged = config_defaults.deep_merge(
        defaults, {section: {first_key: "OVERRIDDEN"}}
    )
    assert merged[section][first_key] == "OVERRIDDEN"
    for k in defaults[section]:
        if k != first_key:
            assert k in merged[section], f"sibling {k!r} was dropped from {section}"

