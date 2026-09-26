"""v3.1.0 latency-adaptive timeout tests.

helpers/latency.py records per-label successful-call durations; the cold
tier of ``_resolve_per_call_timeout`` sizes its budget from observed
p95 x margin, clamped to [latency_floor_s, base_timeout_s]. Conservative
by design: routers and unlimited_paid return before the adaptive path,
the warm fast-path still wins, samples accumulate only on success, and a
genuine timeout CLEARS the label's samples (one bad sizing self-corrects
to the full base).

Run from repo root:
    python -m pytest usr/plugins/model_fallback/tests/test_latency_adaptive_v310.py -q
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(
    os.environ.get("REPO_ROOT_OVERRIDE")
    or Path(__file__).resolve().parents[4]
)
sys.path.insert(0, str(REPO_ROOT))

from usr.plugins.model_fallback import fallback as fb  # noqa: E402
from usr.plugins.model_fallback.helpers import events  # noqa: E402
from usr.plugins.model_fallback.helpers import latency  # noqa: E402
from usr.plugins.model_fallback.helpers import recovery_probe as rp  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_latency_state():
    latency.reset()
    events.reset_events()
    rp.shutdown_probes()
    rp.reset_counters()
    yield
    latency.reset()
    events.reset_events()
    rp.shutdown_probes()
    # test_recovery_probes asserts on these same module-global counters.
    rp.reset_counters()


# ---------------------------------------------------------------------------
# Unit: helpers/latency.py
# ---------------------------------------------------------------------------


def test_record_bounds_window_to_max_samples():
    for i in range(30):
        latency.record("m/a", float(i + 1), max_samples=20)
    snap = latency.snapshot()
    assert snap["m/a"]["samples"] == 20, "rolling window keeps the newest 20"
    # Oldest (1..10) dropped; last recorded sample is 30.
    assert snap["m/a"]["last_s"] == 30.0


def test_record_disabled_is_a_noop():
    latency.record("m/a", 1.0, enabled=False)
    assert latency.snapshot() == {}


def test_record_never_raises_on_garbage():
    latency.record("", -5.0)
    latency.record(None, "not-a-number")  # type: ignore[arg-type]
    # Negative durations are rejected, garbage coerced/dropped -- either
    # way no exception escapes and state stays sane.
    assert latency.sample_count("") in (0, 1)


def test_p95_sizing_clamp_math():
    # 18 fast samples + one 9s outlier (n=19): nearest-rank p95 is the
    # 19th sorted value = 9s.
    for i in range(18):
        latency.record("m/p95", 1.0)
    latency.record("m/p95", 9.0)
    # p95=9 -> est=13.5 -> clamped UP to the floor (45 default) -- the
    # adaptive budget is never sized below latency_floor_s.
    est = latency.p95_timeout_s("m/p95", base_timeout_s=300.0)
    assert est == 45.0
    # Base respected when p95 is large: p95=250 -> 375 -> clamped to 300.
    latency.reset()
    for i in range(19):
        latency.record("m/big", 250.0)
    assert latency.p95_timeout_s("m/big", base_timeout_s=300.0) == 300.0
    # Floor respected when p95 is tiny: p95=0.5 -> 0.75 -> clamped to 45.
    latency.reset()
    for i in range(5):
        latency.record("m/tiny", 0.5)
    assert latency.p95_timeout_s("m/tiny", base_timeout_s=300.0) == 45.0
    # Floor honored as a real knob: floor 1.0 -> est 13.5 survives.
    latency.reset()
    for i in range(18):
        latency.record("m/p95", 1.0)
    latency.record("m/p95", 9.0)
    assert latency.p95_timeout_s("m/p95", base_timeout_s=300.0, floor_s=1.0) == pytest.approx(13.5)


def test_p95_insufficient_samples_returns_none():
    for i in range(4):
        latency.record("m/few", 1.0)
    assert latency.p95_timeout_s("m/few", min_samples=5) is None
    # Exactly min_samples passes: p95=1 -> 1.5 -> floor 45 (default floor).
    latency.record("m/few", 1.0)
    assert latency.p95_timeout_s("m/few", min_samples=5) == 45.0
    # With a tiny floor the raw margin math is visible.
    assert latency.p95_timeout_s("m/few", min_samples=5, floor_s=0.0) == pytest.approx(1.5)


def test_p95_kill_switch_returns_none():
    for i in range(10):
        latency.record("m/off", 1.0)
    assert latency.p95_timeout_s("m/off", enabled=False) is None


def test_clear_label_drops_samples():
    for i in range(6):
        latency.record("m/x", 1.0)
    latency.clear_label("m/x")
    assert latency.sample_count("m/x") == 0
    assert latency.p95_timeout_s("m/x") is None


# ---------------------------------------------------------------------------
# Integration: _resolve_per_call_timeout precedence
# ---------------------------------------------------------------------------

LAT_CFG = {
    "latency_adaptive_enabled": True,
    "latency_min_samples": 5,
    "latency_p95_margin": 1.5,
    "latency_floor_s": 45.0,
    "latency_max_samples": 20,
}


class FakeLog:
    def log(self, type="info", content="", **kwargs):
        pass


@pytest.fixture()
def resolver_env(monkeypatch):
    fb._WARM_LABELS.clear()
    monkeypatch.setattr(
        fb, "_get_plugin_cfg", lambda agent: dict(LAT_CFG), raising=True
    )
    yield monkeypatch


def _resolve(label, base=300.0, warm=20.0, window=600.0, api_base="", allow_warm=True, agent=None):
    return fb._resolve_per_call_timeout(
        label, base, warm, window, agent, api_base, allow_warm
    )


def test_cold_label_with_samples_gets_adaptive_budget(resolver_env):
    for i in range(6):
        latency.record("prov/fast", 2.0)
    # p95=2 -> 3 -> floor 45.
    assert _resolve("prov/fast") == 45.0


def test_cold_label_without_samples_gets_base(resolver_env):
    assert _resolve("prov/cold") == 300.0


def test_warm_label_still_wins_over_adaptive(resolver_env):
    for i in range(6):
        latency.record("prov/warm", 2.0)
    fb._WARM_LABELS["prov/warm"] = time.monotonic()
    assert _resolve("prov/warm") == 20.0


def test_router_budgets_unaffected_by_samples(resolver_env):
    for i in range(10):
        latency.record("omniroute/auto", 2.0)
    resolver_env.setattr(
        fb, "_classify_capacity",
        lambda *a, **k: "router", raising=True,
    )
    # Cold router: max(base, router_cold_call_timeout_s=150 default).
    assert _resolve("omniroute/auto") == 300.0
    # Warm router: router_budget = base (router_call_timeout_s=0).
    fb._WARM_LABELS["omniroute/auto"] = time.monotonic()
    assert _resolve("omniroute/auto") == 300.0


def test_unlimited_paid_unaffected_by_samples(resolver_env):
    for i in range(10):
        latency.record("a0_venice/big", 2.0)
    resolver_env.setattr(
        fb, "_classify_capacity",
        lambda *a, **k: "unlimited_paid", raising=True,
    )
    assert _resolve("a0_venice/big") == 300.0


def test_kill_switch_restores_exact_v300_behavior(resolver_env):
    for i in range(10):
        latency.record("prov/fast", 2.0)
    resolver_env.setattr(
        fb, "_get_plugin_cfg",
        lambda agent: {**LAT_CFG, "latency_adaptive_enabled": False},
        raising=True,
    )
    assert _resolve("prov/fast") == 300.0


def test_clear_on_timeout_falls_back_to_base(resolver_env):
    for i in range(6):
        latency.record("prov/fast", 2.0)
    assert _resolve("prov/fast") == 45.0
    fb.latency.clear_label("prov/fast")
    assert _resolve("prov/fast") == 300.0


# ---------------------------------------------------------------------------
# Integration: engine recording + clear-on-timeout
# ---------------------------------------------------------------------------


class FakeContext:
    def __init__(self, ctx_id):
        self.id = ctx_id
        self.log = FakeLog()


class FakeAgent:
    def __init__(self, ctx_id, primary):
        self.context = FakeContext(ctx_id)
        self._data = {}
        self._primary = primary
        self.rate_limiter_callback = None

    def get_data(self, key, default=None):
        return self._data.get(key, default)

    def set_data(self, key, value):
        self._data[key] = value

    def get_chat_model(self):
        return self._primary

    def get_utility_model(self):
        return self._primary

    async def validate_tool_request(self, tool_request):
        raise AssertionError("validate_tool_request must not fire on plain text")


class FakeModel:
    def __init__(self, name):
        self.model_name = name
        self.provider = "openai"
        self.fail_with = None
        self.sleep_for = 0.0
        self.kwargs = {"api_key": "sk-test"}
        self.unified_kwargs = []

    def unified_call(self, **kwargs):
        self.unified_kwargs.append(kwargs)

        async def _coro(model=self):
            if model.sleep_for:
                await asyncio.sleep(model.sleep_for)
            if model.fail_with is not None:
                raise model.fail_with
            return ("hello-%s" % model.model_name, "")

        return _coro()


ENGINE_CFG = {
    "fallback_utility_timeout_s": 300,
    "fallback_timeout_s": 300,
    "cascade_warm_timeout_s": 20,
    "cascade_warm_window_s": 600,
    "fallback_cycle_delay": 0,
    "fallback_attempt_delay": 0,
    "fallback_max_cycles": 4,
    "max_cycle_delay_s": 0.01,
    "backoff_jitter_s": 0,
    "extended_retry_enabled": False,
    "continuous_fallback": False,
    "latency_adaptive_enabled": True,
    "latency_min_samples": 5,
    "latency_p95_margin": 1.5,
    "latency_floor_s": 45.0,
    "latency_max_samples": 20,
}


@pytest.fixture()
def engine_env(monkeypatch):
    fb._INMEM_COOLDOWNS.clear()
    fb._WARM_LABELS.clear()
    fb._INMEM_DEAD_LABELS.clear()
    fb._INMEM_HEALTHY_LABELS.clear()
    fb._LABEL_API_BASES.clear()
    latency.reset()
    events.reset_events()
    rp.shutdown_probes()
    rp.reset_counters()

    primary = FakeModel("primary-model")
    models = {"primary-model": primary}
    monkeypatch.setattr(
        fb, "_build_candidates",
        lambda model_obj, use_utility_models, agent: [None],
        raising=True,
    )
    monkeypatch.setattr(
        fb, "_build_model",
        lambda spec, model_obj: model_obj if spec is None else models[spec["model"]],
        raising=True,
    )
    monkeypatch.setattr(
        fb, "_resolve_per_call_timeout",
        lambda label, base, warm, window, agent=None, api_base="", allow_warm=True: 5.0,
        raising=True,
    )
    monkeypatch.setattr(
        fb, "_get_plugin_cfg", lambda agent: dict(ENGINE_CFG), raising=True
    )

    class _FakeExtension:
        def __init__(self):
            self.calls = []

        async def call_extensions_async(self, hook, agent, **kwargs):
            self.calls.append(hook)

    monkeypatch.setattr(fb, "extension", _FakeExtension(), raising=True)
    yield models
    latency.reset()
    events.reset_events()
    rp.shutdown_probes()
    rp.reset_counters()


def test_engine_success_records_latency_sample(engine_env):
    agent = FakeAgent("lat-eng-ok", engine_env["primary-model"])
    result = asyncio.run(
        fb._patched_call_utility_model(agent, "sys", "msg")
    )
    assert result == "hello-primary-model"
    assert latency.sample_count("primary-model") == 1
    snap = latency.snapshot()["primary-model"]
    assert snap["samples"] == 1 and snap["last_s"] >= 0


def test_engine_kill_switch_records_nothing(engine_env, monkeypatch):
    monkeypatch.setattr(
        fb, "_get_plugin_cfg",
        lambda agent: {**ENGINE_CFG, "latency_adaptive_enabled": False},
        raising=True,
    )
    agent = FakeAgent("lat-eng-off", engine_env["primary-model"])
    result = asyncio.run(
        fb._patched_call_utility_model(agent, "sys", "msg")
    )
    assert result == "hello-primary-model"
    assert latency.snapshot() == {}, "kill switch: no state accumulates"


def test_engine_timeout_clears_latency_samples(engine_env, monkeypatch):
    agent = FakeAgent("lat-eng-to", engine_env["primary-model"])
    # Seed samples as if previous calls succeeded fast.
    for i in range(6):
        latency.record("primary-model", 1.0)
    assert latency.sample_count("primary-model") == 6
    # Short budget so the hang trips fast (the fixture patches 5.0).
    monkeypatch.setattr(
        fb, "_resolve_per_call_timeout",
        lambda label, base, warm, window, agent=None, api_base="", allow_warm=True: 0.2,
        raising=True,
    )
    # Model hangs past the budget -> TimeoutError path.
    engine_env["primary-model"].sleep_for = 30.0
    with pytest.raises(RuntimeError, match="skipped|exhausted"):
        asyncio.run(fb._patched_call_utility_model(agent, "sys", "msg"))
    assert latency.sample_count("primary-model") == 0, (
        "a genuine timeout clears the label's samples"
    )


def test_turn_success_records_latency_sample_structural():
    # The turn cascade's success block records next to the _WARM_LABELS
    # write (same window as the cascades). Full turn-fixture setup is
    # test_turn_cascade_v28.py territory; here the wiring is pinned
    # structurally: the turn body must contain the record call inside its
    # success region, after the warm write.
    src = Path(fb.__file__).read_text(encoding="utf-8")
    turn_start = src.index("async def _patched_call_chat_model_turn")
    turn_end = src.index("def install_chat_turn_patch")
    turn_body = src[turn_start:turn_end]
    assert "latency.record(" in turn_body
    assert "latency.clear_label(" in turn_body
    # Record sits AFTER the warm-label write in the success block.
    warm_at = turn_body.index("_WARM_LABELS[label] = time.monotonic()")
    assert turn_body.index("latency.record(", warm_at) > warm_at
    # ...and the clear is gated on the timeout shape (the except block
    # catches every non-code error; clearing on 429/5xx would discard
    # good latency data).
    assert "_is_timeout_shaped(e)" in turn_body