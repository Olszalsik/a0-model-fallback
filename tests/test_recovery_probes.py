"""v2.9.0 recovery-probe tests.

Background recovery probes: when a cooldown is booked, the failed
candidate's model wrapper is registered and a background sweep pings
labels whose REMAINING cooldown exceeds the threshold. A successful
probe clears the cooldown early (in place) and marks the label healthy.

Run from repo root:  python -m pytest usr/plugins/_model_fallback/tests/test_recovery_probes.py -q
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

from usr.plugins._model_fallback import fallback as fb  # noqa: E402
from usr.plugins._model_fallback.helpers import recovery_probe as rp  # noqa: E402


class FakeAgent:
    def __init__(self, ctx_id: str = "probe-test-agent"):
        self._data = {}
        self.context = type("Ctx", (), {"id": ctx_id})()

    def get_data(self, key):
        return self._data.get(key)

    def set_data(self, key, value):
        self._data[key] = value


class FakeModel:
    """unified_call-compatible stand-in."""

    def __init__(self, *, delay: float = 0.0, raise_exc: Exception | None = None):
        self.delay = delay
        self.raise_exc = raise_exc
        self.calls: list[dict] = []

    def unified_call(self, **kwargs):
        self.calls.append(kwargs)

        async def _coro():
            if self.delay:
                await asyncio.sleep(self.delay)
            if self.raise_exc is not None:
                raise self.raise_exc
            return "pong"

        return _coro()


@pytest.fixture(autouse=True)
def _clean_probe_state():
    rp.shutdown_probes()
    yield
    rp.shutdown_probes()


# ---------------------------------------------------------------------------
# Registration + lifecycle
# ---------------------------------------------------------------------------


def test_register_creates_target_and_snapshot():
    agent, model = FakeAgent(), FakeModel()
    rp.register_probe_target(agent, "prov/model-a", model, "https://api.example.com")
    assert rp.snapshot()["targets"] == 1
    # Key shape mirrors _INMEM_COOLDOWNS: ((str(context.id),), label).
    assert (("probe-test-agent",), "prov/model-a") in set(rp._TARGETS)


def test_register_ignores_invalid_input():
    rp.register_probe_target(None, "prov/model-a", FakeModel())
    rp.register_probe_target(FakeAgent(), "", FakeModel())
    rp.register_probe_target(FakeAgent(), "<invalid spec: model is list>", None)
    assert rp.snapshot()["targets"] == 0


def test_register_starts_no_task_without_running_loop():
    # Outside asyncio there is no running loop: the task must stay None
    # and the call must not raise (sync booking contexts).
    agent, model = FakeAgent(), FakeModel()
    rp.register_probe_target(agent, "prov/model-a", model)
    assert rp.snapshot()["loop_alive"] is False


def test_shutdown_clears_registry():
    agent, model = FakeAgent(), FakeModel()
    rp.register_probe_target(agent, "prov/model-a", model)
    rp.shutdown_probes()
    assert rp.snapshot()["targets"] == 0


def test_handle_error_cooldown_registers_probe_target(monkeypatch):
    agent = FakeAgent()
    store: dict = {}
    monkeypatch.setattr(fb, "_get_cooldown_store", lambda a: store)
    monkeypatch.setattr(fb, "_save_cooldown_store", lambda a, s: None)
    monkeypatch.setattr(fb, "_record_last_status", lambda *a, **k: None)
    monkeypatch.setattr(fb, "_get_plugin_cfg", lambda a: {})
    monkeypatch.setattr(fb, "_classify_capacity", lambda *a, **k: "free_per_minute")
    monkeypatch.setattr(fb, "_is_rate_limited_error", lambda e: False)
    monkeypatch.setattr(fb, "_is_context_overflow_error", lambda e: False)
    monkeypatch.setattr(fb, "_is_permanently_failed_model", lambda e: False)
    model = FakeModel()
    fb._handle_error_cooldown(
        RuntimeError("boom"), "prov/model-a", store, agent,
        "https://api.example.com", probe_model=model,
    )
    assert rp.snapshot()["targets"] == 1
    # Without probe_model (e.g. code-error / _60 safety-net sites) no target.
    fb._handle_error_cooldown(RuntimeError("boom"), "prov/model-b", store, agent, "")
    assert rp.snapshot()["targets"] == 1


# ---------------------------------------------------------------------------
# Sweep behavior
# ---------------------------------------------------------------------------


def test_sweep_clears_cooldown_on_successful_probe(monkeypatch):
    agent, model = FakeAgent(), FakeModel()
    rp.register_probe_target(agent, "prov/model-a", model)
    store = fb._get_cooldown_store(agent)
    store["prov/model-a"] = time.monotonic() + 300.0  # long remaining

    monkeypatch.setattr(rp, "_cfg", lambda: {**rp._DEFAULTS})

    attempted = asyncio.run(rp.sweep())
    assert attempted == 1
    assert model.calls, "probe must call unified_call through the wrapper"
    assert model.calls[0]["rate_limiter_callback"] is None
    assert model.calls[0]["response_callback"] is None
    assert "prov/model-a" not in store, "successful probe clears early"
    snap = rp.snapshot()
    assert snap["probes_succeeded"] == 1
    assert snap["cooldowns_cleared_early"] == 1
    assert snap["targets"] == 0, "cleared target is dropped"


def test_sweep_keeps_cooldown_on_failed_probe(monkeypatch):
    agent = FakeAgent()
    model = FakeModel(raise_exc=RuntimeError("still down"))
    rp.register_probe_target(agent, "prov/model-a", model)
    store = fb._get_cooldown_store(agent)
    until = time.monotonic() + 300.0
    store["prov/model-a"] = until

    monkeypatch.setattr(rp, "_cfg", lambda: {**rp._DEFAULTS})

    asyncio.run(rp.sweep())
    assert store["prov/model-a"] == until, "failed probe must not touch TTL"
    assert rp.snapshot()["probes_failed"] == 1
    assert rp.snapshot()["targets"] == 1, "still-cooling target stays registered"


def test_sweep_skips_short_cooldowns_and_dead_labels(monkeypatch):
    agent_a, agent_b = FakeAgent("probe-s-a"), FakeAgent("probe-s-b")
    model_a, model_b = FakeModel(), FakeModel()
    rp.register_probe_target(agent_a, "prov/short", model_a)
    rp.register_probe_target(agent_b, "prov/dead", model_b)
    store_a = fb._get_cooldown_store(agent_a)
    store_b = fb._get_cooldown_store(agent_b)
    store_a["prov/short"] = time.monotonic() + 30.0  # below min_cooldown_s
    store_b["prov/dead"] = time.monotonic() + 300.0

    monkeypatch.setattr(rp, "_cfg", lambda: {**rp._DEFAULTS})
    monkeypatch.setattr(fb, "_is_label_dead", lambda label: label == "prov/dead")

    attempted = asyncio.run(rp.sweep())
    assert attempted == 0
    assert not model_a.calls and not model_b.calls
    assert "prov/dead" in store_b  # untouched
    assert rp.snapshot()["targets"] == 2


def test_sweep_drops_target_when_not_in_cooldown(monkeypatch):
    # Unique context id: sibling tests leave cooldowns in the shared
    # "probe-test-agent" store, and a same-key registration would inherit
    # one and probe instead of dropping.
    agent, model = FakeAgent("probe-drops-agent"), FakeModel()
    rp.register_probe_target(agent, "prov/drops-a", model)
    monkeypatch.setattr(rp, "_cfg", lambda: {**rp._DEFAULTS})

    asyncio.run(rp.sweep())  # no cooldown seeded -> nothing to probe
    assert rp.snapshot()["targets"] == 0
    assert not model.calls


def test_sweep_budget_caps_probes_per_cycle(monkeypatch):
    agents = [FakeAgent(f"probe-budget-{i}") for i in range(4)]
    for i, agent in enumerate(agents):
        model = FakeModel()
        rp.register_probe_target(agent, f"prov/m{i}", model)
        fb._get_cooldown_store(agent)["prov/m%d" % i] = time.monotonic() + 300.0

    monkeypatch.setattr(rp, "_cfg", lambda: {**rp._DEFAULTS})

    attempted = asyncio.run(rp.sweep())
    assert attempted == rp._DEFAULTS["recovery_probe_max_targets_per_cycle"]
    assert rp.snapshot()["targets"] == 2  # 2 succeeded+dropped, 2 remain


def test_sweep_disabled_config_is_noop(monkeypatch):
    agent, model = FakeAgent(), FakeModel()
    rp.register_probe_target(agent, "prov/model-a", model)
    store = fb._get_cooldown_store(agent)
    store["prov/model-a"] = time.monotonic() + 300.0
    monkeypatch.setattr(
        rp, "_cfg",
        lambda: {**rp._DEFAULTS, "recovery_probe_enabled": False},
    )
    assert asyncio.run(rp.sweep()) == 0
    assert not model.calls


def test_probe_once_timeout_returns_false():
    model = FakeModel(delay=30.0)
    assert asyncio.run(rp._probe_once(model, timeout_s=0.2)) is False


def test_probe_once_success():
    model = FakeModel()
    assert asyncio.run(rp._probe_once(model, timeout_s=5.0)) is True


def test_probe_once_does_not_leak_unawaited_coroutine():
    import warnings

    model = FakeModel(delay=30.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        asyncio.run(rp._probe_once(model, timeout_s=0.2))