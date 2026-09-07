"""v3.2.0 adaptive generation-budget tests.

A fixed per-call timeout cannot tell a hung connection from a healthy
model writing a long answer -- and guessing wrong kills a mid-generation
call and PUNISHES the label with a cooldown. The v3.2.0 layer learns per
label:

- GROW:  a timeout raises the label's override
         (max(prev, needed) * gen_budget_growth_factor, capped at
         gen_budget_max_s),
- APPLY: the override can only RAISE a resolved budget (any capacity
         class), never shrink it,
- DECAY: every success decays the override (x gen_budget_decay_factor),
         floored at elapsed * gen_budget_success_margin, and the override
         is DELETED once it decays to/below the cold base timeout,
- WALL:  gen_budget_max_s bounds everything.

Sites wired: engine wait_for timeout, engine timeout-shaped branch, turn
path timeout, both success paths, clear_all_cooldowns, hooks.uninstall,
and the outer utility-timeout guard (apply + grow).

Run from repo root:
    python -m pytest usr/plugins/_model_fallback/tests/test_adaptive_gen_budget_v320.py -q
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(
    os.environ.get("REPO_ROOT_OVERRIDE")
    or Path(__file__).resolve().parents[4]
)
sys.path.insert(0, str(REPO_ROOT))

from usr.plugins._model_fallback import fallback as fb  # noqa: E402
from usr.plugins._model_fallback.helpers import events  # noqa: E402
from usr.plugins._model_fallback.helpers import latency  # noqa: E402
from helpers.errors import RepairableException  # noqa: E402
from usr.plugins._model_fallback.helpers import utility_timeout as ut  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_gen_budget_state():
    fb.reset_gen_budgets()
    latency.reset()
    events.reset_events()
    yield
    fb.reset_gen_budgets()
    latency.reset()
    events.reset_events()


@pytest.fixture()
def default_cfg(monkeypatch):
    """Deterministic knob source: the plugin config reads the user's real
    config.json otherwise; force the documented defaults."""
    monkeypatch.setattr(fb, "_get_plugin_cfg", lambda agent=None: {})
    return {}


class _StubAgent:
    """Minimal agent stand-in for clear_all_cooldowns()."""

    def __init__(self):
        self.context = None
        self._data = {}

    def get_data(self, key, default=None):
        return self._data.get(key, default)

    def set_data(self, key, value):
        self._data[key] = value


# ---------------------------------------------------------------------------
# GROW
# ---------------------------------------------------------------------------

def test_grow_compounds_from_needed(default_cfg):
    first = fb._grow_gen_budget("m/a", 60.0, None)
    assert first == pytest.approx(90.0), "60s needed x 1.5 growth"
    second = fb._grow_gen_budget("m/a", 90.0, None)
    assert second == pytest.approx(135.0), "compounds from the raised budget"


def test_grow_caps_at_wall(default_cfg):
    out = fb._grow_gen_budget("m/a", 5000.0, None)
    assert out == 600.0, "gen_budget_max_s is the absolute wall"
    assert fb._GEN_BUDGET_OVERRIDES["m/a"] == 600.0


def test_grow_skips_empty_label_and_zero_signal(default_cfg):
    assert fb._grow_gen_budget("", 60.0, None) is None
    assert fb._grow_gen_budget("m/a", 0.0, None) is None
    assert fb._grow_gen_budget("m/a", None, None) is None
    assert not fb._GEN_BUDGET_OVERRIDES


def test_grow_records_route_event(default_cfg):
    fb._grow_gen_budget("m/a", 60.0, None)
    kinds = [e.get("kind") for e in events.snapshot()]
    assert "gen_budget_grown" in kinds


def test_grow_skips_when_base_covers_need(default_cfg):
    # Stock cold base 300s already covers a 60s need -> no override.
    assert fb._grow_gen_budget("m/a", 60.0, None, base_timeout_s=300.0) is None
    assert not fb._GEN_BUDGET_OVERRIDES
    # Base 80s: 60*1.5=90 exceeds it -> override created.
    out = fb._grow_gen_budget("m/b", 60.0, None, base_timeout_s=80.0)
    assert out == pytest.approx(90.0)
    # A stale override below the (later-raised) base gets cleaned up.
    fb._GEN_BUDGET_OVERRIDES["m/c"] = 90.0
    assert fb._grow_gen_budget("m/c", 10.0, None, base_timeout_s=200.0) is None
    assert "m/c" not in fb._GEN_BUDGET_OVERRIDES


# ---------------------------------------------------------------------------
# APPLY
# ---------------------------------------------------------------------------

def test_apply_raises_any_budget_and_caps(default_cfg):
    fb._GEN_BUDGET_OVERRIDES["m/a"] = 300.0
    assert fb._apply_gen_budget("m/a", 60.0, None) == pytest.approx(300.0)
    assert fb._apply_gen_budget("m/a", 400.0, None) == pytest.approx(400.0), \
        "override never shrinks a bigger resolved budget"
    fb._GEN_BUDGET_OVERRIDES["m/b"] = 900.0
    assert fb._apply_gen_budget("m/b", 60.0, None) == 600.0, \
        "the wall caps even a learned override"


def test_apply_identity_without_override(default_cfg):
    assert fb._apply_gen_budget("m/a", 60.0, None) == pytest.approx(60.0)


def test_kill_switch_disables_grow_apply_and_read(monkeypatch):
    monkeypatch.setattr(
        fb, "_get_plugin_cfg",
        lambda agent=None: {"adaptive_gen_budget_enabled": False},
    )
    assert fb._grow_gen_budget("m/a", 60.0, None) is None
    assert not fb._GEN_BUDGET_OVERRIDES
    fb._GEN_BUDGET_OVERRIDES["m/a"] = 300.0  # stale state from before the toggle
    assert fb._apply_gen_budget("m/a", 60.0, None) == pytest.approx(60.0)
    assert fb._gen_budget_override_for("m/a") == 0.0
    base = fb._resolve_per_call_timeout("m/a", 300.0, 20.0, 600.0, None, "")
    assert base == pytest.approx(300.0)


def test_resolver_wrapper_applies_override(default_cfg, monkeypatch):
    # Cold, non-router label: the base resolution returns the cold base
    # (latency window empty -> no adaptive shrink).
    monkeypatch.setattr(latency, "p95_timeout_s", lambda *a, **k: None)
    fb._GEN_BUDGET_OVERRIDES["m/a"] = 450.0
    out = fb._resolve_per_call_timeout("m/a", 300.0, 20.0, 600.0, None, "")
    assert out == pytest.approx(450.0), "override raises the cold base"
    out2 = fb._resolve_per_call_timeout(
        "m/a", 300.0, 20.0, 600.0, None, "", allow_warm=False,
    )
    assert out2 == pytest.approx(450.0), "turn path (allow_warm=False) too"


# ---------------------------------------------------------------------------
# DECAY
# ---------------------------------------------------------------------------

def test_decay_floors_at_success_margin(default_cfg):
    fb._GEN_BUDGET_OVERRIDES["m/a"] = 90.0
    # Base 50s: a real generation took 80s -> floor 80*1.3 = 104 protects
    # it even though the naive decay (90*0.9=81) would undercut the
    # evidence. (The floor alone must not trigger deletion.)
    out = fb._decay_gen_budget("m/a", 80.0, None, base_timeout_s=50.0)
    assert out == pytest.approx(104.0)
    # A fast success (10s -> floor 13) lets the decay actually shrink it
    # (93.6 still > base 50, so the override stays).
    out2 = fb._decay_gen_budget("m/a", 10.0, None, base_timeout_s=50.0)
    assert out2 == pytest.approx(104.0 * 0.9)


def test_decay_deletes_below_base(default_cfg):
    fb._GEN_BUDGET_OVERRIDES["m/a"] = 90.0
    # decayed 81 <= base 100 AND floor 6.5 <= base 100 -> nothing left to
    # protect that the stock base doesn't cover -> override removed.
    out = fb._decay_gen_budget("m/a", 5.0, None, base_timeout_s=100.0)
    assert out is None, "decayed into stock territory -> override removed"
    assert "m/a" not in fb._GEN_BUDGET_OVERRIDES


def test_decay_disabled_never_mutates(default_cfg, monkeypatch):
    fb._GEN_BUDGET_OVERRIDES["m/a"] = 90.0
    monkeypatch.setattr(
        fb, "_get_plugin_cfg",
        lambda agent=None: {"adaptive_gen_budget_enabled": False},
    )
    assert fb._decay_gen_budget("m/a", 10.0, None, base_timeout_s=300.0) is None
    assert fb._GEN_BUDGET_OVERRIDES["m/a"] == 90.0, "disabled decay never mutates"


# ---------------------------------------------------------------------------
# Lifecycle wiring
# ---------------------------------------------------------------------------

def test_clear_all_cooldowns_clears_overrides(default_cfg):
    agent = _StubAgent()
    fb._GEN_BUDGET_OVERRIDES["m/a"] = 300.0
    fb.clear_all_cooldowns(agent)
    assert not fb._GEN_BUDGET_OVERRIDES, \
        "the documented recovery path must reset learned budgets too"


def test_reset_gen_budgets_clears(default_cfg):
    fb._GEN_BUDGET_OVERRIDES["m/a"] = 300.0
    fb.reset_gen_budgets()
    assert not fb._GEN_BUDGET_OVERRIDES


# ---------------------------------------------------------------------------
# Outer utility-timeout guard parity (integration)
# ---------------------------------------------------------------------------

def test_utility_guard_grows_on_outer_timeout(default_cfg):
    """The outer guard's budget fired -> the label earns a bigger budget."""
    async def scenario():
        # Factory contract (see guarded_call): a SYNC callable returning
        # the awaitable, so wait_for can cancel it cleanly. (An async def
        # here would just RETURN the sleep coroutine unawaited.)
        def inner_factory():
            return asyncio.sleep(0.6)
        try:
            await ut.guarded_call(
                inner_factory,
                model_name="test/grow-label",
                config_overrides={
                    "default_timeout_s": 0.25, "max_wait_s": 0.5,
                    "jitter_s": 0.0,
                },
                agent=None,
            )
        except RepairableException:
            return True
        return False

    fired = asyncio.run(scenario())
    assert fired, "0.6s inner must exceed the 0.25s budget"
    grown = fb._GEN_BUDGET_OVERRIDES.get("test/grow-label")
    assert grown is not None and grown > 0, \
        "outer-guard timeout must feed the grow learner"


def test_utility_guard_lifts_cap_with_override(default_cfg):
    """A learned override lifts BOTH the budget and the max_wait cap: a
    call that outlives the stock cap now SUCCEEDS instead of dying."""
    fb._GEN_BUDGET_OVERRIDES["test/cap-label"] = 5.0

    async def scenario():
        async def inner_factory():
            await asyncio.sleep(0.4)
            return "ok"
        return await ut.guarded_call(
            inner_factory,
            model_name="test/cap-label",
            # Stock cap 0.2s would kill the 0.4s call; the 5s override
            # must lift both the budget and the cap.
            config_overrides={
                "default_timeout_s": 0.1, "max_wait_s": 0.2, "jitter_s": 0.0,
            },
            agent=None,
        )

    assert asyncio.run(scenario()) == "ok"
