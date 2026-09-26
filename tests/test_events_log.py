"""v2.9.1 routing-event-log tests.

helpers/events.py: bounded in-memory ring buffer recording routing
decisions (cooldown bookings/clears, escalations, dead-marks, user
clears, exhaustion) so the /events endpoint can show the timeline.

Run from repo root:  python -m pytest usr/plugins/model_fallback/tests/test_events_log.py -q
"""

from __future__ import annotations

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
from usr.plugins.model_fallback.helpers import recovery_probe as rp  # noqa: E402


class FakeAgent:
    def __init__(self, ctx_id: str = "events-test-agent"):
        self._data = {}
        self.context = type(
            "Ctx",
            (),
            {
                "id": ctx_id,
                "log": type("L", (), {"log": staticmethod(lambda *a, **k: None)})(),
            },
        )()

    def get_data(self, key):
        return self._data.get(key)

    def set_data(self, key, value):
        self._data[key] = value


@pytest.fixture(autouse=True)
def _clean_event_state():
    events.reset_events()
    rp.shutdown_probes()
    rp.reset_counters()
    yield
    events.reset_events()
    rp.shutdown_probes()
    # test_recovery_probes asserts on these same module-global counters;
    # leave them zeroed for the sibling suite.
    rp.reset_counters()


# ---------------------------------------------------------------------------
# Ring-buffer basics
# ---------------------------------------------------------------------------


def test_record_and_snapshot_newest_first():
    agent = FakeAgent()
    for i in range(3):
        events.record_event("cooldown_booked", agent=agent, label=f"m/{i}")
    snap = events.snapshot()
    assert len(snap) == 3
    assert snap[0]["label"] == "m/2", "newest first"
    assert snap[-1]["label"] == "m/0"
    assert snap[0]["kind"] == "cooldown_booked"
    assert snap[0]["context"] == "events-test-agent"
    assert isinstance(snap[0]["ts"], float) and snap[0]["ts"] > 0


def test_record_without_agent_uses_global_context():
    events.record_event("cooldowns_cleared", count=2)
    snap = events.snapshot()
    assert snap[0]["context"] == "__global__"


def test_record_never_raises_on_bad_fields():
    # Non-JSON-safe values get stringified, not raised.
    events.record_event(
        "cooldown_booked", agent=None, label="m/x", weird=object(), status_code=None
    )
    snap = events.snapshot()
    assert snap[0]["status_code"] is None
    assert isinstance(snap[0]["weird"], str)


def test_snapshot_filters_by_kind_label_context():
    a1, a2 = FakeAgent("ev-f1"), FakeAgent("ev-f2")
    events.record_event("cooldown_booked", agent=a1, label="prov/a")
    events.record_event("label_dead", agent=a1, label="prov/b")
    events.record_event("cooldown_booked", agent=a2, label="prov/a")

    assert [e["kind"] for e in events.snapshot(kind="label_dead")] == ["label_dead"]
    assert all(e["label"] == "prov/a" for e in events.snapshot(label="prov/a"))
    assert all(e["context"] == "ev-f2" for e in events.snapshot(context="ev-f2"))
    assert events.snapshot(kind="nope") == []


def test_ring_buffer_caps_at_max():
    for i in range(events._MAX_EVENTS + 50):
        events.record_event("cooldown_booked", label=f"m/{i}")
    snap = events.snapshot(limit=events._MAX_EVENTS)
    assert len(snap) == events._MAX_EVENTS
    assert snap[0]["label"] == f"m/{events._MAX_EVENTS + 49}", "newest kept"
    assert snap[-1]["label"] == "m/50", "oldest evicted"
    # limit clamp
    assert len(events.snapshot(limit=10)) == 10


def test_reset_events_clears_buffer():
    events.record_event("cooldown_booked", label="m/x")
    events.reset_events()
    assert events.snapshot() == []


# ---------------------------------------------------------------------------
# Wrapper: cooldown_booked detection
# ---------------------------------------------------------------------------


def _patch_booking_env(monkeypatch, *, capacity="free_per_minute", cfg=None):
    store: dict = {}
    monkeypatch.setattr(fb, "_get_cooldown_store", lambda a: store)
    monkeypatch.setattr(fb, "_save_cooldown_store", lambda a, s: None)
    monkeypatch.setattr(fb, "_record_last_status", lambda *a, **k: None)
    monkeypatch.setattr(fb, "_get_plugin_cfg", lambda a: dict(cfg or {}))
    monkeypatch.setattr(fb, "_classify_capacity", lambda *a, **k: capacity)
    monkeypatch.setattr(fb, "_is_rate_limited_error", lambda e: False)
    monkeypatch.setattr(fb, "_is_context_overflow_error", lambda e: False)
    monkeypatch.setattr(fb, "_is_permanently_failed_model", lambda e: False)
    monkeypatch.setattr(fb, "_is_format_error", lambda e: False)
    monkeypatch.setattr(fb, "_is_invalid_api_key_400", lambda e: False)
    monkeypatch.setattr(fb, "_mark_label_dead", lambda *a, **k: None)
    return store


def test_wrapper_records_cooldown_booked(monkeypatch):
    agent = FakeAgent("ev-book")
    store = _patch_booking_env(monkeypatch)
    # Unknown error with no status -> falls through to the >=30s floor.
    fb._handle_error_cooldown(RuntimeError("boom"), "prov/a", store, agent, "")
    snap = events.snapshot(kind="cooldown_booked")
    assert len(snap) == 1
    assert snap[0]["label"] == "prov/a"
    assert snap[0]["error"] == "RuntimeError"
    assert snap[0]["cooldown_s"] >= 30.0


def test_wrapper_records_nothing_when_capacity_skips(monkeypatch):
    # 429 on a skip-capacity provider books NO cooldown (returns False).
    agent = FakeAgent("ev-skip")
    _patch_booking_env(monkeypatch, capacity="concurrent_paid")
    monkeypatch.setattr(fb, "_is_rate_limited_error", lambda e: True)
    store: dict = {}
    result = fb._handle_error_cooldown(
        RuntimeError("429"), "prov/a", store, agent, ""
    )
    assert result is False
    assert events.snapshot() == []


def test_wrapper_records_nothing_for_router_dur_zero(monkeypatch):
    # Pure timeout on a router with router_timeout_cooldown_s=0 books
    # nothing -- booked=True but no store write, so no event.
    agent = FakeAgent("ev-router")
    _patch_booking_env(
        monkeypatch,
        capacity="router",
        cfg={"router_timeout_no_cooldown": True, "router_timeout_cooldown_s": 0.0},
    )
    monkeypatch.setattr(fb, "_classify_capacity", lambda *a, **k: "router")
    store: dict = {}
    result = fb._handle_error_cooldown(
        TimeoutError("slow"), "omniroute/auto", store, agent, ""
    )
    assert result is True
    assert store == {}
    assert events.snapshot() == []


def test_wrapper_records_extension_on_rebooking(monkeypatch):
    # A second failure with a longer duration extends the entry: the new
    # value differs from the old one, so it records again.
    agent = FakeAgent("ev-rebook")
    store = _patch_booking_env(monkeypatch)
    fb._handle_error_cooldown(RuntimeError("boom"), "prov/a", store, agent, "")
    until = store["prov/a"]
    store["prov/a"] = until - 10.0  # simulate a shorter stale entry
    fb._handle_error_cooldown(RuntimeError("boom2"), "prov/a", store, agent, "")
    assert len(events.snapshot(kind="cooldown_booked")) == 2


# ---------------------------------------------------------------------------
# Other instrumented sites
# ---------------------------------------------------------------------------


def test_clear_all_cooldowns_records_event():
    agent = FakeAgent("ev-clear")
    store = fb._get_cooldown_store(agent)
    store["prov/a"] = time.monotonic() + 60.0
    store["prov/b"] = time.monotonic() + 60.0
    cleared = fb.clear_all_cooldowns(agent)
    assert cleared == 2
    snap = events.snapshot(kind="cooldowns_cleared")
    assert len(snap) == 1
    assert snap[0]["count"] == 2
    assert snap[0]["cross_context"] is False


def test_clear_all_cooldowns_cross_context_flag_recorded():
    agent = FakeAgent("ev-clear-x")
    fb.clear_all_cooldowns(agent, cross_context=True)
    snap = events.snapshot(kind="cooldowns_cleared")
    assert snap[0]["cross_context"] is True


def test_mark_label_dead_records_event(monkeypatch):
    agent = FakeAgent("ev-dead")
    monkeypatch.setattr(fb, "_is_dead_for_all_agents", lambda e: True)
    monkeypatch.setattr(fb, "_classify_capacity", lambda *a, **k: "free_per_minute")

    class FakeExc(Exception):
        status_code = 404

    fb._mark_label_dead("prov/gone", FakeExc("gone"), agent, "https://x")
    snap = events.snapshot(kind="label_dead")
    assert len(snap) == 1
    assert snap[0]["label"] == "prov/gone"
    assert snap[0]["status_code"] == 404
    assert snap[0]["dead_s"] >= 60.0


def test_emit_fallback_summary_records_exhaustion():
    agent = FakeAgent("ev-exhaust")
    fb._emit_fallback_summary(agent, "utility", [])
    snap = events.snapshot(kind="cascade_exhausted")
    assert len(snap) == 1
    assert snap[0]["kind"] == "cascade_exhausted"
    assert snap[0]["cascade_kind"] == "utility"
    assert snap[0]["candidates"] == 0


def test_probe_clear_records_cleared_early_event():
    agent = FakeAgent("ev-probe-clear")
    store = fb._get_cooldown_store(agent)
    store["prov/a"] = time.monotonic() + 300.0
    rp._clear_cooldown_early(fb, agent, "prov/a")
    snap = events.snapshot(kind="cooldown_cleared_early")
    assert len(snap) == 1
    assert snap[0]["label"] == "prov/a"
    assert "prov/a" not in store


def test_kinds_constant_matches_documented_set():
    assert set(events.KINDS) == {
        "cooldown_booked",
        "cooldown_cleared_early",
        "cooldown_cleared_by_success",
        "primary_skip_escalated",
        "label_dead",
        "cooldowns_cleared",
        "cascade_exhausted",
    }


def test_escalation_site_records_event_structure():
    # Structure check: the unified rotation engine (utility+chat share one
    # copy since v3.0.0) and the turn cascade each write the event right
    # after persisting the escalation.
    src = (Path(fb.__file__)).read_text(encoding="utf-8")
    assert src.count('"primary_skip_escalated"') == 2
    # Every success-path pop records the cleared_by_success event.
    assert src.count('"cooldown_cleared_by_success"') == 2