"""v2.8.4 round-2 flagged-fix tests.

Covers the five items the v2.8.3 audit flagged but did not fix:

  T1  F8: malformed-JSON books the SHORT format_error_cooldown_s (20s),
      not the 300s unknown-error default — and bypasses the 30s floor.
  T2  api/stats.py reads the version from plugin.yaml (no "2.6.8").
  T3  _force_chat_config merges default_config.yaml under config.json
      (the YAML-only force-chat lists are reachable at runtime).
  T4  Turn cascade defines AND calls _maybe_extend_primary_cooldown.
  T5  _strip_a0_only_kwargs(is_primary=True) keeps provider-specific
      kwargs (venice_parameters) on the live primary model object.

Run from repo root:  python -m pytest usr/plugins/_model_fallback/tests/test_v284_flagged_fixes.py -q
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


class FakeAgent:
    """Minimal agent stand-in: get_data/set_data + context.id pattern."""

    def __init__(self):
        self._data = {}
        self.context = type("Ctx", (), {"id": "test-agent-v284"})()

    def get_data(self, key):
        return self._data.get(key)

    def set_data(self, key, value):
        self._data[key] = value


# ---------------------------------------------------------------------------
# T1 — F8: format-slip cooldown
# ---------------------------------------------------------------------------


def test_is_format_error_shapes():
    assert fb._is_format_error(ValueError("Utility model output is not valid JSON"))
    assert fb._is_format_error(ValueError("response could not be parsed"))
    import json as _json

    assert fb._is_format_error(_json.JSONDecodeError("bad", "{", 0))
    # Endpoint-shaped errors must NOT match.
    assert not fb._is_format_error(RuntimeError("connection refused"))
    assert not fb._is_format_error(ValueError("quota exceeded"))


def test_format_slip_cooldown_is_short(tmp_path, monkeypatch):
    agent = FakeAgent()
    # No HTTP status, no Retry-After -> previously 300s.
    e = ValueError("Utility model output is not valid JSON")
    store: dict = {}
    monkeypatch.setattr(fb, "_get_cooldown_store", lambda a: store)
    monkeypatch.setattr(fb, "_save_cooldown_store", lambda a, s: None)
    monkeypatch.setattr(fb, "_record_last_status", lambda *a, **k: None)
    monkeypatch.setattr(
        fb, "_get_plugin_cfg", lambda a: {"format_error_cooldown_s": 20.0}
    )
    monkeypatch.setattr(fb, "_classify_capacity", lambda *a, **k: "free_per_minute")
    monkeypatch.setattr(fb, "_is_rate_limited_error", lambda e: False)
    monkeypatch.setattr(fb, "_is_context_overflow_error", lambda e: False)
    monkeypatch.setattr(fb, "_is_permanently_failed_model", lambda e: False)

    assert fb._handle_error_cooldown(e, "test/model", store, agent) is True
    until = store.get("test/model")
    assert until is not None, "format slip must still book a cooldown"
    remaining = until - time.monotonic()
    assert 0 < remaining <= 25.0, f"expected ~20s cooldown, got {remaining:.0f}s"


def test_format_error_cooldown_zero_disables_booking(tmp_path, monkeypatch):
    agent = FakeAgent()
    e = ValueError("Utility model output is not valid JSON")
    store: dict = {}
    monkeypatch.setattr(fb, "_get_cooldown_store", lambda a: store)
    monkeypatch.setattr(fb, "_save_cooldown_store", lambda a, s: None)
    monkeypatch.setattr(fb, "_record_last_status", lambda *a, **k: None)
    monkeypatch.setattr(fb, "_get_plugin_cfg", lambda a: {"format_error_cooldown_s": 0.0})
    monkeypatch.setattr(fb, "_classify_capacity", lambda *a, **k: "free_per_minute")
    monkeypatch.setattr(fb, "_is_rate_limited_error", lambda e: False)
    monkeypatch.setattr(fb, "_is_context_overflow_error", lambda e: False)
    monkeypatch.setattr(fb, "_is_permanently_failed_model", lambda e: False)

    assert fb._handle_error_cooldown(e, "test/model", store, agent) is True
    assert "test/model" not in store, "dur=0 must not book anything"


def test_cooldown_seconds_for_status_format_path(monkeypatch):
    monkeypatch.setattr(
        fb, "_get_plugin_cfg", lambda a: {"format_error_cooldown_s": 20.0}
    )
    got = fb._cooldown_seconds_for_status(
        None, ValueError("Utility model output is not valid JSON")
    )
    assert got == pytest.approx(20.0)
    # Unknown non-format errors keep the 300s default.
    assert fb._cooldown_seconds_for_status(None, RuntimeError("boom")) == 300.0


# ---------------------------------------------------------------------------
# T2 — api/stats.py version from plugin.yaml
# ---------------------------------------------------------------------------


def test_stats_version_not_hardcoded():
    src = (Path(__file__).resolve().parent.parent / "api" / "stats.py").read_text(
        encoding="utf-8"
    )
    assert '"2.6.8"' not in src, "version must not be hardcoded"
    assert "get_plugin_meta" in src


# ---------------------------------------------------------------------------
# T3 — _force_chat_config merges default_config.yaml
# ---------------------------------------------------------------------------


def test_force_chat_config_reads_yaml_defaults():
    from usr.plugins._model_fallback import models_ext as mx

    providers, patterns, api_bases = mx._force_chat_config(None)
    # default_config.yaml ships non-empty values for all three lists.
    assert providers, "providers list from default_config.yaml must be visible"
    assert api_bases, "api_bases list from default_config.yaml must be visible"


# ---------------------------------------------------------------------------
# T4 — turn-path primary-skip escalation wired
# ---------------------------------------------------------------------------


def test_turn_cascade_defines_and_calls_primary_skip():
    src = (Path(__file__).resolve().parent.parent / "fallback.py").read_text(
        encoding="utf-8"
    )
    turn_start = src.index("async def _patched_call_chat_model_turn")
    turn_end = src.index("def install_chat_turn_patch")
    turn_src = src[turn_start:turn_end]
    assert "_consecutive_primary_failures += 1" in turn_src
    assert turn_src.count("def _maybe_extend_primary_cooldown") == 1
    # The call site sits right after the idx==0 increment.
    # v2.8.5: widened from 400 -- the strike counter now also persists to
    # agent data (DATA_KEY_TURN_PRIMARY_FAILS) between the increment and
    # the escalation call, which pushed the call past the old window.
    # v2.8.6: widened again to 2400 -- the strike-decay block (Z fix:
    # timestamp read + decay + persist) sits between the increment and
    # the escalation call.
    site = turn_src.index("if idx == 0:")
    assert "_maybe_extend_primary_cooldown(reason=" in turn_src[site : site + 2400]


# ---------------------------------------------------------------------------
# T5 — primary kwargs survive the strip
# ---------------------------------------------------------------------------


class _FakeModel:
    def __init__(self, kwargs: dict):
        self.kwargs = kwargs


def test_strip_keeps_provider_specific_on_primary():
    m = _FakeModel(
        {
            "venice_parameters": {"auto_truncation": True},
            "usage": "x",
            "TIMEOUT": 60,
            "api_base": "https://llm.agent-zero.ai/v1",
        }
    )
    fb._strip_a0_only_kwargs(m, is_primary=True)
    assert "venice_parameters" in m.kwargs, "primary keeps its provider feature"
    assert "usage" not in m.kwargs, "usage is always stripped (OpenAI SDK rejects it)"
    assert "TIMEOUT" not in m.kwargs


def test_strip_removes_provider_specific_on_fallback():
    m = _FakeModel(
        {
            "venice_parameters": {"auto_truncation": True},
            "api_base": "https://groq.com/v1",
        }
    )
    fb._strip_a0_only_kwargs(m, is_primary=False)
    assert "venice_parameters" not in m.kwargs


def test_strip_is_backward_compatible_default():
    m = _FakeModel({"venice_parameters": {}})
    fb._strip_a0_only_kwargs(m)
    assert "venice_parameters" not in m.kwargs, "default (wrapper) behavior unchanged"