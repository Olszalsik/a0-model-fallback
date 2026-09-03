"""Characterization tests for the utility + chat rotation cascades (v3.0.0).

Written GREEN against the v2.9.1 code BEFORE the cascade unification, and
kept UNCHANGED through the refactor: they pin the external contract of
``_patched_call_utility_model`` / ``_patched_call_chat_model`` (return
shapes, hooks, rotation, cooldown booking, skip logic, exhaustion, and the
continuous/legacy mode split) so the unified engine must reproduce the
exact same observable behavior.

The turn cascade has its own functional tests (test_turn_cascade_v28.py);
it is NOT part of the v3.0.0 unification.

Test:
    cd /a0
    REPO_ROOT_OVERRIDE="$(pwd)" pytest usr/plugins/_model_fallback/tests/test_cascade_v300.py -v
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
from usr.plugins._model_fallback.helpers import events  # noqa: E402
from usr.plugins._model_fallback.helpers import recovery_probe as rp  # noqa: E402


# ---------------------------------------------------------------------------
# Fakes (pattern from test_turn_cascade_v28.py)
# ---------------------------------------------------------------------------


class FakeLog:
    def __init__(self):
        self.lines: list[tuple] = []

    def log(self, type="info", content="", **kwargs):
        self.lines.append((type, content))


class FakeContext:
    def __init__(self, ctx_id: str):
        self.id = ctx_id
        self.log = FakeLog()


class FakeExtension:
    def __init__(self):
        self.calls: list[str] = []

    async def call_extensions_async(self, hook, agent, **kwargs):
        self.calls.append(hook)


class FakeAgent:
    def __init__(self, ctx_id: str, primary):
        self.context = FakeContext(ctx_id)
        self._data: dict = {}
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
    def __init__(self, name: str):
        self.model_name = name
        self.provider = "openai"
        self.fail_with = None
        self.kwargs = {
            "api_key": "sk-test",
            "api_base": "https://openrouter.ai/api/v1",
        }
        self.unified_kwargs: list[dict] = []

    def unified_call(self, **kwargs):
        self.unified_kwargs.append(kwargs)

        async def _coro(model=self):
            if model.fail_with is not None:
                raise model.fail_with
            return ("hello-%s" % model.model_name, "")

        return _coro()


class FakeRateLimit(Exception):
    """LiteLLM-shaped 429 without needing litellm in the test venv."""

    status_code = 429

    def __init__(self, model: str = "primary-model"):
        self.model = model
        super().__init__(
            f"RateLimitError: 429 upstream_429 on {model} retry_after_seconds: 5"
        )


# Both candidates raise this until flipped per-model.
def _make_models():
    primary = FakeModel("primary-model")
    cand1 = FakeModel("cand1")
    return primary, cand1


CFG = {
    "fallback_timeout_s": 300,
    "fallback_utility_timeout_s": 300,
    "cascade_warm_timeout_s": 20,
    "cascade_warm_window_s": 600,
    "fallback_cycle_delay": 0,
    "fallback_attempt_delay": 0,
    "fallback_max_cycles": 4,
    "max_cycle_delay_s": 0.01,  # empty-pass spin sleeps ~10ms in tests
    "backoff_jitter_s": 0,  # pre-3.0.0: spin jitter has no import random
    "extended_retry_enabled": False,
    "continuous_fallback": False,
}

TEST_CTX = "cascade-v300"


@pytest.fixture()
def env(monkeypatch):
    """Patch the cascades' module-level collaborators. Resolves every
    mutable module global via ``fb.<name>`` at call time (suite-pollution
    gotcha: test_candidate_normalize reloads the module mid-suite)."""
    fb._INMEM_COOLDOWNS.clear()
    fb._WARM_LABELS.clear()
    fb._INMEM_DEAD_LABELS.clear()
    fb._INMEM_HEALTHY_LABELS.clear()
    events.reset_events()
    rp.shutdown_probes()
    rp.reset_counters()

    primary, cand1 = _make_models()
    models = {"primary-model": primary, "cand1": cand1}

    monkeypatch.setattr(
        fb,
        "_build_candidates",
        lambda model_obj, use_utility_models, agent: [None, {"model": "cand1"}],
        raising=True,
    )
    monkeypatch.setattr(
        fb,
        "_build_model",
        lambda spec, model_obj: (
            model_obj if spec is None else models[spec["model"]]
        ),
        raising=True,
    )
    monkeypatch.setattr(
        fb,
        "_resolve_per_call_timeout",
        lambda label, base, warm, window, agent=None, api_base="",
        allow_warm=True: 5.0,
        raising=True,
    )
    monkeypatch.setattr(fb, "_get_plugin_cfg", lambda agent: dict(CFG), raising=True)
    monkeypatch.setattr(fb, "extension", FakeExtension(), raising=True)

    yield models

    events.reset_events()
    rp.shutdown_probes()
    rp.reset_counters()


def _run(kind: str, agent, **kwargs):
    if kind == "utility":
        return asyncio.run(
            fb._patched_call_utility_model(agent, "sys", "msg", **kwargs)
        )
    return asyncio.run(fb._patched_call_chat_model(agent, messages=["hi"], **kwargs))


def _store(agent):
    return fb._INMEM_COOLDOWNS[(agent.context.id,)]


# ---------------------------------------------------------------------------
# 1. Primary succeeds immediately
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["utility", "chat"])
def test_primary_success_return_shape_and_hooks(env, kind):
    agent = FakeAgent(f"{TEST_CTX}-ok", env["primary-model"])
    result = _run(kind, agent)

    if kind == "utility":
        assert result == "hello-primary-model"
        assert env["primary-model"].unified_kwargs[0]["system_message"] == "sys"
        assert env["primary-model"].unified_kwargs[0]["user_message"] == "msg"
        assert env["primary-model"].unified_kwargs[0]["fallbacks"] is None
    else:
        assert result == ("hello-primary-model", "")
        assert env["primary-model"].unified_kwargs[0]["messages"] == ["hi"]
        assert env["primary-model"].unified_kwargs[0]["fallbacks"] is None

    ext = fb.extension
    expected = (
        ["util_model_call_before", "util_model_call_after"]
        if kind == "utility"
        else ["chat_model_call_before", "chat_model_call_after"]
    )
    assert ext.calls == expected, "one before + one after hook on success"
    # No cooldown booked, none cleared
    assert _store(agent) == {}
    assert events.snapshot(kind="cooldown_booked") == []
    # Warm label registered for the successful label
    assert "primary-model" in fb._WARM_LABELS
    # Extended-retry state cleared
    assert agent.get_data(fb.DATA_KEY_EXT_RETRY_ATTEMPTS) == 0


# ---------------------------------------------------------------------------
# 2. Primary 429 -> rotation to candidate 1
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["utility", "chat"])
def test_primary_429_rotates_to_candidate(env, kind):
    models = env
    models["primary-model"].fail_with = FakeRateLimit("primary-model")
    agent = FakeAgent(f"{TEST_CTX}-rot", models["primary-model"])

    result = _run(kind, agent)

    expected = "hello-cand1"
    assert result == expected if kind == "utility" else result == (expected, "")
    # Both models were actually called
    assert len(models["primary-model"].unified_kwargs) == 1
    assert len(models["cand1"].unified_kwargs) == 1
    # Primary booked a cooldown (Retry-After hint 5s)
    store = _store(agent)
    assert "primary-model" in store
    assert store["primary-model"] > time.monotonic()
    # Success label NOT in cooldown, and NOT logged as cleared-by-success
    # (it was never cooling)
    assert events.snapshot(kind="cooldown_cleared_by_success") == []
    assert events.snapshot(kind="cooldown_booked")[0]["label"] == "primary-model"
    # Switching warning logged
    warns = [c for t, c in agent.context.log.lines if "switching" in str(c)]
    assert warns, "fallback rotation logs a switching warning"
    # Hooks: two before (one per attempt) + one after
    hooks = fb.extension.calls
    assert hooks.count(hooks[0]) == 2
    assert hooks[-1].endswith("_after")


# ---------------------------------------------------------------------------
# 2b. Success on a label that WAS in cooldown -> cleared_by_success event
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["utility", "chat"])
def test_success_after_cooldown_records_cleared_event(env, kind):
    # Real production trigger: the cooldown EXPIRED but the store entry was
    # not popped yet (entries are only popped by a skip check on an active
    # cooldown or by the success pop). The label is then callable again and
    # its success pops the stale entry -> cleared_by_success event.
    agent = FakeAgent(f"{TEST_CTX}-rec", env["primary-model"])
    store = fb._get_cooldown_store(agent)
    store["primary-model"] = time.monotonic() - 1.0  # expired, unpopped

    result = _run(kind, agent)
    assert result == "hello-primary-model" if kind == "utility" else (
        result == ("hello-primary-model", "")
    )
    cleared = events.snapshot(kind="cooldown_cleared_by_success")
    assert len(cleared) == 1
    assert cleared[0]["label"] == "primary-model"
    assert "primary-model" not in _store(agent)


# ---------------------------------------------------------------------------
# 3. All candidates fail (legacy mode) -> RuntimeError exhausted
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["utility", "chat"])
def test_all_fail_legacy_raises_exhausted(env, kind, monkeypatch):
    models = env
    models["primary-model"].fail_with = FakeRateLimit("primary-model")
    models["cand1"].fail_with = FakeRateLimit("cand1")
    monkeypatch.setattr(fb, "_get_plugin_cfg",
                        lambda agent: {**CFG, "fallback_max_cycles": 1}, raising=True)
    agent = FakeAgent(f"{TEST_CTX}-exh", models["primary-model"])

    with pytest.raises(RuntimeError, match="exhausted"):
        _run(kind, agent)

    # Both candidates booked cooldowns; exhaustion event recorded
    store = _store(agent)
    assert "primary-model" in store and "cand1" in store
    exhausted = events.snapshot(kind="cascade_exhausted")
    assert len(exhausted) == 1


# ---------------------------------------------------------------------------
# 4. Cancellation propagates (no cooldown, no rotation)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["utility", "chat"])
def test_cancelled_error_propagates(env, kind):
    models = env

    class Cancelled(Exception):
        pass

    async def unified_call(self=None, **kwargs):
        raise asyncio.CancelledError()

    models["primary-model"].unified_call = unified_call
    agent = FakeAgent(f"{TEST_CTX}-cancel", models["primary-model"])

    with pytest.raises(asyncio.CancelledError):
        _run(kind, agent)
    assert _store(agent) == {}, "no cooldown booked on external cancellation"
    assert fb.extension.calls == [fb.extension.calls[0]], "before-hook only"


# ---------------------------------------------------------------------------
# 4b. Code error fails fast (no rotation)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["utility", "chat"])
def test_code_error_fails_fast(env, kind):
    models = env
    models["primary-model"].fail_with = ImportError("no module named broken")
    agent = FakeAgent(f"{TEST_CTX}-code", models["primary-model"])

    with pytest.raises(Exception) as exc_info:
        _run(kind, agent)
    assert type(exc_info.value).__name__ == "CodeError"
    # Only the primary was called (fail fast, no rotation)
    assert models["cand1"].unified_kwargs == []
    assert "primary-model" in _store(agent)


# ---------------------------------------------------------------------------
# 5. Cooldown skip: cooled candidate is skipped, exhausted without a call
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["utility", "chat"])
def test_cooldown_skipped_candidate_not_called(env, kind, monkeypatch):
    models = env
    models["primary-model"].fail_with = FakeRateLimit("primary-model")
    monkeypatch.setattr(fb, "_get_plugin_cfg",
                        lambda agent: {**CFG, "fallback_max_cycles": 1}, raising=True)
    agent = FakeAgent(f"{TEST_CTX}-skip", models["primary-model"])
    # Pre-cool candidate 1 so the cascade skips it without calling
    store = fb._get_cooldown_store(agent)
    store["cand1"] = time.monotonic() + 60.0

    with pytest.raises(RuntimeError, match="exhausted"):
        _run(kind, agent)

    assert models["cand1"].unified_kwargs == [], "cooled candidate skipped"
    skip_logs = [c for t, c in agent.context.log.lines if "skipping" in str(c)]
    assert skip_logs, "cooldown skip is logged (deduped)"


# ---------------------------------------------------------------------------
# 6. All-skipped roster -> legacy exhaustion after 3 empty passes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["utility", "chat"])
def test_all_skipped_exhausts_in_legacy_mode(env, kind):
    models = env
    agent = FakeAgent(f"{TEST_CTX}-empty", models["primary-model"])
    store = fb._get_cooldown_store(agent)
    store["primary-model"] = time.monotonic() + 60.0
    store["cand1"] = time.monotonic() + 60.0

    with pytest.raises(RuntimeError, match="skipped"):
        _run(kind, agent)

    # No model was called at all
    assert models["primary-model"].unified_kwargs == []
    assert models["cand1"].unified_kwargs == []