"""Tests for v2.8.0 — turn-path fallback cascade + transient-error safety net.

Root cause under test: since the v2.10/2.11 upstream merge the main agent
loop calls ``Agent.call_chat_model_turn`` (monologue -> ``unified_turn``
-> ``LiteLLMTransport.astream``), which had NO fallback coverage. A
shared-pool OpenRouter 429 was litellm-retried against the SAME model and
the ``RateLimitError`` escaped to ``handle_exception`` -> agent stopped.

Covered here:
  1. ``_patched_call_chat_model_turn`` rotates to the next candidate when
     the primary returns a 429, and returns the original's LLMResult.
  2. The failed primary label lands in the shared cooldown store.
  3. When ALL candidates 429, the cascade raises ``RetryAfterHours``
     (never the raw RateLimitError) so the existing _70 extension can
     swallow it and keep the agent alive.
  4. If a response chunk already streamed to the UI, the cascade does NOT
     rotate (would duplicate partial output) -- it re-raises after
     booking the cooldown.
  5. ``install_chat_turn_patch`` is idempotent and captures the original.
  6. The ``_60_handle_transient_llm_error`` safety net swallows a bare
     429 that escaped everything (data["exception"] -> None) and books
     the cooldown.
  7. (v3.1.1) The safety net's swallow bound is configurable via the
     ``transient_swallow_max`` / ``transient_swallow_reset_window_s``
     plugin knobs; unreadable values fall back to the historic 5 / 300s.

Test:
    cd /a0
    REPO_ROOT_OVERRIDE="$(pwd)" pytest usr/plugins/model_fallback/tests/test_turn_cascade_v28.py -v
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(os.environ.get("REPO_ROOT_OVERRIDE") or "/a0")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest

from usr.plugins.model_fallback import fallback
from usr.plugins.model_fallback.fallback import (
    _patched_call_chat_model_turn,
    install_chat_turn_patch,
)

# NOTE: resolve module-level state (cooldown dicts, RetryAfterHours) via
# ``fallback.<name>`` at call time, never via import-time from-imports.
# ``test_candidate_normalize`` reloads the fallback module mid-suite, which
# REBINDS every module global -- an import-time reference to _INMEM_COOLDOWNS
# or RetryAfterHours would be a stale object and the cooldown/class asserts
# would fail on suite ordering (the exact suite-pollution class of bug fixed
# once before for the langchain shim tests).


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeLog:
    def __init__(self):
        self.lines: list[tuple] = []

    def log(self, type="info", content="", **kwargs):
        self.lines.append((type, content))


class FakeContext:
    id = "test-agent-1"

    def __init__(self):
        self.log = FakeLog()


class FakeAgent:
    def __init__(self, primary):
        self.context = FakeContext()
        self._data: dict = {}
        self._primary = primary

    def get_data(self, key, default=None):
        return self._data.get(key, default)

    def set_data(self, key, value):
        self._data[key] = value

    def get_chat_model(self):
        return self._primary


class FakeModel:
    def __init__(self, name: str):
        self.model_name = name
        self.provider = "openai"
        self.kwargs = {
            "api_key": "sk-test",
            "api_base": "https://openrouter.ai/api/v1",
        }


class FakeRateLimit(Exception):
    """LiteLLM-shaped 429 without needing litellm in the test venv."""

    status_code = 429

    def __init__(self, model: str = "openrouter/z-ai/glm-5.2:free"):
        self.model = model
        super().__init__(
            f"RateLimitError: 429 upstream_429 on {model} retry_after_seconds: 5"
        )


class FakeLLMResult:
    def __init__(self, response: str):
        self.response = response
        self.reasoning = ""
        self.capability = {}


@pytest.fixture()
def isolation(monkeypatch):
    """Patch the cascade's module-level collaborators so no real model
    building / litellm happens."""
    # Fresh in-memory cooldowns per test
    fallback._INMEM_COOLDOWNS.clear()
    fallback._WARM_LABELS.clear()
    fallback._INMEM_DEAD_LABELS.clear()

    primary = FakeModel("openrouter/z-ai/glm-5.2:free")
    cand1 = FakeModel("openrouter/z-ai/glm-4.7:free")
    cand2 = FakeModel("openrouter/deepseek/deepseek-v4:free")
    models = {primary.model_name: primary, "cand1": cand1, "cand2": cand2}

    monkeypatch.setattr(
        fallback,
        "_build_candidates",
        lambda primary_obj, use_utility_models, agent: [
            None,
            {"model": "cand1"},
            {"model": "cand2"},
        ],
        raising=True,
    )
    monkeypatch.setattr(
        fallback,
        "_build_model",
        lambda spec, model_obj: (
            model_obj
            if spec is None
            else models[spec["model"]]
        ),
        raising=True,
    )
    monkeypatch.setattr(
        fallback,
        "_resolve_per_call_timeout",
        # v2.8.5: the turn cascade passes allow_warm=False (kill-path fix),
        # so the stub must accept the kwarg.
        lambda label, base, warm, window, agent=None, api_base="",
        allow_warm=True: 5.0,
        raising=True,
    )
    monkeypatch.setattr(
        fallback,
        "_get_plugin_cfg",
        lambda agent: {
            "fallback_timeout_s": 300,
            "fallback_attempt_delay": 0,
            "cascade_warm_timeout_s": 20,
            "cascade_warm_window_s": 600,
        },
        raising=True,
    )
    return models


# ---------------------------------------------------------------------------
# 1-2. Rotation on primary 429 + cooldown booking
# ---------------------------------------------------------------------------


def test_turn_cascade_rotates_on_primary_429(isolation, monkeypatch):
    models = isolation
    agent = FakeAgent(models["openrouter/z-ai/glm-5.2:free"])
    calls: list[str] = []

    async def original(self, **kwargs):
        model = self.get_chat_model()
        calls.append(model.model_name)
        if model.model_name.endswith("glm-5.2:free"):
            raise FakeRateLimit(model.model_name)
        return FakeLLMResult(f"answer from {model.model_name}")

    monkeypatch.setattr(fallback, "_ORIGINAL_CALL_CHAT_MODEL_TURN", original)

    llm_result = asyncio.run(
        _patched_call_chat_model_turn(agent, messages=["hi"])
    )

    assert llm_result.response == "answer from openrouter/z-ai/glm-4.7:free"
    assert calls[0].endswith("glm-5.2:free")  # primary tried first
    # Primary label booked in the shared cooldown store (Retry-After: 5 -> 5s)
    store = fallback._INMEM_COOLDOWNS[("test-agent-1",)]
    assert "openrouter/z-ai/glm-5.2:free" in store
    assert store["openrouter/z-ai/glm-5.2:free"] > time.monotonic()
    # Instance attribute shadow removed after the call
    assert "get_chat_model" not in agent.__dict__
    # Success clears extended-retry state
    assert agent.get_data(fallback.DATA_KEY_EXT_RETRY_ATTEMPTS) == 0
    assert agent.get_data(fallback.DATA_KEY_EXT_RETRY_PHASE) == 0
    assert agent.get_data(fallback.DATA_KEY_EXT_RETRY_NOTIFIED) is False


# ---------------------------------------------------------------------------
# 3. All candidates 429 -> RetryAfterHours (NOT the raw error)
# ---------------------------------------------------------------------------


def test_turn_cascade_raises_retry_after_hours_when_all_fail(isolation, monkeypatch):
    models = isolation
    agent = FakeAgent(models["openrouter/z-ai/glm-5.2:free"])

    async def original(self, **kwargs):
        raise FakeRateLimit(self.get_chat_model().model_name)

    monkeypatch.setattr(fallback, "_ORIGINAL_CALL_CHAT_MODEL_TURN", original)

    with pytest.raises(fallback.RetryAfterHours) as excinfo:
        asyncio.run(_patched_call_chat_model_turn(agent, messages=["hi"]))

    assert not isinstance(excinfo.value, FakeRateLimit)
    assert excinfo.value.retry_after >= 60.0
    # All three labels (primary + 2 candidates) are in cooldown for the
    # next pass
    store = fallback._INMEM_COOLDOWNS[("test-agent-1",)]
    assert len(store) == 3
    # Shadow cleaned up
    assert "get_chat_model" not in agent.__dict__


# ---------------------------------------------------------------------------
# 4. Mid-stream failure after UI-visible output -> re-raise, no rotation
# ---------------------------------------------------------------------------


def test_turn_cascade_rejects_rotation_after_streamed_output(isolation, monkeypatch):
    models = isolation
    agent = FakeAgent(models["openrouter/z-ai/glm-5.2:free"])
    streamed: list[str] = []

    async def response_cb(chunk: str, total: str):
        streamed.append(chunk)
        return None

    async def original(self, response_callback=None, **kwargs):
        await response_callback("partial ", "partial ")
        raise FakeRateLimit(self.get_chat_model().model_name)

    monkeypatch.setattr(fallback, "_ORIGINAL_CALL_CHAT_MODEL_TURN", original)

    with pytest.raises(FakeRateLimit):
        asyncio.run(
            _patched_call_chat_model_turn(
                agent,
                messages=["hi"],
                response_callback=response_cb,
            )
        )

    assert streamed == ["partial "]
    # Cooldown was still booked for the primary
    store = fallback._INMEM_COOLDOWNS[("test-agent-1",)]
    assert "openrouter/z-ai/glm-5.2:free" in store


# ---------------------------------------------------------------------------
# 5. Installer idempotency + original capture
# ---------------------------------------------------------------------------


def test_install_chat_turn_patch_idempotent(isolation, monkeypatch):
    class FakeAgentCls:
        async def call_chat_model_turn(self, **kwargs):
            return "original"

    sentinel = FakeAgentCls()
    assert install_chat_turn_patch(FakeAgentCls) is True
    assert fallback._ORIGINAL_CALL_CHAT_MODEL_TURN is not None
    assert getattr(FakeAgentCls.call_chat_model_turn, "_fallback_turn_patched", False)
    # Second install is a no-op (does not stack layers / re-capture)
    before = fallback._ORIGINAL_CALL_CHAT_MODEL_TURN
    assert install_chat_turn_patch(FakeAgentCls) is False
    assert fallback._ORIGINAL_CALL_CHAT_MODEL_TURN is before
    # The patched method delegates to the captured original
    inst = sentinel


# ---------------------------------------------------------------------------
# 6. Safety net: escaped 429 is swallowed and cooled down
# ---------------------------------------------------------------------------


def _load_safety_net():
    path = (
        Path(__file__).parent.parent
        / "extensions/python/_functions/agent/Agent/handle_exception/end/"
        "_60_handle_transient_llm_error.py"
    )
    spec = importlib.util.spec_from_file_location("safety_net_60", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_safety_net_swallows_escaped_429(isolation, monkeypatch):
    models = isolation
    agent = FakeAgent(models["openrouter/z-ai/glm-5.2:free"])
    mod = _load_safety_net()

    async def fake_sleep(_s):
        return None

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    ext = mod.HandleTransientLLMError(agent=agent)
    data = {"exception": FakeRateLimit()}
    asyncio.run(ext.execute(data=data))

    assert data["exception"] is None
    assert agent.get_data(mod.DATA_KEY_SWALLOW_COUNT) == 1
    # Cooldown booked on the label extracted from exc.model
    store = fallback._INMEM_COOLDOWNS[("test-agent-1",)]
    assert "openrouter/z-ai/glm-5.2:free" in store


def test_safety_net_ignores_code_errors(isolation, monkeypatch):
    models = isolation
    agent = FakeAgent(models["openrouter/z-ai/glm-5.2:free"])
    mod = _load_safety_net()

    async def fake_sleep(_s):
        return None

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    ext = mod.HandleTransientLLMError(agent=agent)
    boom = ModuleNotFoundError("No module named 'foo'")
    data = {"exception": boom}
    asyncio.run(ext.execute(data=data))

    assert data["exception"] is boom  # untouched -> _90 will surface it
    assert agent.get_data(mod.DATA_KEY_SWALLOW_COUNT) is None


def test_safety_net_caps_consecutive_swallows(isolation, monkeypatch):
    models = isolation
    agent = FakeAgent(models["openrouter/z-ai/glm-5.2:free"])
    mod = _load_safety_net()

    async def fake_sleep(_s):
        return None

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    ext = mod.HandleTransientLLMError(agent=agent)
    # Pre-set the counter at the cap (v3.1.1: the bound is resolved from
    # plugin config at exception time; the historic default is 5)
    cap, _window = mod._swallow_limits(agent)
    agent.set_data(mod.DATA_KEY_SWALLOW_COUNT, cap)
    agent.set_data(mod.DATA_KEY_SWALLOW_AT, time.monotonic())

    data = {"exception": FakeRateLimit()}
    asyncio.run(ext.execute(data=data))

    # Over the cap: exception left for the critical handler
    assert data["exception"] is not None


def test_safety_net_uses_configured_swallow_cap(isolation, monkeypatch):
    """v3.1.1: the swallow bound comes from plugin config, not a
    hard-coded constant -- slow free-tier presets raise both knobs."""
    import helpers.plugins as plugin_helpers

    models = isolation
    agent = FakeAgent(models["openrouter/z-ai/glm-5.2:free"])
    mod = _load_safety_net()

    async def fake_sleep(_s):
        return None

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(
        plugin_helpers,
        "get_plugin_config",
        lambda name, agent=None: {"transient_swallow_max": 2},
        raising=True,
    )

    ext = mod.HandleTransientLLMError(agent=agent)
    for _ in range(2):
        data = {"exception": FakeRateLimit()}
        asyncio.run(ext.execute(data=data))
        assert data["exception"] is None  # inside the configured cap
    # Third consecutive failure is past the configured cap of 2
    data = {"exception": FakeRateLimit()}
    asyncio.run(ext.execute(data=data))
    assert data["exception"] is not None


def test_safety_net_bad_config_falls_back_to_defaults(isolation, monkeypatch):
    """v3.1.1: unreadable knob values keep the historic 5 / 300s bound --
    the safety net must never become the bug."""
    import helpers.plugins as plugin_helpers

    mod = _load_safety_net()
    monkeypatch.setattr(
        plugin_helpers,
        "get_plugin_config",
        lambda name, agent=None: {
            "transient_swallow_max": "garbage",
            "transient_swallow_reset_window_s": -5,
        },
        raising=True,
    )
    assert mod._swallow_limits(None) == (5, 300.0)