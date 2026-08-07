"""Tests for v2.6 per-provider capacity inference (added 2026-07-28).

``_classify_capacity(label)`` returns one of:

  - ``concurrent_paid``     -- e.g. local ``ollama/*``. 429 means a competing
    agent is holding the slot; release in seconds. No cooldown written.
  - ``unlimited_paid``      -- e.g. ``a0_venice/*``. Quota-exhaustion 429, use
    configured cooldown (default 30s, same as free tier per user's
    2026-07-28 conservative call).
  - ``free_per_minute``     -- default for everything else (nvidia_nim,
    openrouter, groq, mistral, cohere, together, ollama_cloud, ...).
    Use configured cooldown.

Three wiring sites in fallback.py are also exercised:

  1. ``_handle_error_cooldown`` rate-limit branch returns False (no cooldown
     written) for ``concurrent_paid`` labels.
  2. ``_handle_error_cooldown`` writes the configured cooldown for
     ``unlimited_paid`` and ``free_per_minute`` (NOT 24h, NOT the legacy 60s).
  3. ``_maybe_extend_primary_cooldown`` returns immediately for
     ``concurrent_paid`` labels (no escalation, no counter increment).

To run:
    cd /a0
    REPO_ROOT_OVERRIDE="$(pwd)" pytest usr/plugins/_model_fallback/tests/test_capacity_v26.py -v
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(os.environ.get("REPO_ROOT_OVERRIDE") or "/a0")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from usr.plugins._model_fallback import fallback as _fb_mod
from usr.plugins._model_fallback.fallback import (
    _classify_capacity,
    _handle_error_cooldown,
    _DEFAULT_RATE_LIMIT_NO_RETRY_AFTER_COOLDOWN_S,
)


# Test-time patch for ``_get_plugin_cfg``. The real implementation walks
# the project filesystem looking for ``_model_fallback/config.json``
# (~32s wall-clock in tests). The contract under test is the rate-limit
# branch's cooldown arithmetic, not plugin-config discovery; a minimal
# patch lets the rate-limit branch run its real ``dur = ...`` lines in
# microseconds.
_PATCHED_PLUGIN_CFG = {
    "rate_limit_no_retry_after_cooldown_s":
        _DEFAULT_RATE_LIMIT_NO_RETRY_AFTER_COOLDOWN_S,
    "cascade_warm_timeout_s": 20.0,
    "cascade_warm_window_s": 600.0,
    "primary_skip_cooldown_s": 120,
    "primary_skip_strikes": 2,
    "health_horizon_s": 60.0,
}


class _FakeRateLimitError(Exception):
    """Stand-in for a LiteLLM 429. Carries status_code=429 and an empty
    Retry-After header so the rate-limit branch uses the configured
    no-Retry-After cooldown."""

    status_code = 429

    def __init__(self, message: str = "Rate limited"):
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return self.message


class _FakeAgent:
    """Minimal stand-in for an Agent for _handle_error_cooldown."""

    def __init__(self, ctx_id: str = "test-agent"):
        class _Ctx:
            id = ctx_id

            class log:
                @staticmethod
                def log(level: str, msg: str):
                    pass

        self.context = _Ctx()
        self.config = type("_Cfg", (), {"profile": ""})()

    def get_data(self, key, default=None):
        return default

    def set_data(self, key, value):
        pass


# Patch context manager the tests use to skip the 32s filesystem walk.
# Apply with ``with _patched_cfg():``.
from contextlib import contextmanager

@contextmanager
def _patched_cfg():
    with patch.object(
        _fb_mod, "_get_plugin_cfg", lambda agent: dict(_PATCHED_PLUGIN_CFG)
    ):
        yield


# ---------------------------------------------------------------------------
# Pure _classify_capacity tests (no framework state)
# ---------------------------------------------------------------------------


def test_capacity_ollama_returns_concurrent_paid():
    """Local ollama on localhost is genuinely concurrent; release in
    seconds on 429, don't cooldown."""
    assert _classify_capacity("ollama/gemma4:31b-cloud") == "concurrent_paid"
    assert _classify_capacity("ollama/llama3.2") == "concurrent_paid"
    assert _classify_capacity("ollama/qwen2.5-coder:7b") == "concurrent_paid"


def test_capacity_ollama_cloud_returns_free_per_minute():
    """``ollama_cloud`` and any other ``ollama_*`` provider is metered
    (NOT local), so a 429 is real throttling."""
    assert _classify_capacity("ollama_cloud/nemotron-3-super:cloud") == "free_per_minute"
    assert _classify_capacity("ollama_production/some-model") == "free_per_minute"


def test_capacity_a0_venice_returns_unlimited_paid():
    """Venice's daily credit window is generous but a quota-exhaustion
    429 still means we're done for now."""
    assert _classify_capacity("a0_venice/google-gemma-4-26b-a4b-it") == "unlimited_paid"
    assert _classify_capacity("a0_venice/llama-3.3-70b") == "unlimited_paid"


def test_capacity_openrouter_three_segment_returns_free_per_minute():
    """Three-segment labels (provider/model/variant) work with the
    first-segment rule."""
    assert _classify_capacity(
        "openrouter/google/gemma-4-31b-it:free"
    ) == "free_per_minute"
    assert _classify_capacity(
        "openrouter/anthropic/claude-3.5-sonnet"
    ) == "free_per_minute"


def test_capacity_nvidia_nim_three_segment_returns_free_per_minute():
    """Real production label from the user's setup."""
    assert _classify_capacity(
        "nvidia_nim/stepfun-ai/step-3.7-flash"
    ) == "free_per_minute"
    assert _classify_capacity(
        "nvidia_nim/meta/llama-3.1-70b-instruct"
    ) == "free_per_minute"


def test_capacity_unknown_provider_returns_free_per_minute():
    """Unknown providers default to ``free_per_minute`` (safe)."""
    assert _classify_capacity("meta-llama/llama-3") == "free_per_minute"
    assert _classify_capacity("anthropic/claude-3-opus") == "free_per_minute"
    assert _classify_capacity("some-future-provider/some-model") == "free_per_minute"


def test_capacity_label_without_slash_returns_free_per_minute():
    """A label without a ``/`` has no provider prefix; safe default
    is ``free_per_minute``."""
    assert _classify_capacity("gpt-4") == "free_per_minute"
    assert _classify_capacity("claude-3-opus") == "free_per_minute"


def test_capacity_case_insensitive():
    """Provider comparison is case-insensitive (lower() applied)."""
    assert _classify_capacity("OLLAMA/llama3.2") == "concurrent_paid"
    assert _classify_capacity("Ollama/llama3.2") == "concurrent_paid"
    assert _classify_capacity("A0_Venice/foo") == "unlimited_paid"


# ---------------------------------------------------------------------------
# _handle_error_cooldown wiring tests (real function, fake agent)
# ---------------------------------------------------------------------------


def test_handle_error_cooldown_no_op_for_concurrent_paid():
    """For a local ollama label, a 429 produces NO cooldown written
    and the function returns False."""
    agent = _FakeAgent("test-agent-1")
    store = _fb_mod._INMEM_COOLDOWNS
    store[("test-agent-1",)] = {}
    e = _FakeRateLimitError("Rate limited")

    try:
        with _patched_cfg():
            result = _handle_error_cooldown(e, "ollama/llama3.2", store, agent)
        assert result is False, (
            f"concurrent_paid 429 should return False (no cooldown written), "
            f"got {result!r}"
        )
        # Store must remain empty -- no cooldown entry, no _last_status entry.
        assert store[("test-agent-1",)] == {}, (
            f"store should be empty after concurrent_paid 429, "
            f"got {store[('test-agent-1',)]}"
        )
    finally:
        store[("test-agent-1",)] = {}


def test_handle_error_cooldown_30s_for_unlimited_paid():
    """For a Venice label, a 429 (no Retry-After) writes the configured
    ``rate_limit_no_retry_after_cooldown_s`` cooldown (default 30s, NOT
    24h, NOT the legacy 60s default)."""
    agent = _FakeAgent("test-agent-2")
    store = _fb_mod._INMEM_COOLDOWNS
    store[("test-agent-2",)] = {}
    e = _FakeRateLimitError("Quota exhausted")

    try:
        before = time.monotonic()
        with _patched_cfg():
            result = _handle_error_cooldown(
                e, "a0_venice/google-gemma-4-26b-a4b-it", store, agent,
            )
        after = time.monotonic()

        assert result is True
        cooldown_until = store[("test-agent-2",)].get(
            "a0_venice/google-gemma-4-26b-a4b-it"
        )
        assert cooldown_until is not None, "cooldown should have been written"
        cooldown_remaining = cooldown_until - before
        assert (
            _DEFAULT_RATE_LIMIT_NO_RETRY_AFTER_COOLDOWN_S - 1.0
            <= cooldown_remaining
            <= _DEFAULT_RATE_LIMIT_NO_RETRY_AFTER_COOLDOWN_S + 1.0
        ), (
            f"Venice cooldown should be ~{_DEFAULT_RATE_LIMIT_NO_RETRY_AFTER_COOLDOWN_S}s "
            f"(conservative per user's 2026-07-28 call), got {cooldown_remaining:.1f}s"
        )
        # Must NOT be 24h
        assert cooldown_remaining < 3600.0, (
            "Venice cooldown must NOT be 24h -- user picked conservative 30s"
        )
    finally:
        store[("test-agent-2",)] = {}


def test_handle_error_cooldown_30s_for_free_per_minute():
    """For an OpenRouter label, a 429 (no Retry-After) writes the same
    configured cooldown as Venice (30s default)."""
    agent = _FakeAgent("test-agent-3")
    store = _fb_mod._INMEM_COOLDOWNS
    store[("test-agent-3",)] = {}
    e = _FakeRateLimitError("Free tier limit")

    try:
        before = time.monotonic()
        with _patched_cfg():
            result = _handle_error_cooldown(
                e, "openrouter/anthropic/claude-3.5-sonnet", store, agent,
            )
        assert result is True
        cooldown_until = store[("test-agent-3",)].get(
            "openrouter/anthropic/claude-3.5-sonnet"
        )
        assert cooldown_until is not None
        cooldown_remaining = cooldown_until - before
        assert (
            _DEFAULT_RATE_LIMIT_NO_RETRY_AFTER_COOLDOWN_S - 1.0
            <= cooldown_remaining
            <= _DEFAULT_RATE_LIMIT_NO_RETRY_AFTER_COOLDOWN_S + 1.0
        ), (
            f"free_per_minute cooldown should be ~"
            f"{_DEFAULT_RATE_LIMIT_NO_RETRY_AFTER_COOLDOWN_S}s, got "
            f"{cooldown_remaining:.1f}s"
        )
    finally:
        store[("test-agent-3",)] = {}


def test_handle_error_cooldown_no_op_for_ollama_cloud():
    """Sanity: ``ollama_cloud/*`` (metered, not local) DOES get the
    configured cooldown. This is the user-flagged split."""
    agent = _FakeAgent("test-agent-4")
    store = _fb_mod._INMEM_COOLDOWNS
    store[("test-agent-4",)] = {}
    e = _FakeRateLimitError("ollama cloud metered")

    try:
        before = time.monotonic()
        with _patched_cfg():
            result = _handle_error_cooldown(
                e, "ollama_cloud/nemotron-3-super:cloud", store, agent,
            )
        assert result is True
        cooldown_until = store[("test-agent-4",)].get(
            "ollama_cloud/nemotron-3-super:cloud"
        )
        assert cooldown_until is not None
        cooldown_remaining = cooldown_until - before
        assert (
            _DEFAULT_RATE_LIMIT_NO_RETRY_AFTER_COOLDOWN_S - 1.0
            <= cooldown_remaining
            <= _DEFAULT_RATE_LIMIT_NO_RETRY_AFTER_COOLDOWN_S + 1.0
        ), (
            f"ollama_cloud (metered) cooldown should be ~"
            f"{_DEFAULT_RATE_LIMIT_NO_RETRY_AFTER_COOLDOWN_S}s, got "
            f"{cooldown_remaining:.1f}s"
        )
    finally:
        store[("test-agent-4",)] = {}