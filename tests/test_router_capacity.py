"""Tests for the v2.6.2 ``router`` capacity class (added 2026-08-08).

OmniRoute (``omniroute/*``) is a self-healing local gateway that re-routes the
next request to a healthy upstream tier, so a 429/5xx does NOT mean the model
is broken. ``_classify_capacity`` returns ``"router"`` for it, and the
``_capacity_skips_cooldown`` helper centralizes the "no 429 cooldown / no
primary-skip escalation" policy for ``concurrent_paid`` + ``router``.

Covered:

  1. ``_classify_capacity("omniroute/*")`` -> ``"router"`` (incl. 3-segment
     labels and case-insensitivity).
  2. OpenRouter is NOT a router -- it serves a specific requested model and a
     429 on ``openrouter/x:free`` is a real free-tier limit, so it stays
     ``free_per_minute``.
  3. ``_capacity_skips_cooldown`` is True for ``ollama/*`` + ``omniroute/*``
     and False for everything else.
  4. ``_handle_error_cooldown`` writes NO cooldown (returns False) for a 429
     on an ``omniroute/*`` label.
  5. ``_handle_error_cooldown`` writes a short ``router_cooldown_s`` (default
     5s, NOT the 120s a normal 500 gets) for a 5xx on an ``omniroute/*`` label.
  6. Sanity: a 500 on a non-router label (``nvidia_nim/*``) still gets the
     normal 120s transient cooldown -- the router short-cooldown does not leak.
  7. v2.6.3: ``_resolve_per_call_timeout`` gives a router label the cold
     ``base_timeout_s`` even when it's "warm" -- routers fronting free/slow
     tiers aren't fast warm cloud calls, and the 20s warm ceiling was
     timing out the OmniRoute utility model (``auto/coding:free`` etc.).

To run:
    cd /a0
    REPO_ROOT_OVERRIDE="$(pwd)" pytest usr/plugins/_model_fallback/tests/test_router_capacity.py -v
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
    _capacity_skips_cooldown,
    _handle_error_cooldown,
    _DEFAULT_ROUTER_COOLDOWN_S,
)


# Minimal plugin cfg for the patched ``_get_plugin_cfg``. The router 5xx path
# reads ``router_cooldown_s`` with ``_DEFAULT_ROUTER_COOLDOWN_S`` as fallback,
# so omitting it here exercises the default. Include the other knobs the
# rate-limit / transient branches reference so nothing KeyErrors.
_PATCHED_PLUGIN_CFG = {
    "rate_limit_no_retry_after_cooldown_s": 30.0,
    "cascade_warm_timeout_s": 20.0,
    "cascade_warm_window_s": 600.0,
    "primary_skip_cooldown_s": 120,
    "primary_skip_strikes": 2,
    "health_horizon_s": 60.0,
}


class _FakeRateLimitError(Exception):
    """Stand-in for a LiteLLM 429 (status_code=429, no Retry-After)."""

    status_code = 429

    def __init__(self, message: str = "Rate limited"):
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return self.message


class _FakeTransientError(Exception):
    """Stand-in for a LiteLLM 5xx (status_code=500, plain message). Not a
    rate-limit error (no 429, no rate-limit phrase) and not a permanent fail
    (no overflow/payload/invalid-key), so ``_handle_error_cooldown`` reaches
    the transient tail where the v2.6.2 router short-cooldown lives."""

    status_code = 500

    def __init__(self, message: str = "Internal Server Error"):
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return self.message


class _FakeAuthError(Exception):
    """Stand-in for a LiteLLM 401/403/404 (status_code on the exception).
    ``_is_permanently_failed_model`` treats 401-404 as permanent, so this
    exercises the v2.6.7 Fix B router exemption: a router-class label must
    NOT take the 300s permanent cooldown — it falls through to the short
    ``router_cooldown_s`` block instead."""

    def __init__(self, status_code: int = 401, message: str = "Unauthorized"):
        super().__init__(message)
        self.status_code = status_code
        self.message = message

    def __str__(self) -> str:
        return self.message


class _FakeAgent:
    """Minimal stand-in for an Agent for _handle_error_cooldown."""

    def __init__(self, ctx_id: str = "test-router-agent"):
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


from contextlib import contextmanager


@contextmanager
def _patched_cfg():
    with patch.object(
        _fb_mod, "_get_plugin_cfg", lambda agent: dict(_PATCHED_PLUGIN_CFG)
    ):
        yield


# ---------------------------------------------------------------------------
# Pure classification tests
# ---------------------------------------------------------------------------


def test_classify_omniroute_returns_router():
    """OmniRoute labels classify as ``router`` (the self-healing gateway)."""
    assert _classify_capacity("omniroute/auto") == "router"
    assert _classify_capacity("omniroute/openai/gpt-4o") == "router"
    assert _classify_capacity("omniroute/veo-free/deepseek-r1") == "router"


def test_classify_omniroute_case_insensitive():
    """Provider comparison is case-insensitive (lower() applied)."""
    assert _classify_capacity("OmniRoute/auto") == "router"
    assert _classify_capacity("OMNIROUTE/auto") == "router"


def test_classify_openrouter_stays_free_per_minute():
    """OpenRouter is NOT a router: it serves a specific requested model and a
    429 on a free model is a real free-tier limit, so it stays
    ``free_per_minute``. Guards against over-classifying routers."""
    assert _classify_capacity("openrouter/google/gemma-4-31b-it:free") == "free_per_minute"
    assert _classify_capacity("openrouter/anthropic/claude-3.5-sonnet") == "free_per_minute"


def test_capacity_skips_cooldown_helper():
    """``_capacity_skips_cooldown`` is True for concurrent_paid (ollama) and
    router (omniroute); False for everything else."""
    assert _capacity_skips_cooldown("ollama/llama3.2") is True
    assert _capacity_skips_cooldown("omniroute/auto") is True
    assert _capacity_skips_cooldown("a0_venice/some-model") is False
    assert _capacity_skips_cooldown("openrouter/x:free") is False
    assert _capacity_skips_cooldown("nvidia_nim/meta/llama-3.1-70b") is False
    assert _capacity_skips_cooldown("some-future-provider/x") is False


# ---------------------------------------------------------------------------
# _handle_error_cooldown wiring tests (real function, fake agent)
# ---------------------------------------------------------------------------


def test_handle_error_429_no_cooldown_for_router():
    """A 429 on an ``omniroute/*`` label writes NO cooldown and returns
    False -- the gateway re-routes on the next call, so locking it out would
    be wrong. Mirrors the concurrent_paid (local ollama) skip."""
    agent = _FakeAgent("router-429")
    store = _fb_mod._INMEM_COOLDOWNS
    store[("router-429",)] = {}
    e = _FakeRateLimitError("Rate limited")

    try:
        with _patched_cfg():
            result = _handle_error_cooldown(e, "omniroute/auto", store, agent)
        assert result is False, (
            f"router 429 should return False (no cooldown written), got {result!r}"
        )
        assert store[("router-429",)] == {}, (
            f"store should be empty after router 429, got {store[('router-429',)]}"
        )
    finally:
        store[("router-429",)] = {}


def test_handle_error_500_short_cooldown_for_router():
    """A 5xx on an ``omniroute/*`` label writes the short
    ``router_cooldown_s`` (default 5s) -- NOT the 120s a normal 500 gets --
    so we space out a 5xx storm without locking the router out."""
    agent = _FakeAgent("router-500")
    store = _fb_mod._INMEM_COOLDOWNS
    store[("router-500",)] = {}
    e = _FakeTransientError("Internal Server Error")
    label = "omniroute/auto"

    try:
        before = time.monotonic()
        with _patched_cfg():
            result = _handle_error_cooldown(e, label, store, agent)
        after = time.monotonic()

        assert result is True, "router 5xx should return True (cooldown written)"
        cooldown_until = store[("router-500",)].get(label)
        assert cooldown_until is not None, "router 5xx cooldown should have been written"
        remaining = cooldown_until - before
        # Default router_cooldown_s is 5s; allow slack for the clamp + timing.
        assert remaining <= _DEFAULT_ROUTER_COOLDOWN_S + 1.0, (
            f"router 5xx cooldown should be ~{_DEFAULT_ROUTER_COOLDOWN_S}s, "
            f"got {remaining:.1f}s"
        )
        # The decisive assertion: must NOT be the 120s a normal 500 gets.
        assert remaining < 60.0, (
            f"router 5xx cooldown must be the short router_cooldown_s, not the "
            f"normal 120s transient cooldown; got {remaining:.1f}s"
        )
        assert remaining > 0.0, "router 5xx cooldown should be > 0 (spacing)"
    finally:
        store[("router-500",)] = {}


def test_handle_error_500_normal_cooldown_for_non_router():
    """Sanity: a 500 on a non-router label (``nvidia_nim/*``) still gets the
    normal 120s transient cooldown from ``_DEFAULT_COOLDOWNS_S[500]``. The
    router short-cooldown must not leak to non-router providers."""
    agent = _FakeAgent("nonrouter-500")
    store = _fb_mod._INMEM_COOLDOWNS
    store[("nonrouter-500",)] = {}
    e = _FakeTransientError("Internal Server Error")
    label = "nvidia_nim/meta/llama-3.1-70b-instruct"

    try:
        before = time.monotonic()
        with _patched_cfg():
            result = _handle_error_cooldown(e, label, store, agent)
        assert result is True
        cooldown_until = store[("nonrouter-500",)].get(label)
        assert cooldown_until is not None
        remaining = cooldown_until - before
        # _DEFAULT_COOLDOWNS_S[500] == 120.0; allow timing slack.
        assert 115.0 <= remaining <= 125.0, (
            f"non-router 500 cooldown should be ~120s (the normal transient "
            f"cooldown), got {remaining:.1f}s"
        )
    finally:
        store[("nonrouter-500",)] = {}


def test_handle_error_401_short_cooldown_for_router():
    """v2.6.7 Fix B: a 401 (or 403/404) on an ``omniroute/*`` label is
    ``_is_permanently_failed_model`` True, but a self-healing router must NOT
    take the 300s permanent cooldown — the gateway re-routes the next call
    (and 401 often just means one upstream tier's key tier exhausted). The
    router exemption in the permanent-fail block makes it fall through to
    the short ``router_cooldown_s`` (5s) instead. Guards the regression where
    a gateway 401 locked the router out for 5 minutes per outage."""
    agent = _FakeAgent("router-401")
    store = _fb_mod._INMEM_COOLDOWNS
    store[("router-401",)] = {}
    e = _FakeAuthError(status_code=401, message="Unauthorized")
    label = "omniroute/auto"

    try:
        before = time.monotonic()
        with _patched_cfg():
            result = _handle_error_cooldown(e, label, store, agent)
        after = time.monotonic()

        assert result is True, "router 401 should return True (cooldown written)"
        cooldown_until = store[("router-401",)].get(label)
        assert cooldown_until is not None, "router 401 cooldown should have been written"
        remaining = cooldown_until - before
        # The decisive assertion: the short router_cooldown_s (~5s), NOT the
        # 300s permanent cooldown a non-router 401 would get.
        assert remaining < 60.0, (
            f"router 401 cooldown must be the short router_cooldown_s, not the "
            f"300s permanent cooldown; got {remaining:.1f}s"
        )
        assert remaining <= _DEFAULT_ROUTER_COOLDOWN_S + 1.0, (
            f"router 401 cooldown should be ~{_DEFAULT_ROUTER_COOLDOWN_S}s, "
            f"got {remaining:.1f}s"
        )
    finally:
        store[("router-401",)] = {}


def test_handle_error_401_permanent_cooldown_for_non_router():
    """Sanity companion to Fix B: a 401 on a NON-router label still gets the
    permanent 300s cooldown (``_DEFAULT_COOLDOWNS_S[401]``) -- the router
    exemption must NOT leak to non-router providers."""
    agent = _FakeAgent("nonrouter-401")
    store = _fb_mod._INMEM_COOLDOWNS
    store[("nonrouter-401",)] = {}
    e = _FakeAuthError(status_code=401, message="Unauthorized")
    label = "nvidia_nim/meta/llama-3.1-70b-instruct"

    try:
        before = time.monotonic()
        with _patched_cfg():
            result = _handle_error_cooldown(e, label, store, agent)
        assert result is True
        cooldown_until = store[("nonrouter-401",)].get(label)
        assert cooldown_until is not None
        remaining = cooldown_until - before
        # _DEFAULT_COOLDOWNS_S[401] == 300.0; allow timing slack.
        assert 295.0 <= remaining <= 305.0, (
            f"non-router 401 cooldown should be ~300s (permanent), "
            f"got {remaining:.1f}s"
        )
    finally:
        store[("nonrouter-401",)] = {}


# ---------------------------------------------------------------------------
# v2.6.3: router warm-timeout exemption
# ---------------------------------------------------------------------------


def test_resolve_per_call_timeout_router_uses_cold_even_when_warm():
    """v2.6.3: a router-class label (omniroute/*) gets the cold
    ``base_timeout_s`` even when it's "warm" -- routers fronting free/slow
    tiers aren't fast warm cloud calls, and the 20s warm ceiling was
    timing out the OmniRoute utility model. The warm timestamp must NOT
    coax it into the 20s fast-path."""
    label = "omniroute/auto"
    saved = _fb_mod._WARM_LABELS.get(label)
    try:
        _fb_mod._WARM_LABELS[label] = time.monotonic()  # mark warm
        got = _fb_mod._resolve_per_call_timeout(
            label, base_timeout_s=300.0, warm_timeout_s=20.0, warm_window_s=600.0,
        )
        assert got == 300.0, (
            f"router label must use cold base (300s) even when warm, got {got}"
        )
    finally:
        if saved is None:
            _fb_mod._WARM_LABELS.pop(label, None)
        else:
            _fb_mod._WARM_LABELS[label] = saved


def test_resolve_per_call_timeout_non_router_warm_uses_warm():
    """Sanity: a warm ``free_per_minute`` label still gets the short warm
    timeout -- the fast-path is intact for free-tier providers. Guards
    against the router / unlimited_paid exemptions accidentally disabling
    the warm path for everyone. (``a0_venice`` is ``unlimited_paid`` and is
    intentionally exempt -- see test_resolve_per_call_timeout_unlimited_paid_warm_uses_base.)"""
    label = "ollama_cloud/some-model"
    saved = _fb_mod._WARM_LABELS.get(label)
    try:
        _fb_mod._WARM_LABELS[label] = time.monotonic()  # mark warm
        got = _fb_mod._resolve_per_call_timeout(
            label, base_timeout_s=300.0, warm_timeout_s=20.0, warm_window_s=600.0,
        )
        assert got == 20.0, (
            f"warm free_per_minute must use warm_timeout_s (20s), got {got}"
        )
    finally:
        if saved is None:
            _fb_mod._WARM_LABELS.pop(label, None)
        else:
            _fb_mod._WARM_LABELS[label] = saved


def test_resolve_per_call_timeout_unlimited_paid_warm_uses_base():
    """v2.6.7: a warm ``unlimited_paid`` label (e.g. ``a0_venice`` = Agent Zero
    API) gets the full cold ``base_timeout_s`` even when warm. Paid models
    legitimately run long utility prompts; the 20s warm ceiling was timing
    them out on the 2nd call and forcing a 300s cooldown, so the agent
    dropped off after ONE successful request. Only ``free_per_minute``
    labels keep the aggressive warm fast-path."""
    label = "a0_venice/some-model"
    saved = _fb_mod._WARM_LABELS.get(label)
    try:
        _fb_mod._WARM_LABELS[label] = time.monotonic()  # mark warm
        got = _fb_mod._resolve_per_call_timeout(
            label, base_timeout_s=300.0, warm_timeout_s=20.0, warm_window_s=600.0,
        )
        assert got == 300.0, (
            f"warm unlimited_paid must use cold base (300s), got {got}"
        )
    finally:
        if saved is None:
            _fb_mod._WARM_LABELS.pop(label, None)
        else:
            _fb_mod._WARM_LABELS[label] = saved


def test_resolve_per_call_timeout_cold_label_uses_base():
    """A label with no warm timestamp (cold) gets the base timeout --
    unchanged legacy behavior (router exemption is warm-only)."""
    label = "a0_venice/__never_warm_test__"
    _fb_mod._WARM_LABELS.pop(label, None)
    try:
        got = _fb_mod._resolve_per_call_timeout(
            label, base_timeout_s=300.0, warm_timeout_s=20.0, warm_window_s=600.0,
        )
        assert got == 300.0, f"cold label must use base (300s), got {got}"
    finally:
        _fb_mod._WARM_LABELS.pop(label, None)


def test_classify_a0_venice_api_base_unlimited_paid():
    """v2.6.8: the a0_venice primary ("Agent Zero API") is registered with
    ``litellm_provider: openai``, so its cascade label arrives as
    ``openai/deepseek-v4-flash`` -- the ``a0_venice`` provider-segment check
    never matches and it would fall through to ``free_per_minute``. The
    durable signal is its api_base (``llm.agent-zero.ai``): classify by where
    the call actually goes, so the warm-path exemption reaches it."""
    # Without the api_base, the openai-aliased label is free_per_minute
    # (the regression we're guarding against).
    assert _classify_capacity("openai/deepseek-v4-flash") == "free_per_minute"
    # With the a0_venice api_base, it is unlimited_paid.
    assert _classify_capacity(
        "openai/deepseek-v4-flash", api_base="https://llm.agent-zero.ai/v1"
    ) == "unlimited_paid"
    # Substring match -- trailing path / scheme variants all hit.
    assert _classify_capacity(
        "openai/deepseek-v4-flash", api_base="http://llm.agent-zero.ai:443/v1/chat"
    ) == "unlimited_paid"
    # Case-insensitive.
    assert _classify_capacity(
        "openai/deepseek-v4-flash", api_base="HTTPS://LLM.AGENT-ZERO.AI/V1"
    ) == "unlimited_paid"
    # A genuinely-free openai-aliased label with a different api_base stays
    # free_per_minute (the matcher is specific, not a blanket openai catch).
    assert _classify_capacity(
        "openai/some-free-model", api_base="https://free.example.com/v1"
    ) == "free_per_minute"


def test_resolve_per_call_timeout_a0_venice_api_base_warm_uses_base():
    """v2.6.8 end-to-end: a warm ``openai/deepseek-v4-flash`` whose api_base is
    the a0_venice endpoint (``llm.agent-zero.ai``) gets the full cold
    ``base_timeout_s`` -- NOT the 20s warm ceiling. This is the real
    "works once, then drops off after one request" symptom: the 2nd call
    timed out at 20s -> 300s cooldown. The api_base matcher classifies it
    ``unlimited_paid`` so the warm-path exemption applies."""
    label = "openai/deepseek-v4-flash"
    api_base = "https://llm.agent-zero.ai/v1"
    saved = _fb_mod._WARM_LABELS.get(label)
    try:
        _fb_mod._WARM_LABELS[label] = time.monotonic()  # mark warm
        got = _fb_mod._resolve_per_call_timeout(
            label, base_timeout_s=300.0, warm_timeout_s=20.0, warm_window_s=600.0,
            api_base=api_base,
        )
        assert got == 300.0, (
            f"warm a0_venice (api_base-matched unlimited_paid) must use cold "
            f"base (300s), got {got}"
        )
    finally:
        if saved is None:
            _fb_mod._WARM_LABELS.pop(label, None)
        else:
            _fb_mod._WARM_LABELS[label] = saved