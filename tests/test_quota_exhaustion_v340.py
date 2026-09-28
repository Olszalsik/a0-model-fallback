"""v3.4.0 -- quota-exhaustion awareness (the ollama.com "session usage limit" case).

Reproduces the reported production failure and pins the new contract:

  * ollama.com answers an exhausted plan with a 429 whose message is
    "you (X) have reached your session usage limit, upgrade for higher
    limits ... or add usage credits ..." and, critically, with NO Retry-After
    and NO reset timestamp.
  * The plugin used to treat that as a transient 429: book 30s, re-attempt
    every 30s (each attempt paying litellm's own two retries), and when every
    candidate was down raise ``RetryAfterHours(retry_after=60)`` whose message
    reads "All model candidates are currently unavailable" -- a hard-stopper
    error for a condition that heals on its own.

Run from repo root:
    python -m pytest usr/plugins/model_fallback/tests/test_quota_exhaustion_v340.py -q
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(os.environ.get("REPO_ROOT_OVERRIDE") or Path(__file__).resolve().parents[4])
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from usr.plugins.model_fallback import fallback as fb  # noqa: E402
from usr.plugins.model_fallback import models_ext as mx  # noqa: E402

# The verbatim error from the production report.
OLLAMA_MSG = (
    "litellm.exceptions.RateLimitError: litellm.RateLimitError: RateLimitError: "
    "OpenAIException - you (olszalsik) have reached your session usage limit, "
    "upgrade for higher limits: https://ollama.com/upgrade or add usage "
    "credits: https://ollama.com/settings (ref: a4d87270-9d87-4886-b50c-503914cf4b38) "
    "LiteLLM Retried: 2 times"
)


class RateLimit(Exception):
    status_code = 429


class Bare(Exception):
    """A 429 as re-wrapped by litellm -- message only, no status_code."""


class FakeAgent:
    def __init__(self, ctx_id: str = "test-agent-quota"):
        self._data = {}
        self.config = type("_Cfg", (), {"profile": ""})()

        class _Ctx:
            id = ctx_id

            class log:
                @staticmethod
                def log(level, msg, **kw):
                    pass

        self.context = _Ctx()

    def get_data(self, key, default=None):
        return self._data.get(key, default)

    def set_data(self, key, value):
        self._data[key] = value


def _cfg(**over):
    base = {
        "rate_limit_no_retry_after_cooldown_s": 30.0,
        "quota_cooldown_max_s": 43200.0,
    }
    base.update(over)
    return base


def _cooldown_for(exc, agent=None, cfg=None):
    """Book a cooldown for `exc` and return its duration in seconds."""
    store = fb._get_cooldown_store(agent)
    store.clear()
    with patch.object(fb, "_get_plugin_cfg", lambda *a, **k: dict(cfg or _cfg())):
        fb._handle_error_cooldown(exc, "some/model", store, agent)
    until = store.get("some/model")
    if until is None:
        return None
    return until - time.monotonic()


# --- Q1/Q2: classification --------------------------------------------------


def test_ollama_session_usage_limit_is_quota_exhaustion():
    e = RateLimit(OLLAMA_MSG)
    assert mx._is_rate_limited_error(e)
    assert mx._is_quota_exhaustion_error(e)
    assert mx.detect_quota_scope(e) == "session"


def test_quota_detected_without_status_code():
    # litellm re-wraps 429s into OpenAIError / MidStreamFallbackError, which
    # do not carry status_code. Phrase detection is the only signal left.
    e = Bare("Error code: 429 - you have reached your session usage limit")
    assert getattr(e, "status_code", None) != 429
    assert mx._is_rate_limited_error(e), "must still read as a rate limit"
    assert mx._is_quota_exhaustion_error(e), "must still read as a quota exhaustion"
    assert mx.detect_quota_scope(e) == "session"


# --- Q3: never rotation-permanent ------------------------------------------


def test_quota_is_not_rotation_permanent():
    # A quota error IS cooldown-"permanent" (book it, move on) but must NOT be
    # permanent for ROTATION, which is what would re-raise the raw
    # RateLimitError past the structured RetryAfterHours handling.
    e = RateLimit(OLLAMA_MSG)
    assert mx._is_permanently_failed_model(e)


# --- Q6: reset-header parsing -----------------------------------------------


def test_retry_after_header_variants():
    def probe(headers, on_response=False):
        e = RateLimit("boom")
        if on_response:
            e.response = type("R", (), {"headers": headers})()
        else:
            e.headers = headers
        return mx.extract_retry_after_seconds(e)

    assert probe({"Retry-After": "120"}) == 120.0
    assert probe({"retry-after": "77"}) == 77.0, "must be case-insensitive"
    assert probe({"x-ratelimit-reset-requests": "8.64s"}) == 8.64
    assert probe({"x-ratelimit-reset-tokens": "1m12s"}) == 72.0
    assert probe({"retry-after": "45"}, on_response=True) == 45.0, (
        "headers set only on .response.headers were previously invisible"
    )
    # x-ratelimit-reset as an epoch must convert to a delta, not be used raw.
    epoch = probe({"x-ratelimit-reset": str(int(time.time()) + 300)})
    assert epoch is not None and 295.0 < epoch < 305.0, epoch
    # HTTP-date form of Retry-After (RFC 9110).
    assert probe({"Retry-After": "Wed, 21 Oct 2099 07:28:00 GMT"}) is not None
    # A genuine Retry-After must win over an ambiguous x-ratelimit-reset
    # regardless of dict ordering.
    assert probe({"x-ratelimit-reset": "600", "Retry-After": "30"}) == 30.0

    # v3.4.1 regression: real litellm exceptions carry httpx.Headers, which
    # subclasses collections.abc.Mapping -- NOT dict. The old
    # isinstance(headers, dict) check made this entire subsystem yield
    # nothing for exactly the exception types it was written for, while
    # every variant above passed because they all used plain dicts.
    try:
        import httpx

        httpx_headers = probe(httpx.Headers({"Retry-After": "90"}))
        assert httpx_headers == 90.0, (
            "httpx.Headers must be accepted (it is a Mapping, not a dict)"
        )
    except ImportError:
        pass  # httpx is a litellm dependency; absent only in bare envs


# --- Q7: hints, caps and config overrides -----------------------------------


def test_provider_hint_beats_guessed_scope():
    agent = FakeAgent("q7")
    e = RateLimit(OLLAMA_MSG)
    e.headers = {"x-ratelimit-reset-requests": "3h"}
    dur = _cooldown_for(e, agent)
    assert dur is not None and 10790.0 < dur < 10810.0, dur


def test_quota_cooldown_is_capped():
    agent = FakeAgent("q7b")
    e = RateLimit(OLLAMA_MSG)
    e.headers = {"retry-after": "999999"}  # absurd header
    dur = _cooldown_for(e, agent, cfg=_cfg(quota_cooldown_max_s=3600.0))
    assert dur is not None and dur <= 3601.0, dur


def test_scope_overrides_from_config():
    agent = FakeAgent("q7c")
    dur = _cooldown_for(
        RateLimit(OLLAMA_MSG),
        agent,
        cfg=_cfg(quota_scope_cooldown_s={"session": 1234.0}),
    )
    assert dur is not None and 1230.0 < dur < 1240.0, dur


# --- Q8: the user-facing message --------------------------------------------


def test_retry_after_hours_quota_message_is_not_a_hard_stopper():
    e = fb.RetryAfterHours(retry_after=900, reason="quota_session")
    text = str(e)
    assert e.reason == "quota_session"
    assert "quota" in text.lower()
    assert "All model candidates are currently unavailable" not in text, (
        "a self-healing quota window must not read as a total failure"
    )
    assert "15 min" in text, text


def test_retry_after_hours_transient_message_unchanged():
    e = fb.RetryAfterHours(retry_after=60)
    assert "All model candidates are currently unavailable" in str(e)
    assert e.reason == ""


# --- Q10: the n<=1 quota branch must survive a config-read failure ----------


def test_quota_branch_survives_config_read_failure():
    """Regression for a real bug found in review.

    The turn cascade's ``n <= 1`` branch read plugin config inside a
    try/except and then, outside it, dereferenced ``cfg0`` in the v3.4.0
    quota branch. If the config read raised, ``cfg0`` was never bound and
    the expression raised ``NameError`` -- crashing the exact code path
    whose entire purpose is to keep the agent alive.

    ``cfg0`` is now pre-initialised to ``{}`` and the cap read has its own
    fallback, so a config failure degrades to the module default instead of
    raising.
    """
    import inspect

    src = inspect.getsource(fb._patched_call_chat_model_turn)

    # cfg0 must be bound before the try/except that may fail.
    assert "cfg0: dict = {}" in src, (
        "cfg0 must be pre-initialised before the config try/except"
    )
    # The quota cap read must be independently guarded.
    assert "cap = _DEFAULT_QUOTA_COOLDOWN_MAX_S" in src, (
        "the quota cap read needs its own fallback"
    )
    # No bare `float(cfg0.get(` dereference may remain unguarded.
    assert "float(cfg0.get(" not in src, (
        "a bare float(cfg0.get(...)) outside the guard is the NameError bug"
    )


def test_quota_delay_is_defined_when_config_is_unavailable():
    """Behavioural half of the regression: with config reading broken, the
    quota delay computation still yields a sane positive value."""
    with patch.object(fb, "_get_plugin_cfg", lambda *a, **k: {}):
        dur = fb._quota_cooldown_seconds(RateLimit(OLLAMA_MSG), None)
    assert dur is not None and dur > 0, dur


def test_quota_cooldown_max_cannot_be_bypassed_by_nonsense_override():
    agent = FakeAgent("q10b")
    dur = _cooldown_for(
        RateLimit(OLLAMA_MSG),
        agent,
        cfg=_cfg(quota_cooldown_max_s=10.0),  # below the 60s floor
    )
    # The 60s floor still applies, but must not exceed the cap by orders.
    assert dur is not None and dur <= 61.0, dur


# --- Q4: transient 429s are not quota ---------------------------------------


def test_transient_429_is_not_quota():
    for msg in (
        "RateLimitError: 429 Too Many Requests",
        "Rate limit exceeded, slow down",
        "The server is busy, try again later",
    ):
        assert mx._is_rate_limited_error(RateLimit(msg)), msg
        assert mx.detect_quota_scope(RateLimit(msg)) == "", msg
        assert not mx._is_quota_exhaustion_error(RateLimit(msg)), msg


# --- Q5: the booking is scope-aware, not 30s --------------------------------


def test_quota_exhaustion_books_scope_cooldown_not_30s():
    agent = FakeAgent("q5")
    dur = _cooldown_for(RateLimit(OLLAMA_MSG), agent)
    assert dur is not None, "a cooldown must be booked"
    assert 890.0 < dur < 910.0, f"expected ~900s session quota cooldown, got {dur}"


def test_transient_429_still_books_30s():
    agent = FakeAgent("q5b")
    dur = _cooldown_for(RateLimit("429 Too Many Requests"), agent)
    assert dur is not None
    assert 29.0 < dur < 31.0, f"transient 429 must keep the 30s cooldown, got {dur}"
