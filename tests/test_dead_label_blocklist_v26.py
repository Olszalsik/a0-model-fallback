"""Tests for the v2.6.8 cross-agent permanent-fail blocklist.

_INMEM_DEAD_LABELS is the inverse of _INMEM_HEALTHY_LABELS: when ANY agent
in the process hits a genuinely permanent failure on a label (404 gone,
401/402/403 auth-quota, 400-invalid-key), the label is marked dead in a
shared process-global index so other agents skip it without re-paying the
failure tax. The user's 2026-08-25 caveat is enforced: entries EXPIRE
(5 min for 401/403 quota -> Venice's daily free tier is re-probed and
recovers after midnight; 24 h for a gone 404 slug), so this is never a
forever-block. 429 / 5xx / pure timeouts are excluded (transient).

These tests are pure-Python and exercise the helpers directly. To run:
    cd /a0
    REPO_ROOT_OVERRIDE="$(pwd)" pytest \
        usr/plugins/model_fallback/tests/test_dead_label_blocklist_v26.py -v
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(os.environ.get("REPO_ROOT_OVERRIDE") or "/a0")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from usr.plugins.model_fallback import fallback as _fb_mod
from usr.plugins.model_fallback.fallback import (
    _is_dead_for_all_agents,
    _is_label_dead,
    _mark_label_dead,
    _mark_label_healthy,
)


def _dead_labels() -> dict:
    """Read the module's _INMEM_DEAD_LABELS through the module attribute so
    we always hit the current module instance (other tests reload
    fallback.py between runs)."""
    return _fb_mod._INMEM_DEAD_LABELS


def _healthy_labels() -> dict:
    return _fb_mod._INMEM_HEALTHY_LABELS


class _FakeExc(Exception):
    """Exception with a settable status_code, mirroring LiteLLM errors."""

    def __init__(self, status_code, message=""):
        super().__init__(message)
        self.status_code = status_code


class _FakeAgent:
    """Minimal stand-in for an Agent (only the .context.log attribute is
    touched by _mark_label_dead's optional log)."""

    def __init__(self):
        class _Ctx:
            class log:
                @staticmethod
                def log(*a, **k):
                    pass

        self.context = _Ctx()


# --- _is_dead_for_all_agents: the qualifying set ----------------------------


def test_dead_set_includes_404_and_auth_quota():
    """404/401/402/403 are dead for every agent (shared key/credentials)."""
    for sc in (401, 402, 403, 404):
        assert _is_dead_for_all_agents(_FakeExc(sc)) is True, sc


def test_dead_set_excludes_transient_statuses():
    """429 / 5xx / no-status must NOT be cross-agent blocked."""
    for sc in (429, 500, 502, 503, 504, 408, None):
        assert _is_dead_for_all_agents(_FakeExc(sc)) is False, sc


def test_dead_set_includes_invalid_api_key_400():
    """A 400 whose body signals an auth/key problem is dead for all agents."""
    exc = _FakeExc(400, "invalid api key provided")
    assert _is_dead_for_all_agents(exc) is True


def test_dead_set_excludes_plain_400():
    """A 400 that is NOT an auth/key problem (e.g. bad request body) is
    not a shared-permanent failure -- it may be prompt-specific."""
    assert _is_dead_for_all_agents(_FakeExc(400, "bad json")) is False


# --- _mark_label_dead + _is_label_dead: expiry is the "not forever" guarantee


def test_mark_dead_then_skip():
    label = "test-dead-omniroute/utility-free"
    _dead_labels().pop(label, None)
    _mark_label_dead(label, _FakeExc(404), _FakeAgent())
    assert _is_label_dead(label) is True
    # Entry is (status, expires_at).
    status, expires_at = _dead_labels()[label]
    assert status == 404
    assert expires_at > time.monotonic()
    _dead_labels().pop(label, None)


def test_mark_dead_noops_for_transient():
    """A 429 must not land in the shared blocklist."""
    label = "test-dead-transient"
    _dead_labels().pop(label, None)
    _mark_label_dead(label, _FakeExc(429), _FakeAgent())
    assert label not in _dead_labels()
    assert _is_label_dead(label) is False


def test_dead_entry_expires_and_auto_evicts():
    """The user's 'don't block forever' caveat: an expired entry is evicted
    on read and the label becomes eligible again."""
    label = "test-dead-expiry"
    _dead_labels().pop(label, None)
    _mark_label_dead(label, _FakeExc(404), _FakeAgent())
    # Force the entry into the past.
    status, _expires = _dead_labels()[label]
    _dead_labels()[label] = (status, time.monotonic() - 1.0)
    assert _is_label_dead(label) is False
    # Auto-evicted.
    assert label not in _dead_labels()


def test_venice_no_quota_uses_short_5min_window():
    """Venice's 403 'no quota left' must get the 5-min (300s) window, NOT
    24h, so it re-probes and recovers after midnight UTC."""
    label = "a0_venice/deepseek-v4-flash"
    _dead_labels().pop(label, None)
    _mark_label_dead(label, _FakeExc(403), _FakeAgent())
    _status, expires_at = _dead_labels()[label]
    dur = expires_at - time.monotonic()
    # 300s default for 403, floored at 60s, capped at 86400. Allow slack.
    assert 60.0 <= dur <= 360.0, f"Venice no-quota dead window {dur}s"
    _dead_labels().pop(label, None)


def test_404_gone_uses_long_window():
    """A gone 404 slug gets the 24h window (truly gone)."""
    label = "test-dead-404-gone"
    _dead_labels().pop(label, None)
    _mark_label_dead(label, _FakeExc(404), _FakeAgent())
    _status, expires_at = _dead_labels()[label]
    dur = expires_at - time.monotonic()
    assert dur >= 80000.0, f"404 dead window {dur}s expected ~24h"
    _dead_labels().pop(label, None)


# --- symmetry: success clears dead, dead clears healthy ---------------------


def test_mark_healthy_clears_dead_entry():
    """A success on a previously-dead label (Venice quota refreshed after
    midnight, or a 404 model came back) must clear the dead blocklist so
    the label is used again immediately."""
    label = "test-dead-then-healthy"
    _dead_labels().pop(label, None)
    _healthy_labels().pop(label, None)
    _mark_label_dead(label, _FakeExc(403), _FakeAgent())
    assert _is_label_dead(label) is True
    _mark_label_healthy(_FakeAgent(), label)
    assert _is_label_dead(label) is False
    assert label not in _dead_labels()
    _dead_labels().pop(label, None)
    _healthy_labels().pop(label, None)


def test_mark_dead_clears_healthy_entry():
    """A fresh dead-mark supersedes a stale healthy signal (symmetric)."""
    label = "test-healthy-then-dead"
    _dead_labels().pop(label, None)
    _healthy_labels().pop(label, None)
    _mark_label_healthy(_FakeAgent(), label)
    assert label in _healthy_labels()
    _mark_label_dead(label, _FakeExc(404), _FakeAgent())
    assert label not in _healthy_labels()
    _dead_labels().pop(label, None)
    _healthy_labels().pop(label, None)