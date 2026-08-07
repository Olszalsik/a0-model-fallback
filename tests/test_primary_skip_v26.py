"""Tests for the v2.6 "reset primary-skip on cross-agent healthy signal" rule.

Added 2026-07-28. When ``_INMEM_HEALTHY_LABELS[label]`` shows that a
peer agent has recently proven this label healthy (within
``health_horizon_s``), and the local per-agent cooldown is stale (the
only-cleared-never-overwritten invariant from v2.5.2), this agent's
local strike counter resets to zero on the next primary failure.

The behavior under test is the *contract* of
``_maybe_clear_cooldown_for_healthy_label`` -- the actual Phase 5
change in fallback.py just wires this existing helper into the
``_maybe_extend_primary_cooldown`` short-circuit path. These tests
verify the contract that Phase 5 relies on.

To run:
    cd /a0
    REPO_ROOT_OVERRIDE="$(pwd)" pytest usr/plugins/_model_fallback/tests/test_primary_skip_v26.py -v
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(os.environ.get("REPO_ROOT_OVERRIDE") or "/a0")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from usr.plugins._model_fallback import fallback as _fb_mod
from usr.plugins._model_fallback.fallback import (
    _maybe_clear_cooldown_for_healthy_label,
    _INMEM_HEALTHY_LABELS,
)


def _healthy_labels() -> dict:
    """Read the current module's _INMEM_HEALTHY_LABELS through the module
    attribute so we always hit the current module instance (test_candidate_
    normalize.py reloads fallback.py between tests)."""
    return _fb_mod._INMEM_HEALTHY_LABELS


class _FakeAgent:
    """Minimal stand-in for an Agent for the helper signature."""

    def __init__(self):
        # The helper reads agent.data via _get_cooldown_store, which
        # uses getattr(agent, "context", None).context.id. We mock that.
        class _Ctx:
            id = "test-agent"
        self.context = _Ctx()


def test_primary_skip_reset_when_healthy_index_set():
    """When _INMEM_HEALTHY_LABELS[label] is set (peer agent proved
    healthy) AND the local cooldown is stale (only-cleared-never-
    overwritten invariant), _maybe_clear_cooldown_for_healthy_label
    returns True -- this is the trigger that Phase 5 wires into the
    primary-skip short-circuit."""
    label = "test-primary-label"
    agent = _FakeAgent()
    healthy_labels = _healthy_labels()

    # Set up: local cooldown is stale (in the past), peer healthy index
    # is in the future.
    _fb_mod._INMEM_COOLDOWNS[("test-agent",)] = {label: time.monotonic() - 60.0}
    healthy_labels[label] = time.monotonic() + 60.0

    try:
        cleared = _maybe_clear_cooldown_for_healthy_label(agent, label)
        assert cleared is True, (
            "stale local cooldown + active peer healthy index should clear"
        )
        # Verify the local cooldown was actually popped
        assert label not in _fb_mod._INMEM_COOLDOWNS[("test-agent",)]
    finally:
        healthy_labels.pop(label, None)
        _fb_mod._INMEM_COOLDOWNS[("test-agent",)] = {}


def test_primary_skip_does_not_reset_when_local_cooldown_active():
    """When the local cooldown is fresh (in the future), the
    only-cleared-never-overwritten invariant says the peer healthy
    signal must NOT clear it. _maybe_clear_cooldown_for_healthy_label
    returns False -- Phase 5's short-circuit doesn't fire and the
    escalation proceeds normally."""
    label = "test-active-cooldown-label"
    agent = _FakeAgent()
    healthy_labels = _healthy_labels()

    # Local cooldown is fresh (in the future), peer healthy index also
    # active. Fresh cooldown wins.
    _fb_mod._INMEM_COOLDOWNS[("test-agent",)] = {label: time.monotonic() + 60.0}
    healthy_labels[label] = time.monotonic() + 60.0

    try:
        cleared = _maybe_clear_cooldown_for_healthy_label(agent, label)
        assert cleared is False, (
            "active local cooldown must NOT be cleared by peer healthy signal "
            "-- only-cleared-never-overwritten invariant"
        )
        # Local cooldown should still be present
        assert label in _fb_mod._INMEM_COOLDOWNS[("test-agent",)]
    finally:
        healthy_labels.pop(label, None)
        _fb_mod._INMEM_COOLDOWNS[("test-agent",)] = {}


def test_primary_skip_only_resets_for_idx_zero():
    """The Phase 5 short-circuit is checked only for the primary
    candidate (idx 0). For non-primary candidates the helper still
    runs as before, but Phase 5's wiring uses idx to skip it. We
    verify the contract by confirming the helper itself is label-only
    (it doesn't care about idx -- the idx gating happens in
    _maybe_extend_primary_cooldown)."""
    label = "test-non-primary-label"
    agent = _FakeAgent()
    healthy_labels = _healthy_labels()

    # Stale local cooldown + peer healthy index -- would clear regardless
    # of idx; the idx gating is upstream.
    _fb_mod._INMEM_COOLDOWNS[("test-agent",)] = {label: time.monotonic() - 60.0}
    healthy_labels[label] = time.monotonic() + 60.0

    try:
        cleared = _maybe_clear_cooldown_for_healthy_label(agent, label)
        assert cleared is True
    finally:
        healthy_labels.pop(label, None)
        _fb_mod._INMEM_COOLDOWNS[("test-agent",)] = {}


def test_primary_skip_no_healthy_index_returns_false():
    """When _INMEM_HEALTHY_LABELS has no entry for this label, the
    helper returns False and Phase 5's short-circuit doesn't fire."""
    label = "test-no-healthy-index-label"
    agent = _FakeAgent()
    healthy_labels = _healthy_labels()

    # Make sure no entry exists for this label
    healthy_labels.pop(label, None)
    _fb_mod._INMEM_COOLDOWNS[("test-agent",)] = {
        label: time.monotonic() - 60.0,
    }

    try:
        cleared = _maybe_clear_cooldown_for_healthy_label(agent, label)
        assert cleared is False
    finally:
        _fb_mod._INMEM_COOLDOWNS[("test-agent",)] = {}


def test_primary_skip_stale_healthy_index_returns_false():
    """When _INMEM_HEALTHY_LABELS[label] is set but the deadline has
    passed, the helper returns False."""
    label = "test-stale-healthy-label"
    agent = _FakeAgent()
    healthy_labels = _healthy_labels()

    # Local cooldown stale, healthy index also stale (past)
    _fb_mod._INMEM_COOLDOWNS[("test-agent",)] = {label: time.monotonic() - 60.0}
    healthy_labels[label] = time.monotonic() - 10.0

    try:
        cleared = _maybe_clear_cooldown_for_healthy_label(agent, label)
        assert cleared is False, "stale healthy index should not trigger clear"
    finally:
        healthy_labels.pop(label, None)
        _fb_mod._INMEM_COOLDOWNS[("test-agent",)] = {}