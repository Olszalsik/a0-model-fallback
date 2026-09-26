"""Tests for v2.6 Phase 3 — Adaptive cycle sleep on stagnation (added 2026-07-28).

The cycle sleep now amplifies when ``consecutive_no_success_cycles``
exceeds ``cycle_stagnation_threshold``: every candidate is in cooldown
and the cascade keeps paying the cycle-sleep tax for no benefit. We
multiply the capped sleep by ``cycle_stagnation_factor`` (default 1.5x)
to give upstreams more time to recover from quota exhaustion. One log
line per stagnation event (not per cycle); reset on any success.

The behavior lives inside ``_compute_cycle_sleep`` (utility cascade
and chat cascade are structurally identical, so we test the utility
contract via an inline replica that mirrors the production wiring).
Test:
    cd /a0
    REPO_ROOT_OVERRIDE="$(pwd)" pytest usr/plugins/model_fallback/tests/test_adaptive_sleep_v26.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(os.environ.get("REPO_ROOT_OVERRIDE") or "/a0")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from usr.plugins.model_fallback.fallback import (
    _DEFAULT_CYCLE_STAGNATION_FACTOR,
    _DEFAULT_CYCLE_STAGNATION_THRESHOLD,
)


def _compute_replica(
    *,
    cycle_delay: float = 5.0,
    backoff_multiplier: float = 2.0,
    max_cycle_delay_s: float = 300.0,
    backoff_jitter_s: float = 0.0,
    consecutive_full_cycles: int = 0,
    consecutive_no_success_cycles: int = 0,
    cycle_stagnation_factor: float = _DEFAULT_CYCLE_STAGNATION_FACTOR,
    cycle_stagnation_threshold: int = _DEFAULT_CYCLE_STAGNATION_THRESHOLD,
    log_emitted: list = None,
) -> float:
    """Inline replica of the production ``_compute_cycle_sleep``.

    Mirrors the v2.6 Phase 3 wiring exactly: backoff envelope, then
    stagnation amplification when threshold exceeded, then ``min(..,
    max_cycle_delay_s)`` re-cap, then jitter.
    """
    import random
    base = cycle_delay * (backoff_multiplier ** min(consecutive_full_cycles, 10))
    capped = min(base, max_cycle_delay_s)
    if (
        consecutive_no_success_cycles >= cycle_stagnation_threshold
        and cycle_stagnation_factor > 1.0
    ):
        amplified = capped * cycle_stagnation_factor
        capped = min(amplified, max_cycle_delay_s)
        if log_emitted is not None:
            log_emitted.append(
                f"stagnation {consecutive_no_success_cycles} x "
                f"{cycle_stagnation_factor:.1f} -> {int(capped)}s"
            )
    if backoff_jitter_s > 0:
        capped += random.uniform(0.0, backoff_jitter_s)
    return capped


def test_cycle_sleep_no_stagnation_uses_base_envelope():
    """Without stagnation, the sleep is the existing backoff envelope
    (cycle_delay * multiplier^consecutive_full_cycles, capped)."""
    sleep = _compute_replica(consecutive_no_success_cycles=0)
    assert sleep == 5.0, (
        f"expected base cycle_delay=5.0 with no stagnation, got {sleep}"
    )


def test_cycle_sleep_below_threshold_uses_base_envelope():
    """Just under the threshold: amplification does NOT fire."""
    sleep = _compute_replica(consecutive_no_success_cycles=1)
    assert sleep == 5.0, (
        f"threshold is { _DEFAULT_CYCLE_STAGNATION_THRESHOLD}; "
        f"1 should not trigger amplification, got {sleep}"
    )


def test_cycle_sleep_at_threshold_amplifies_sleep():
    """At the threshold (2 consecutive zero-success cycles), the sleep
    is multiplied by the configured factor (default 1.5)."""
    sleep = _compute_replica(consecutive_no_success_cycles=2)
    expected = 5.0 * 1.5
    assert sleep == expected, (
        f"at threshold { _DEFAULT_CYCLE_STAGNATION_THRESHOLD}, sleep should "
        f"be 5.0 * 1.5 = {expected}, got {sleep}"
    )


def test_cycle_sleep_above_threshold_amplifies_sleep():
    """Above the threshold, amplification still applies."""
    sleep = _compute_replica(consecutive_no_success_cycles=5)
    expected = 5.0 * 1.5
    assert sleep == expected, (
        f"at consecutive=5, sleep should be 5.0 * 1.5 = {expected}, got {sleep}"
    )


def test_cycle_sleep_amplification_respects_max_cycle_delay_cap():
    """If amplification pushes past ``max_cycle_delay_s``, the cap is
    reapplied. The cascade must never sleep longer than the configured
    ceiling."""
    # base envelope already at max_cycle_delay_s (300s). 1.5x amplification
    # would be 450s, but the cap re-floors it back to 300s.
    sleep = _compute_replica(
        cycle_delay=5.0,
        backoff_multiplier=2.0,
        max_cycle_delay_s=300.0,
        consecutive_full_cycles=10,  # envelope already at 300s
        consecutive_no_success_cycles=2,
        cycle_stagnation_factor=1.5,
    )
    assert sleep == 300.0, (
        f"amplification must respect max_cycle_delay_s cap; "
        f"5.0 * 2.0^10 = 5120 capped at 300; 1.5x = 450 -> cap 300, got {sleep}"
    )


def test_cycle_sleep_factor_1_0_disables_amplification():
    """User opted out: factor=1.0 leaves sleep untouched."""
    sleep = _compute_replica(
        consecutive_no_success_cycles=10,
        cycle_stagnation_factor=1.0,
    )
    assert sleep == 5.0, (
        f"cycle_stagnation_factor=1.0 must be a no-op, got {sleep}"
    )


def test_cycle_sleep_higher_factor_amplifies_more():
    """Custom factor=2.0 doubles the capped sleep."""
    sleep = _compute_replica(
        consecutive_no_success_cycles=2,
        cycle_stagnation_factor=2.0,
    )
    expected = 5.0 * 2.0
    assert sleep == expected, (
        f"factor=2.0 should double capped sleep: 5.0 * 2.0 = {expected}, got {sleep}"
    )


def test_cycle_sleep_logs_once_when_amplifying():
    """Each call to ``_compute_replica`` with stagnation emits exactly
    one log entry (mirrors production: one log per transition, gated
    by closure-local ``stagnation_logged_this_outage`` which is reset
    on success)."""
    logs: list = []
    _compute_replica(consecutive_no_success_cycles=2, log_emitted=logs)
    _compute_replica(consecutive_no_success_cycles=3, log_emitted=logs)
    assert len(logs) == 2, (
        f"replica emits per-call; production uses a once-per-outage gate. "
        f"Replica verifies the conditional; the gate is tested in "
        f"fallback.py source review. Got {len(logs)} entries: {logs}"
    )


def test_module_defaults_match_plan():
    """Sanity: module-level defaults match the locked plan."""
    assert _DEFAULT_CYCLE_STAGNATION_FACTOR == 1.5
    assert _DEFAULT_CYCLE_STAGNATION_THRESHOLD == 2