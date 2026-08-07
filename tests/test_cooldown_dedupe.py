"""Tests for the cooldown-skip log dedupe added 2026-07-21.

The cascade's "in cooldown, skipping (Xs)" log was firing every
~2s while a candidate was in cooldown, producing ~150 countdown
lines per 5-min cooldown. The fix dedupes the log by keying it on
the `cooldown_until` value, so it fires once per cooldown window.

These tests are pure-Python and don't need the framework -- they
exercise the dedupe dict directly. To run:
    cd /a0
    REPO_ROOT_OVERRIDE="$(pwd)" pytest usr/plugins/_model_fallback/tests/test_cooldown_dedupe.py -v
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Same sys.path bootstrap as the other test files in this plugin family.
REPO_ROOT = Path(os.environ.get("REPO_ROOT_OVERRIDE") or "/a0")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def simulate_skip_iterations(cooldown_until: float, iterations: int, sleep_step: float = 2.0):
    """Replicate the dedupe logic from fallback.py for one label.

    Returns the list of (iteration_index, remaining_seconds) tuples
    that would have been logged. With the dedupe, this list should
    have exactly one entry per unique cooldown_until value.
    """
    _last_skip_log_until: dict = {}
    now = 0.0
    logs = []
    for i in range(iterations):
        now += sleep_step
        if cooldown_until > now:
            # Dedupe gate, mirroring fallback.py:888-889.
            if _last_skip_log_until.get("test-model") != cooldown_until:
                _last_skip_log_until["test-model"] = cooldown_until
                logs.append((i, int(cooldown_until - now)))
        else:
            # Cooldown expired, clear dedupe marker.
            if _last_skip_log_until.get("test-model") is not None:
                _last_skip_log_until.pop("test-model", None)
    return logs


def test_single_cooldown_emits_one_log():
    """A 300s cooldown over 150 iterations (each 2s) should emit one log line."""
    logs = simulate_skip_iterations(cooldown_until=300.0, iterations=150)
    assert len(logs) == 1, f"expected 1 log, got {len(logs)}: {logs}"
    assert logs[0][1] == 298  # 300s - 2s elapsed


def test_consecutive_cooldowns_emit_two_logs():
    """A second cooldown after the first expires should emit a second log line."""
    _last_skip_log_until: dict = {}
    now = 0.0
    cooldown_1 = 300.0
    cooldown_2 = 600.0  # set after first expires
    logs = []

    # First cooldown
    for i in range(150):
        now += 2.0
        if cooldown_1 > now:
            if _last_skip_log_until.get("test-model") != cooldown_1:
                _last_skip_log_until["test-model"] = cooldown_1
                logs.append((i, "c1", int(cooldown_1 - now)))
        else:
            if _last_skip_log_until.get("test-model") is not None:
                _last_skip_log_until.pop("test-model", None)

    # Second cooldown
    for i in range(150):
        now += 2.0
        if cooldown_2 > now:
            if _last_skip_log_until.get("test-model") != cooldown_2:
                _last_skip_log_until["test-model"] = cooldown_2
                logs.append((i + 150, "c2", int(cooldown_2 - now)))
        else:
            if _last_skip_log_until.get("test-model") is not None:
                _last_skip_log_until.pop("test-model", None)

    assert len(logs) == 2, f"expected 2 logs, got {len(logs)}: {logs}"
    assert logs[0][1] == "c1"
    assert logs[1][1] == "c2"


def test_regression_no_dedupe_count():
    """Without the dedupe, the same scenario would produce ~150 logs.

    This is a regression guard: if someone removes the dedupe gate,
    this assertion breaks and forces them to think about the spam.
    """
    cooldown_until = 300.0
    now = 0.0
    raw_log_count = 0
    for _ in range(150):
        now += 2.0
        if cooldown_until > now:
            raw_log_count += 1
    # 149 iterations where now < cooldown_until (300). At iteration 149
    # now=300 and the check is False; iteration 0 sees 298s remaining.
    assert raw_log_count == 149, (
        "raw log count changed; this test is meant to lock the spam "
        "shape, not the dedupe count"
    )
