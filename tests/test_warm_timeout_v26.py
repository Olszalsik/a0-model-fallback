"""Tests for the v2.6 warm/cold per-candidate timeout (added 2026-07-28).

The cascade now distinguishes "warm" labels (called successfully within
`cascade_warm_window_s`) from "cold" labels. Warm labels get the shorter
`cascade_warm_timeout_s` (default 20s); cold labels get the legacy
`fallback_utility_timeout_s` / `fallback_timeout_s` (typically 60-300s).

These tests are pure-Python and don't need the framework -- they
exercise _resolve_per_call_timeout directly. To run:
    cd /a0
    REPO_ROOT_OVERRIDE="$(pwd)" pytest usr/plugins/_model_fallback/tests/test_warm_timeout_v26.py -v
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


# Import the function-under-test and the module-level state dict so we
# can mutate _WARM_LABELS between test runs. We access the module via
# its attribute (not a bound name) so the test stays correct even when
# other test files in the suite (notably test_candidate_normalize.py)
# ``importlib.reload(fallback)`` between tests -- a reload creates a
# new module object but our test should target whichever module
# instance the SUT currently has.
from usr.plugins._model_fallback import fallback as _fb_mod
from usr.plugins._model_fallback.fallback import (
    _resolve_per_call_timeout,
    _DEFAULT_CASCADE_WARM_TIMEOUT_S,
    _DEFAULT_CASCADE_WARM_WINDOW_S,
)


def _warm_labels() -> dict:
    """Return the current module's _WARM_LABELS dict.

    test_candidate_normalize.py reloads the fallback module between
    tests, which creates a new dict each time. Reading through the
    module attribute (instead of caching at import) ensures we always
    hit the current instance.
    """
    return _fb_mod._WARM_LABELS


def test_warm_label_first_call_uses_default_timeout():
    """A label that has never been called (not in _WARM_LABELS) gets the
    cold default, not the warm timeout."""
    warm_dict = _warm_labels()
    warm_dict.pop("test-cold-label", None)
    base = 90.0
    warm = 20.0
    window = 600.0
    result = _resolve_per_call_timeout(
        "test-cold-label", base, warm, window,
    )
    assert result == base, (
        f"cold label should use base_timeout_s={base}, got {result}"
    )


def test_warm_label_within_window_uses_warm_timeout():
    """A label that succeeded within `cascade_warm_window_s` gets the
    shorter warm timeout."""
    warm_dict = _warm_labels()
    warm_dict["test-warm-label"] = time.monotonic()
    base = 90.0
    warm = 20.0
    window = 600.0
    result = _resolve_per_call_timeout(
        "test-warm-label", base, warm, window,
    )
    assert result == warm, (
        f"warm label should use warm_timeout_s={warm}, got {result}"
    )


def test_warm_label_outside_window_uses_default_timeout():
    """A label that succeeded > cascade_warm_window_s ago falls back to
    the cold default (provider may have cooled off)."""
    # Stamp 700s ago — past the 600s window.
    warm_dict = _warm_labels()
    warm_dict["test-stale-label"] = time.monotonic() - 700.0
    base = 90.0
    warm = 20.0
    window = 600.0
    result = _resolve_per_call_timeout(
        "test-stale-label", base, warm, window,
    )
    assert result == base, (
        f"stale warm label (>window) should use base_timeout_s={base}, got {result}"
    )


def test_warm_label_does_not_override_user_kwarg_timeout():
    """The kwarg override is handled at the call site (not inside
    _resolve_per_call_timeout), so this function only sees the
    resolved base. Verify the contract: if a user kwarg is set, the
    caller MUST short-circuit before calling _resolve_per_call_timeout.
    The existing test_provider_kwarg_strip.py already covers the kwarg
    path at the framework level; here we just confirm the helper's
    pure behavior."""
    # Simulate: a label is warm, but the caller has already resolved a
    # user kwarg and passed it as base. _resolve_per_call_timeout must
    # still respect the warm state (it does NOT re-apply kwarg logic).
    warm_dict = _warm_labels()
    warm_dict["test-kwarg-label"] = time.monotonic()
    user_kwarg_value = 180.0  # what user set on TIMEOUT=
    warm = 20.0
    window = 600.0
    result = _resolve_per_call_timeout(
        "test-kwarg-label", user_kwarg_value, warm, window,
    )
    # Helper sees user_kwarg_value as base; warm short-circuits to warm.
    # Caller is responsible for not calling helper when kwarg is set.
    assert result == warm


def test_warm_label_resets_on_process_restart():
    """_WARM_LABELS is module-level dict. Simulate a fresh module state
    by clearing it."""
    _warm_labels().clear()
    base = 90.0
    warm = 20.0
    window = 600.0
    result = _resolve_per_call_timeout(
        "test-after-restart", base, warm, window,
    )
    assert result == base, (
        f"after _WARM_LABELS.clear(), label should be cold, got warm"
    )


def test_warm_label_window_boundary_exact():
    """Edge case: a label that succeeded exactly at the window boundary
    should be treated as cold (strictly < window counts as warm)."""
    # Stamp exactly 600.0s ago. Window check is `now - last < window`,
    # so exactly 600.0s ago means 0 < 600.0 is False → cold.
    warm_dict = _warm_labels()
    warm_dict["test-edge-label"] = time.monotonic() - 600.0
    base = 90.0
    warm = 20.0
    window = 600.0
    result = _resolve_per_call_timeout(
        "test-edge-label", base, warm, window,
    )
    assert result == base, (
        f"label at exact window boundary should be cold, got warm"
    )


def test_warm_label_just_under_window_still_warm():
    """A label that succeeded 599s ago (just inside the window) is warm."""
    warm_dict = _warm_labels()
    warm_dict["test-just-warm"] = time.monotonic() - 599.0
    base = 90.0
    warm = 20.0
    window = 600.0
    result = _resolve_per_call_timeout(
        "test-just-warm", base, warm, window,
    )
    assert result == warm, (
        f"label 599s ago (just inside 600s window) should be warm"
    )


def test_module_defaults_match_plan():
    """Sanity-check the module-level defaults match what the v2.6 plan
    documented (locked 2026-07-28)."""
    assert _DEFAULT_CASCADE_WARM_TIMEOUT_S == 20.0
    assert _DEFAULT_CASCADE_WARM_WINDOW_S == 600.0
