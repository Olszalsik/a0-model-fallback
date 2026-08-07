"""Tests that the cascade utility/chat timeouts stay above 60s.

Added 2026-07-21 (Laci) after the docker log showed all 5 utility
candidates timing out at 30s even though every provider responded in
<2s from the host. Root cause: the 30s value was tight enough that a
cold container's first request (which can take 10s+ to establish
TLS to the provider) hit the ceiling before the response came back,
and the cascade cycled through all 5 candidates before any of them
got a real response.

Local repro (scripts/diag_cascade_one_call.py):
  Run 1 (cold start): 10.23s
  Run 2 (warm):        1.22s
  Run 3 (warm):        1.41s

Fix: raise the per-candidate ceilings to 90s (utility) and 180s
(chat), and the outer guard's max_wait_s to 180s so it doesn't
silently clamp the raised default back down to 120. See
config.json and default_config.yaml for the new values.

These tests guard against a future config edit silently dropping the
timeouts back to 30s. They do not enforce a specific upper bound --
only a floor at 60s -- so config can be tuned up or down within
reason without breaking the test.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(os.environ.get("REPO_ROOT_OVERRIDE") or "/a0")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


_FLOOR_S = 60.0


def _load_yaml(path: Path) -> dict:
    """Load default_config.yaml via PyYAML.

    The file uses inline comments (``key: value  # comment``) which
    would fool a regex parser. PyYAML handles them natively.
    """
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def test_fallback_utility_timeout_s_above_floor():
    """config.json: fallback_utility_timeout_s (per-candidate utility) must be >= 60s."""
    cfg_path = REPO_ROOT / "usr" / "plugins" / "_model_fallback" / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    val = cfg.get("fallback_utility_timeout_s")
    assert isinstance(val, (int, float)), (
        f"fallback_utility_timeout_s missing or non-numeric: {val!r}"
    )
    assert val >= _FLOOR_S, (
        f"fallback_utility_timeout_s = {val}s is below the {_FLOOR_S}s floor; "
        "the 30s default was too tight for cold-start network calls. Raise "
        "it or update the floor if you have a measured reason to go lower."
    )


def test_fallback_timeout_s_above_floor():
    """config.json: fallback_timeout_s (per-candidate chat) must be >= 60s."""
    cfg_path = REPO_ROOT / "usr" / "plugins" / "_model_fallback" / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    val = cfg.get("fallback_timeout_s")
    assert isinstance(val, (int, float)), (
        f"fallback_timeout_s missing or non-numeric: {val!r}"
    )
    assert val >= _FLOOR_S, (
        f"fallback_timeout_s = {val}s is below the {_FLOOR_S}s floor; "
        "the chat cascade needs the same headroom as the utility cascade."
    )


def test_utility_timeout_guard_default_above_floor():
    """default_config.yaml: utility_timeout_guard.default_timeout_s >= 60s."""
    cfg_path = REPO_ROOT / "usr" / "plugins" / "_model_fallback" / "default_config.yaml"
    block = _load_yaml(cfg_path).get("utility_timeout_guard", {})
    val = block.get("default_timeout_s")
    assert isinstance(val, (int, float)), (
        f"utility_timeout_guard.default_timeout_s missing: {block!r}"
    )
    assert val >= _FLOOR_S, (
        f"utility_timeout_guard.default_timeout_s = {val}s is below the "
        f"{_FLOOR_S}s floor. The outer guard's default_to caps the inner "
        "cascade's wait_for; if it's 30s, a 30s inner timeout is moot."
    )


def test_utility_timeout_guard_max_wait_covers_default():
    """utility_timeout_guard.max_wait_s must be >= default_timeout_s.

    utility_timeout.py:163 does ``min(max(default_to, 0), max_wait)``,
    so if max_wait is smaller than default_to, the effective ceiling
    is silently clamped to max_wait. The 2026-07-21 fix raised both
    together; this test guards against one being raised without the
    other.
    """
    cfg_path = REPO_ROOT / "usr" / "plugins" / "_model_fallback" / "default_config.yaml"
    block = _load_yaml(cfg_path).get("utility_timeout_guard", {})
    default_to = block.get("default_timeout_s")
    max_wait = block.get("max_wait_s")
    assert isinstance(default_to, (int, float)) and isinstance(max_wait, (int, float)), (
        f"default_timeout_s or max_wait_s missing/non-numeric: {block!r}"
    )
    assert max_wait >= default_to, (
        f"utility_timeout_guard.max_wait_s = {max_wait} is below "
        f"default_timeout_s = {default_to}. The outer guard would "
        "silently clamp the raised default back down. Keep max_wait "
        ">= default_timeout_s."
    )
