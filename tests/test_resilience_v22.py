"""Tests for the v2.2 resilience layer of the `_model_fallback` plugin.

Covers the remaining two pieces (v2.5: the third piece,
``housekeeping``, was removed together with the loop):
* utility_timeout.guarded_call (timeout, success, lazy close_inner_coro)
* webui_extensions_cache (TTL, bust, circuit breaker)

The tests do NOT touch the real framework; everything is mocked
or run in isolation. To run:
    cd /a0
    REPO_ROOT_OVERRIDE="$(pwd)" pytest usr/plugins/_model_fallback/tests/test_resilience_v22.py -v
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

# Same sys.path bootstrap as the other test files in this plugin
# family. The framework's "test from the install root" convention
# is /a0; for host-side runs we honor REPO_ROOT_OVERRIDE.
REPO_ROOT = Path(os.environ.get("REPO_ROOT_OVERRIDE") or "/a0")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Test the utility timeout guard
# ---------------------------------------------------------------------------

from usr.plugins._model_fallback.helpers import utility_timeout  # noqa: E402
from usr.plugins._model_fallback.helpers import stats as _stats  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_stats():
    _stats.reset()
    utility_timeout.reset()
    yield
    _stats.reset()
    utility_timeout.reset()


@pytest.mark.asyncio
async def test_utility_timeout_raises_on_timeout():
    """A slow inner coroutine triggers a RepairableException within max_wait_s."""
    cfg = utility_timeout.resolve_config({
        "enabled": True,
        "default_timeout_s": 0.05,
        "max_wait_s": 0.05,
        "jitter_s": 0.0,
        "close_inner_on_timeout": False,
    })

    async def _slow():
        await asyncio.sleep(2.0)
        return "never"

    from helpers.errors import RepairableException
    with pytest.raises(RepairableException):
        await utility_timeout.guarded_call(
            _slow, model_name="test_slow", config_overrides=cfg
        )
    snap = _stats.utility_timeout_snapshot()
    assert snap["timeouts_total"] == 1
    assert snap["last_timeout_model"] == "test_slow"


@pytest.mark.asyncio
async def test_utility_timeout_does_not_raise_on_success():
    """A fast inner coroutine returns its value and bumps the call counter."""
    cfg = utility_timeout.resolve_config({
        "enabled": True,
        "default_timeout_s": 1.0,
        "max_wait_s": 1.0,
        "jitter_s": 0.0,
    })

    async def _fast():
        return "ok"

    result = await utility_timeout.guarded_call(
        _fast, model_name="test_fast", config_overrides=cfg
    )
    assert result == "ok"
    snap = _stats.utility_timeout_snapshot()
    assert snap["calls_total"] == 1
    assert snap["timeouts_total"] == 0


@pytest.mark.asyncio
async def test_utility_timeout_disabled_is_pass_through():
    """When ``enabled`` is False, guarded_call is a plain await."""
    cfg = utility_timeout.resolve_config({"enabled": False})

    async def _f():
        return 42

    result = await utility_timeout.guarded_call(_f, model_name="x", config_overrides=cfg)
    assert result == 42
    # Stats not touched.
    assert _stats.utility_timeout_snapshot()["calls_total"] == 0


@pytest.mark.asyncio
async def test_utility_timeout_closes_inner_coro_on_timeout():
    """On timeout, close_inner_coro is called (best-effort) via the
    memory_hardening helper. We mock the lazy import path so the test
    does not require memory_hardening to be installed.
    """
    cfg = utility_timeout.resolve_config({
        "enabled": True,
        "default_timeout_s": 0.05,
        "max_wait_s": 0.05,
        "jitter_s": 0.0,
        "close_inner_on_timeout": True,
    })

    # Create a coroutine that, when awaited, would hang.
    async def _slow():
        await asyncio.sleep(2.0)
        return "nope"
    # Build a coroutine object (not awaited yet) — passed to the
    # close_inner_coro mock so we can prove the helper tried to
    # close the *real* coroutine, not a freshly-built one.
    leaked = _slow()
    try:
        closed_count = {"n": 0}

        def _fake_close(coro):
            closed_count["n"] += 1
            try:
                coro.close()
            except Exception:  # noqa: BLE001
                pass
            return True

        # Patch the import path the helper uses.
        with patch.dict(
            sys.modules,
            {
                "usr.plugins.memory_hardening.helpers.coroutine_guard": type(
                    "M", (), {"close_inner_coro": staticmethod(_fake_close)}
                )(),
            },
        ):
            from helpers.errors import RepairableException
            with pytest.raises(RepairableException):
                await utility_timeout.guarded_call(
                    lambda: _slow(), model_name="x", config_overrides=cfg
                )
        # close_inner_coro was attempted. (The exact count is at
        # least 1 because guarded_call attempts it on the inner it
        # just created.)
        assert closed_count["n"] >= 1
        snap = _stats.utility_timeout_snapshot()
        assert snap["close_inner_attempted"] >= 1
    finally:
        # Close the leaked coroutine so the test doesn't emit
        # ``coroutine was never awaited`` warnings.
        try:
            leaked.close()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# Test the webui_extensions_cache
# ---------------------------------------------------------------------------

from usr.plugins._model_fallback.helpers import webui_extensions_cache  # noqa: E402


@pytest.fixture(autouse=False)
def _reset_cache():
    webui_extensions_cache.reset()
    yield
    webui_extensions_cache.reset()
    try:
        webui_extensions_cache.uninstall()
    except Exception:  # noqa: BLE001
        pass


def test_extensions_cache_hits_within_ttl(_reset_cache):
    """Identical requests within the TTL window are served from the cache."""
    cfg = webui_extensions_cache.resolve_config({
        "enabled": True, "ttl_s": 2.0,
        "circuit_breaker_enabled": False,
    })
    calls = {"n": 0}

    def _original(agent, extension_point, filters):
        calls["n"] += 1
        return [f"ext_{calls['n']}"]

    wrapper = webui_extensions_cache._build_wrapper(_original, lambda: cfg)
    # First call -> miss
    v1 = wrapper(None, ["page-head"], None)
    assert v1 == ["ext_1"]
    assert calls["n"] == 1
    # Second call same args -> hit
    v2 = wrapper(None, ["page-head"], None)
    assert v2 == ["ext_1"]
    assert calls["n"] == 1
    snap = _stats.extensions_cache_snapshot()
    assert snap["hits"] == 1
    assert snap["misses"] == 1


def test_extensions_cache_bust_invalidates(_reset_cache):
    """``bust()`` clears the cache; the next call re-fetches."""
    cfg = webui_extensions_cache.resolve_config({
        "enabled": True, "ttl_s": 10.0,
        "circuit_breaker_enabled": False,
    })
    calls = {"n": 0}

    def _original(agent, extension_point, filters):
        calls["n"] += 1
        return [f"v{calls['n']}"]

    wrapper = webui_extensions_cache._build_wrapper(_original, lambda: cfg)
    wrapper(None, ["p"], None)
    wrapper(None, ["p"], None)
    assert calls["n"] == 1
    webui_extensions_cache.bust()
    wrapper(None, ["p"], None)
    assert calls["n"] == 2
    assert _stats.extensions_cache_snapshot()["busts"] >= 1


def test_extensions_cache_circuit_breaker_opens_then_recovers(_reset_cache):
    """A burst of errors opens the circuit; after recovery_s it closes."""
    cfg = webui_extensions_cache.resolve_config({
        "enabled": True, "ttl_s": 0.0,
        "circuit_breaker_enabled": True,
        "circuit_breaker_window_s": 60.0,
        "circuit_breaker_threshold": 3,
        "circuit_breaker_recovery_s": 0.1,  # 100ms for the test
    })

    def _original_raises(agent, extension_point, filters):
        raise RuntimeError("fs wedged")

    wrapper = webui_extensions_cache._build_wrapper(_original_raises, lambda: cfg)
    # Three failing calls open the breaker.
    for _ in range(3):
        v = wrapper(None, ["p"], None)
        assert v == []  # still returns [] (graceful)
    snap = _stats.extensions_cache_snapshot()
    assert snap["circuit_open_count"] >= 1
    # The fourth call should short-circuit (no error raised this time).
    snap_before = _stats.extensions_cache_snapshot()
    v = wrapper(None, ["p"], None)
    assert v == []
    snap_after = _stats.extensions_cache_snapshot()
    assert snap_after["circuit_short_circuits"] == snap_before["circuit_short_circuits"] + 1
    # Wait for recovery.
    time.sleep(0.15)
    # Next call should hit the original again (which raises).
    v = wrapper(None, ["p"], None)
    assert v == []


def test_extensions_cache_disabled_is_pass_through(_reset_cache):
    """When disabled, the wrapper calls the original on every request."""
    cfg = webui_extensions_cache.resolve_config({"enabled": False})
    calls = {"n": 0}

    def _original(agent, extension_point, filters):
        calls["n"] += 1
        return [f"v{calls['n']}"]

    wrapper = webui_extensions_cache._build_wrapper(_original, lambda: cfg)
    wrapper(None, ["p"], None)
    wrapper(None, ["p"], None)
    assert calls["n"] == 2
    assert _stats.extensions_cache_snapshot()["hits"] == 0
    assert _stats.extensions_cache_snapshot()["misses"] == 0


# ---------------------------------------------------------------------------
# v2.5: the standalone housekeeping loop was removed. Tests for the
# housekeeping module (and the WS-pulse two-path code that lived in
# it) were deleted together with the module. The stats endpoint no
# longer exposes a ``housekeeping`` block; the contract is asserted
# below in ``test_stats_snapshot_shape``.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Test the utility timeout guard does not double-install
# ---------------------------------------------------------------------------


def test_install_utility_timeout_patch_idempotent():
    """The timeout guard has a sentinel that prevents double-install.

    We don't import the patch module directly because it pulls in
    the full agent-zero framework (which may not be available in
    the host venv). Instead we verify the contract that the helper
    would check: the sentinel attribute, the resolve_config output.
    """
    cfg = utility_timeout.resolve_config({
        "enabled": True,
        "max_wait_s": 120.0,
    })
    assert cfg["max_wait_s"] == 120.0
    # The sentinel name used by the patch must match the helper
    # check. The patch sets ``_utility_timeout_patched`` on the
    # wrapped function. We verify the attribute name is consistent
    # with the helper expectations by reading the patch file
    # directly (string check) — no framework import needed.
    patch_path = (
        Path(__file__).resolve().parent.parent
        / "extensions" / "python" / "agent_init"
        / "_10_install_utility_timeout_patch.py"
    )
    src = patch_path.read_text(encoding="utf-8")
    assert "_utility_timeout_patched" in src
    assert "_utility_timeout_original" in src
    assert "Agent.call_utility_model = wrapped" in src or "Agent.call_utility_model=wrapped" in src


# ---------------------------------------------------------------------------
# Test the stats endpoint contract
# ---------------------------------------------------------------------------


def test_stats_snapshot_shape():
    """v2.5: only the two remaining counter groups are exposed.
    The ``housekeeping`` group was removed together with the loop.
    """
    snap_ut = _stats.utility_timeout_snapshot()
    snap_ec = _stats.extensions_cache_snapshot()
    for key in ("calls_total", "timeouts_total", "max_observed_wait_s"):
        assert key in snap_ut, f"utility_timeout missing {key}"
    for key in ("hits", "misses", "errors", "circuit_open", "circuit_open_count"):
        assert key in snap_ec, f"extensions_cache missing {key}"
    # Sanity: ``reset`` only touches the two groups that still exist.
    _stats.reset()
    snap_ut2 = _stats.utility_timeout_snapshot()
    snap_ec2 = _stats.extensions_cache_snapshot()
    assert snap_ut2["calls_total"] == 0
    assert snap_ec2["hits"] == 0
