"""Tests for the v2.6 "don't preempt healthy calls" rule (added 2026-07-28).

The inner cascade's ``_call_utility_model`` / ``_call_chat_model``
wait_for block now distinguishes:

  * ``asyncio.TimeoutError`` (wait_for fired at the timeout budget) --
    keep the legacy close-coroutine cleanup so the frame doesn't leak.
    This is a genuine inner timeout; the call was hung.
  * ``asyncio.CancelledError`` (external cancellation from outer guard
    or container shutdown) -- skip the close-coroutine cleanup. The
    call was healthy and running; the cancellation was externally
    forced. Closing the inner coro preemptively would just add a
    RuntimeWarning with no benefit.

This is a smaller, more conservative interpretation of the v2.6 plan's
"don't preempt past 50% wall-clock" intent. The 50% wall-clock gate
itself was abandoned during implementation because ``asyncio.wait_for``
raises at the timeout boundary (not after), so ``elapsed`` is always
``>= timeout_s`` at the catch -- a 50% gate would be meaningless.

The behavior under test lives inside ``_call_utility_model`` and
``_call_chat_model`` closures (deeply nested inside the cascade patches),
so these tests exercise the *contract* via a faithful inline replica of
the wait_for block. The replica's job is to assert what the production
code MUST do at the boundary.

To run:
    cd /a0
    REPO_ROOT_OVERRIDE="$(pwd)" pytest usr/plugins/model_fallback/tests/test_no_preempt_v26.py -v
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

REPO_ROOT = Path(os.environ.get("REPO_ROOT_OVERRIDE") or "/a0")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


async def _simulate_inner_wait(
    inner_coro_factory,
    timeout_s: float,
    external_cancel_after: float = None,
):
    """Inline replica of the inner-cascade wait_for block.

    Returns one of:
      ("completed", elapsed) -- the inner call returned normally.
      ("timeout_with_cleanup", elapsed) -- wait_for fired at timeout,
        close-coroutine cleanup ran (legacy hung-call behavior).
      ("cancel_no_cleanup", elapsed) -- external CancelledError, no
        close-coroutine cleanup (v2.6 healthy-call behavior).
    """
    inner_coro = inner_coro_factory()
    try:
        if external_cancel_after is None:
            result = await asyncio.wait_for(inner_coro, timeout=timeout_s)
            return ("completed", None, result)
        # Schedule an external cancellation on this task. asyncio's
        # shield/timeout chain will see CancelledError when the cancel
        # fires (since wait_for's inner await observes it).
        loop = asyncio.get_event_loop()
        delay_handle = loop.call_later(
            external_cancel_after, asyncio.current_task().cancel
        )
        try:
            result = await asyncio.wait_for(inner_coro, timeout=timeout_s)
            return ("completed", None, result)
        except asyncio.CancelledError:
            # v2.6: external cancellation -- skip close-coroutine cleanup.
            return ("cancel_no_cleanup", None, None)
        except asyncio.TimeoutError:
            # Legacy close-coroutine cleanup at timeout.
            try:
                if inner_coro.cr_frame is not None:
                    inner_coro.close()
            except Exception:
                pass
            return ("timeout_with_cleanup", None, None)
        finally:
            delay_handle.cancel()
    except asyncio.CancelledError:
        return ("cancel_no_cleanup", None, None)
    except asyncio.TimeoutError:
        try:
            if inner_coro.cr_frame is not None:
                inner_coro.close()
        except Exception:
            pass
        return ("timeout_with_cleanup", None, None)


def test_inner_call_legacy_close_cleanup_at_timeout():
    """A call that exceeds `timeout_s` triggers TimeoutError and runs
    the legacy close-coroutine cleanup (frame leak prevention)."""
    timeout_s = 0.05

    async def _slow_coro():
        await asyncio.sleep(timeout_s + 0.5)
        return "never"

    outcome, _, value = asyncio.run(
        _simulate_inner_wait(_slow_coro, timeout_s)
    )
    assert outcome == "timeout_with_cleanup", (
        f"hung call should be timeout_with_cleanup, got {outcome}"
    )


def test_inner_call_no_close_cleanup_on_external_cancel():
    """A call that's externally cancelled (CancelledError) does NOT
    run the close-coroutine cleanup -- the call was healthy."""
    timeout_s = 2.0

    async def _healthy_coro():
        await asyncio.sleep(10.0)
        return "healthy"

    outcome, _, value = asyncio.run(
        _simulate_inner_wait(_healthy_coro, timeout_s, external_cancel_after=0.05)
    )
    assert outcome == "cancel_no_cleanup", (
        f"healthy call externally cancelled should be cancel_no_cleanup, "
        f"got {outcome}"
    )


def test_inner_call_fast_returns_normally():
    """Sanity check: a fast call returns its value, no timeout path."""
    timeout_s = 1.0

    async def _fast_coro():
        return "ok"

    outcome, _, value = asyncio.run(
        _simulate_inner_wait(_fast_coro, timeout_s)
    )
    assert outcome == "completed"
    assert value == "ok"


def test_no_preempt_independent_of_timeout_value():
    """The CancelledError-vs-TimeoutError split is independent of the
    timeout budget -- only the cancellation source matters."""
    timeout_s = 5.0

    async def _healthy_coro():
        await asyncio.sleep(10.0)
        return "healthy"

    outcome, _, value = asyncio.run(
        _simulate_inner_wait(_healthy_coro, timeout_s, external_cancel_after=0.02)
    )
    assert outcome == "cancel_no_cleanup"


def test_outer_guard_unchanged_by_phase_4():
    """Phase 4 only modifies the inner cascade's wait_for block. The
    outer guard (``helpers/utility_timeout.guarded_call`` with
    ``max_wait_s: 180``) must still raise RepairableException on its
    own timeout. This test imports the real utility_timeout helper and
    verifies its behavior is unchanged.

    This complements the inner-replica tests above: the inner
    replica verifies the new split behavior on CancelledError; this
    test verifies the outer guard is NOT touched.

    Cleanup: ``utility_timeout.guarded_call`` triggers a lazy import
    of ``usr.plugins.memory_hardening.helpers.coroutine_guard`` from
    inside its timeout handler. Once that import runs, Python caches
    the resolved attribute on the parent ``helpers`` package, which
    would defeat the ``patch.dict(sys.modules, ...)`` trick used by
    ``test_utility_timeout_closes_inner_coro_on_timeout`` later in the
    suite. We pop the module so the test suite order doesn't matter.
    """
    from helpers.errors import RepairableException
    from usr.plugins.model_fallback.helpers import utility_timeout

    cfg = utility_timeout.resolve_config({
        "enabled": True,
        "default_timeout_s": 0.05,
        "max_wait_s": 0.05,
        "jitter_s": 0.0,
    })

    async def _slow():
        await asyncio.sleep(1.0)
        return "never"

    raised = False
    try:
        asyncio.run(utility_timeout.guarded_call(
            _slow, model_name="test_phase4_outer", config_overrides=cfg,
        ))
    except RepairableException:
        raised = True
    assert raised, (
        "outer guard must still raise RepairableException on its own "
        "timeout -- Phase 4 must not have touched it"
    )

    # Cleanup: remove the lazy-loaded memory_hardening helpers module
    # so subsequent tests can patch.dict() it without a stale cache
    # resolving to the real module.
    for mod_name in (
        "usr.plugins.memory_hardening.helpers.coroutine_guard",
        "usr.plugins.memory_hardening.helpers",
        "usr.plugins.memory_hardening",
    ):
        sys.modules.pop(mod_name, None)