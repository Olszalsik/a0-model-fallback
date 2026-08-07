"""Utility-model timeout guard for the v2.2 resilience layer.

Wraps ``Agent.call_utility_model`` so a hung utility call (typically
ollama on CPU, or any model whose first byte takes > 30s) cannot
freeze the agent loop indefinitely. Lives entirely in this plugin; no
core file is touched.

How it integrates with the existing cascade
-------------------------------------------
``call_utility_model`` is decorated ``@extension.extensible`` in
agent.py:827. The implicit ``start`` extension point fires before
the function body, with a ``data`` dict whose ``"result"`` slot
extensions are free to set; if it's set, the decorator returns that
value instead of calling the wrapped function.

The hook at
``extensions/python/_functions/agent/Agent/call_utility_model/start/_10_utility_max_wait.py``
substitutes ``data["result"]`` with a coroutine that wraps the
existing cascade (which is itself monkey-patched by
``_00_install_fallback_patches.py``) in ``asyncio.wait_for(..., timeout)``.

On ``asyncio.TimeoutError`` we:

1. Best-effort close the inner litellm coroutine chain via the
   ``memory_hardening`` plugin's ``close_inner_coro`` helper. Lazy
   import; if the plugin is disabled the call is a no-op.
2. Raise ``RepairableException("utility_model_timeout")`` so the
   outer ``call_utility_model`` propagates it, the agent's
   ``monologue`` exception handler catches it, and the LLM is told
   the utility call failed (the LLM can then rephrase or skip).
3. Record a stat so the WebUI can show "this model timed out N times
   today" — useful for picking a smaller ollama model.

Why not patch ``call_utility_model`` directly?
---------------------------------------------
``_model_fallback`` already does that (its cascade lives in
``fallback.py``). Adding a second monkey-patch on top would require
either nested wrappers (slow, fragile) or replacing the cascade
entirely. Hooking the ``start`` extension point lets the timeout
guard sit OUTSIDE the cascade: the cascade's own ``wait_for``
governs cascade-internal sleeps; this guard governs the outermost
``asyncio.wait_for`` on the whole utility call.

Migration
---------
If a future agent-zero version stops using ``@extensible`` on
``call_utility_model`` (or renames the implicit hook path), this
plugin will need to either re-add the monkey-patch pattern or move
to a different entry point. The migration is documented in
``AGENTS.md`` so the next maintainer can do it in one place.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, Optional

from helpers.errors import RepairableException
from usr.plugins._model_fallback.helpers import stats

_log = logging.getLogger("model_fallback.utility_timeout")

# Default config. Mirrored from default_config.yaml so unit tests that
# don't load the plugin config still get sane numbers.
DEFAULTS: Dict[str, Any] = {
    "enabled": True,
    "default_timeout_s": 30.0,
    "max_wait_s": 120.0,
    "jitter_s": 1.0,
    "close_inner_on_timeout": True,
}

# Cached per-process resolved config; reset() clears it.
_resolved: Dict[str, Any] = {}


def resolve_config(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return the effective config for this module.

    ``overrides`` is a partial dict; only the keys it contains win
    over the defaults. Used by the start hook which reads from the
    plugin's own config.json (per-scope) and falls back to defaults.
    """
    cfg = dict(DEFAULTS)
    if overrides:
        for k, v in overrides.items():
            if v is not None:
                cfg[k] = v
    cfg["default_timeout_s"] = float(cfg.get("default_timeout_s") or 30.0)
    cfg["max_wait_s"] = float(cfg.get("max_wait_s") or 120.0)
    cfg["jitter_s"] = float(cfg.get("jitter_s") or 0.0)
    cfg["enabled"] = bool(cfg.get("enabled", True))
    cfg["close_inner_on_timeout"] = bool(cfg.get("close_inner_on_timeout", True))
    return cfg


def set_resolved(cfg: Dict[str, Any]) -> None:
    """Cache the resolved config so the hot path doesn't re-read."""
    global _resolved
    _resolved = dict(cfg)


def get_resolved() -> Dict[str, Any]:
    if not _resolved:
        _resolved = dict(DEFAULTS)
    return _resolved


# ---------------------------------------------------------------------------
# Coroutine cleanup (lazy, optional)
# ---------------------------------------------------------------------------

def _try_close_inner_coro(coro: Any) -> bool:
    """Best-effort: close the inner litellm coroutine chain on timeout.

    Returns True if we actually closed something. The lazy import means
    this is a no-op when ``memory_hardening`` is disabled.
    """
    if coro is None:
        return False
    try:
        from usr.plugins.memory_hardening.helpers import coroutine_guard  # type: ignore
        closed = bool(coroutine_guard.close_inner_coro(coro))
        stats.utility_timeout_record_close_inner(closed)
        return closed
    except Exception as exc:  # noqa: BLE001
        stats.utility_timeout_record_close_inner(False)
        _log.debug("close_inner_coro unavailable: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Public: build the timeout coroutine that the start hook substitutes
# ---------------------------------------------------------------------------

async def guarded_call(
    inner_coro_factory,
    *,
    model_name: str = "",
    config_overrides: Optional[Dict[str, Any]] = None,
) -> Any:
    """Run ``inner_coro_factory()`` under ``asyncio.wait_for``.

    The factory pattern is important: we want to create the inner
    coroutine on THIS task so ``asyncio.wait_for`` can cancel it
    cleanly. If we accepted a pre-built coroutine, we'd risk closing
    a coroutine that another task is already driving.

    On ``asyncio.TimeoutError`` we close the inner chain, record a
    stat, and re-raise as ``RepairableException`` so the LLM sees a
    recoverable error rather than a hard timeout.
    """
    cfg = resolve_config(config_overrides or get_resolved())
    if not cfg.get("enabled", True):
        return await inner_coro_factory()

    max_wait = float(cfg.get("max_wait_s") or 120.0)
    default_to = float(cfg.get("default_timeout_s") or 30.0)
    # Use the larger of the two so a configured "default" never exceeds
    # the hard cap; this keeps the contract: "max_wait_s" is a real cap.
    timeout_s = min(max(default_to, 0.0), max_wait)
    # Add a small random spread so concurrent agents don't hit the
    # wire lockstep.
    jitter = float(cfg.get("jitter_s") or 0.0)
    if jitter > 0:
        import random
        timeout_s += random.uniform(0.0, jitter)

    inner = inner_coro_factory()
    started = time.monotonic()
    try:
        result = await asyncio.wait_for(inner, timeout=timeout_s)
    except asyncio.TimeoutError:
        elapsed = time.monotonic() - started
        stats.utility_timeout_record_timeout(model_name)
        if cfg.get("close_inner_on_timeout", True):
            _try_close_inner_coro(inner)
        _log.warning(
            "utility call exceeded %.1fs (max_wait_s=%.1f, model=%s); "
            "raising RepairableException for cascade failover",
            elapsed, max_wait, model_name or "<unknown>",
        )
        raise RepairableException(
            f"utility_model_timeout: {model_name or 'unknown'} "
            f"exceeded {elapsed:.1f}s (cap {max_wait:.0f}s)"
        ) from None
    except asyncio.CancelledError:
        # The framework cancelled us (e.g. user intervention). Pass
        # it through but still attempt the inner coroutine cleanup
        # so we don't leak a half-built litellm chain.
        if cfg.get("close_inner_on_timeout", True):
            _try_close_inner_coro(inner)
        raise
    except Exception as exc:
        # Deterministic upstream errors (PAYLOAD_TOO_LARGE, context
        # overflow, invalid api key, etc.) come back from litellm in
        # milliseconds. Letting them bubble out of utility_timeout
        # is fine in principle, but the cascade that called us has
        # already classified them as permanent and short-circuited.
        # For the cases where the cascade DIDN'T catch it (older
        # plugin versions, non-fallback callers, or exceptions
        # wrapped in a way the cascade doesn't recognize), we still
        # want to surface them as a RepairableException so the agent
        # loop doesn't waste a full max_wait_s waiting for a response
        # that will never arrive. We deliberately do NOT include the
        # raw exception text in the RepairableException -- it can be
        # huge (full prompt dump) and would spam the log.
        if _is_deterministic_upstream_error(exc):
            elapsed = time.monotonic() - started
            stats.utility_timeout_record_timeout(model_name)
            if cfg.get("close_inner_on_timeout", True):
                _try_close_inner_coro(inner)
            _log.warning(
                "utility call rejected deterministically in %.3fs "
                "(model=%s): %s; raising RepairableException for fast-fail",
                elapsed, model_name or "<unknown>", type(exc).__name__,
            )
            raise RepairableException(
                f"utility_model_rejected: {model_name or 'unknown'} "
                f"({type(exc).__name__})"
            ) from exc
        raise
    else:
        elapsed = time.monotonic() - started
        stats.utility_timeout_record_call(elapsed)
        return result
    finally:
        # Defensive: if ``inner`` is still a coroutine reference and
        # not yet closed (e.g. an outer cancel came in between our
        # try and the framework's catch), try once more.
        try:
            if asyncio.iscoroutine(inner) and not inner.cr_frame:
                # Already closed/finalized; nothing to do.
                pass
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Deterministic error detection (used by guarded_call fast-fail path)
# ---------------------------------------------------------------------------

# String fragments in litellm exception messages that indicate the upstream
# rejected the call for a reason that retrying will not fix. The cascade in
# models_ext already classifies most of these as "permanent", but we keep
# a defense-in-depth list here so callers that bypass the cascade still
# fast-fail within the utility timeout guard instead of waiting max_wait_s.
_DETERMINISTIC_PHRASES = (
    "payload_too_large",
    "request body too large",
    "request entity too large",
    "context_length_exceeded",
    "context length exceeded",
    "maximum context length",
    "reduce the length",
    "invalid_api_key",
    "incorrect api key",
    "model_not_found",
    "model not found",
    "insufficient_quota",
    "quota exceeded",
    "credit balance",
    "permission_denied",
    "401 ",
    "402 ",
    "403 ",
    "404 ",
    "413 ",
)


def _is_deterministic_upstream_error(exc: BaseException) -> bool:
    """True when ``exc`` is a known permanent-failure from a litellm-backed
    provider. Used to fast-fail the outer timeout guard so the agent
    loop does not wait the full max_wait_s for an error that already
    came back in milliseconds.
    """
    if exc is None:
        return False
    # Common case: litellm types we know about. Imported lazily so the
    # module loads even when litellm is absent.
    try:
        import litellm.exceptions as _le  # type: ignore

        if isinstance(
            exc,
            (
                _le.BadRequestError,
                _le.AuthenticationError,
                _le.PermissionDeniedError,
                _le.NotFoundError,
                _le.UnprocessableEntityError,
                _le.ContextWindowExceededError,
                _le.ContentPolicyViolationError,
                _le.BudgetExceededError,
            ),
        ):
            return True
    except Exception:  # noqa: BLE001
        pass
    msg = (str(exc) or "").lower()
    if not msg:
        return False
    return any(phrase in msg for phrase in _DETERMINISTIC_PHRASES)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def reset() -> None:
    """Reset the module's cached config. Called by hooks.uninstall()."""
    global _resolved
    _resolved = {}
