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
``model_fallback`` already does that (its cascade lives in
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
from usr.plugins.model_fallback.helpers import stats

_log = logging.getLogger("model_fallback.utility_timeout")

# Default config. Mirrored from default_config.yaml so unit tests that
# don't load the plugin config still get sane numbers.
# v2.8.3: synced to the YAML values (60s/180s, raised from 30/120 on
# 2026-07-23). The YAML raise never took effect at runtime because
# get_plugin_config does not merge default_config.yaml and config.json
# had no nested section -- _resolve_config now merges the YAML, and
# these DEFAULTS stay in sync as the last fallback.
DEFAULTS: Dict[str, Any] = {
    "enabled": True,
    "default_timeout_s": 60.0,
    "max_wait_s": 180.0,
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
    # v3.4.1: the stale 30/120 literals here silently overrode the
    # synced DEFAULTS (60/180) whenever a value arrived as 0/""/None --
    # an explicit `or` treats a CONFIGURED 0 as "use the stale literal"
    # instead of "use the default". `is not None` semantics: a real
    # value (including 0 -> clamped by max_wait_s logic below) wins;
    # only absent/None falls back to DEFAULTS. Non-numeric junk
    # degrades to the default via the except.
    for key, fallback in (
        ("default_timeout_s", DEFAULTS["default_timeout_s"]),
        ("max_wait_s", DEFAULTS["max_wait_s"]),
        ("jitter_s", 0.0),
    ):
        raw = cfg.get(key)
        try:
            cfg[key] = float(raw) if raw is not None else float(fallback)
        except (TypeError, ValueError):
            cfg[key] = float(fallback)
    cfg["enabled"] = bool(cfg.get("enabled", True))
    cfg["close_inner_on_timeout"] = bool(cfg.get("close_inner_on_timeout", True))
    return cfg


def set_resolved(cfg: Dict[str, Any]) -> None:
    """Cache the resolved config so the hot path doesn't re-read."""
    global _resolved
    _resolved = dict(cfg)


def get_resolved() -> Dict[str, Any]:
    # v3.1.1 fix: this function ASSIGNS to ``_resolved`` on the
    # cache-miss path, so without ``global`` Python compiled the name
    # as a local for the whole function -- the read on the first line
    # raised ``UnboundLocalError: cannot access local variable
    # '_resolved'`` on EVERY call, crashing every utility-model call
    # routed through guarded_call (e.g. the memory plugin's
    # memorize hook). set_resolved()/reset() already declared global;
    # this one was missing it.
    global _resolved
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
    agent: Any = None,
) -> Any:
    """Run ``inner_coro_factory()`` under ``asyncio.wait_for``.

    The factory pattern is important: we want to create the inner
    coroutine on THIS task so ``asyncio.wait_for`` can cancel it
    cleanly. If we accepted a pre-built coroutine, we'd risk closing
    a coroutine that another task is already driving.

    On ``asyncio.TimeoutError`` we close the inner chain, record a
    stat, book a cooldown for the utility label (v2.8.5, needs
    ``agent``), and re-raise as ``RepairableException`` so the LLM
    sees a recoverable error rather than a hard timeout.
    """
    cfg = resolve_config(config_overrides or get_resolved())
    if not cfg.get("enabled", True):
        return await inner_coro_factory()

    # v3.4.1: the stale 120/30 literals here silently re-overrode the
    # resolved values behind resolve_config's back (a resolved 0 fell
    # back to the literal, not the DEFAULT). resolve_config guarantees
    # non-None floats, so use them verbatim (tests drive sub-second
    # budgets through this path; no floor).
    max_wait = float(cfg.get("max_wait_s") or 0.0)
    default_to = float(cfg.get("default_timeout_s") or 0.0)
    # v2.8.5: raise the budget to cover the inner cascade's largest
    # legitimate per-call timeout. A cold router call gets up to
    # router_cold_call_timeout_s (150s, v2.7.0), injected here as the
    # private ``_router_cold_s`` key by
    # _10_install_utility_timeout_patch._resolve_config; without this
    # the OUTER guard fired first on every utility call to a cold
    # router, killed the call, and the cascade never got to book its
    # own cooldown. Normal labels keep the default ceiling, and the
    # max_wait_s hard cap still wins for everything -- so this stays
    # min(budget, max_wait), not an unbounded max.
    router_cold = float(cfg.get("_router_cold_s") or 0.0)
    budget = max(max(default_to, 0.0), max(router_cold, 0.0))
    # v3.2.0: learned generation-budget parity with the cascades. A long
    # utility generation (memory summarization, JSON validation on big
    # payloads) must not die at the outer cap when the label has PROVEN it
    # needs longer. Both the budget and the max_wait cap lift to the
    # override (still bounded by the fallback module's gen_budget_max_s
    # wall); override 0 -> exactly the old arithmetic.
    gen_override = 0.0
    try:
        from usr.plugins.model_fallback import fallback as _fb_cfg
        gen_override = float(_fb_cfg._gen_budget_override_for(model_name, agent))
    except Exception:  # noqa: BLE001
        gen_override = 0.0
    if gen_override > 0:
        budget = max(budget, gen_override)
        max_wait = max(max_wait, gen_override)
    timeout_s = min(budget, max_wait)
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
        # v3.2.0: the outer guard's budget fired -- feed the same
        # grow-on-timeout learner the cascades use so the label earns a
        # bigger budget instead of being re-killed at the same ceiling on
        # every subsequent utility call. Best-effort; never blocks the
        # RepairableException path below.
        try:
            from usr.plugins.model_fallback import fallback as _fb_grow
            _fb_grow._grow_gen_budget(model_name, elapsed, agent)
        except Exception:  # noqa: BLE001
            pass
        if cfg.get("close_inner_on_timeout", True):
            _try_close_inner_coro(inner)
        # v2.8.5: book the timeout cooldown for this utility label so the
        # next utility call routes around it. Before, the outer guard
        # killed the call and booked NOTHING -- a candidate that hung past
        # the budget was retried from scratch on every subsequent call and
        # utility traffic starved. Router labels keep their policy: a pure
        # timeout books a 0s cooldown (router_timeout_no_cooldown), handled
        # inside _handle_error_cooldown via the label->api_base registry.
        # The cascade also usually books its own cooldown -- but only when
        # ITS wait_for fires first, which this budget change makes the
        # normal case; this is the safety net for the race the other way.
        try:
            if agent is not None and model_name:
                from usr.plugins.model_fallback import fallback as _fb

                timeout_exc = TimeoutError(
                    f"utility timeout guard: {model_name} exceeded "
                    f"{elapsed:.1f}s"
                )
                _fb._evict_warm_on_timeout(timeout_exc, model_name)
                # v3.1.0: parity with the cascade timeout sites -- drop the
                # label's latency samples so the next sizing falls back to
                # the full base timeout (the budget that just fired was too
                # tight).
                try:
                    from usr.plugins.model_fallback.helpers import latency as _lat

                    _lat.clear_label(model_name)
                except Exception:  # noqa: BLE001
                    pass
                _fb._handle_error_cooldown(
                    timeout_exc, model_name, _fb._get_cooldown_store(agent),
                    agent,
                )
        except Exception:  # noqa: BLE001
            pass
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
