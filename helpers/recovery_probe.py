"""Background recovery probes (v2.9.0).

Problem this solves: a cooled-down label is only re-probed by REAL agent
traffic -- the first post-cooldown call pays the full timeout/429 again.
If the provider recovered 5 minutes into a 10-minute cooldown, every
cascade in every agent keeps routing around a healthy label for the
remaining 5 minutes, and the "recovery" is rediscovered by paying for a
failed attempt.

Mechanism: when a cooldown is booked, the cascade registers the failed
candidate's model wrapper here (STRONG ref -- see
register_probe_target). A single background asyncio
task sweeps the registry every ``recovery_probe_interval_s``:

  - targets whose cooldown is gone -> dropped (recovered or cleared)
  - labels with remaining cooldown < ``recovery_probe_min_cooldown_s``
    -> left alone (real traffic re-probes them soon anyway)
  - dead-marked labels -> skipped (recovery needs user action, e.g. a
    fixed API key; probing 404/invalid-key every 45s is pure waste)
  - otherwise: ONE cheap call via the same wrapper the cascade uses
    (``model.unified_call(system_message, user_message, ...)`` -- same
    transport, kwargs and api_key as the real path, no cascade re-entry,
    no litellm hand-assembly), under ``asyncio.wait_for``

A successful probe clears the cooldown EARLY (in place, same object the
live cascades hold -- the v2.8.4 in-place invariant), marks the label
healthy cross-agent, and logs. A failed probe changes nothing except the
counters; the cooldown keeps its original expiry.

Bounded provider burn: max ``recovery_probe_max_targets_per_cycle``
probes per sweep, each at most ``recovery_probe_timeout_s`` long, and
only labels whose remaining cooldown exceeds the threshold are touched.
"""
from __future__ import annotations

import asyncio
import time
import weakref
from typing import Any, Dict, Optional, Tuple

# (context_key, label) -> {"model": weakref, "agent": weakref,
#                          "api_base": str, "registered_at": float}
_TARGETS: Dict[Tuple[tuple, str], Dict[str, Any]] = {}

# Counters (module-local; exposed via snapshot() for api/stats.py).
_counters: Dict[str, Any] = {
    "probes_attempted": 0,
    "probes_succeeded": 0,
    "probes_failed": 0,
    "cooldowns_cleared_early": 0,
    "targets_registered": 0,
    "last_probe_at": 0.0,
    "last_probe_label": "",
    "last_probe_result": "",
}

_TASK: Optional[asyncio.Task] = None

_DEFAULTS = {
    "recovery_probe_enabled": True,
    "recovery_probe_interval_s": 45.0,
    "recovery_probe_min_cooldown_s": 120.0,
    "recovery_probe_timeout_s": 15.0,
    "recovery_probe_max_targets_per_cycle": 2,
}


def _cfg() -> dict:
    """Merged plugin config for the probe knobs.

    Uses the fallback module's merged resolver (YAML defaults UNDER
    config.json) when an agent is available from a target; falls back to
    the code defaults above when config cannot be read. The knobs are
    also in default_config.yaml so WebUI edits reach config.json.
    """
    out = dict(_DEFAULTS)
    try:
        from usr.plugins.model_fallback import fallback as fb
        # Any live target's agent works for config resolution; None is
        # tolerated by _get_plugin_cfg's own try/except.
        agent = None
        for tgt in _TARGETS.values():
            agent = tgt.get("agent")()
            if agent is not None:
                break
        merged = fb._get_plugin_cfg(agent)
        for key in _DEFAULTS:
            if key in merged:
                out[key] = merged[key]
    except Exception:  # noqa: BLE001
        pass
    # Clamps (defensive against hand-edited config).
    out["recovery_probe_enabled"] = bool(out["recovery_probe_enabled"])
    out["recovery_probe_interval_s"] = max(10.0, float(out["recovery_probe_interval_s"]))
    out["recovery_probe_min_cooldown_s"] = max(0.0, float(out["recovery_probe_min_cooldown_s"]))
    out["recovery_probe_timeout_s"] = min(60.0, max(5.0, float(out["recovery_probe_timeout_s"])))
    out["recovery_probe_max_targets_per_cycle"] = max(
        1, int(out["recovery_probe_max_targets_per_cycle"])
    )
    return out


def register_probe_target(agent, label: str, model, api_base: str = "") -> None:
    """Remember a cooled-down candidate's wrapper so the sweep can probe it.

    Called from ``_handle_error_cooldown`` at booking time. ``model`` is
    the candidate wrapper that JUST failed (it carries api_key/api_base/
    provider kwargs), so the probe replays the same transport.

    The model wrapper is a STRONG reference on purpose: per-call wrappers
    are usually unreachable after the cascade iteration that booked the
    cooldown, so a weakref would be collected before the first sweep and
    the whole feature would be a silent no-op. Wrappers are small config
    objects and the registry is bounded by distinct (context, label)
    pairs; a target is dropped as soon as its label leaves cooldown. The
    AGENT stays a weakref -- agents are heavy, and once the chat/context
    is gone there is nobody to route the recovery to.
    """
    if agent is None or model is None or not label or label.startswith("<"):
        return
    try:
        ctx = getattr(agent, "context", None)
        ctx_id = getattr(ctx, "id", None)
        key = ("__global__",) if ctx_id is None else (str(ctx_id),)
        try:
            agent_ref = weakref.ref(agent)
        except TypeError:  # noqa: PERF203 -- non-weakref-able agent stand-in
            agent_ref = lambda a=agent: a  # noqa: E731
        entry: Dict[str, Any] = {
            "agent": agent_ref,
            "model": model,
            "api_base": api_base,
            "registered_at": time.monotonic(),
        }
        _TARGETS[((key, label))] = entry
        _counters["targets_registered"] = _counters["targets_registered"] + 1
    except Exception:  # noqa: BLE001
        pass
    ensure_loop()


def ensure_loop() -> None:
    """Start the background sweep task once (idempotent, loop-safe).

    A no-op when there is no running event loop (sync booking contexts) --
    the next registration inside async context retries.
    """
    global _TASK
    try:
        if _TASK is not None and not _TASK.done():
            return
        loop = asyncio.get_running_loop()
        _TASK = loop.create_task(_probe_loop())
    except Exception:  # noqa: BLE001
        _TASK = None


async def _probe_loop() -> None:
    while True:
        try:
            await sweep()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            pass
        try:
            interval = _cfg()["recovery_probe_interval_s"]
        except Exception:  # noqa: BLE001
            interval = _DEFAULTS["recovery_probe_interval_s"]
        await asyncio.sleep(interval)


async def sweep() -> int:
    """One pass over the registered targets. Returns probes attempted.

    All per-target failures are contained: a broken target or a probe
    error only skips that label; the loop keeps running.
    """
    cfg = _cfg()
    if not cfg["recovery_probe_enabled"]:
        return 0
    try:
        from usr.plugins.model_fallback import fallback as fb
    except Exception:  # noqa: BLE001
        return 0

    now = time.monotonic()
    attempted = 0
    budget = int(cfg["recovery_probe_max_targets_per_cycle"])
    for key, tgt in list(_TARGETS.items()):
        if attempted >= budget:
            break
        agent = tgt.get("agent")()
        model = tgt.get("model")  # strong ref -- no weakref call
        if agent is None or model is None:
            _TARGETS.pop(key, None)  # agent collected / invalid -> drop
            continue
        label = key[1]
        try:
            store = fb._get_cooldown_store(agent)
            until = store.get(label) if isinstance(store, dict) else None
        except Exception:  # noqa: BLE001
            continue
        if until is None:
            # Not in cooldown anymore (recovered on its own, cleared by
            # the user, or popped by a live cascade) -- nothing to probe.
            _TARGETS.pop(key, None)
            continue
        if until <= now:
            # v3.4.1: expired-but-still-stored cooldown -- real traffic
            # can use the label again; probing it here only pays for a
            # call no cascade is routing to, and the target would
            # otherwise linger for the rest of the sweep's life.
            _TARGETS.pop(key, None)
            continue
        # v3.4.1: per-target probe backoff. A failed probe used to be
        # retried on EVERY sweep (45s), so a still-down label burned a
        # probe + timeout per sweep for the whole cooldown. Escalate
        # 60s -> 120s -> ... capped at 15min; reset implicitly when the
        # target is re-registered (a fresh booking makes a fresh entry).
        next_probe_at = float(tgt.get("next_probe_at") or 0.0)
        if now < next_probe_at:
            continue
        if (until - now) < cfg["recovery_probe_min_cooldown_s"]:
            # Short cooldown: real traffic re-probes soon anyway.
            continue
        try:
            if fb._is_label_dead(label):
                # Dead mark = invalid key / model gone; recovery needs a
                # user action, not a probe. Re-checking every 45s against
                # a 404 just burns quota.
                continue
        except Exception:  # noqa: BLE001
            pass

        _counters["probes_attempted"] = _counters["probes_attempted"] + 1
        attempted += 1
        _counters["last_probe_at"] = time.time()
        _counters["last_probe_label"] = label
        ok = await _probe_once(model, cfg["recovery_probe_timeout_s"])
        if ok:
            _counters["probes_succeeded"] = _counters["probes_succeeded"] + 1
            _counters["last_probe_result"] = "ok"
            _clear_cooldown_early(fb, agent, label)
            _TARGETS.pop(key, None)
        else:
            _counters["probes_failed"] = _counters["probes_failed"] + 1
            _counters["last_probe_result"] = "fail"
            streak = int(tgt.get("fail_streak") or 0) + 1
            tgt["fail_streak"] = streak
            tgt["next_probe_at"] = time.monotonic() + min(60.0 * (2 ** min(streak, 4)), 900.0)
    return attempted


async def _probe_once(model, timeout_s: float) -> bool:
    """One cheap health-check call through the candidate's own wrapper.

    ``unified_call`` is the same entry point the cascades use, so auth,
    api_base, provider prefixes and (via models_ext) the Responses ->
    chat-completions fallback all behave exactly as in real traffic.
    No streaming callback and no rate-limiter callback: a probe must not
    touch the UI or the agent's rate limiter.
    """
    try:
        coro = model.unified_call(
            system_message="You are a health check.",
            user_message="ping",
            response_callback=None,
            rate_limiter_callback=None,
            fallbacks=None,
        )
        await asyncio.wait_for(coro, timeout_s)
        return True
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001
        return False


def _clear_cooldown_early(fb, agent, label: str) -> None:
    """Clear ``label``'s cooldown in place + persist + cross-agent healthy.

    Mutates the LIVE store object (the same dict a running cascade holds
    -- the v2.8.4 in-place invariant), never swaps a literal.
    """
    try:
        store = fb._get_cooldown_store(agent)
        if isinstance(store, dict) and label in store:
            store.pop(label, None)
            fb._save_cooldown_store(agent, store)
            _counters["cooldowns_cleared_early"] = (
                _counters["cooldowns_cleared_early"] + 1
            )
            # v2.9.1: route event -- distinguishes probe recovery from a
            # live-call recovery (cooldown_cleared_by_success).
            try:
                from usr.plugins.model_fallback.helpers import events
                events.record_event(
                    "cooldown_cleared_early", agent=agent, label=label
                )
            except Exception:  # noqa: BLE001
                pass
        fb._mark_label_healthy(agent, label)
        try:
            agent.context.log.log(
                "info",
                f"Recovery probe succeeded on [{label}] -- cooldown "
                "cleared early.",
            )
        except Exception:  # noqa: BLE001
            pass
    except Exception:  # noqa: BLE001
        pass


def shutdown_probes() -> None:
    """Cancel the background task and drop the registry (hooks.uninstall)."""
    global _TASK
    if _TASK is not None:
        try:
            _TASK.cancel()
        except Exception:  # noqa: BLE001
            pass
        _TASK = None
    _TARGETS.clear()


def snapshot() -> Dict[str, Any]:
    """Counters + registry size for api/stats.py."""
    out = dict(_counters)
    out["targets"] = len(_TARGETS)
    out["loop_alive"] = bool(_TASK is not None and not _TASK.done())
    if out["last_probe_at"]:
        out["seconds_since_last_probe"] = round(time.time() - out["last_probe_at"], 1)
    return out


def reset_counters() -> None:
    for key in _counters:
        if key in ("last_probe_at",):
            _counters[key] = 0.0
        elif key in ("last_probe_label", "last_probe_result"):
            _counters[key] = ""
        else:
            _counters[key] = 0