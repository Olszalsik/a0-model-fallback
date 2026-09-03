"""Outer timeout guard for ``call_utility_model`` (v2.2 resilience).

Installed at ``agent_init`` time, AFTER ``_00_install_fallback_patches.py``
which monkey-patches the cascade onto ``Agent.call_utility_model``. Both
run in the same ``call_extensions_sync("agent_init", ...)`` call from
``Agent.__init__``; the framework sorts extensions by module name, so
``_00`` < ``_10`` means the cascade is in place by the time we run.

We wrap the (now-cascaded) ``Agent.call_utility_model`` with
``asyncio.wait_for(..., timeout=...)`` so a hung utility call (typically
ollama on CPU) cannot freeze the agent loop indefinitely. On
``asyncio.TimeoutError`` we:

1. Best-effort close the inner litellm coroutine chain via
   ``memory_hardening.helpers.coroutine_guard.close_inner_coro`` (lazy
   import, no-op if the plugin is disabled).
2. Raise ``RepairableException("utility_model_timeout")`` so the
   agent's monologue exception handler catches it, the LLM is told the
   utility call failed, and the user sees a recoverable error rather
   than a hard timeout.

This wrapper sits OUTSIDE the cascade: the cascade's own ``wait_for``
governs per-candidate timeouts and rotations; this guard governs the
absolute wall-clock wait for the whole utility call.

Idempotent
----------
A sentinel attribute (``_utility_timeout_patched``) prevents the
wrapper from being installed twice (e.g. if the framework re-runs
``agent_init``). If the sentinel is present we leave the function
alone.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict

from helpers.extension import Extension
from agent import Agent

_log = logging.getLogger("model_fallback.utility_timeout.patch")


def _resolve_config(agent) -> Dict[str, Any]:
    try:
        from helpers import plugins as plugin_helpers  # type: ignore
        cfg = plugin_helpers.get_plugin_config("_model_fallback", agent) or {}
        # v2.8.3: get_plugin_config returns config.json WITHOUT merging
        # default_config.yaml, so the YAML's ``utility_timeout_guard:``
        # section (60s/180s, raised 2026-07-23) never reached the
        # runtime and DEFAULTS' stale 30s/120s won instead. Merge the
        # YAML defaults UNDER config.json so YAML edits are effective
        # and user config.json values still win.
        try:
            defaults = plugin_helpers.get_default_plugin_config(
                "_model_fallback"
            ) or {}
            if isinstance(defaults, dict):
                merged = dict(defaults)
                if isinstance(cfg, dict):
                    merged.update(cfg)
                cfg = merged
        except Exception:  # noqa: BLE001
            pass
    except Exception:  # noqa: BLE001
        cfg = {}
    # v2.5 WebUI: top-level ``utility_timeout_guard_enabled`` wins
    # over the nested section's ``enabled`` so a user toggling OFF
    # in the WebUI takes effect immediately on the next agent_init.
    from usr.plugins._model_fallback.helpers import toggles
    if not toggles.resolve_toggle(cfg, "utility_timeout_guard", default=True):
        return {"enabled": False}
    overrides = cfg.get("utility_timeout_guard") if isinstance(cfg, dict) else None
    if not isinstance(overrides, dict):
        overrides = {}
    # v2.8.5: the outer guard's budget must accommodate the inner cascade's
    # largest legitimate per-call timeout -- a COLD router call gets
    # router_cold_call_timeout_s (150s warm-up budget, v2.7.0). With the
    # flat default the outer guard fired first on every utility call to a
    # cold router, killed the call, and (before v2.8.5 booked nothing)
    # the next utility call re-tried the same router for another full
    # budget -- utility traffic starved while chat/turn traffic worked.
    # Injected as a private key; guarded_call raises its effective budget
    # to max(default, router cold), still capped by max_wait_s.
    try:
        router_cold_s = float(cfg.get("router_cold_call_timeout_s") or 0.0)
    except (TypeError, ValueError):
        router_cold_s = 0.0
    if router_cold_s > 0:
        overrides["_router_cold_s"] = router_cold_s
    from usr.plugins._model_fallback.helpers import utility_timeout
    return utility_timeout.resolve_config(overrides)


def _model_name(agent) -> str:
    try:
        if agent is None:
            return ""
        model = agent.get_utility_model()
        name = getattr(model, "name", None) or getattr(model, "model_name", None)
        if isinstance(name, str):
            return name
    except Exception:  # noqa: BLE001
        pass
    return ""


def _plugin_version() -> str:
    """Best-effort plugin version for the install-guard stamp (AD)."""
    try:
        from helpers import plugins as _plugins
        return str(
            getattr(_plugins.get_plugin_meta("_model_fallback"), "version", "") or ""
        )
    except Exception:  # noqa: BLE001
        return ""


def _install(agent: Agent | None) -> bool:
    """Install the timeout guard on ``Agent.call_utility_model``.

    Returns True if a fresh install happened, False if the function
    was already wrapped or the config is disabled.
    """
    cfg = _resolve_config(agent)
    from usr.plugins._model_fallback.helpers import utility_timeout
    if not cfg.get("enabled", True):
        # v2.8.5: refresh the resolved config BEFORE returning so a
        # wrapper that is already live (installed by an earlier agent_init
        # while the guard was enabled) sees the toggle-off -- guarded_call
        # re-checks ``cfg.get("enabled")`` per call against this cache.
        # The old code returned False with the stale enabled=True config
        # still cached, so disabling the guard never took effect until a
        # process restart.
        utility_timeout.set_resolved(cfg)
        return False

    stamp = _plugin_version()
    current = Agent.call_utility_model
    if getattr(current, "_utility_timeout_patched", False):
        # Already wrapped.
        # v2.8.5 (AD): version-stamp the wrapper. After a plugin UPDATE the
        # old version's wrapper (closure over old code) would stay live for
        # the rest of the process; on a version mismatch, unwrap back to
        # the true original and fall through to a fresh install.
        if getattr(current, "_utility_timeout_patch_version", "") == stamp:
            utility_timeout.set_resolved(cfg)
            return False
        original = getattr(current, "_utility_timeout_original", None)
        if original is not None:
            Agent.call_utility_model = original  # type: ignore[assignment]
            current = Agent.call_utility_model
            if getattr(current, "_utility_timeout_patched", False):
                # Original itself was a stale wrapper (shouldn't happen:
                # re-installs always wrap the true original). Leave it.
                utility_timeout.set_resolved(cfg)
                return False

    async def wrapped(self: Agent, *args: Any, **kwargs: Any) -> Any:
        # Re-resolve the model name at call time; the agent's
        # utility model can be changed between calls.
        model_name = _model_name(self)
        # Build the inner coroutine factory. The factory pattern is
        # important: we want to create the inner coroutine on THIS
        # task so asyncio.wait_for can cancel it cleanly.
        def _inner():
            return current(self, *args, **kwargs)
        return await utility_timeout.guarded_call(
            _inner,
            model_name=model_name,
            # v2.8.3: pass None so guarded_call reads get_resolved() --
            # the closure dict ``cfg`` is frozen at install time, so a
            # later config change (refreshed via set_resolved on the
            # sentinel branch) never reached the wrapper.
            config_overrides=None,
            # v2.8.5: needed so a guard timeout can book a cooldown for
            # this utility label (see guarded_call) -- without it the
            # outer guard killed the call and left no state behind.
            agent=self,
        )

    wrapped._utility_timeout_patched = True  # type: ignore[attr-defined]
    wrapped._utility_timeout_original = current  # type: ignore[attr-defined]
    wrapped._utility_timeout_patch_version = stamp  # type: ignore[attr-defined]
    Agent.call_utility_model = wrapped  # type: ignore[assignment]
    utility_timeout.set_resolved(cfg)
    _log.info(
        "utility timeout guard installed (default=%.1fs, max=%.1fs, jitter=%.1fs)",
        cfg.get("default_timeout_s"), cfg.get("max_wait_s"), cfg.get("jitter_s"),
    )
    return True


def uninstall() -> bool:
    """Restore the original ``call_utility_model`` if we wrapped it.

    Used by ``hooks.uninstall()`` so a plugin disable returns to
    baseline behavior.
    """
    current = Agent.call_utility_model
    original = getattr(current, "_utility_timeout_original", None)
    if original is None:
        return False
    Agent.call_utility_model = original  # type: ignore[assignment]
    return True


class InstallUtilityTimeoutPatch(Extension):
    def execute(self, **kwargs: Any) -> None:
        try:
            _install(self.agent)
        except Exception as exc:  # noqa: BLE001
            # Never let our patch crash the agent loop.
            _log.debug("install utility timeout patch failed: %s", exc)
