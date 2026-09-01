"""Backup patch installer: directly monkey-patches Agent methods at init.
This ensures fallback.py patches are active even if _functions overrides fail to load.

v2.8.0: also patches ``Agent.call_chat_model_turn`` -- since the v2.10/2.11
upstream merge the main agent loop uses the turn path, which previously had
NO fallback coverage (429s stopped the agent). See fallback.py
install_chat_turn_patch()."""
from helpers.extension import Extension
from agent import Agent


class InstallFallbackPatches(Extension):
    def execute(self, **kwargs):
        # Lazy import to avoid circular dependencies during module load
        from usr.plugins._model_fallback.fallback import (
            _patched_call_utility_model,
            _patched_call_chat_model,
            install_chat_turn_patch,
        )

        # Only patch if not already patched (avoid double-patching on reload)
        if getattr(Agent.call_utility_model, "_fallback_patched", False):
            return
        if getattr(Agent.call_chat_model, "_fallback_patched", False):
            return

        Agent.call_utility_model = _patched_call_utility_model
        Agent.call_chat_model = _patched_call_chat_model
        _patched_call_utility_model._fallback_patched = True  # type: ignore[attr-defined]
        _patched_call_chat_model._fallback_patched = True  # type: ignore[attr-defined]

        # Turn path (v2.10+ monologue). install_chat_turn_patch captures the
        # @extensible wrapper before overwriting, so the original's own
        # extension points keep firing. Idempotent on its own marker.
        turn_patched = install_chat_turn_patch(Agent)

        if self.agent and self.agent.context:
            msg = "Model Fallback System patches installed via agent_init."
            if turn_patched:
                msg += " (turn path: call_chat_model_turn cascade active)"
            self.agent.context.log.log("info", msg)
