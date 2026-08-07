"""Backup patch installer: directly monkey-patches Agent methods at init.
This ensures fallback.py patches are active even if _functions overrides fail to load."""
from helpers.extension import Extension
from agent import Agent


class InstallFallbackPatches(Extension):
    def execute(self, **kwargs):
        # Lazy import to avoid circular dependencies during module load
        from usr.plugins._model_fallback.fallback import (
            _patched_call_utility_model,
            _patched_call_chat_model,
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

        if self.agent and self.agent.context:
            self.agent.context.log.log("info", "Model Fallback System patches installed via agent_init.")
