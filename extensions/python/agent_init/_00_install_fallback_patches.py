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
        # Lazy import to avoid circular dependencies during module load.
        # v2.8.1: resolve via the module object with getattr, NOT from-imports.
        # A stale fallback.py (e.g. a .pyc left over after a partial update on
        # a slow bind mount) must degrade gracefully -- an ImportError here
        # would kill Agent.__init__ and with it every new chat.
        # v2.8.3: the import itself is guarded too -- call_extensions_sync
        # does not catch exceptions, so a SyntaxError/ImportError raised BY
        # the fallback module import (broken sibling import chain, truncated
        # file on a slow mount) would still kill Agent.__init__ for every
        # new chat. No fallback coverage is better than no agent at all.
        try:
            import usr.plugins._model_fallback.fallback as fb
        except Exception as import_exc:  # noqa: BLE001
            try:
                self.agent.context.log.log(
                    "error",
                    "Model Fallback System: failed to import fallback.py "
                    f"({type(import_exc).__name__}: {import_exc}). No "
                    "fallback coverage active this session.",
                )
            except Exception:
                pass
            return

        _patched_call_utility_model = getattr(
            fb, "_patched_call_utility_model", None)
        _patched_call_chat_model = getattr(fb, "_patched_call_chat_model", None)
        install_chat_turn_patch = getattr(fb, "install_chat_turn_patch", None)
        if _patched_call_utility_model is None or _patched_call_chat_model is None:
            try:
                self.agent.context.log.log(
                    "error",
                    "Model Fallback System: fallback.py is missing its patch "
                    "functions (stale install?). No fallback coverage active.",
                )
            except Exception:
                pass
            return

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
        turn_patched = False
        if install_chat_turn_patch is not None:
            try:
                turn_patched = install_chat_turn_patch(Agent)
            except Exception as e:
                try:
                    self.agent.context.log.log(
                        "warning",
                        f"Model Fallback System: turn-path cascade not "
                        f"installed ({type(e).__name__}: {e}). Chat cascade "
                        f"still active.",
                    )
                except Exception:
                    pass
        elif self.agent and self.agent.context:
            self.agent.context.log.log(
                "warning",
                "Model Fallback System: fallback.py predates the turn-path "
                "cascade (v2.8.0) -- main-loop calls have chat-cascade "
                "coverage only.",
            )

        if self.agent and self.agent.context:
            msg = "Model Fallback System patches installed via agent_init."
            if turn_patched:
                msg += " (turn path: call_chat_model_turn cascade active)"
            self.agent.context.log.log("info", msg)
