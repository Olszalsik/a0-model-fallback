"""Backup patch installer: directly monkey-patches Agent methods at init.
This ensures fallback.py patches are active even if _functions overrides fail to load.

v2.8.0: also patches ``Agent.call_chat_model_turn`` -- since the v2.10/2.11
upstream merge the main agent loop uses the turn path, which previously had
NO fallback coverage (429s stopped the agent). See fallback.py
install_chat_turn_patch().

v2.8.6: (V) per-method, version-stamped idempotence -- a partial state
(utility patched, chat not) is now repairable instead of early-returning
on utility's sentinel alone, and a cascade left by an older plugin
version is re-installed instead of being protected by its forever-True
sentinel. (N) the TRUE originals of call_utility_model / call_chat_model
are captured on the Agent class so hooks.uninstall() can fully restore
it (module-level uninstall() below)."""
from helpers.extension import Extension
from agent import Agent


def _plugin_version() -> str:
    """Best-effort plugin version for the install-guard stamp (v2.8.5 AD
    pattern, mirrored from _10_install_utility_timeout_patch)."""
    try:
        from helpers import plugins as _plugins
        return str(
            getattr(_plugins.get_plugin_meta("model_fallback"), "version", "") or ""
        )
    except Exception:  # noqa: BLE001
        return ""


def uninstall() -> bool:
    """v2.8.6 (N): full restore chain for hooks.uninstall().

    Order matters: the utility timeout patch (_10) wraps the cascade
    installed here, so its wrapper must be unwrapped first (that restores
    the cascade onto the class), then the class gets its true originals
    back. The turn original is the one captured by
    ``fallback.install_chat_turn_patch`` (module global
    ``_ORIGINAL_CALL_CHAT_MODEL_TURN``).

    Returns True when at least one Agent method was restored.
    """
    try:
        from usr.plugins.model_fallback.extensions.python.agent_init import (
            _10_install_utility_timeout_patch,
        )
        _10_install_utility_timeout_patch.uninstall()
    except Exception:  # noqa: BLE001
        pass
    restored = False
    for attr, orig_attr in (
        ("call_utility_model", "_mfb_original_call_utility_model"),
        ("call_chat_model", "_mfb_original_call_chat_model"),
    ):
        original = getattr(Agent, orig_attr, None)
        if original is not None:
            try:
                setattr(Agent, attr, original)
                delattr(Agent, orig_attr)
                restored = True
            except Exception:  # noqa: BLE001
                pass
    try:
        import usr.plugins.model_fallback.fallback as fb
        turn_orig = getattr(fb, "_ORIGINAL_CALL_CHAT_MODEL_TURN", None)
        if (
            turn_orig is not None
            and getattr(Agent.call_chat_model_turn, "_fallback_turn_patched", False)
        ):
            Agent.call_chat_model_turn = turn_orig
            restored = True
    except Exception:  # noqa: BLE001
        pass
    return restored


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
            import usr.plugins.model_fallback.fallback as fb
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

        # v2.8.6 (V): per-method, version-stamped idempotence. The old
        # double-sentinel early-returned on utility's marker BEFORE
        # checking chat, so a partial state (utility patched, chat not --
        # an exception between the two assigns, or a hand-toggled class)
        # was unfixable until restart; and a cascade left over from an
        # older plugin version kept its sentinel True forever, so a
        # plugin update could never refresh the installed code. Each
        # method now re-installs independently whenever its sentinel is
        # missing OR its version stamp predates this build.
        _stamp = _plugin_version()
        current_utility = Agent.call_utility_model
        current_chat = Agent.call_chat_model

        def _stale(current) -> bool:
            if not getattr(current, "_fallback_patched", False):
                return True  # not ours -- install
            return getattr(current, "_fallback_patched_version", "") != _stamp

        need_utility = _stale(current_utility)
        need_chat = _stale(current_chat)

        # v2.8.6 (N): on FIRST install (the current method is not ours)
        # capture the true original on the class so uninstall() can fully
        # restore it. On a version-bump re-install the current method is
        # our own cascade -- the original stays whatever it was.
        if need_utility and not getattr(current_utility, "_fallback_patched", False):
            try:
                Agent._mfb_original_call_utility_model = current_utility
            except Exception:  # noqa: BLE001
                pass
        if need_chat and not getattr(current_chat, "_fallback_patched", False):
            try:
                Agent._mfb_original_call_chat_model = current_chat
            except Exception:  # noqa: BLE001
                pass

        if need_utility:
            Agent.call_utility_model = _patched_call_utility_model
            _patched_call_utility_model._fallback_patched = True  # type: ignore[attr-defined]
            try:
                _patched_call_utility_model._fallback_patched_version = _stamp  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass
        if need_chat:
            Agent.call_chat_model = _patched_call_chat_model
            _patched_call_chat_model._fallback_patched = True  # type: ignore[attr-defined]
            try:
                _patched_call_chat_model._fallback_patched_version = _stamp  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass

        # Turn path (v2.10+ monologue). install_chat_turn_patch captures the
        # @extensible wrapper before overwriting, so the original's own
        # extension points keep firing. v2.8.6: version-aware -- a stale
        # turn cascade from an older plugin version is re-assigned (the
        # captured original is NOT overwritten on a re-assign).
        turn_patched = False
        if install_chat_turn_patch is not None:
            try:
                turn_patched = install_chat_turn_patch(Agent, version=_stamp)
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

        if self.agent and self.agent.context and (need_utility or need_chat or turn_patched):
            msg = "Model Fallback System patches installed via agent_init."
            if turn_patched:
                msg += " (turn path: call_chat_model_turn cascade active)"
            self.agent.context.log.log("info", msg)
