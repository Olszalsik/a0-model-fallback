"""Public API and lifecycle hooks for the Model Fallback System plugin."""

# v2.8.3: real data-key names (the old literal strings
# "ext_retry_phase"/"ext_retry_at" didn't match what the cascade
# reads/writes, so reset_fallback_settings never cleared them).
# v2.8.5: the dead "mfb_ext_retry_at" constant (nothing in fallback.py
# writes it; the live key is mfb_ext_retry_attempts) was removed and
# reset_fallback_settings now clears the real keys via the fallback module
# constants.


async def install():
    """Plugin installation hook.

    v2.5 update: install the langchain v0 -> v1 import shim here at
    process start, not just at agent_init. The agent_init extension
    still runs as a defensive backup for reloads; this primary install
    ensures the shim is active BEFORE the first utility-model call
    fires, which is the only path that triggers the
    ``ModuleNotFoundError: No module named 'langchain.prompts'``
    before any agent exists. Idempotent; respects the
    ``langchain_compat_enabled`` toggle (default ON).
    """
    try:
        # Lazy import so a broken shim never blocks plugin load.
        from usr.plugins.model_fallback.helpers import langchain_compat

        if langchain_compat.already_installed_in_process():
            return

        # Resolve the langchain_compat toggle at process start. We
        # read default_config.yaml + config.json directly because
        # plugin_helpers.get_plugin_config requires an agent, and
        # install() runs before any agent exists. Resolution order
        # mirrors helpers.toggles.resolve_toggle.
        enabled = _read_langchain_compat_toggle()
        if not enabled:
            return

        results = langchain_compat.install_shim()
        installed = [k for k, v in results.items() if v]
        # v2.8.5 (wiring#12): only record the install marker when at least
        # one shim actually installed. Marking unconditionally made
        # already_installed_in_process() return True after a total failure
        # (e.g. langchain_core absent), so the agent_init extension's
        # defensive retry was skipped for the life of the process.
        if installed:
            langchain_compat.mark_installed()
            import logging
            logging.getLogger("model_fallback.langchain_compat.install").info(
                "langchain compat shim installed at process start: %s",
                ", ".join(installed),
            )
    except Exception:  # noqa: BLE001
        # Never block plugin install on shim failure. The agent_init
        # extension will retry on the first agent creation.
        pass

    # v3.3.0: defensive re-install of the two merged guards.
    #
    # These normally install from
    # extensions/python/startup_migration/_00_install_early_guards.py at
    # boot. This block is a safety net for the paths that never reach
    # startup_migration (embedded/test harnesses, an extension reload).
    #
    # It is NOT the primary install site: ``install()`` itself only runs
    # when a plugin is installed or updated through the installer, never
    # on an ordinary restart, which is exactly why the standalone
    # ``_asyncio_guard`` plugin was dead code.
    try:
        from usr.plugins.model_fallback.helpers import asyncio_guard
        from usr.plugins.model_fallback.helpers import import_guard
        from usr.plugins.model_fallback.helpers import toggles as _toggles

        cfg = {}
        try:
            from usr.plugins.model_fallback import fallback as _fb

            loaded = _fb._get_plugin_cfg(None)
            cfg = loaded if isinstance(loaded, dict) else {}
        except Exception:  # noqa: BLE001
            cfg = {}
        if _toggles.resolve_toggle(cfg, "extension_import_guard", default=True):
            import_guard.install()
        if _toggles.resolve_toggle(cfg, "asyncio_read_ready_guard", default=True):
            asyncio_guard.install()
    except Exception:  # noqa: BLE001
        pass


def _read_langchain_compat_toggle() -> bool:
    """Best-effort read of the langchain_compat toggle at install().

    Mirrors helpers.toggles.resolve_toggle resolution order:
    1. ``langchain_compat_enabled`` in config.json (top-level).
    2. ``langchain_compat_enabled`` in default_config.yaml.
    3. Built-in default: ON (shim is a no-op on real v0 langchain).

    Returns True on any read error so the shim still installs. The
    shim itself is a safe no-op if langchain_core is not present
    (e.g. test envs without any langchain build), so defaulting ON
    cannot break a working environment.
    """
    try:
        import json
        from pathlib import Path

        here = Path(__file__).resolve().parent
        # 1. User override (config.json) wins.
        user_cfg_path = here / "config.json"
        if user_cfg_path.is_file():
            try:
                with user_cfg_path.open("r", encoding="utf-8") as f:
                    user_cfg = json.load(f)
                if "langchain_compat_enabled" in user_cfg:
                    return bool(user_cfg["langchain_compat_enabled"])
            except Exception:  # noqa: BLE001
                pass
        # 2. Default config.
        default_cfg_path = here / "default_config.yaml"
        if default_cfg_path.is_file():
            try:
                with default_cfg_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        stripped = line.strip()
                        if not stripped or stripped.startswith("#"):
                            continue
                        if stripped.startswith("langchain_compat_enabled:"):
                            value = stripped.split(":", 1)[1].strip().split("#")[0].strip()
                            if value.lower() in ("false", "0", "no", "off"):
                                return False
                            if value.lower() in ("true", "1", "yes", "on"):
                                return True
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        pass
    # 3. Default ON.
    return True


async def pre_update():
    """Pre-update hook called before plugin code is updated."""
    pass


def uninstall():
    """Plugin disable / process-shutdown hook.

    v2.8.5: made SYNC. The framework calls sync hooks directly but runs
    coroutine hooks via ``asyncio.run`` (helpers/plugins.py:917-920) --
    and the only live trigger for uninstall is the plugin-delete API
    endpoint, which executes INSIDE a running event loop. ``asyncio.run``
    there raised RuntimeError, the API turned it into a 500, and every
    piece of uninstall cleanup below was silently skipped. The body was
    always fully synchronous; the async declaration was the only problem.
    (The async ``install`` hook is safe: it only fires from the sync
    plugin-installer path.)

    v2.4 update: the resilience layer (utility timeout monkey-patch,
    langchain import shim) holds process-global resources that must
    be released on disable so a re-enable starts fresh. Per-agent
    cooldown / retry state still survives the disable/re-enable cycle
    by design (see the v0.x.x comment).

    v2.6.6 update: the server-side WebUI extensions cache was migrated
    to the ``ui_loader_optimizer`` plugin (v3.5.0); its uninstall is
    no longer handled here.

    v2.5 update: the standalone housekeeping loop and its helper
    module were removed (it was a never-tested same-day fix that
    manifested as a per-minute ``housekeeping loop not alive``
    respawn log line). Nothing to clean up on that front.
    """
    # Best-effort cleanup. Each piece catches its own errors so a
    # bug in one does not block the others. Synchronous helpers
    # because uninstall is fired from a sync context in v2.5.
    # v2.8.6 (N): the full patch-restore chain lives in
    # _00_install_fallback_patches.uninstall() -- it unwraps the utility
    # timeout wrapper FIRST (it wraps the cascade installed by _00), then
    # restores the true originals for call_utility_model / call_chat_model
    # / call_chat_model_turn. Previously only the top wrapper was undone,
    # so a "disable" left the plugin's own cascades on the Agent class.
    # Falls back to the old _10-only restore if _00 predates v2.8.6.
    restored_chain = False
    try:
        from usr.plugins.model_fallback.extensions.python.agent_init import (
            _00_install_fallback_patches,
        )
        restored_chain = bool(_00_install_fallback_patches.uninstall())
    except Exception:  # noqa: BLE001
        pass
    if not restored_chain:
        try:
            from usr.plugins.model_fallback.extensions.python.agent_init import (
                _10_install_utility_timeout_patch,
            )
            _10_install_utility_timeout_patch.uninstall()
        except Exception:  # noqa: BLE001
            pass
    # v2.9.0: the background recovery-probe loop holds a process-global
    # asyncio task + a weakref registry -- cancel and drop on disable so a
    # re-enable starts fresh and no probe fires while the plugin is off.
    try:
        from usr.plugins.model_fallback.helpers import recovery_probe
        recovery_probe.shutdown_probes()
    except Exception:  # noqa: BLE001
        pass
    try:
        from usr.plugins.model_fallback.helpers import langchain_compat
        langchain_compat.uninstall_shim()
        langchain_compat.clear_installed_marker()
    except Exception:  # noqa: BLE001
        pass
    try:
        from usr.plugins.model_fallback.helpers import stats
        stats.reset()
    except Exception:  # noqa: BLE001
        pass
    # v3.4.1: the sticky Responses-5xx set is process-global runtime state
    # like the latency samples above -- a disable must not leave stale
    # force-chat-completions marks behind for endpoints whose /v1/responses
    # recovered (or whose config changed) between disable and re-enable.
    try:
        from usr.plugins.model_fallback import models_ext as _mx_reset
        _mx_reset.reset_responses_5xx_seen()
    except Exception:  # noqa: BLE001
        pass
    # v3.1.0: per-label latency samples are process-global runtime state
    # -- drop them on disable so a re-enable starts fresh.
    try:
        from usr.plugins.model_fallback.helpers import latency
        latency.reset()
    except Exception:  # noqa: BLE001
        pass
    # v3.2.0: learned generation budgets are process-global runtime state
    # too -- a plugin disable must not leave stale ceilings behind.
    try:
        from usr.plugins.model_fallback import fallback as _fb_reset
        _fb_reset.reset_gen_budgets()
        # v3.3.0: belt-and-braces -- reset_gen_budgets() already clears
        # the memoised config, but a disable must not leave a warm cache
        # behind regardless of what else ran.
        _fb_reset.invalidate_plugin_cfg_cache()
    except Exception:  # noqa: BLE001
        pass
    # v3.3.0: restore the upstream extension loader so a disable actually
    # reverts the per-file import guard. The asyncio guard deliberately
    # stays installed (see helpers/asyncio_guard.uninstall): removing a
    # safety monkeypatch from a live process is riskier than leaving it.
    try:
        from usr.plugins.model_fallback.helpers import import_guard

        import_guard.uninstall()
    except Exception:  # noqa: BLE001
        pass
    return None


def get_fallback_settings(agent) -> dict:
    """Get the fallback settings an agent's cascades actually use.

    v2.8.5: read them from the merged plugin config
    (fallback._get_plugin_cfg = default_config.yaml + config.json). The
    old version read agent-data keys the runtime never writes -- the
    cascades resolve every knob from plugin config (model kwargs first),
    so "get" displayed hardcoded defaults (delay 5.0, attempts 60) that
    contradicted both config.json and the YAML defaults.
    """
    try:
        from usr.plugins.model_fallback import fallback as _fb

        # v3.3.0: use_cache=False -- this renders settings back to the user
        # in the WebUI panel, so it must never show a value that is up to
        # one cache TTL behind a config they just saved.
        cfg = _fb._get_plugin_cfg(agent, use_cache=False)
    except Exception:  # noqa: BLE001
        cfg = {}

    def _as_int(key: str, default: int) -> int:
        # v3.4.1: coercions degrade instead of raising -- a string-typed
        # value in config.json (hand-edited) used to raise ValueError out
        # of the settings API. Display defaults mirror default_config.yaml
        # (initial_cycle_attempts: 20 -- the old hardcoded 60 was masked by
        # config.json and contradicted the shipped default).
        try:
            return int(cfg.get(key, default))
        except (TypeError, ValueError):
            return default

    def _as_float(key: str, default: float) -> float:
        try:
            return float(cfg.get(key, default))
        except (TypeError, ValueError):
            return default

    def _as_bool(key: str, default: bool) -> bool:
        val = cfg.get(key, default)
        if isinstance(val, bool):
            return val
        try:
            return bool(int(val))
        except (TypeError, ValueError):
            return default

    return {
        "max_cycles": _as_int("fallback_max_cycles", 4),
        "cycle_delay": _as_float("fallback_cycle_delay", 5.0),
        "extended_retry_enabled": _as_bool("extended_retry_enabled", True),
        "phase_a_delay_s": _as_float("phase_a_delay_s", 900.0),
        "phase_b_delay_s": _as_float("phase_b_delay_s", 3600.0),
        "initial_cycle_attempts": _as_int("initial_cycle_attempts", 20),
    }


def reset_fallback_settings(agent):
    """Reset the runtime fallback state to defaults.

    v2.8.5: the cascade knobs live in plugin config (edited in the WebUI),
    not agent data -- the old version wrote dead data keys AND cleared a
    nonexistent "mfb_ext_retry_at" while leaving the live
    "mfb_ext_retry_attempts" stale, so "reset" never restarted the
    extended-retry burst budget (the next exhaustion immediately promoted
    phase A -> B into the 3600s wait). Now: clear cooldowns + the real
    extended-retry phase state. Config knobs remain user-managed.
    """
    clear_cooldowns(agent)
    try:
        from usr.plugins.model_fallback import fallback as _fb

        agent.set_data(_fb.DATA_KEY_EXT_RETRY_PHASE, 0)
        agent.set_data(_fb.DATA_KEY_EXT_RETRY_ATTEMPTS, 0)
        agent.set_data(_fb.DATA_KEY_EXT_RETRY_NOTIFIED, False)
        agent.set_data(_fb.DATA_KEY_TURN_PRIMARY_FAILS, 0)
        # v2.8.6 (Z): the strike-decay timestamp must reset with the
        # counter, or a stale stamp keeps the decay check armed.
        agent.set_data(_fb.DATA_KEY_TURN_PRIMARY_FAILS_AT, 0)
    except Exception:  # noqa: BLE001
        pass
    # v2.9.1: the event timeline is runtime state -- reset goes with it.
    try:
        from usr.plugins.model_fallback.helpers import events
        events.reset_events()
    except Exception:  # noqa: BLE001
        pass
    return get_fallback_settings(agent)


def clear_cooldowns(agent, cross_context: bool = False):
    """Clear all model cooldowns (useful for testing or after adding credits).

    v2.8.3: the authoritative store is the in-memory _INMEM_COOLDOWNS
    dict (seeded from agent.data only on first use after a restart) --
    the old version wiped the legacy ``_model_cooldowns`` data key and
    left the live store untouched, making this a silent no-op.

    v2.8.6 (wiring#11): ``cross_context=True`` clears every context's
    in-memory store (plus the process-global dead/warm indexes), not just
    this agent's -- sub-agents and other chats sharing a broken
    credential were unreachable from a per-context clear.
    """
    try:
        from usr.plugins.model_fallback import fallback as _fb
        cleared = _fb.clear_all_cooldowns(agent, cross_context=bool(cross_context))
    except Exception:  # noqa: BLE001
        cleared = 0
    # v2.8.5: also wipe the persisted key by its current name. The old
    # literal "_model_cooldowns" was the pre-v2.8.5 (underscore-prefixed)
    # name -- persist_chat strips underscore keys, so nothing ever
    # persisted under it anyway.
    #
    # v3.3.0 bug fix: this block referenced ``_fb``, which is only bound
    # INSIDE the first ``try``. If that import failed, the second block
    # raised NameError and its own ``except Exception`` swallowed it --
    # so the persisted key was silently left behind. Re-import locally so
    # the two blocks are independent.
    try:
        from usr.plugins.model_fallback import fallback as _fb_keys

        agent.set_data(_fb_keys.DATA_KEY_COOLDOWNS, {})
    except Exception:  # noqa: BLE001
        pass
    return cleared


def get_cooldowns(agent) -> dict:
    """Get current model cooldowns for inspection.

    v2.8.3: reads the live in-memory store (falling back to the
    persisted snapshot) instead of the legacy data key nothing writes.
    """
    try:
        from usr.plugins.model_fallback import fallback as _fb
        live = _fb.snapshot_cooldowns(agent)
        if live:
            return live
    except Exception:  # noqa: BLE001
        pass
    # v2.8.5: fall back to the persisted key under its CURRENT name (the
    # legacy "_model_cooldowns" literal predated the rename).
    try:
        from usr.plugins.model_fallback import fallback as _fb2
        return agent.get_data(_fb2.DATA_KEY_COOLDOWNS) or {}
    except Exception:  # noqa: BLE001
        return {}
