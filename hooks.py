"""Public API and lifecycle hooks for the Model Fallback System plugin."""


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
        from usr.plugins._model_fallback.helpers import langchain_compat

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
        langchain_compat.mark_installed()
        installed = [k for k, v in results.items() if v]
        if installed:
            import logging
            logging.getLogger("model_fallback.langchain_compat.install").info(
                "langchain compat shim installed at process start: %s",
                ", ".join(installed),
            )
    except Exception:  # noqa: BLE001
        # Never block plugin install on shim failure. The agent_init
        # extension will retry on the first agent creation.
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


async def uninstall():
    """Plugin disable / process-shutdown hook.

    v2.4 update: the resilience layer (extensions cache, utility
    timeout monkey-patch, langchain import shim) holds process-global
    resources that must be released on disable so a re-enable starts
    fresh. Per-agent cooldown / retry state still survives the
    disable/re-enable cycle by design (see the v0.x.x comment).

    v2.5 update: the standalone housekeeping loop and its helper
    module were removed (it was a never-tested same-day fix that
    manifested as a per-minute ``housekeeping loop not alive``
    respawn log line). Nothing to clean up on that front.
    """
    # Best-effort cleanup. Each piece catches its own errors so a
    # bug in one does not block the others. Synchronous helpers
    # because uninstall is fired from a sync context in v2.5.
    try:
        from usr.plugins._model_fallback.extensions.python.agent_init import (
            _10_install_utility_timeout_patch,
        )
        _10_install_utility_timeout_patch.uninstall()
    except Exception:  # noqa: BLE001
        pass
    try:
        from usr.plugins._model_fallback.helpers import (
            webui_extensions_cache,
        )
        webui_extensions_cache.uninstall()
    except Exception:  # noqa: BLE001
        pass
    try:
        from usr.plugins._model_fallback.helpers import langchain_compat
        langchain_compat.uninstall_shim()
        langchain_compat.clear_installed_marker()
    except Exception:  # noqa: BLE001
        pass
    try:
        from usr.plugins._model_fallback.helpers import stats
        stats.reset()
    except Exception:  # noqa: BLE001
        pass
    return None


def get_fallback_settings(agent) -> dict:
    """Get current fallback settings for an agent."""
    return {
        "max_cycles": int(agent.get_data("fallback_max_cycles") or 4),
        "cycle_delay": float(agent.get_data("fallback_cycle_delay") or 5.0),
        "extended_retry_enabled": bool(agent.get_data("extended_retry_enabled") or True),
        "phase_a_delay_s": float(agent.get_data("phase_a_delay_s") or 9000.0),
        "phase_b_delay_s": float(agent.get_data("phase_b_delay_s") or 43200.0),
        "initial_cycle_attempts": int(agent.get_data("initial_cycle_attempts") or 60),
    }


def reset_fallback_settings(agent):
    """Reset fallback settings to defaults."""
    agent.set_data("fallback_max_cycles", 4)
    agent.set_data("fallback_cycle_delay", 5.0)
    agent.set_data("extended_retry_enabled", True)
    agent.set_data("phase_a_delay_s", 9000.0)
    agent.set_data("phase_b_delay_s", 43200.0)
    agent.set_data("initial_cycle_attempts", 60)
    agent.set_data("_model_cooldowns", {})
    agent.set_data("ext_retry_phase", None)
    agent.set_data("ext_retry_substep", None)
    agent.set_data("ext_retry_at", None)
    return get_fallback_settings(agent)


def clear_cooldowns(agent):
    """Clear all model cooldowns (useful for testing or after adding credits)."""
    agent.set_data("_model_cooldowns", {})


def get_cooldowns(agent) -> dict:
    """Get current model cooldowns for inspection."""
    return agent.get_data("_model_cooldowns") or {}
