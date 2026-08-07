"""Install the langchain v0 -> v1 import compatibility shim (v2.4).

This extension lives in the existing ``_model_fallback`` plugin
(merged from the standalone ``_langchain_compat`` plugin in v2.4).
The user preferred to keep all LLM-error-handling fixes in one
plugin so a single toggle disables everything.

The shim is installed at ``agent_init`` time, NOT at plugin install
time, because we need it active BEFORE any v0-style langchain
import runs in the agent loop. ``agent_init`` is the right hook
for that: it fires after the framework has loaded the plugin
manager but before the first agent monologue, so any subsequent
``from langchain.prompts import ...`` (whether in core code or in
a user plugin) sees the shim.

The install path is in ``helpers/langchain_compat.py`` and is
shared with the stats endpoint. The extension wrapper here is a
thin try/except that never crashes agent init: a bug in the shim
or a missing langchain_core package logs a debug line and
continues.
"""

from __future__ import annotations

import logging
from typing import Any

from helpers.extension import Extension

_log = logging.getLogger("model_fallback.langchain_compat.install")


def _resolve_config(agent) -> dict:
    try:
        from helpers import plugins as plugin_helpers  # type: ignore
        cfg = plugin_helpers.get_plugin_config("_model_fallback", agent) or {}
    except Exception:  # noqa: BLE001
        cfg = {}
    # v2.5 WebUI: top-level ``langchain_compat_enabled`` wins
    # over the nested section's ``enabled``. The piece defaults
    # to ON (the shim is a no-op if v0 modules are already
    # present), so explicit False disables.
    from usr.plugins._model_fallback.helpers import toggles
    if not toggles.resolve_toggle(cfg, "langchain_compat", default=True):
        return {"enabled": False}
    return {"enabled": True}


class InstallLangchainCompatShim(Extension):
    def execute(self, **kwargs: Any) -> None:
        try:
            cfg = _resolve_config(self.agent)
            if not cfg.get("enabled", True):
                _log.debug("langchain compat shim disabled by config; skipping")
                return
            from usr.plugins._model_fallback.helpers import langchain_compat
            if langchain_compat.already_installed_in_process():
                return
            results = langchain_compat.install_shim()
            langchain_compat.mark_installed()
            installed = [k for k, v in results.items() if v]
            if installed:
                _log.info(
                    "langchain compat shim installed for: %s",
                    ", ".join(installed),
                )
            else:
                _log.debug(
                    "langchain compat shim found no targets to install "
                    "(v1 source modules missing?)"
                )
            if self.agent and getattr(self.agent, "context", None):
                try:
                    self.agent.context.log.log(
                        "info",
                        f"LangChain compat shim installed: "
                        f"{', '.join(installed) or 'none'}",
                    )
                except Exception:
                    pass
        except Exception as exc:  # noqa: BLE001
            _log.debug("langchain compat shim install failed: %s", exc)
