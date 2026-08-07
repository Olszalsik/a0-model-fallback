"""Install the ``get_webui_extensions`` cache + circuit breaker.

Fired at ``run_ui/init_a0/start`` so the patch is in place before
the HTTP server starts accepting requests. Idempotent: the
underlying ``webui_extensions_cache.install`` checks for the
sentinel attribute and returns early if the wrapper is already in
place.

If the framework ever re-runs init_a0 (e.g. test harness, plugin
reload), the cache is refreshed in place; the original
``get_webui_extensions`` reference is preserved so the wrapper
stack stays at exactly one layer.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from helpers.extension import Extension

_log = logging.getLogger("model_fallback.extensions_cache.install")


def _resolve_config() -> Dict[str, Any]:
    try:
        from helpers import plugins as plugin_helpers  # type: ignore
        cfg = plugin_helpers.get_plugin_config("_model_fallback", None) or {}
    except Exception:  # noqa: BLE001
        cfg = {}
    # v2.5 WebUI: top-level ``webui_extensions_cache_enabled``
    # wins over the nested section's ``enabled`` so a user
    # toggling OFF in the WebUI takes effect on the next
    # init_a0.
    from usr.plugins._model_fallback.helpers import toggles
    if not toggles.resolve_toggle(cfg, "webui_extensions_cache", default=True):
        return {"enabled": False}
    overrides = cfg.get("webui_extensions_cache") if isinstance(cfg, dict) else None
    if not isinstance(overrides, dict):
        overrides = {}
    from usr.plugins._model_fallback.helpers import webui_extensions_cache
    return webui_extensions_cache.resolve_config(overrides)


def _config_provider_factory():
    """Return a callable that yields the live config each time it's
    called. The wrapper invokes this on every request so a config
    change (e.g. user toggles the cache) takes effect without a
    restart.
    """
    return _resolve_config


class InstallExtensionsCache(Extension):
    def execute(self, **kwargs: Any) -> None:
        try:
            from usr.plugins._model_fallback.helpers import webui_extensions_cache
            installed = webui_extensions_cache.install(
                config_provider=_config_provider_factory()
            )
            if installed:
                _log.info("extensions cache installed at init_a0/start")
        except Exception as exc:  # noqa: BLE001
            _log.debug("extensions cache install failed: %s", exc)
