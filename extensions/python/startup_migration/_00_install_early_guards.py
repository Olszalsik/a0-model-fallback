"""Install the early runtime guards at ``startup_migration`` (v3.3.0).

Why this extension point
------------------------
Both guards merged into this plugin used to install from places that never
run at boot:

* the asyncio guard patched asyncio at *import time of its own
  ``hooks.py``* -- but the only importer of a plugin ``hooks.py`` is
  ``helpers.plugins.call_plugin_hook``, which is lazy, and nothing calls
  ``install()`` for enabled plugins during startup;
* the extension import guard installed from ``agent_init``, which is
  itself loaded by the very ``load_classes_from_folder`` it patches -- so
  its own first (cold-cache) enumeration ran unpatched.

``startup_migration`` is dispatched from ``initialize.initialize_migration``
in ``run()`` before the web server is prepared and before any agent exists,
and it resolves plugin extension folders with ``agent=None``. It is
therefore the earliest hook that reliably fires, which fixes both problems:
the asyncio guard is live before the first HTTPS call, and the import guard
now protects the ``agent_init`` enumeration too.

Ordering
--------
Numbered ``_00`` so it sorts first among this plugin's extensions. The
import guard is installed before the asyncio guard because it hardens the
loader that every later extension load (including this plugin's own)
goes through.
"""

from helpers.extension import Extension


def _log(msg: str) -> None:
    try:
        from helpers.print_style import PrintStyle

        PrintStyle().print(f"[model_fallback] {msg}")
    except Exception:  # noqa: BLE001
        pass


class InstallEarlyGuards(Extension):
    def execute(self, **kwargs) -> None:
        # 1) Extension import guard first: it hardens the class loader, so
        #    every subsequent extension enumeration is already protected.
        try:
            from usr.plugins.model_fallback.helpers import import_guard
            from usr.plugins.model_fallback.helpers import toggles

            if toggles.resolve_toggle(
                _safe_cfg(), "extension_import_guard", default=True
            ):
                if import_guard.install():
                    _log("Extension import guard active (broken files skipped).")
            else:
                _log("Extension import guard disabled by config.")
        except Exception as exc:  # noqa: BLE001
            _log(f"Extension import guard not installed: {type(exc).__name__}: {exc}")

        # 2) Asyncio read-ready guard.
        try:
            from usr.plugins.model_fallback.helpers import asyncio_guard
            from usr.plugins.model_fallback.helpers import toggles

            if toggles.resolve_toggle(
                _safe_cfg(), "asyncio_read_ready_guard", default=True
            ):
                asyncio_guard.install()
            else:
                _log("Asyncio read-ready guard disabled by config.")
        except Exception as exc:  # noqa: BLE001
            _log(f"Asyncio read-ready guard not installed: {type(exc).__name__}: {exc}")


def _safe_cfg() -> dict:
    """Merged plugin config, tolerating a missing/broken config.

    ``get_plugin_config`` does not merge ``default_config.yaml`` under
    ``config.json``; ``fallback._get_plugin_cfg`` does. We use the plugin's
    own resolver so a YAML-only toggle is honoured. Returns ``{}`` on any
    failure, which makes ``resolve_toggle`` fall back to the built-in
    default (ON for both guards).
    """
    try:
        from usr.plugins.model_fallback import fallback as fb

        cfg = fb._get_plugin_cfg(None)
        return cfg if isinstance(cfg, dict) else {}
    except Exception:  # noqa: BLE001
        return {}
