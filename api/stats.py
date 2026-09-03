"""Read-only observability endpoint for the v2.2 resilience layer.

Route: GET /api/plugins/_model_fallback/stats

Returns a flat JSON dict with the resilience counter group
(utility_timeout) plus the context_size_guard and langchain_compat
readouts. A top-level ``version`` and ``enabled`` flag let the WebUI
tile render a clear on/off indicator.

v2.5 housekeeping removal
-------------------------
The ``housekeeping`` block in the response was removed when the
standalone housekeeping loop was deleted. The block used to
expose ``loop_alive``, ``last_tick_at``, and a derived
``seconds_since_tick``; the loop that produced those counters is
gone, so the block is gone too.

v2.6.6 extensions-cache removal
-------------------------------
The ``extensions_cache`` block was removed when the server-side
WebUI extensions cache was migrated to the ``ui_loader_optimizer``
plugin (v3.5.0), which now exposes its own ``extensions_cache``
snapshot. This endpoint retains only the utility-timeout counters.

Auth: the framework's default requires_auth=True is inherited
from ApiHandler; the WebUI's logged-in session passes that
transparently. The endpoint is read-only; no POST/PUT.
"""

from __future__ import annotations

import time
from typing import Any, Dict

from helpers.api import ApiHandler, Request
from usr.plugins._model_fallback.helpers import stats


class Stats(ApiHandler):
    async def process(self, input: Dict[str, Any], request: Request) -> Dict[str, Any]:
        ut = stats.utility_timeout_snapshot()

        # Context size guard counters. v2.8.5: they live in helpers/stats
        # -- importing the EXTENSION module here used to create a second
        # synthetic module instance with a fresh zero counter, so this
        # endpoint reported phantom zeros forever while the live hook
        # counted in its own instance.
        context_size: Dict[str, Any] = {}
        try:
            context_size = stats.context_guard_snapshot()
        except Exception:  # noqa: BLE001
            context_size = {"trims": 0, "messages_dropped": 0,
                            "last_kept": 0, "last_dropped": 0}

        # LangChain v1 import shim status. The shim is in
        # helpers/langchain_compat.py; lazy import so a missing
        # helper never breaks the stats endpoint.
        langchain_shim: Dict[str, Any] = {"installed": False, "shims": {}}
        try:
            from usr.plugins._model_fallback.helpers import langchain_compat as _lcc
            langchain_shim = _lcc.shim_status()
        except Exception:  # noqa: BLE001
            pass

        # v2.8.4: read the version from plugin.yaml instead of a hardcoded
        # string that silently drifts out of date on every release.
        version: str = "unknown"
        try:
            from helpers import plugins as _plugins

            meta = _plugins.get_plugin_meta("_model_fallback")
            version = str(getattr(meta, "version", "") or "unknown")
        except Exception:  # noqa: BLE001
            pass

        return {
            "version": version,
            "utility_timeout": ut,
            "context_size_guard": context_size,
            "langchain_compat": langchain_shim,
        }
