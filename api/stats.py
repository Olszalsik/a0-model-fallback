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

        # Context size guard counters live in a separate module (the
        # trim hook). Lazy import so a missing module never breaks
        # the stats endpoint.
        context_size: Dict[str, Any] = {}
        try:
            from usr.plugins._model_fallback.extensions.python.message_loop_prompts_after import (
                _10_context_size_guard as _csg,
            )
            context_size = _csg.get_counter().snapshot()
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

        return {
            "version": "2.6.8",
            "utility_timeout": ut,
            "context_size_guard": context_size,
            "langchain_compat": langchain_shim,
        }
