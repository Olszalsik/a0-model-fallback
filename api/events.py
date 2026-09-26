"""Read-only event-timeline endpoint (v2.9.1).

Route: POST /api/plugins/model_fallback/events

Returns the newest-first ring-buffer of routing events recorded by
helpers/events.py -- cooldown bookings and clears (probe vs live-call
recovery), primary-skip escalations, cross-agent dead-marks,
user-initiated clears, and cascade exhaustions. This is the "story"
view on top of the counters that /stats exposes.

Input (all optional):
  limit   -- max events returned, 1..500 (default 100)
  kind    -- filter to one event kind (see helpers/events.KINDS)
  context -- filter to one agent-context id
  label   -- filter to one model label

Response: {"version": ..., "kinds": [...], "count": N, "events": [...]}
Newest event first. The buffer is in-memory only and resets on
container restart (routing state is transient by design).

Auth/CSRF: inherited defaults from ApiHandler (auth + CSRF), matching
the /stats endpoint in this plugin.
"""

from __future__ import annotations

from typing import Any, Dict

from helpers.api import ApiHandler, Request


class Events(ApiHandler):
    async def process(self, input: Dict[str, Any], request: Request) -> Dict[str, Any]:
        from usr.plugins.model_fallback.helpers import events

        try:
            limit = int(input.get("limit", 100))
        except Exception:  # noqa: BLE001
            limit = 100
        limit = max(1, min(limit, 500))

        kind = input.get("kind") or None
        context = input.get("context") or None
        label = input.get("label") or None

        return {
            "version": _version(),
            "kinds": list(events.KINDS),
            "count": len(events.snapshot(limit=limit, kind=kind, context=context, label=label)),
            "events": events.snapshot(limit=limit, kind=kind, context=context, label=label),
        }


def _version() -> str:
    """plugin.yaml version (same resolution as api/stats.py)."""
    try:
        from helpers import plugins

        meta = plugins.get_plugin_meta("model_fallback")
        return str(getattr(meta, "version", "") or "unknown")
    except Exception:  # noqa: BLE001
        return "unknown"