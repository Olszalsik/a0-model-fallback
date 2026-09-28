"""In-memory ring-buffer event log for the fallback plugin (v2.9.1).

Problem this solves: the plugin's state is scattered across counters
(api/stats.py), log lines (agent context logs, lost with the context),
and in-memory dicts (cooldowns, dead labels, warm labels). Answering
"what actually happened to model routing in the last hour" meant
reconstructing it from three sources. This module keeps ONE bounded,
queryable sequence of routing events so the stats endpoint can show
the story, not just the tally.

Design:
  - Pure in-memory ring buffer (collections.deque, maxlen 500). No
    persistence by design: events describe transient routing decisions,
    the durable counters already exist, and a restart legitimately
    resets cooldown state anyway.
  - ``record_event`` NEVER raises and never blocks the cascade: every
    failure is swallowed, and the lock is a threading.Lock held only
    for a list append.
  - Entries are JSON-safe by construction (``_sanitize`` coerces
    anything unexpected to a truncated string), so the API endpoint
    can serialize the buffer directly.
  - Bounded memory: 500 entries x ~200 bytes is ~100KB worst case.

Event kinds (see KINDS):
  - cooldown_booked              a candidate entered/extended cooldown
  - cooldown_cleared_early       recovery probe succeeded (v2.9.0)
  - cooldown_cleared_by_success  a live cascade SUCCEEDED on a label
                                 that was in cooldown (real recovery)
  - primary_skip_escalated       primary-skip escalation wrote a long
                                 cooldown for candidate 0
  - label_dead                   cross-agent dead-mark written
  - cooldowns_cleared            user-initiated clear (button/API)
  - cascade_exhausted            all candidates failed, RetryAfterHours
                                 is about to be raised
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional

_MAX_EVENTS = 500
_LOCK = threading.Lock()
_EVENTS: Deque[Dict[str, Any]] = deque(maxlen=_MAX_EVENTS)

KINDS = (
    "cooldown_booked",
    "cooldown_cleared_early",
    "cooldown_cleared_by_success",
    "primary_skip_escalated",
    "label_dead",
    "cooldowns_cleared",
    "cascade_exhausted",
    # v3.4.1: recorded by _grow_gen_budget since v3.2.0 but missing from
    # this registry -- a kind-filtered query for it returned nothing.
    "gen_budget_grown",
)


def _sanitize(value: Any) -> Any:
    """Coerce an event field to something JSON-serializable."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, float):  # unreachable, kept for clarity
        return value
    return str(value)[:200]


def record_event(
    kind: str,
    *,
    agent: Any = None,
    label: str = "",
    **fields: Any,
) -> None:
    """Append one routing event to the ring buffer. Never raises.

    ``agent`` (optional) is used only to stamp the context id so events
    from different chats/sub-agents can be told apart and filtered.
    """
    try:
        ctx = getattr(agent, "context", None)
        ctx_id = getattr(ctx, "id", None)
        entry: Dict[str, Any] = {
            "ts": round(time.time(), 3),
            "kind": str(kind),
            "label": str(label or ""),
            "context": str(ctx_id) if ctx_id is not None else "__global__",
        }
        for key, value in fields.items():
            entry[key] = _sanitize(value)
        with _LOCK:
            _EVENTS.append(entry)
    except Exception:  # noqa: BLE001 -- observability must never throw
        pass


def snapshot(
    limit: int = 100,
    *,
    kind: Optional[str] = None,
    context: Optional[str] = None,
    label: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Newest-first view of the buffer with optional filters."""
    with _LOCK:
        items = list(_EVENTS)
    out: List[Dict[str, Any]] = []
    for entry in reversed(items):
        if kind is not None and entry.get("kind") != kind:
            continue
        if context is not None and entry.get("context") != context:
            continue
        if label is not None and entry.get("label") != label:
            continue
        out.append(entry)
        if len(out) >= limit:
            break
    return out


def reset_events() -> None:
    """Clear the buffer (hooks.reset_fallback_settings / tests)."""
    with _LOCK:
        _EVENTS.clear()