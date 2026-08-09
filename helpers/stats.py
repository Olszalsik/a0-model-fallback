"""Process-global counters for the v2.2 resilience layer.

This module is intentionally tiny: a single mutable dict per counter
group, plus accessor helpers. The utility timeout guard bumps
counters here and reads them through the helpers in this file.

Why module-level state instead of an instance on the Agent class:
the utility timeout guard wraps every ``call_utility_model`` call,
and a single set of counters is enough for the WebUI's traffic-light
tile. Module-level dicts are the simplest correct option and are
reset by ``reset()`` from ``uninstall``.

Why not the existing ``_model_fallback.fallback`` stats: those are
agent-scoped (one dict per agent, persisted to ``agent.data``). The
resilience counters are PROCESS-scoped. Two distinct scopes, two
distinct stores.

Thread safety: this is async code on a single event loop. There is
no preemption between ``+=`` and the dict read. The bookkeeping
counters are read by the ``/api/plugins/_model_fallback/stats``
endpoint, which runs on the same loop. No locks needed.

Memory: the dict grows at the rate of model calls. With the defaults
(utility timeout: 0-50 calls/chat) the per-day footprint is well
under 1KB. We never record per-call payloads — only the count and
the aggregate stats the WebUI needs to render its traffic-light tile.

v2.5 housekeeping removal
-------------------------
The ``_HOUSEKEEPING`` counter group and the
``housekeeping_snapshot`` / ``housekeeping_mark_*`` accessors were
removed in this branch. The standalone housekeeping loop is gone;
the counters would have stayed frozen at zero forever. The
``/api/plugins/_model_fallback/stats`` endpoint no longer surfaces
a ``housekeeping`` block.

v2.6.6 extensions-cache removal
-------------------------------
The ``_EXTENSIONS_CACHE`` counter group and its
``extensions_cache_*`` accessors were removed when the server-side
WebUI extensions cache was migrated to the ``ui_loader_optimizer``
plugin (v3.5.0). The cache now owns its own self-contained counters
+ ``snapshot()``. This module retains only the utility-timeout
counters.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Counter groups
# ---------------------------------------------------------------------------

_UTILITY_TIMEOUT: Dict[str, Any] = {
    "calls_total": 0,               # every guarded utility call
    "timeouts_total": 0,            # calls that hit max_wait_s
    "max_observed_wait_s": 0.0,     # largest (await asyncio.wait_for) duration seen
    "last_timeout_at": 0.0,         # time.monotonic() of the most recent timeout
    "last_timeout_model": "",       # model name of the most recent timeout (best-effort)
    "close_inner_attempted": 0,     # how many times we tried close_inner_coro
    "close_inner_succeeded": 0,     # how many times it actually closed a coroutine
}


# ---------------------------------------------------------------------------
# Utility timeout accessors
# ---------------------------------------------------------------------------

def utility_timeout_snapshot() -> Dict[str, Any]:
    return dict(_UTILITY_TIMEOUT)


def utility_timeout_record_call(wait_s: float) -> None:
    _UTILITY_TIMEOUT["calls_total"] += 1
    if wait_s > _UTILITY_TIMEOUT["max_observed_wait_s"]:
        _UTILITY_TIMEOUT["max_observed_wait_s"] = float(wait_s)


def utility_timeout_record_timeout(model_name: str) -> None:
    _UTILITY_TIMEOUT["timeouts_total"] += 1
    _UTILITY_TIMEOUT["last_timeout_at"] = time.monotonic()
    _UTILITY_TIMEOUT["last_timeout_model"] = str(model_name or "")


def utility_timeout_record_close_inner(success: bool) -> None:
    _UTILITY_TIMEOUT["close_inner_attempted"] += 1
    if success:
        _UTILITY_TIMEOUT["close_inner_succeeded"] += 1


# ---------------------------------------------------------------------------
# Reset (called by hooks.uninstall)
# ---------------------------------------------------------------------------

def reset() -> None:
    """Reset all counters. Called by ``uninstall()`` so a disable/re-enable
    cycle starts with a clean slate.
    """
    for k in list(_UTILITY_TIMEOUT.keys()):
        v = _UTILITY_TIMEOUT[k]
        if isinstance(v, bool):
            _UTILITY_TIMEOUT[k] = False
        elif isinstance(v, int):
            _UTILITY_TIMEOUT[k] = 0
        elif isinstance(v, float):
            _UTILITY_TIMEOUT[k] = 0.0
        else:
            _UTILITY_TIMEOUT[k] = ""
    # Re-initialize to declared defaults so the reset is complete even
    # for keys whose type is "falsy numeric".
    _UTILITY_TIMEOUT.update({
        "calls_total": 0, "timeouts_total": 0, "max_observed_wait_s": 0.0,
        "last_timeout_at": 0.0, "last_timeout_model": "",
        "close_inner_attempted": 0, "close_inner_succeeded": 0,
    })
