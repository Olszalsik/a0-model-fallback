"""Process-global counters for the v2.2 resilience layer.

This module is intentionally tiny: a single mutable dict per counter
group, plus accessor helpers. Every other piece of the v2.2 work
(extensions cache, utility timeout guard) bumps counters here and
reads them through the helpers in this file.

Why module-level state instead of an instance on the Agent class:
the utility timeout guard wraps every ``call_utility_model`` call,
the extensions cache serves every WebUI request, and a single set
of counters is enough for the WebUI's traffic-light tile. Module-
level dicts are the simplest correct option and are reset by
``reset()`` from ``uninstall``.

Why not the existing ``_model_fallback.fallback`` stats: those are
agent-scoped (one dict per agent, persisted to ``agent.data``). The
resilience counters are PROCESS-scoped. Two distinct scopes, two
distinct stores.

Thread safety: this is async code on a single event loop. There is
no preemption between ``+=`` and the dict read. The bookkeeping
counters are read by the ``/api/plugins/_model_fallback/stats``
endpoint, which runs on the same loop. No locks needed.

Memory: the dicts grow at the rate of model calls + cache events.
With the defaults (utility timeout: 0-50 calls/chat, cache: tens
of hits/sec at peak) the per-day footprint is well under 1KB. We
never record per-call payloads — only the count and the aggregate
stats the WebUI needs to render its traffic-light tile.

v2.5 housekeeping removal
-------------------------
The ``_HOUSEKEEPING`` counter group and the
``housekeeping_snapshot`` / ``housekeeping_mark_*`` accessors were
removed in this branch. The standalone housekeeping loop is gone;
the counters would have stayed frozen at zero forever. The
``/api/plugins/_model_fallback/stats`` endpoint no longer surfaces
a ``housekeeping`` block.
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

_EXTENSIONS_CACHE: Dict[str, Any] = {
    "hits": 0,
    "misses": 0,
    "errors": 0,                    # get_webui_extensions raised
    "circuit_opened_at": 0.0,       # time.monotonic() when circuit last opened
    "circuit_open_count": 0,        # total times the circuit has tripped
    "circuit_short_circuits": 0,    # requests that hit the open circuit
    "last_error": "",               # last exception message (truncated)
    "busts": 0,                     # times the cache was invalidated externally
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
# Extensions cache accessors
# ---------------------------------------------------------------------------

def extensions_cache_snapshot() -> Dict[str, Any]:
    snap = dict(_EXTENSIONS_CACHE)
    # Surface a derived flag: is the circuit currently OPEN? (not yet recovered)
    opened = snap.get("circuit_opened_at") or 0.0
    snap["circuit_open"] = bool(opened)
    return snap


def extensions_cache_record_hit() -> None:
    _EXTENSIONS_CACHE["hits"] += 1


def extensions_cache_record_miss() -> None:
    _EXTENSIONS_CACHE["misses"] += 1


def extensions_cache_record_error(message: str) -> None:
    _EXTENSIONS_CACHE["errors"] += 1
    # Truncate so a verbose traceback doesn't bloat process memory.
    _EXTENSIONS_CACHE["last_error"] = (message or "")[:200]


def extensions_cache_record_bust() -> None:
    _EXTENSIONS_CACHE["busts"] += 1


def extensions_cache_record_circuit_open() -> None:
    now = time.monotonic()
    _EXTENSIONS_CACHE["circuit_opened_at"] = now
    _EXTENSIONS_CACHE["circuit_open_count"] += 1


def extensions_cache_record_short_circuit() -> None:
    _EXTENSIONS_CACHE["circuit_short_circuits"] += 1


# ---------------------------------------------------------------------------
# Reset (called by hooks.uninstall)
# ---------------------------------------------------------------------------

def reset() -> None:
    """Reset all counters. Called by ``uninstall()`` so a disable/re-enable
    cycle starts with a clean slate.
    """
    for d in (_UTILITY_TIMEOUT, _EXTENSIONS_CACHE):
        for k in list(d.keys()):
            if isinstance(d[k], (int, float)):
                d[k] = 0 if isinstance(d[k], int) else 0.0
            elif isinstance(d[k], bool):
                d[k] = False
            else:
                d[k] = "" if k.endswith("_at") or k == "last_error" else ""
    # Re-initialize the dicts to their declared defaults so the reset is
    # complete even for keys whose type is "falsy numeric".
    _UTILITY_TIMEOUT.update({
        "calls_total": 0, "timeouts_total": 0, "max_observed_wait_s": 0.0,
        "last_timeout_at": 0.0, "last_timeout_model": "",
        "close_inner_attempted": 0, "close_inner_succeeded": 0,
    })
    _EXTENSIONS_CACHE.update({
        "hits": 0, "misses": 0, "errors": 0, "circuit_opened_at": 0.0,
        "circuit_open_count": 0, "circuit_short_circuits": 0,
        "last_error": "", "busts": 0,
    })
