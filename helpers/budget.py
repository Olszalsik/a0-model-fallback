"""Per-label budget-pressure windows for candidate preference (v3.2.0).

Problem this solves: the rotation cascade always starts at the persisted
rotation index and walks candidates in order. When the primary (or an
early candidate) is absorbing heavy traffic — a tight per-minute quota
or fresh 429s — the cascade keeps paying that label's rate-limit errors
instead of starting with a roster peer that has headroom. This module
keeps a small per-label sliding window of observed request and 429
timestamps so the cascade can PREFER a low-pressure start. No artificial
delays, no throttling (user decision): track + prefer only.

Design (mirrors helpers/latency.py / helpers/events.py):
  - Process-global dict of windows. NOT agent.data: pressure is a
    property of the model+provider pair, not of a chat, and it is
    inherently short-lived (the window is seconds) — persisting it would
    resurrect stale pressure after a restart.
  - ``record_request`` fires at the cascade pre-call point (every real
    request the cascade is about to make, including the Responses-5xx
    retry); ``record_429`` fires in the cooldown ladder for any 429,
    INCLUDING capacity classes that book no cooldown (routers /
    concurrent_paid) — their 429s are exactly the pressure signal we
    would otherwise lose.
  - Pressure math: with a known per-minute request limit,
    ``req_ratio = window_requests / limit``; without one (fallback
    candidates have no ModelConfig — build_fallback_wrapper passes no
    model_config, so the core limiter never runs for them), 429 density
    alone drives the score: ``min(429_count * weight, 1.0)``. The final
    pressure is the max of the two, clamped to [0, 1].
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Deque, Dict, Optional

_LOCK = threading.Lock()
_WINDOWS: Dict[str, "_Window"] = {}

# Fallback window when the caller does not pass budget_window_s.
_DEFAULT_WINDOW_S = 60.0


class _Window:
    __slots__ = ("requests", "rate_limits")

    def __init__(self) -> None:
        self.requests: Deque[float] = deque()
        self.rate_limits: Deque[float] = deque()

    def trim(self, window_s: float, now: float) -> None:
        cutoff = now - window_s
        while self.requests and self.requests[0] <= cutoff:
            self.requests.popleft()
        while self.rate_limits and self.rate_limits[0] <= cutoff:
            self.rate_limits.popleft()


def _window_for(label: str) -> _Window:
    win = _WINDOWS.get(label)
    if win is None:
        win = _Window()
        _WINDOWS[label] = win
    return win


def record_request(label: str, *, enabled: bool = True, window_s: float = _DEFAULT_WINDOW_S) -> None:
    """Record one request the cascade is about to make. Never raises."""
    try:
        if not enabled:
            return
        now = time.monotonic()
        with _LOCK:
            win = _window_for(str(label))
            win.requests.append(now)
            win.trim(float(window_s) if window_s and float(window_s) > 0 else _DEFAULT_WINDOW_S, now)
    except Exception:  # noqa: BLE001 -- observability must never throw
        pass


def record_429(label: str, *, enabled: bool = True, window_s: float = _DEFAULT_WINDOW_S) -> None:
    """Record one 429 on ``label``. Never raises."""
    try:
        if not enabled:
            return
        now = time.monotonic()
        with _LOCK:
            win = _window_for(str(label))
            win.rate_limits.append(now)
            win.trim(float(window_s) if window_s and float(window_s) > 0 else _DEFAULT_WINDOW_S, now)
    except Exception:  # noqa: BLE001 -- observability must never throw
        pass


def pressure(
    label: str,
    limit_requests: Optional[int] = None,
    *,
    weight_429: float = 0.4,
    window_s: float = _DEFAULT_WINDOW_S,
) -> float:
    """Budget pressure for ``label`` in [0, 1]; 0 means plenty of room.

    ``limit_requests`` comes from the caller's limit lookup (the live
    primary's ModelConfig, or None for fallback candidates — see the
    module docstring). With no limit and no 429s the pressure is 0.
    """
    try:
        now = time.monotonic()
        with _LOCK:
            win = _WINDOWS.get(str(label))
            if win is None:
                return 0.0
            win.trim(float(window_s) if window_s and float(window_s) > 0 else _DEFAULT_WINDOW_S, now)
            n_req = len(win.requests)
            n_429 = len(win.rate_limits)
        req_ratio = 0.0
        try:
            limit = int(limit_requests) if limit_requests is not None else 0
        except Exception:  # noqa: BLE001
            limit = 0
        if limit > 0 and n_req:
            req_ratio = n_req / float(limit)
        p429 = min(n_429 * float(weight_429), 1.0) if weight_429 > 0 else 0.0
        return min(max(req_ratio, p429), 1.0)
    except Exception:  # noqa: BLE001
        return 0.0


def snapshot() -> Dict[str, Any]:
    """JSON-safe per-label window summary for api/stats.py (counts only;
    timestamps are monotonic and not exposed)."""
    out: Dict[str, Any] = {}
    try:
        now = time.monotonic()
        with _LOCK:
            items = [(label, win) for label, win in _WINDOWS.items()]
        for label, win in items:
            win.trim(_DEFAULT_WINDOW_S, now)
            out[label] = {
                "requests_in_window": len(win.requests),
                "rate_limits_in_window": len(win.rate_limits),
            }
    except Exception:  # noqa: BLE001
        return {}
    return out


def reset() -> None:
    """Clear all windows (hooks.uninstall / tests). Mutates in place so
    live references are not orphaned."""
    try:
        with _LOCK:
            _WINDOWS.clear()
    except Exception:  # noqa: BLE001
        pass