"""Per-label latency samples for adaptive per-call timeouts (v3.1.0).

Problem this solves: the warm/cold two-tier timeout
(``_resolve_per_call_timeout``) only knows two sizes -- the 20s warm
ceiling for labels that succeeded recently, and the full cold base
(default 300s) for everything else. A label that is NOT warm (last
success outside the warm window) but has historically fast calls keeps
paying the full cold timeout on a hung connection before the cascade
rotates. This module records how long successful calls ACTUALLY took
per label and lets the resolver size the cold timeout from observed
reality: p95 x margin, clamped to ``[floor, base]``.

Design (mirrors helpers/events.py):
  - Process-global dict of bounded sample lists. NOT agent.data: latency
    is a property of the model+provider pair, not of a chat, and there
    is nothing to persist (a restart legitimately starts fresh, same as
    the warm-labels cache).
  - ``record`` NEVER raises and is O(1): append + trim. Samples are
    only recorded on SUCCESS -- a timed-out call is censored data (we
    only know it exceeded the timeout, not how long it would have
    taken), so it is never used to size the budget.
  - ``clear_label`` on a genuine timeout: the sizing that just timed out
    was too tight, so drop the label's samples and fall back to the
    full base timeout until enough fresh samples accumulate again (one
    bad sizing self-corrects).
  - ``p95`` is nearest-rank on the sorted sample list -- a single fast
    outlier cannot shrink the budget below the 95th percentile of
    recent reality.
"""
from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

_LOCK = threading.Lock()
_SAMPLES: Dict[str, List[float]] = {}

# Fallback cap used when the caller does not pass max_samples (tests,
# snapshot callers); the resolver passes the configured knob.
_DEFAULT_MAX_SAMPLES = 20


def record(label: str, elapsed_s: float, *, enabled: bool = True, max_samples: int = _DEFAULT_MAX_SAMPLES) -> None:
    """Record one successful call duration for ``label``. Never raises.

    Gated on ``enabled`` so the kill-switch knob stops state from
    accumulating at all (a true no-op path, matching v3.0.0 behavior).
    """
    try:
        if not enabled:
            return
        elapsed = float(elapsed_s)
        if elapsed < 0:
            return
        with _LOCK:
            samples = _SAMPLES.setdefault(str(label), [])
            samples.append(elapsed)
            # Bound the window (rolling): drop the OLDEST samples first.
            cap = int(max_samples) if max_samples and int(max_samples) > 0 else _DEFAULT_MAX_SAMPLES
            if len(samples) > cap:
                del samples[: len(samples) - cap]
    except Exception:  # noqa: BLE001 -- observability must never throw
        pass


def p95_timeout_s(
    label: str,
    *,
    min_samples: int = 5,
    margin: float = 1.5,
    floor_s: float = 45.0,
    base_timeout_s: float = 300.0,
    enabled: bool = True,
) -> Optional[float]:
    """Adaptive cold-timeout budget for ``label``, or None when the
    caller should keep the plain base timeout.

    Conservative by construction: returns None when disabled, when
    fewer than ``min_samples`` successes are on file, and the result is
    clamped to ``[floor_s, base_timeout_s]`` -- the adaptive path can
    only ever SHRINK the cold base (fail faster on a hung call), never
    grow it beyond what the config already allows, and never below the
    configured floor.
    """
    try:
        if not enabled:
            return None
        need = int(min_samples)
        if need <= 0:
            need = 1
        with _LOCK:
            samples = list(_SAMPLES.get(str(label), ()))
        if len(samples) < need:
            return None
        samples.sort()
        # Nearest-rank p95.
        rank = max(1, -(-len(samples) * 95 // 100))
        p95 = samples[min(rank, len(samples)) - 1]
        est = p95 * float(margin)
        floor = float(floor_s)
        if floor > 0:
            est = max(est, floor)
        base = float(base_timeout_s)
        if base > 0:
            est = min(est, base)
        return est
    except Exception:  # noqa: BLE001 -- sizing must never break the resolver
        return None


def clear_label(label: str) -> None:
    """Drop ``label``'s samples (called on a genuine timeout so the next
    resolution falls back to the full base timeout). Never raises."""
    try:
        with _LOCK:
            _SAMPLES.pop(str(label), None)
    except Exception:  # noqa: BLE001
        pass


def sample_count(label: str) -> int:
    """How many samples are on file for ``label`` (test/debug aid)."""
    try:
        with _LOCK:
            return len(_SAMPLES.get(str(label), ()))
    except Exception:  # noqa: BLE001
        return 0


def snapshot() -> Dict[str, Any]:
    """JSON-safe per-label summary for api/stats.py."""
    out: Dict[str, Any] = {}
    try:
        with _LOCK:
            items = [(label, list(samples)) for label, samples in _SAMPLES.items()]
        for label, samples in items:
            if not samples:
                continue
            ordered = sorted(samples)
            n = len(ordered)
            rank = max(1, -(-n * 95 // 100))
            out[label] = {
                "samples": n,
                "last_s": round(ordered[-1], 3),
                "p50_s": round(ordered[max(0, -(-n * 50 // 100)) - 1], 3),
                "p95_s": round(ordered[rank - 1], 3),
            }
    except Exception:  # noqa: BLE001
        return {}
    return out


def reset() -> None:
    """Clear all samples (hooks.uninstall / tests). Mutates in place
    where possible so live references are not orphaned."""
    try:
        with _LOCK:
            _SAMPLES.clear()
    except Exception:  # noqa: BLE001
        pass