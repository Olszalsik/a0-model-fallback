"""TTL cache + circuit breaker for ``helpers.extension.get_webui_extensions``.

Background
----------
The WebUI calls ``/api/load_webui_extensions`` whenever Alpine
re-renders an extension point. During page lifecycle (init, settings
open, plugin toggle, project switch) this can fire 10+ times per
second, and each call walks every plugin's ``extensions/webui/``
folder via ``files.find_existing_paths_by_pattern``. With a slow
filesystem mount or a plugin that exposes many slots, the response
time climbs, the client polls faster, and we get a positive-feedback
storm — the user sees it as "the WebUI is hanging."

The 2s TTL cache breaks the storm: at most one filesystem walk per
2s, identical responses are served from RAM. The cache auto-busts
when the extension watchdog fires (plugin enable/disable, file
edit, etc.) so user-visible state stays fresh.

The circuit breaker is a second, defensive layer. When the
filesystem is slow enough that ``get_webui_extensions`` itself
raises (or hangs past the breaker threshold), the breaker opens
and we return ``[]`` immediately for ``recovery_s`` seconds. This
keeps the WebUI responsive even when the framework's FS layer is
wedged.

Plugin-locality
---------------
The cache is installed via a module-level monkey-patch of
``helpers.extension.get_webui_extensions`` from the
``run_ui/init_a0/start`` extension point. This is the same pattern
``_model_fallback`` already uses for its cascade. The patch is
idempotent (guarded by a sentinel attribute) so re-running
``init_a0`` (e.g. in a test harness) does not stack wrappers.

Future-proofing
---------------
If a future agent-zero version adds a built-in TTL cache to
``get_webui_extensions``, this module can be deleted and the
init_a0 hook simplified to a no-op. The dependency is one-way:
no core code knows about this cache.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from usr.plugins._model_fallback.helpers import stats

_log = logging.getLogger("model_fallback.extensions_cache")

DEFAULTS: Dict[str, Any] = {
    "enabled": True,
    "ttl_s": 2.0,
    "circuit_breaker_enabled": True,
    "circuit_breaker_window_s": 10.0,
    "circuit_breaker_threshold": 5,
    "circuit_breaker_recovery_s": 30.0,
}

# ---------------------------------------------------------------------------
# Resolved config (cached per-process)
# ---------------------------------------------------------------------------

_resolved: Dict[str, Any] = {}


def resolve_config(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = dict(DEFAULTS)
    if overrides:
        for k, v in overrides.items():
            if v is not None:
                cfg[k] = v
    cfg["enabled"] = bool(cfg.get("enabled", True))
    cfg["ttl_s"] = max(0.0, float(cfg.get("ttl_s") or 2.0))
    cfg["circuit_breaker_enabled"] = bool(cfg.get("circuit_breaker_enabled", True))
    cfg["circuit_breaker_window_s"] = max(0.1, float(cfg.get("circuit_breaker_window_s") or 10.0))
    cfg["circuit_breaker_threshold"] = max(1, int(cfg.get("circuit_breaker_threshold") or 5))
    cfg["circuit_breaker_recovery_s"] = max(0.0, float(cfg.get("circuit_breaker_recovery_s") or 30.0))
    return cfg


def set_resolved(cfg: Dict[str, Any]) -> None:
    global _resolved
    _resolved = dict(cfg)


def get_resolved() -> Dict[str, Any]:
    if not _resolved:
        _resolved = dict(DEFAULTS)
    return _resolved


# ---------------------------------------------------------------------------
# Cache + circuit-breaker state
# ---------------------------------------------------------------------------

# (args, expires_at). args is a (tuple-of-points, tuple-of-filters) key.
_cache: Dict[str, Any] = {
    "args": None,
    "value": None,
    "expires_at": 0.0,
}

# Sliding window of error timestamps for the breaker.
_error_window: Deque[float] = deque(maxlen=64)


def _make_key(extension_point: Any, filters: Any) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    return (
        tuple(extension_point or []),
        tuple(filters or []) or ("*",),
    )


def _circuit_is_open(cfg: Dict[str, Any], now: float) -> bool:
    """Return True if the breaker is currently open (we should short-circuit)."""
    if not cfg.get("circuit_breaker_enabled", True):
        return False
    opened = stats.extensions_cache_snapshot().get("circuit_opened_at") or 0.0
    if not opened:
        return False
    recovery = float(cfg.get("circuit_breaker_recovery_s") or 30.0)
    if (now - opened) >= recovery:
        return False
    return True


def _record_error(now: float, cfg: Dict[str, Any], exc: BaseException) -> None:
    _error_window.append(now)
    stats.extensions_cache_record_error(str(exc)[:200])
    if not cfg.get("circuit_breaker_enabled", True):
        return
    window = float(cfg.get("circuit_breaker_window_s") or 10.0)
    threshold = int(cfg.get("circuit_breaker_threshold") or 5)
    # Count errors in the last ``window`` seconds.
    recent = [t for t in _error_window if (now - t) <= window]
    if len(recent) >= threshold:
        stats.extensions_cache_record_circuit_open()
        _log.warning(
            "extensions cache circuit breaker OPEN after %d errors in %.1fs; "
            "short-circuiting for %.1fs",
            len(recent), window,
            float(cfg.get("circuit_breaker_recovery_s") or 30.0),
        )


def bust() -> None:
    """Invalidate the cache. Called from the extension watchdog hook
    so plugin enable/disable/file-edit is reflected immediately.
    """
    _cache["args"] = None
    _cache["value"] = None
    _cache["expires_at"] = 0.0
    stats.extensions_cache_record_bust()


# ---------------------------------------------------------------------------
# The wrapper that the init_a0/start hook installs
# ---------------------------------------------------------------------------

def _build_wrapper(
    original: Callable[..., List[str]],
    config_provider: Callable[[], Dict[str, Any]],
) -> Callable[..., List[str]]:
    """Return a drop-in replacement for ``get_webui_extensions``.

    The wrapper is a regular function (not a coroutine) because the
    original is synchronous. It returns a fresh list each call so
    callers can't accidentally mutate the cache.
    """
    sentinel = object()

    def wrapper(
        agent: Any = None,
        extension_point: Any = None,
        filters: Any = None,
    ) -> List[str]:
        cfg = config_provider()

        # Disabled == no-op wrapper (pass-through). Useful for tests
        # that want to opt out of caching.
        if not cfg.get("enabled", True):
            return list(original(agent, extension_point, filters) or [])

        now = time.monotonic()

        # Circuit breaker: when open, return [] without touching the FS.
        if _circuit_is_open(cfg, now):
            stats.extensions_cache_record_short_circuit()
            return []

        key = _make_key(extension_point, filters)
        cached_args = _cache.get("args")
        cached_value = _cache.get("value")
        cached_expires = _cache.get("expires_at") or 0.0

        if cached_args == key and cached_expires > now and cached_value is not sentinel:
            stats.extensions_cache_record_hit()
            return list(cached_value)

        stats.extensions_cache_record_miss()
        try:
            value = original(agent, extension_point, filters)
        except Exception as exc:  # noqa: BLE001
            _record_error(now, cfg, exc)
            # Return empty list (matches the framework's "no extensions"
            # semantics) so the WebUI can still render; the breaker
            # will open if this keeps happening.
            return []
        if value is None:
            value = []
        # Cache for ttl_s. ttl_s == 0 means "don't cache" (still record).
        ttl = float(cfg.get("ttl_s") or 0.0)
        if ttl > 0:
            _cache["args"] = key
            _cache["value"] = list(value)
            _cache["expires_at"] = now + ttl
        return list(value)

    wrapper._extensions_cache_wrapped = True  # type: ignore[attr-defined]
    wrapper._extensions_cache_original = original  # type: ignore[attr-defined]
    return wrapper


def install(config_provider: Optional[Callable[[], Dict[str, Any]]] = None) -> bool:
    """Monkey-patch ``helpers.extension.get_webui_extensions``.

    Idempotent: calling install() twice replaces the wrapper with a
    fresh one (the original reference is preserved), so the function
    stack stays at exactly one wrapper layer.

    Returns True if a fresh install happened, False if the function
    was already wrapped.
    """
    from helpers import extension as ext_helper  # type: ignore

    current = ext_helper.get_webui_extensions
    if getattr(current, "_extensions_cache_wrapped", False):
        # Already wrapped. Refresh the resolved config so the new
        # call uses the latest values (e.g. user changed a knob).
        if config_provider is not None:
            set_resolved(resolve_config(config_provider()))
        return False

    original = current
    if config_provider is None:
        provider = get_resolved
    else:
        def provider() -> Dict[str, Any]:  # type: ignore[misc]
            return resolve_config(config_provider())

    wrapper = _build_wrapper(original, provider)
    ext_helper.get_webui_extensions = wrapper  # type: ignore[assignment]
    if config_provider is not None:
        set_resolved(resolve_config(config_provider()))
    _log.info("extensions cache + circuit breaker installed (ttl=%.1fs)", get_resolved().get("ttl_s"))
    return True


def uninstall() -> None:
    """Restore the original ``get_webui_extensions``. Called by
    ``hooks.uninstall()`` so a plugin disable restores baseline behavior.
    """
    from helpers import extension as ext_helper  # type: ignore
    current = getattr(ext_helper, "get_webui_extensions", None)
    if current is None:
        return
    original = getattr(current, "_extensions_cache_original", None)
    if original is None:
        return
    ext_helper.get_webui_extensions = original  # type: ignore[assignment]
    bust()
    _error_window.clear()


def is_installed() -> bool:
    from helpers import extension as ext_helper  # type: ignore
    return bool(getattr(ext_helper.get_webui_extensions, "_extensions_cache_wrapped", False))


# ---------------------------------------------------------------------------
# Reset (called by hooks.uninstall())
# ---------------------------------------------------------------------------

def reset() -> None:
    """Clear cache state and circuit breaker history. Does NOT
    restore the original function — call uninstall() for that.
    """
    global _resolved
    bust()
    _error_window.clear()
    _resolved = {}
