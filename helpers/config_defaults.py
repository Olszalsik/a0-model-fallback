"""Canonical plugin-config resolution: default_config.yaml UNDER the live config.

**Why this module exists.** ``helpers.plugins.get_plugin_config`` is
either/or: it returns ``config.json`` **XOR** ``default_config.yaml``, never
both (see ``helpers/plugins.py::get_plugin_config``, the ``default_used``
flag). Every YAML-only knob is therefore dead unless the user hand-copies it
into ``config.json``.

That single framework gap produced **three separate production incidents** in
this plugin's own history, all documented in AGENTS.md:

- v2.6.7 — router detection, then the v2.6.8 / v2.6.9 follow-ons
- v2.8.3 F1 — the utility guard ran at 30s instead of the YAML's 60/180
- v2.8.4 — ``force_chat_completions_providers`` was dead at runtime

The workaround was to re-merge the defaults at each call site. That
produced **two more copies** of the merge (in ``fallback.py`` and
``models_ext.py``), which then drifted. And both copies were **shallow**:
``dict(defaults); merged.update(cfg)``. A shallow merge is itself a bug for
this plugin's config, because ``default_config.yaml`` has NESTED sections
(``utility_timeout_guard``, ``context_size_guard``, ``unlimited_paid_api_bases``
users, ...). A user who overrides ONE key inside a nested section silently
loses every sibling default in that section:

    default_config.yaml   utility_timeout_guard: {enabled: true, default_timeout_s: 60, max_wait_s: 180}
    user config.json      utility_timeout_guard: {default_timeout_s: 120}
    shallow merge         utility_timeout_guard: {default_timeout_s: 120}      <- enabled/max_wait_s GONE

That is the same failure mode as v2.8.3 F1 (a nested section whose siblings
vanished, so a code constant took over), one level down.

**Contract**

- ``deep_merge`` is recursive; the override wins. Nested mappings merge key by
  key. Lists and scalars are **replaced**, never appended (a user who lists
  three providers means those three, not "defaults plus three").
- ``load_defaults`` is mtime-keyed cached, so this is not a YAML parse per
  call. An mtime change is picked up without a restart.
- ``resolve_config`` is the single resolver. Both former call sites
  (``fallback._get_plugin_cfg`` and ``models_ext._force_chat_config``) delegate
  here, so the merge semantics cannot drift again.
- Never raises. A broken config must degrade to "use the defaults", not take
  a live model call down.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any

PLUGIN_NAME = "model_fallback"

_MERGE_CACHE_TTL_S = 1.0
_MERGE_CACHE_MAX = 64

_lock = threading.RLock()
# v3.4.1: "miss" marks a cached absent-YAML result so stamp=None hits the
# cache instead of re-reading the plugin dir on every call.
_defaults_cache: dict = {"stamp": None, "value": None, "miss": False}
_merge_cache: dict = {}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` onto ``base``. Override always wins.

    Nested mappings merge key-by-key so a partial nested override keeps its
    siblings. Lists and scalars are replaced outright.
    """
    out: dict = dict(base or {})
    for key, value in (override or {}).items():
        existing = out.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            out[key] = _deep_merge(existing, value)
        else:
            out[key] = value
    return out


# Public alias: the merge semantics are a contract, not an implementation detail.
deep_merge = _deep_merge


def invalidate_config_cache() -> None:
    """Drop every cache here. Call after any config write, reset, or disable."""
    with _lock:
        _defaults_cache["stamp"] = None
        _defaults_cache["value"] = None
        _defaults_cache["miss"] = False
        _merge_cache.clear()


def load_defaults(plugin_name: str = PLUGIN_NAME) -> dict:
    """default_config.yaml contents, mtime-keyed cached. Never raises."""
    path = ""
    try:
        from helpers import files as _files
        from helpers import plugins as _plugins

        plugin_dir = _plugins.find_plugin_dir(plugin_name)
        if plugin_dir:
            path = _files.get_abs_path(
                plugin_dir, _plugins.CONFIG_DEFAULT_FILE_NAME
            )
    except Exception:  # noqa: BLE001
        path = ""

    stamp: Any = None
    if path:
        try:
            st = os.stat(path)
            stamp = (st.st_mtime_ns, st.st_size)
        except OSError:
            stamp = None

    with _lock:
        # v3.4.1: a MISS (YAML absent) is cached too. The old hit
        # condition required ``stamp is not None``, so with no YAML on
        # disk every call re-ran the plugin-config read -- the sentinel
        # ("miss" flag) makes the absent-YAML result cached like any
        # other.
        if _defaults_cache["stamp"] == stamp and (
            stamp is not None or _defaults_cache["miss"]
        ):
            cached = _defaults_cache["value"]
            if isinstance(cached, dict):
                return cached

    try:
        from helpers import plugins as _plugins

        defaults = _plugins.get_default_plugin_config(plugin_name)
    except Exception:  # noqa: BLE001
        defaults = {}
    if not isinstance(defaults, dict):
        defaults = {}

    with _lock:
        _defaults_cache["stamp"] = stamp
        _defaults_cache["value"] = defaults
        _defaults_cache["miss"] = stamp is None
    return defaults


def _cache_key(plugin_name: str, agent: Any) -> str:
    """Stable per-scope cache key for the merged plugin config.

    A key that is too COARSE is a cross-chat config leak, which is worse than a
    cache miss. The context id is therefore part of the key even when project
    and profile both resolve: a scoped config is a function of all three.
    """
    project_name = ""
    agent_profile = ""
    context_id = ""
    if agent is not None:
        try:
            from helpers import projects

            project_name = (
                projects.get_context_project_name(getattr(agent, "context", None))
                or ""
            )
        except Exception:  # noqa: BLE001
            project_name = ""
        try:
            agent_profile = (
                getattr(getattr(agent, "config", None), "profile", "") or ""
            )
        except Exception:  # noqa: BLE001
            agent_profile = ""
        try:
            context_id = str(
                getattr(getattr(agent, "context", None), "id", "") or ""
            )
        except Exception:  # noqa: BLE001
            context_id = ""
    return (
        f"{plugin_name}::{project_name}::{agent_profile}::{context_id}"
    )


def resolve_config(
    plugin_name: str = PLUGIN_NAME,
    agent: Any = None,
    *,
    use_cache: bool = True,
) -> dict:
    """The single resolver: default_config.yaml UNDER the live scoped config.

    ``use_cache=False`` forces a fresh read. Anything that renders config back
    to a user (the settings panel, the stats endpoint) must pass False, or it
    can display a value up to one TTL stale.
    """
    key = _cache_key(plugin_name, agent)
    now = time.monotonic()
    if use_cache:
        entry = _merge_cache.get(key)
        if entry is not None:
            expires_at, merged = entry
            if now < expires_at:
                # v3.4.1: hand the caller a COPY. The cached dict is shared
                # across every reader of this scope; a caller that mutated
                # the returned mapping (e.g. injected a private key) used
                # to poison the cache for every subsequent reader.
                return dict(merged)

    cfg: dict = {}
    try:
        from helpers import plugins as _plugins

        cfg = _plugins.get_plugin_config(plugin_name, agent) or {}
    except Exception:  # noqa: BLE001
        cfg = {}
    if not isinstance(cfg, dict):
        cfg = {}

    merged = _deep_merge(load_defaults(plugin_name), cfg)

    with _lock:
        if len(_merge_cache) >= _MERGE_CACHE_MAX:
            # Drop the soonest-to-expire entry rather than growing forever.
            oldest = min(_merge_cache, key=lambda k: _merge_cache[k][0])
            _merge_cache.pop(oldest, None)
        _merge_cache[key] = (now + _MERGE_CACHE_TTL_S, merged)
    # v3.4.1: same copy-on-return contract for the freshly computed merge.
    return dict(merged)


def resolve_section(
    section: str,
    plugin_name: str = PLUGIN_NAME,
    agent: Any = None,
    *,
    use_cache: bool = True,
) -> dict:
    """One nested section of the resolved config, or ``{}`` when absent."""
    value = resolve_config(plugin_name, agent, use_cache=use_cache).get(section)
    return value if isinstance(value, dict) else {}


__all__ = [
    "PLUGIN_NAME",
    "deep_merge",
    "invalidate_config_cache",
    "load_defaults",
    "resolve_config",
    "resolve_section",
]
