"""Per-feature top-level toggle resolver (v2.5 WebUI surface).

The WebUI binds to flat top-level booleans (e.g.
``utility_timeout_guard_enabled``) instead of nested section
keys, because Alpine ``x-model`` and a flat settings dict are
simpler to bind. The runtime helpers still accept the nested
section (``<piece>.enabled``) for back-compat with hand-edited
config.yaml files; if both are present the **top-level toggle
wins** so the WebUI always reflects what the user just toggled.

Why a single helper module
--------------------------
Three pieces read config (utility_timeout_guard, context_size_guard,
langchain_compat) and each used to copy the same
``cfg.get("<piece>")`` pattern. With the new top-level keys,
the resolution order becomes:

    1. ``<piece>_enabled`` (top-level) — the WebUI toggle.
    2. ``<piece>.enabled`` (nested) — the legacy key, kept for
       hand-edited configs that never went through the WebUI.
    3. The piece's DEFAULTS dict.

This module exports one function per piece so each call site reads
its own override dict from the merged plugin config and the
per-piece helper returns a normalized boolean.

When to delete this file
------------------------
If the v2.5 WebUI contract is ever changed to use the nested
``<piece>.enabled`` keys directly, the top-level keys can be
removed and this file goes away. The piece extensions fall back to
the nested key, so behavior is preserved.

v2.5 housekeeping removal
-------------------------
The standalone event-loop housekeeping loop was removed in this
branch (it was implemented the same day, never tested, and
turned into a per-minute safety-net respawn log line that
provided no user-visible benefit). The ``housekeeping_enabled``
and ``second_pulse_path_enabled`` keys are still recognised for
back-compat with hand-edited configs but are ignored at runtime
-- the corresponding extensions no longer exist.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


# Top-level toggle key for each piece. Order matters: the
# ``legacy_section`` argument is only consulted if the top-level
# key is absent.
#
# ``housekeeping`` is intentionally still listed here so that a
# hand-edited config containing ``housekeeping_enabled`` does not
# raise a KeyError during toggle resolution; the value is returned
# but the runtime no longer reads it.
_PIECE_TOGGLES = {
    "utility_timeout_guard": "utility_timeout_guard_enabled",
    "housekeeping": "housekeeping_enabled",  # ignored at runtime; back-compat only
    "context_size_guard": "context_size_guard_enabled",
    "langchain_compat": "langchain_compat_enabled",
}


def resolve_toggle(
    cfg: Dict[str, Any],
    piece: str,
    default: Optional[bool] = None,
) -> bool:
    """Return the effective on/off boolean for a piece.

    Resolution order
    ----------------
    1. ``cfg[<piece>_enabled]`` if the key is present (even if None
       or False — explicit user choice wins over the legacy key).
    2. ``cfg[piece]["enabled"]`` if the nested section is present.
    3. ``default`` if the caller passed one explicitly.
    4. The per-piece built-in default (True for most pieces, False
       for ``context_size_guard`` which is opt-in).

    The first rule's "even if None" caveat is important: a user
    toggling OFF in the WebUI saves ``False``, not the key
    deleted, so the second rule must not override a deliberate
    False. We use ``"in"`` semantics via ``.get`` only for
    existence checks, not values.
    """
    if not isinstance(cfg, dict):
        return default if default is not None else _BUILTIN_DEFAULTS.get(piece, True)
    top_key = _PIECE_TOGGLES.get(piece)
    if top_key and top_key in cfg:
        return bool(cfg[top_key])
    nested = cfg.get(piece)
    if isinstance(nested, dict) and "enabled" in nested:
        return bool(nested["enabled"])
    if default is not None:
        return default
    return _BUILTIN_DEFAULTS.get(piece, True)


# Built-in per-piece defaults. The WebUI displays these on
# first load; the runtime falls back to them when neither the
# top-level key nor the nested section is present.
_BUILTIN_DEFAULTS: Dict[str, bool] = {
    "utility_timeout_guard": True,   # ON; cascade alone is not enough for slow ollama CPU
    "housekeeping": True,            # ignored at runtime; back-compat default (no loop exists)
    "context_size_guard": False,     # OFF; opt-in because aggressive trimming can confuse the LLM
    "langchain_compat": True,        # ON; shim is a no-op on langchain v0.x
}


def is_second_pulse_path_enabled(cfg: Dict[str, Any]) -> bool:
    """Back-compat shim: the old ``second_pulse_path_enabled`` toggle
    gated an alternate WebSocket pulse path that lived inside the
    housekeeping loop. The loop is gone, so this is a no-op read:
    any config that still carries the key returns the user's last
    value, but nothing on the runtime side consults the result.

    Kept so a hand-edited config doesn't break the WebUI's
    settings-store schema validation (which probes every known
    key on load).
    """
    if not isinstance(cfg, dict):
        return True
    if "second_pulse_path_enabled" in cfg:
        return bool(cfg["second_pulse_path_enabled"])
    nested = cfg.get("housekeeping")
    if isinstance(nested, dict) and "second_pulse_path_enabled" in nested:
        return bool(nested["second_pulse_path_enabled"])
    return True
