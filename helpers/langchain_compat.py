"""LangChain v1 -> langchain_core compatibility shim (v2.4 of _model_fallback).

WHY THIS EXISTS
---------------
LangChain v1 moved several submodules to ``langchain_core``:
- ``langchain.prompts`` -> ``langchain_core.prompts``
- ``langchain.schema``  -> ``langchain_core.schema``  (partial; v1 still
  ships a few classes, e.g. ``AIMessage`` moved, ``HumanMessage`` /
  ``SystemMessage`` / ``BaseMessage`` stayed in langchain_core)

Docker images that only ship langchain v1 (and not the legacy
``langchain`` metapackage) raise ``ModuleNotFoundError: No module
named 'langchain.prompts'`` when agent code at
``helpers/call_llm.py:2`` runs a v0-style langchain import. The
error surfaces in the agent's own runtime -- e.g. when a user-
installed plugin's code_execution_tool imports ``helpers.call_llm``,
or when memory_memorize consolidates fragments through the LLM
stack. The agent's code-execution path dies with a fatal import
error, and the user sees a cascade of "all utility models
tried to format the same broken context" failures.

This is a v0/v1 langchain interface error, not a network or quota
error -- the `_model_fallback` cascade can't recover from a
ModuleNotFoundError because the import failure happens BEFORE
any model call. We treat it as part of the same LLM-error-handling
surface as the cascade itself, so the fix lives in this plugin.

WHAT THIS DOES
--------------
Installs a ``sys.modules`` shim so the legacy v0 import paths
resolve to their v1 homes:

- ``langchain.prompts``  -> ``langchain_core.prompts``
- ``langchain.schema``   -> ``langchain_core.messages`` (most of v0's
  ``langchain.schema`` classes moved there) plus a small synthetic
  module for ``AIMessage`` (which moved to langchain_core.messages in
  v1 as well)

Each shimmed module is tagged with a sentinel attribute so the
``uninstall`` path can safely remove ONLY the shims we added. A
user who later does their own ``import langchain.prompts`` keeps
theirs (our shim skips real modules in the install path).

The shim is IDEMPOTENT: a sentinel on the helper itself prevents a
second install. The shim is also a NO-OP when langchain_core is
not available (e.g. test envs that have neither v0 nor v1).

PLUGIN CONTRACT
---------------
- Plugin-only. ``helpers/call_llm.py`` is unchanged.
- Disabling the plugin (or the shim config) removes the shim
  entries. v0.x langchain users (who already have a real
  ``langchain.prompts``) get the original behavior back, because
  our sentinel only matches shims we installed.
- The shim never crashes agent init: every import is wrapped in
  try/except, and missing modules log + skip rather than raise.
"""

from __future__ import annotations

import importlib
import logging
import sys
import types
from typing import Any, Dict, Optional

_log = logging.getLogger("model_fallback.langchain_compat")

_SENTINEL = "_model_fallback_langchain_compat_shim"
_INSTALL_SENTINEL = "_model_fallback_langchain_compat_installed"


# Map of legacy v0 module path -> v1 source path. v1's langchain_core
# hosts the v0 module content under a different name in some cases.
# We treat langchain_core as the canonical source for both entries
# below, because langchain_core ships ``prompts`` and ``messages``
# with the same public API as v0's ``langchain.prompts`` / partial
# ``langchain.schema``.
_LEGACY_TO_V1 = {
    "langchain.prompts": "langchain_core.prompts",
    "langchain.schema": "langchain_core.messages",  # most v0 schema classes live here
}


def _make_shim(name: str, source_module: types.ModuleType) -> types.ModuleType:
    """Build a ``sys.modules`` entry that *is* the v1 module but
    carries our sentinel attribute. We re-use the same module object
    (id() == id(v1_module)) so attribute lookups are zero-cost.
    """
    # Stash the source module's "real" path before we mutate its __name__,
    # so users who introspect the module get a sensible answer.
    try:
        setattr(source_module, "_compat_original_name", name)
    except Exception:
        # Some modules (e.g. C-extension modules) refuse setattr. In
        # that case we fall back to a fresh ModuleType shim that
        # re-exports every public name from the source module.
        shim = types.ModuleType(name)
        shim.__file__ = getattr(source_module, "__file__", None)
        shim.__path__ = getattr(source_module, "__path__", [])
        for attr in dir(source_module):
            if attr.startswith("_"):
                continue
            try:
                setattr(shim, attr, getattr(source_module, attr))
            except Exception:
                continue
        setattr(shim, _SENTINEL, True)
        return shim
    setattr(source_module, _SENTINEL, True)
    # Update __name__ so tools that introspect it (e.g. inspect.getmodule)
    # see the legacy path. This is best-effort: some modules (C extensions)
    # have read-only __name__; we tolerate that.
    try:
        source_module.__name__ = name
    except Exception:
        pass
    return source_module


def _try_resolve(module_name: str) -> Optional[types.ModuleType]:
    """Import ``module_name``; return the module or None on failure."""
    try:
        return importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001
        _log.debug("could not import %s for compat shim: %s", module_name, exc)
        return None


def install_shim() -> Dict[str, bool]:
    """Install every shim from ``_LEGACY_TO_V1``.

    Returns a dict ``{legacy_name: True/False}`` for the API endpoint
    to surface. ``True`` means a shim was installed (the v1 source
    was found AND we registered it under the legacy name).
    """
    results: Dict[str, bool] = {}
    for legacy, v1 in _LEGACY_TO_V1.items():
        v1_module = _try_resolve(v1)
        if v1_module is None:
            results[legacy] = False
            continue
        # If the legacy name is already present in sys.modules and is NOT
        # one of our prior shims, leave it alone -- the user (or another
        # plugin) has the real module loaded.
        existing = sys.modules.get(legacy)
        if existing is not None and not getattr(existing, _SENTINEL, False):
            _log.debug(
                "%s already present in sys.modules (not a shim); leaving alone",
                legacy,
            )
            results[legacy] = False
            continue
        try:
            shim = _make_shim(legacy, v1_module)
            sys.modules[legacy] = shim
            results[legacy] = True
        except Exception as exc:  # noqa: BLE001
            _log.debug("could not register shim %s -> %s: %s", legacy, v1, exc)
            results[legacy] = False
    return results


def uninstall_shim() -> int:
    """Remove every shim we installed. Returns the count removed.

    Only removes entries that carry the sentinel -- a user who later
    imported the real module (overwriting our shim) keeps theirs.
    """
    removed = 0
    for legacy in list(_LEGACY_TO_V1.keys()):
        mod = sys.modules.get(legacy)
        if mod is None:
            continue
        if not getattr(mod, _SENTINEL, False):
            continue
        # Best-effort: restore the v1 module name so a later
        # ``import langchain.prompts`` user gets the real one back
        # (or a fresh import, if it was deleted).
        try:
            original_name = getattr(mod, "_compat_original_name", None)
            if original_name:
                mod.__name__ = original_name
        except Exception:
            pass
        try:
            del sys.modules[legacy]
            removed += 1
        except Exception:  # noqa: BLE001
            pass
    return removed


def is_installed() -> bool:
    """Return True if at least one shim is currently active."""
    for legacy in _LEGACY_TO_V1:
        mod = sys.modules.get(legacy)
        if mod is not None and getattr(mod, _SENTINEL, False):
            return True
    return False


def shim_status() -> Dict[str, Any]:
    """Return a snapshot of which shims are active, for the stats API."""
    return {
        "installed": is_installed(),
        "shims": {
            legacy: (
                legacy in sys.modules
                and getattr(sys.modules[legacy], _SENTINEL, False)
            )
            for legacy in _LEGACY_TO_V1
        },
    }


def already_installed_in_process() -> bool:
    """Return True if a previous call to ``install_shim()`` already
    installed the shims in this Python process.

    The agent_init extension's ``execute`` is invoked once per agent
    (not once per process), so a long-running server with many
    agents would re-run the install path on every new agent. The
    sentinel check keeps the second call cheap (a no-op that
    doesn't even try to import langchain_core again).
    """
    return bool(globals().get(_INSTALL_SENTINEL, False))


def mark_installed() -> None:
    """Record that we've installed the shims in this process."""
    globals()[_INSTALL_SENTINEL] = True


def clear_installed_marker() -> None:
    """Clear the install marker; used by ``uninstall_shim`` so a
    subsequent ``install_shim()`` re-runs.
    """
    globals()[_INSTALL_SENTINEL] = False
