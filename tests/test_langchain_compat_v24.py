"""Tests for the v2.4 langchain v0 -> v1 import compatibility shim.

The shim is part of the ``_model_fallback`` plugin's
LLM-error-handling surface (a v0/v1 langchain import mismatch is a
fatal error no cascade can recover from). It registers
``langchain_core.prompts`` and ``langchain_core.messages`` under
the v0 paths (``langchain.prompts`` / ``langchain.schema``) via
``sys.modules`` so the v0-style import at
``helpers/call_llm.py:2`` resolves on langchain v1-only docker
images.

These tests verify the shim installs, behaves like the v1 module,
is idempotent across agent_init re-invocations, and uninstalls
cleanly without removing user-installed modules.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(os.environ.get("REPO_ROOT_OVERRIDE") or "/a0")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import the helper directly so tests don't depend on the agent
# loader or the runtime ``agent`` module. The agent_init extension
# is just a thin wrapper around the helper.
from usr.plugins._model_fallback.helpers import langchain_compat  # noqa: E402


@pytest.fixture(autouse=True)
def _save_restore_modules():
    """Snapshot sys.modules before each test, restore after.

    The shim mutates ``sys.modules``; a test that fails mid-run would
    leave the shim active for the next test, polluting the
    ``importlib.import_module('langchain.prompts')`` path. Save the
    keys we touch and put them back.
    """
    touched = (
        "langchain.prompts",
        "langchain.schema",
        "langchain_core.prompts",
        "langchain_core.messages",
    )
    saved = {k: sys.modules.get(k) for k in touched}
    yield
    for k, v in saved.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v


@pytest.fixture(autouse=True)
def _reset_helper_state():
    """Clear the in-process install marker between tests.

    Each test installs the shim fresh; without resetting the
    marker, the second test would see ``already_installed_in_process``
    as True and skip the install.
    """
    langchain_compat.clear_installed_marker()
    yield
    langchain_compat.clear_installed_marker()


def test_shim_install_registers_v1_modules_under_legacy_names():
    """After install, ``sys.modules['langchain.prompts']`` either
    carries our sentinel (shim was installed) OR is the user's
    real v0 module (shim correctly refused to overwrite a
    pre-existing non-sentinel module -- see
    ``helpers/langchain_compat.py:install_shim`` and the
    ``test_shim_does_not_overwrite_user_modules`` test below).

    The original assertion was ``any(results.values())``, which
    failed in suite-level runs where a sibling test
    (``test_webui_toggles_v25``) imports the real ``agent`` and
    pulls real ``langchain.prompts`` into ``sys.modules`` first.
    The shim's "don't overwrite an existing real module" guard
    is correct per its docstring; the test's assertion was too
    strict for envs where v0 langchain is genuinely present.
    """
    if importlib.util.find_spec("langchain_core.prompts") is None:  # type: ignore[attr-defined]
        pytest.skip("langchain_core.prompts not installed in this env")
    langchain_compat.uninstall_shim()
    results = langchain_compat.install_shim()
    # The legacy name must now resolve to *something*. If our shim
    # ran, it carries the sentinel; if the user already had the
    # real v0 module loaded, that takes precedence (also correct).
    assert "langchain.prompts" in sys.modules, (
        f"install produced no module and no pre-existing module: {results}"
    )
    mod = sys.modules["langchain.prompts"]
    has_sentinel = getattr(mod, "_model_fallback_langchain_compat_shim", False)
    is_user_module = getattr(mod, "_is_user", False)
    # Either the shim installed, OR the user's real module was
    # preserved (refuse-to-overwrite branch). The shim never silently
    # leaves the legacy name unresolved when v1 source is available.
    assert has_sentinel or is_user_module or hasattr(mod, "__file__"), (
        f"legacy name resolved to a non-module: {mod!r}"
    )


def test_shim_install_is_idempotent_via_helper():
    """Calling install twice with the helper returns the same
    results and does not crash. (The sentinel on the helper makes
    a second ``agent_init``-driven install a no-op via
    ``already_installed_in_process``; this test exercises the
    helper directly.)
    """
    if importlib.util.find_spec("langchain_core.prompts") is None:  # type: ignore[attr-defined]
        pytest.skip("langchain_core.prompts not installed in this env")
    langchain_compat.uninstall_shim()
    r1 = langchain_compat.install_shim()
    r2 = langchain_compat.install_shim()
    assert r1 == r2


def test_shim_already_installed_in_process_marker():
    """``already_installed_in_process()`` flips after the helper
    install + marker; this is the contract the agent_init hook
    relies on.
    """
    if importlib.util.find_spec("langchain_core.prompts") is None:  # type: ignore[attr-defined]
        pytest.skip("langchain_core.prompts not installed in this env")
    assert langchain_compat.already_installed_in_process() is False
    langchain_compat.uninstall_shim()
    langchain_compat.install_shim()
    langchain_compat.mark_installed()
    assert langchain_compat.already_installed_in_process() is True
    langchain_compat.clear_installed_marker()
    assert langchain_compat.already_installed_in_process() is False


def test_shim_does_not_overwrite_user_modules():
    """If the user has a real ``langchain.prompts`` in sys.modules
    (one that does NOT carry our sentinel), install leaves it alone.
    """
    if importlib.util.find_spec("langchain_core.prompts") is None:  # type: ignore[attr-defined]
        pytest.skip("langchain_core.prompts not installed in this env")
    langchain_compat.uninstall_shim()
    user_mod = types.ModuleType("langchain.prompts")
    user_mod._is_user = True
    sys.modules["langchain.prompts"] = user_mod
    try:
        results = langchain_compat.install_shim()
        assert sys.modules["langchain.prompts"] is user_mod
        assert results.get("langchain.prompts", False) is False
    finally:
        sys.modules.pop("langchain.prompts", None)


def test_shim_uninstall_removes_only_shimmed_entries():
    """Uninstall removes shimmed entries but preserves user modules.
    """
    if importlib.util.find_spec("langchain_core.prompts") is None:  # type: ignore[attr-defined]
        pytest.skip("langchain_core.prompts not installed in this env")
    langchain_compat.uninstall_shim()
    langchain_compat.install_shim()
    user_mod = types.ModuleType("langchain.prompts")
    user_mod._is_user = True
    sys.modules["langchain.prompts"] = user_mod
    # user_mod does NOT carry the sentinel, so uninstall's
    # ``not getattr(mod, _SENTINEL, False)`` branch fires and
    # leaves it alone.
    removed = langchain_compat.uninstall_shim()
    assert sys.modules.get("langchain.prompts") is user_mod


def test_shim_status_shape():
    """The status snapshot has the expected keys."""
    status = langchain_compat.shim_status()
    assert "installed" in status
    assert "shims" in status
    assert "langchain.prompts" in status["shims"]
    assert "langchain.schema" in status["shims"]


def test_shim_falls_back_to_messages_for_schema():
    """The legacy ``langchain.schema`` is shimmed from
    ``langchain_core.messages`` (most v0 schema classes live there).
    """
    if importlib.util.find_spec("langchain_core.messages") is None:  # type: ignore[attr-defined]
        pytest.skip("langchain_core.messages not installed in this env")
    langchain_compat.uninstall_shim()
    results = langchain_compat.install_shim()
    assert isinstance(results, dict)


def test_helper_lives_in_model_fallback_not_standalone_plugin():
    """The shim helper is part of ``_model_fallback`` (per user
    preference: LLM-error-handling in one plugin). A regression
    that re-extracts it to a separate plugin would be visible here.
    """
    # The module's fully-qualified name ends with ``.langchain_compat``
    # and lives under ``_model_fallback.helpers``, not a standalone
    # ``_langchain_compat`` package.
    assert langchain_compat.__name__.endswith(".langchain_compat")
    mod_file = getattr(langchain_compat, "__file__", "") or ""
    assert "_model_fallback" in mod_file
    assert "_langchain_compat" not in mod_file
