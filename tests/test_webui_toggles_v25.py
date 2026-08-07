"""Tests for the v2.5 WebUI feature toggles.

The WebUI binds to flat top-level booleans on ``context.settings``
(e.g. ``utility_timeout_guard_enabled``) instead of nested
section keys, because Alpine ``x-model`` and a flat settings dict
are simpler to bind. The runtime helpers in
``usr/plugins/_model_fallback/helpers/toggles.py`` read the
top-level key first and fall back to the nested section for
back-compat with hand-edited configs.

These tests verify:
* The top-level key wins when present, even if False.
* The nested section's ``enabled`` is honoured when the top-level
  key is absent.
* The defaults match the conservative defaults documented in
  ``AGENTS.md`` (utility_timeout_guard ON, webui_extensions_cache
  ON, context_size_guard OFF, langchain_compat ON; the
  ``housekeeping`` and ``second_pulse_path_enabled`` keys still
  resolve to a default of True for back-compat but the runtime
  ignores them — the loop was removed in v2.5).
* The two helpers (per-piece toggle and second-pulse-path
  sub-toggle) have the right precedence rules.
* Each of the 4 remaining extension-level ``_resolve_config``
  functions returns the early-disable dict when the toggle is OFF.

The test does NOT touch the real framework; the extension
_resolve_config functions each fetch the plugin config via
``helpers.plugins.get_plugin_config`` which is mocked.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(os.environ.get("REPO_ROOT_OVERRIDE") or "/a0")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Stub out ``agent`` and ``models`` BEFORE importing any extension
# that does ``from agent import Agent`` at module load. The
# repository's host environment may not have a compatible
# huggingface-hub for the real ``models.py`` import chain, and
# the extensions only need ``Agent`` as a type reference for the
# monkey-patch. A lightweight class is enough.
class _StubAgent:  # noqa: D401
    """Minimal stand-in for ``agent.Agent`` so module-level
    ``from agent import Agent`` resolves in tests."""

    def __init__(self, *args, **kwargs):
        pass

    def get_utility_model(self):  # mirrors the real Agent method
        return None


class _StubCallUtilityModel:  # type: ignore[no-redef]
    """Stand-in for ``Agent.call_utility_model``."""


class _StubAgentModule(types.ModuleType):
    Agent = _StubAgent

    def __getattr__(self, name):
        if name == "call_utility_model":
            return _StubCallUtilityModel
        raise AttributeError(name)


# Install the stub into sys.modules only if the real module is
# not already loadable. The repo runs CI against a venv that
# can import the real agent; tests run in the host venv where
# it may not. We probe by trying to import the real agent; on
# ImportError we install the stub.
if "agent" not in sys.modules:
    try:
        import agent  # noqa: F401
    except Exception:  # noqa: BLE001
        sys.modules["agent"] = _StubAgentModule("agent")

from usr.plugins._model_fallback.helpers import toggles  # noqa: E402


# ---------------------------------------------------------------------------
# The toggle helper itself
# ---------------------------------------------------------------------------

class TestResolveToggleTopLevelWins:
    """When the top-level key is present, its value wins — even
    if the value is False. This is important: a user toggling
    OFF in the WebUI must not be silently overridden by an
    enabled nested section.
    """

    @pytest.mark.parametrize("piece", [
        "utility_timeout_guard",
        "housekeeping",
        "webui_extensions_cache",
        "context_size_guard",
        "langchain_compat",
    ])
    def test_top_level_true_wins(self, piece):
        cfg = {f"{piece}_enabled": True, piece: {"enabled": False}}
        assert toggles.resolve_toggle(cfg, piece) is True

    @pytest.mark.parametrize("piece", [
        "utility_timeout_guard",
        "housekeeping",
        "webui_extensions_cache",
        "context_size_guard",
        "langchain_compat",
    ])
    def test_top_level_false_wins(self, piece):
        cfg = {f"{piece}_enabled": False, piece: {"enabled": True}}
        assert toggles.resolve_toggle(cfg, piece) is False


class TestResolveToggleFallback:
    """When the top-level key is absent, the nested section's
    ``enabled`` is honoured.
    """

    @pytest.mark.parametrize("piece", [
        "utility_timeout_guard",
        "housekeeping",
        "webui_extensions_cache",
        "context_size_guard",
        "langchain_compat",
    ])
    def test_nested_true_falls_through(self, piece):
        cfg = {piece: {"enabled": True}}
        assert toggles.resolve_toggle(cfg, piece) is True

    @pytest.mark.parametrize("piece", [
        "utility_timeout_guard",
        "housekeeping",
        "webui_extensions_cache",
        "context_size_guard",
        "langchain_compat",
    ])
    def test_nested_false_falls_through(self, piece):
        cfg = {piece: {"enabled": False}}
        assert toggles.resolve_toggle(cfg, piece) is False


class TestResolveToggleDefaults:
    """When neither the top-level nor the nested key is present,
    the per-piece default applies. Five of six are ON by default
    (conservative); ``context_size_guard`` is OFF (opt-in because
    aggressive trimming can confuse the LLM).
    """

    def test_utility_timeout_guard_default_on(self):
        assert toggles.resolve_toggle({}, "utility_timeout_guard") is True

    def test_housekeeping_default_on(self):
        assert toggles.resolve_toggle({}, "housekeeping") is True

    def test_webui_extensions_cache_default_on(self):
        assert toggles.resolve_toggle({}, "webui_extensions_cache") is True

    def test_context_size_guard_default_off(self):
        assert toggles.resolve_toggle({}, "context_size_guard") is False

    def test_langchain_compat_default_on(self):
        assert toggles.resolve_toggle({}, "langchain_compat") is True

    def test_garbage_config_returns_default(self):
        # Bad config shape (None, string, list) must not raise; it
        # returns the per-piece default.
        assert toggles.resolve_toggle(None, "housekeeping") is True
        assert toggles.resolve_toggle("not a dict", "housekeeping") is True
        assert toggles.resolve_toggle(["a", "b"], "context_size_guard") is False

    def test_explicit_default_override(self):
        # The default argument overrides the per-piece default.
        assert toggles.resolve_toggle({}, "context_size_guard", default=True) is True
        assert toggles.resolve_toggle({}, "housekeeping", default=False) is False


# ---------------------------------------------------------------------------
# The second-pulse-path sub-toggle
# ---------------------------------------------------------------------------

class TestSecondPulsePath:
    """The ``second_pulse_path_enabled`` is a sub-feature of
    housekeeping (it gates the socketio.emit path in the WS
    keepalive pulse). It is intentionally a top-level key, not a
    nested ``housekeeping.second_pulse_path_enabled``, because
    the WebUI surface is a flat dict and the sub-feature is
    enabled/disabled on its own toggle row.
    """

    def test_top_level_true(self):
        assert toggles.is_second_pulse_path_enabled(
            {"second_pulse_path_enabled": True}
        ) is True

    def test_top_level_false(self):
        assert toggles.is_second_pulse_path_enabled(
            {"second_pulse_path_enabled": False}
        ) is False

    def test_nested_fallback(self):
        assert toggles.is_second_pulse_path_enabled(
            {"housekeeping": {"second_pulse_path_enabled": True}}
        ) is True
        assert toggles.is_second_pulse_path_enabled(
            {"housekeeping": {"second_pulse_path_enabled": False}}
        ) is False

    def test_top_level_wins_over_nested(self):
        # Top-level False beats nested True.
        assert toggles.is_second_pulse_path_enabled(
            {
                "second_pulse_path_enabled": False,
                "housekeeping": {"second_pulse_path_enabled": True},
            }
        ) is False

    def test_default_on(self):
        # The second pulse path is ON by default. The first pulse
        # path (the WsManager broadcast) is always on while
        # housekeeping is enabled, so the default for the second
        # path being OFF would not break the keepalive — it would
        # just leave the reconnect-storm window unprotected.
        assert toggles.is_second_pulse_path_enabled({}) is True


# ---------------------------------------------------------------------------
# Extension-level resolve_config functions
# ---------------------------------------------------------------------------

class TestExtensionResolveConfigEarlyDisable:
    """Each extension's ``_resolve_config`` returns
    ``{"enabled": False}`` when the WebUI toggle is OFF, and the
    full config dict when the toggle is ON. The runtime checks
    ``cfg.get("enabled", True)`` and bails when False.

    The extensions do ``from helpers import plugins as
    plugin_helpers`` inside the function, so the test patches
    ``helpers.plugins.get_plugin_config`` directly.
    """

    def test_utility_timeout_disabled(self):
        from usr.plugins._model_fallback.extensions.python.agent_init import (
            _10_install_utility_timeout_patch as mod,
        )
        with patch("helpers.plugins.get_plugin_config") as gpc:
            gpc.return_value = {
                "utility_timeout_guard_enabled": False,
                "utility_timeout_guard": {
                    "enabled": True, "max_wait_s": 120, "default_timeout_s": 30,
                },
            }
            cfg = mod._resolve_config(None)
            assert cfg == {"enabled": False}

    def test_utility_timeout_enabled_falls_through_to_full_config(self):
        from usr.plugins._model_fallback.extensions.python.agent_init import (
            _10_install_utility_timeout_patch as mod,
        )
        with patch("helpers.plugins.get_plugin_config") as gpc:
            gpc.return_value = {
                "utility_timeout_guard_enabled": True,
                "utility_timeout_guard": {"max_wait_s": 90, "default_timeout_s": 25},
            }
            cfg = mod._resolve_config(None)
            # When ON, the inner knobs flow through.
            assert cfg.get("enabled") is True
            assert cfg.get("max_wait_s") == 90
            assert cfg.get("default_timeout_s") == 25

    def test_webui_extensions_cache_disabled(self):
        from usr.plugins._model_fallback.extensions.python._functions.run_ui.init_a0.start import (
            _10_install_extensions_cache as mod,
        )
        with patch("helpers.plugins.get_plugin_config") as gpc:
            gpc.return_value = {
                "webui_extensions_cache_enabled": False,
                "webui_extensions_cache": {"enabled": True, "ttl_s": 2.0},
            }
            cfg = mod._resolve_config()
            assert cfg == {"enabled": False}

    def test_langchain_compat_disabled(self):
        from usr.plugins._model_fallback.extensions.python.agent_init import (
            _00_install_langchain_shim as mod,
        )
        with patch("helpers.plugins.get_plugin_config") as gpc:
            gpc.return_value = {
                "langchain_compat_enabled": False,
                "langchain_compat": {"enabled": True},
            }
            cfg = mod._resolve_config(None)
            assert cfg == {"enabled": False}

    def test_context_size_guard_disabled(self):
        from usr.plugins._model_fallback.extensions.python.message_loop_prompts_after import (
            _10_context_size_guard as mod,
        )
        with patch("helpers.plugins.get_plugin_config") as gpc:
            gpc.return_value = {
                "context_size_guard_enabled": False,
                "context_size_guard": {"enabled": True, "max_chars": 50000},
            }
            cfg = mod._resolve_runtime_config(None)
            assert cfg == {"enabled": False}


# ---------------------------------------------------------------------------
# v2.5: the housekeeping module and the second-pulse-path
# sub-feature were removed together with the loop. The
# ``TestHousekeepingSecondPulsePath`` class and the
# ``test_housekeeping_disabled`` / ``test_housekeeping_nested_only``
# tests below this comment are deleted.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Regression: back-compat with hand-edited config.yaml
# ---------------------------------------------------------------------------

class TestHandEditedConfigBackCompat:
    """A user who never went through the WebUI still has a
    hand-edited ``config.json`` that only contains the nested
    sections (e.g. ``{"utility_timeout_guard": {"enabled": true}}``).
    The runtime must honour the nested key so the migration
    doesn't break their existing setup.
    """

    def test_utility_timeout_nested_only(self):
        from usr.plugins._model_fallback.extensions.python.agent_init import (
            _10_install_utility_timeout_patch as mod,
        )
        with patch("helpers.plugins.get_plugin_config") as gpc:
            gpc.return_value = {
                "utility_timeout_guard": {"enabled": True, "max_wait_s": 60},
            }
            cfg = mod._resolve_config(None)
            assert cfg.get("enabled") is True
            assert cfg.get("max_wait_s") == 60

    def test_langchain_compat_nested_only(self):
        from usr.plugins._model_fallback.extensions.python.agent_init import (
            _00_install_langchain_shim as mod,
        )
        with patch("helpers.plugins.get_plugin_config") as gpc:
            gpc.return_value = {
                "langchain_compat": {"enabled": True},
            }
            cfg = mod._resolve_config(None)
            assert cfg.get("enabled") is True
