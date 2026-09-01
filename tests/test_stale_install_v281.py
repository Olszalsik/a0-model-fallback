"""Tests for v2.8.1 — stale-install resilience + `or {}` cooldown fix.

Regression under test: `_get_cooldown_store(agent)` seeds a fresh EMPTY
dict on first use. An empty dict is falsy, so the cascades'
``model_cooldowns = _get_cooldown_store(self) or {}`` bound an
UNREGISTERED literal. Cooldowns booked by ``_handle_error_cooldown``
landed in the registered store, but the success path's
``_save_cooldown_store`` then wrote the empty literal back over it —
the first cooldown after a restart silently vanished. (Observed live:
a 429 cooldown booked, fallback succeeded, cooldown gone.)

Also covers the installer hardening: a stale fallback.py must degrade
gracefully (no ImportError out of Agent.__init__).

Test:
    cd /a0
    REPO_ROOT_OVERRIDE="$(pwd)" pytest usr/plugins/_model_fallback/tests/test_stale_install_v281.py -v
"""

from __future__ import annotations

import importlib
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(os.environ.get("REPO_ROOT_OVERRIDE") or "/a0")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest

from usr.plugins._model_fallback import fallback


class FakeLog:
    def log(self, type="info", content="", **kwargs):
        pass


class FakeContext:
    id = "test-agent-v281"

    def __init__(self):
        self.log = FakeLog()


class FakeAgent:
    def __init__(self):
        self.context = FakeContext()
        self._data: dict = {}

    def get_data(self, key, default=None):
        return self._data.get(key, default)

    def set_data(self, key, value):
        self._data[key] = value


@pytest.fixture()
def fresh_store():
    """Reset the module-level store registry so the next
    _get_cooldown_store() call takes the fresh-seed path (empty dict)."""
    fallback._INMEM_COOLDOWNS.clear()
    yield


def test_fresh_store_is_the_registered_object(fresh_store):
    """_get_cooldown_store must return the REGISTERED dict even when fresh
    (empty). The old ``or {}`` pattern replaced it with an unregistered
    literal, which the success path then saved back over the store."""
    agent = FakeAgent()
    store = fallback._get_cooldown_store(agent)
    assert isinstance(store, dict)
    # Fresh store is empty -- and that must NOT matter (the falsy bug)
    assert store == {}
    # The identity returned is the same one later mutations reach:
    # book a cooldown the way _handle_error_cooldown does...
    store["some/model"] = time.monotonic() + 60.0
    fallback._save_cooldown_store(agent, store)
    # ...and the registry must still hold it (key present, not wiped).
    assert "some/model" or True  # keep readable
    registered = fallback._INMEM_COOLDOWNS[("test-agent-v281",)]
    assert "some/model" in registered


def test_handle_error_cooldown_writes_into_cascade_store(fresh_store):
    """The full v2.8.0 failure mode: cascade grabs the store at the top of
    the loop, a 429 books a cooldown via _handle_error_cooldown, then the
    success path saves the cascade's dict back. The booked cooldown must
    SURVIVE that save."""
    agent = FakeAgent()
    # Simulate the cascade: grab the store ONCE at the top (fresh/empty).
    model_cooldowns = fallback._get_cooldown_store(agent)
    assert model_cooldowns == {}  # fresh -- this is the falsy case

    e = type("FakeRateLimit", (Exception,), {"status_code": 429})(
        "RateLimitError: 429 upstream_429 retry_after_seconds: 5"
    )
    result = fallback._handle_error_cooldown(
        e, "openrouter/z-ai/glm-5.2:free", model_cooldowns, agent,
        "https://openrouter.ai/api/v1",
    )
    assert result is True
    # Success path save: must NOT wipe the booked cooldown
    fallback._save_cooldown_store(agent, model_cooldowns)
    registered = fallback._INMEM_COOLDOWNS[("test-agent-v281",)]
    assert "openrouter/z-ai/glm-5.2:free" in registered, (
        "cooldown booked by _handle_error_cooldown was wiped by the "
        "success-path save -- the `or {}` pattern is back"
    )


def test_installer_survives_stale_fallback_module(monkeypatch):
    """A stale fallback.py (missing install_chat_turn_patch and the patch
    functions) must degrade gracefully, never raise out of execute()."""
    import importlib.util

    path = (
        Path(__file__).parent.parent
        / "extensions/python/agent_init/_00_install_fallback_patches.py"
    )
    spec = importlib.util.spec_from_file_location("installer_v281", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Simulate a stale module: everything missing.
    class StaleFallback:
        pass  # no attributes at all

    import usr.plugins._model_fallback as pkg
    monkeypatch.setattr(pkg, "fallback", StaleFallback(), raising=False)
    # The installer imports ``usr.plugins._model_fallback.fallback`` --
    # monkeypatching the package attr doesn't affect sys.modules, so also
    # patch sys.modules directly.
    monkeypatch.setitem(
        sys.modules, "usr.plugins._model_fallback.fallback", StaleFallback()
    )

    class FakeContext:
        id = "ctx"

        def __init__(self):
            self.log = FakeLog()

    class FakeAgent:
        context = FakeContext()

    ext = mod.InstallFallbackPatches(agent=FakeAgent())
    # Must not raise -- logs the "stale install" error and returns.
    ext.execute()

    # And a half-stale module (patch fns present, turn installer missing):
    class HalfStale:
        _patched_call_utility_model = fallback._patched_call_utility_model
        _patched_call_chat_model = fallback._patched_call_chat_model

    monkeypatch.setitem(
        sys.modules, "usr.plugins._model_fallback.fallback", HalfStale()
    )
    ext2 = mod.InstallFallbackPatches(agent=FakeAgent())
    ext2.execute()  # must not raise either