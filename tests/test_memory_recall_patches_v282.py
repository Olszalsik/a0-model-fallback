"""Tests for v2.8.2 — memory recall patches must bind the FRAMEWORK classes.

Regression under test: the framework loads extension files via
``helpers.modules.import_module`` as SYNTHETIC modules (file basename,
never in sys.modules). The old implementation imported the canonical
dotted paths (``plugins._memory.extensions...``), creating phantom
module + class objects the dispatcher never instantiates. All four
runtime patches were silent no-ops live:

- RecallWait.execute wrap   -> never installed (30s recall TimeoutError
  still killed the agent loop, 2026-08-27 crash)
- SEARCH_TIMEOUT = 90       -> never applied (framework module kept 30)
- MemorizeMemories/Solutions.wrap -> never installed

The fix resolves classes through ``helpers.extension._get_extension_classes``
-- the exact cached list the dispatcher iterates.

Test:
    cd <repo root>
    REPO_ROOT_OVERRIDE="$(pwd)" python -m pytest usr/plugins/model_fallback/tests/test_memory_recall_patches_v282.py -v
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
from pathlib import Path

REPO_ROOT = Path(os.environ.get("REPO_ROOT_OVERRIDE") or "/a0")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest


EXT_PATH = (
    Path(__file__).parent.parent
    / "extensions/python/monologue_start/_10_memory_recall_patches.py"
)


def _load_ext_module():
    spec = importlib.util.spec_from_file_location("mfb_memory_patches_v282", EXT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeLog:
    def log(self, *a, **kw):
        pass


class FakeContext:
    def __init__(self):
        self.log = FakeLog()


class FakeAgent:
    def __init__(self):
        self.context = FakeContext()


# ---------------------------------------------------------------------------
# Fakes mimicking the framework's synthetic modules
# ---------------------------------------------------------------------------


def _make_fake_extension_classes():
    """Build fake framework classes + module globals for the _memory plugin."""

    # _91_recall_wait.RecallWait -- execute raises TimeoutError (crash path)
    class RecallWait:
        pass

    async def recall_wait_execute(self, loop_data=None, **kwargs):
        raise asyncio.TimeoutError("30s budget")

    RecallWait.__module__ = "_91_recall_wait"
    RecallWait.execute = recall_wait_execute

    # _50_recall_memories.RecallMemories -- module globals reachable via
    # a method's __globals__ (synthetic modules are not in sys.modules).
    # exec() gives the fake its OWN globals dict, mirroring a real
    # synthetic module (a nested def's __globals__ would be the test
    # module's dict, leaking the _mfb_timeout_patched flag across tests).
    _fifty_globals = {"SEARCH_TIMEOUT": 30, "__name__": "_50_recall_memories"}
    exec(
        "def search_memories(self, *a, **kw):\n"
        "    return {}\n",
        _fifty_globals,
    )

    class RecallMemories:
        pass

    RecallMemories.search_memories = _fifty_globals["search_memories"]
    RecallMemories.__module__ = "_50_recall_memories"

    # monologue_end memorize classes
    class MemorizeMemories:
        pass

    class MemorizeSolutions:
        pass

    for cls in (MemorizeMemories, MemorizeSolutions):
        async def memorize(self, loop_data, log_item, **kwargs):
            return "memorized"

        cls.memorize = memorize
    MemorizeMemories.__module__ = "_50_memorize_fragments"
    MemorizeSolutions.__module__ = "_51_memorize_solutions"

    return {
        "message_loop_prompts_after": [RecallMemories, RecallWait],
        "monologue_end": [MemorizeMemories, MemorizeSolutions],
        "classes": {
            "RecallWait": RecallWait,
            "RecallMemories": RecallMemories,
            "MemorizeMemories": MemorizeMemories,
            "MemorizeSolutions": MemorizeSolutions,
        },
    }


@pytest.fixture()
def framework(monkeypatch):
    fakes = _make_fake_extension_classes()

    from helpers import extension as ext

    def fake_get_classes(extension_point, agent=None, **kwargs):
        return fakes.get(extension_point, [])

    monkeypatch.setattr(ext, "_get_extension_classes", fake_get_classes)
    return fakes


@pytest.fixture()
def ext_cls():
    mod = _load_ext_module()
    return mod.MemoryRecallPatches


def test_recall_wait_wraps_framework_class(framework, ext_cls):
    """The wrap must land on the framework's class, not a phantom."""
    RW = framework["classes"]["RecallWait"]
    inst = ext_cls(agent=FakeAgent())
    asyncio.run(inst.execute())

    assert getattr(RW, "_mfb_memory_patched", False) is True
    # wrapped, not replaced: the original is preserved
    assert getattr(RW.execute, "__wrapped__", None) is not None

    # a recall timeout must not escape the wrapper
    inst2 = RW()
    inst2.agent = FakeAgent()
    asyncio.run(inst2.execute())  # must not raise


def test_recall_wait_preserves_result_application(framework, ext_cls):
    """The wrapper must CALL the original (v2.11 upstream applies the
    recall result after await task -- replacing execute would drop that)."""
    RW = framework["classes"]["RecallWait"]
    called = {}

    async def good_execute(self, loop_data=None, **kwargs):
        called["ran"] = True
        return "result"

    RW.execute = good_execute
    RW._mfb_memory_patched = False

    inst = ext_cls(agent=FakeAgent())
    asyncio.run(inst.execute())
    inst2 = RW()
    inst2.agent = FakeAgent()
    assert asyncio.run(inst2.execute()) == "result"
    assert called["ran"] is True


def test_search_timeout_patches_framework_module_globals(framework, ext_cls):
    """SEARCH_TIMEOUT=90 must land in the framework module's globals dict
    (reached through a method's __globals__), not a phantom module."""
    inst = ext_cls(agent=FakeAgent())
    asyncio.run(inst.execute())
    sm = framework["classes"]["RecallMemories"].search_memories
    assert sm.__globals__["SEARCH_TIMEOUT"] == 90, sm.__globals__
    assert sm.__globals__["_mfb_timeout_patched"] is True


def test_memorize_classes_wrapped(framework, ext_cls):
    MM = framework["classes"]["MemorizeMemories"]
    MS = framework["classes"]["MemorizeSolutions"]
    inst = ext_cls(agent=FakeAgent())
    asyncio.run(inst.execute())
    assert getattr(MM, "mfb_memorize_patched", False) is True
    assert getattr(MS, "mfb_memorize_patched", True) is True


def test_idempotent_reapplication(framework, ext_cls):
    """Second execute() must not stack another wrapper layer."""
    RW = framework["classes"]["RecallWait"]
    inst = ext_cls(agent=FakeAgent())
    asyncio.run(inst.execute())
    first = RW.execute
    asyncio.run(inst.execute())
    assert RW.execute is first