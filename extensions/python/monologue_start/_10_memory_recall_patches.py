"""
Memory recall patches - merged from standalone memory_fix plugin.

Prevents the built-in _memory plugin's TimeoutError from crashing the agent loop.
Applied at monologue_start (BEFORE message_loop_prompts_after where the crash
path lives), so patches are in effect on every iteration. Self-healing: after
framework updates that overwrite /a0/plugins/_memory/, the next monologue
automatically re-applies all fixes.

Patches applied:
  1. RecallWait.execute: wraps `await task` in try/except to silently catch
     asyncio.TimeoutError, asyncio.CancelledError, and any other Exception --
     memory recall failure no longer crashes the agent loop.
  2. SEARCH_TIMEOUT: 30s -> 90s (in _50_recall_memories module).
  3. MAX_MSGS_CHARS: 80000 -> 50000 (in both _50_memorize_fragments and
     _51_memorize_solutions, applied to disk for next process AND wrapped at
     runtime via concat_messages for current session).
  4. config.json: creates safe defaults if missing (delayed recall on, shorter
     history length, consolidation off).
"""
import os
import asyncio
import json
import inspect

from helpers.extension import Extension
from agent import LoopData


# Safe defaults written to /a0/plugins/_memory/config.json when missing.
_DEFAULT_MEMORY_CONFIG = {
    "memory_recall_history_len": 4000,
    "memory_recall_delayed": True,
    "memory_memorize_consolidation": False,
}


def _find_framework_class(agent, extension_point, class_name, module_suffix):
    """Resolve the extension class the framework will ACTUALLY instantiate.

    A0 loads extension files via helpers.modules.import_module -- a
    synthetic module named after the file basename, never registered in
    sys.modules. A canonical dotted-path import therefore creates a
    phantom module + class that the dispatcher never calls: wrapping or
    patching it is a silent no-op (v0.5.2 / earlier versions of this file
    all had this bug live). The authoritative list is
    helpers.extension._get_extension_classes -- the same cache the
    dispatcher iterates.
    """
    try:
        from helpers import extension as _ext

        classes = _ext._get_extension_classes(extension_point, agent=agent)
        for cls in classes or []:
            if cls.__name__ != class_name:
                continue
            if str(getattr(cls, "__module__", "")).endswith(module_suffix):
                return cls
    except Exception:
        pass
    return None


class MemoryRecallPatches(Extension):
    async def execute(self, loop_data: LoopData = LoopData(), **kwargs):
        # Apply patches silently -- never let our fix crash the agent loop.
        try:
            self._patch_recall_wait()
        except Exception:
            pass
        try:
            self._patch_search_timeout()
        except Exception:
            pass
        try:
            self._patch_memorize_files()
        except Exception:
            pass
        try:
            self._patch_memorize_runtime()
        except Exception:
            pass
        try:
            self._ensure_config()
        except Exception:
            pass

    def _patch_recall_wait(self):
        """Wrap RecallWait.execute so a recall failure cannot crash the loop.

        WRAPS (does not replace) the upstream execute: upstream v2.11's
        _91_recall_wait applies the recall result after ``await task``, and
        replacing the method wholesale would silently drop that.
        """
        RecallWait = _find_framework_class(
            self.agent, "message_loop_prompts_after", "RecallWait", "_91_recall_wait"
        )
        if RecallWait is None:
            return

        if getattr(RecallWait, "_mfb_memory_patched", False):
            return

        original = RecallWait.execute

        async def safe_execute(self_recall, loop_data=None, **kwargs):
            if loop_data is None:
                loop_data = LoopData()
            try:
                return await original(self_recall, loop_data, **kwargs)
            except asyncio.CancelledError:
                # Shutdown cancellation must propagate (never swallow it);
                # the recall task itself cannot deliver one here because
                # upstream's 30s wait_for converts to TimeoutError first.
                raise
            except (asyncio.TimeoutError, Exception):
                # Memory recall is best-effort; a timeout/error must not
                # kill the agent loop.
                return

        safe_execute.__wrapped__ = original
        RecallWait.execute = safe_execute
        RecallWait._mfb_memory_patched = True

    def _patch_search_timeout(self):
        """Increase SEARCH_TIMEOUT from 30 to 90 seconds.

        The framework's _50_recall_memories module is synthetic (not in
        sys.modules), so we reach its globals dict through a method's
        ``__globals__`` on the resolved framework class.
        """
        RecallMemories = _find_framework_class(
            self.agent,
            "message_loop_prompts_after",
            "RecallMemories",
            "_50_recall_memories",
        )
        if RecallMemories is None:
            return

        module_globals = None
        for attr in list(vars(RecallMemories).values()):
            fn = getattr(attr, "__func__", attr)
            if inspect.isfunction(fn) and "SEARCH_TIMEOUT" in fn.__globals__:
                module_globals = fn.__globals__
                break
        if module_globals is None:
            return

        if module_globals.get("_mfb_timeout_patched", False):
            return

        module_globals["SEARCH_TIMEOUT"] = 90
        module_globals["_mfb_timeout_patched"] = True

    def _patch_memorize_files(self):
        """Patch MAX_MSGS_CHARS in memorize files on disk for next process."""
        files = [
            "/a0/plugins/_memory/extensions/python/monologue_end/_50_memorize_fragments.py",
            "/a0/plugins/_memory/extensions/python/monologue_end/_51_memorize_solutions.py",
        ]
        for f in files:
            try:
                with open(f, 'r') as fh:
                    content = fh.read()
                if 'MAX_MSGS_CHARS = 80000' in content:
                    content = content.replace('MAX_MSGS_CHARS = 80000', 'MAX_MSGS_CHARS = 50000')
                    with open(f, 'w') as fh:
                        fh.write(content)
            except Exception:
                pass

    def _patch_memorize_runtime(self):
        """Patch memorize methods at runtime to enforce 50k char limit on current session."""
        for point, name, suffix in (
            ("monologue_end", "MemorizeMemories", "_50_memorize_fragments"),
            ("monologue_end", "MemorizeSolutions", "_51_memorize_solutions"),
        ):
            cls = _find_framework_class(self.agent, point, name, suffix)
            if cls is None or getattr(cls, "_mfb_memorize_patched", False):
                continue

            original = cls.memorize

            def make_safe(orig):
                async def safe_memorize(self_mem, loop_data, log_item, **kwargs):
                    agent = self_mem.agent
                    if agent:
                        orig_concat = agent.concat_messages
                        def limited_concat(history):
                            text = orig_concat(history)
                            if len(text) > 50000:
                                text = text[-50000:]
                            return text
                        agent.concat_messages = limited_concat
                        try:
                            await orig(self_mem, loop_data, log_item, **kwargs)
                        finally:
                            agent.concat_messages = orig_concat
                    else:
                        await orig(self_mem, loop_data, log_item, **kwargs)
                return safe_memorize

            cls.memorize = make_safe(original)
            cls._mfb_memorize_patched = True

    def _ensure_config(self):
        """Ensure /a0/plugins/_memory/config.json exists with safe defaults."""
        config_path = "/a0/plugins/_memory/config.json"
        if os.path.exists(config_path):
            return
        try:
            with open(config_path, 'w') as fh:
                json.dump(_DEFAULT_MEMORY_CONFIG, fh, indent=4)
        except Exception:
            pass
