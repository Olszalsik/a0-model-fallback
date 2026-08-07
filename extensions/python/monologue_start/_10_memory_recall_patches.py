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

from helpers.extension import Extension
from agent import LoopData


# Safe defaults written to /a0/plugins/_memory/config.json when missing.
_DEFAULT_MEMORY_CONFIG = {
    "memory_recall_history_len": 4000,
    "memory_recall_delayed": True,
    "memory_memorize_consolidation": False,
}


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
        """Wrap await task in try/except to prevent TimeoutError crash."""
        from plugins._memory.extensions.python.message_loop_prompts_after._91_recall_wait import RecallWait
        from plugins._memory.extensions.python.message_loop_prompts_after._50_recall_memories import (
            DATA_NAME_TASK as _TASK,
            DATA_NAME_ITER as _ITER,
        )
        from helpers import plugins

        if getattr(RecallWait, '_mfb_memory_patched', False):
            return

        async def safe_execute(self_recall, loop_data=LoopData(), **kwargs):
            if not self_recall.agent:
                return

            cfg = plugins.get_plugin_config("_memory", self_recall.agent)
            if not cfg:
                return None

            task = self_recall.agent.get_data(_TASK)
            iter_val = self_recall.agent.get_data(_ITER) or 0

            if task and not task.done():
                if cfg.get("memory_recall_delayed", False):
                    if iter_val == loop_data.iteration:
                        delay_text = self_recall.agent.read_prompt("memory.recall_delay_msg.md")
                        loop_data.extras_temporary["memory_recall_delayed"] = delay_text
                        return

                # CRITICAL FIX: catch exceptions to prevent agent loop crash.
                try:
                    await task
                except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                    pass

        RecallWait.execute = safe_execute
        RecallWait._mfb_memory_patched = True

    def _patch_search_timeout(self):
        """Increase SEARCH_TIMEOUT from 30 to 90 seconds."""
        from plugins._memory.extensions.python.message_loop_prompts_after import _50_recall_memories as rm

        if getattr(rm, '_mfb_timeout_patched', False):
            return

        rm.SEARCH_TIMEOUT = 90
        rm._mfb_timeout_patched = True

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
        try:
            from plugins._memory.extensions.python.monologue_end._50_memorize_fragments import MemorizeMemories
            from plugins._memory.extensions.python.monologue_end._51_memorize_solutions import MemorizeSolutions
        except ImportError:
            return

        for cls in (MemorizeMemories, MemorizeSolutions):
            if getattr(cls, '_mfb_memorize_patched', False):
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
