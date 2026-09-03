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

# v2.8.5 (AD): version-stamp the install guards. A bare ``True`` sentinel
# survives a plugin UPDATE: the wrapper (closure) from the old code stays
# live on the framework class for the rest of the process even though its
# logic is stale. Patches now also record _mfb_*_patch_version; a version
# mismatch unwraps the old wrapper (via the stored original) and re-applies.
_PATCH_VERSION = "2.8.5"


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


def _resolve_memory_cfg(agent) -> dict:
    """Resolve the plugin's memory_* knobs (wiring#6, v2.8.5).

    The YAML documented ``memory_recall_timeout_s`` /
    ``memory_memorize_max_chars`` / ``memory_recall_delayed`` /
    ``memory_memorize_consolidation`` but every consumer hardcoded its
    value, so editing the config did nothing. Merged config
    (default_config.yaml under config.json), same precedence as the
    other extensions.
    """
    try:
        from helpers import plugins as plugin_helpers  # type: ignore
        cfg = plugin_helpers.get_plugin_config("_model_fallback", agent) or {}
        try:
            defaults = plugin_helpers.get_default_plugin_config(
                "_model_fallback"
            ) or {}
            if isinstance(defaults, dict):
                merged = dict(defaults)
                if isinstance(cfg, dict):
                    merged.update(cfg)
                cfg = merged
        except Exception:  # noqa: BLE001
            pass
    except Exception:  # noqa: BLE001
        cfg = {}
    try:
        timeout_s = max(30.0, min(float(cfg.get("memory_recall_timeout_s") or 90), 600.0))
    except (TypeError, ValueError):
        timeout_s = 90.0
    try:
        max_chars = max(1000, min(int(cfg.get("memory_memorize_max_chars") or 50000), 500000))
    except (TypeError, ValueError):
        max_chars = 50000
    return {
        "recall_timeout_s": int(timeout_s),
        "memorize_max_chars": int(max_chars),
        "recall_delayed": bool(cfg.get("memory_recall_delayed", True)),
        "memorize_consolidation": bool(cfg.get("memory_memorize_consolidation", False)),
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
        # v2.8.5 (wiring#9): the disk-touching patches ran SYNCHRONOUSLY on
        # the event loop every monologue (two file reads + a stat + a
        # possible write). Offload them to a thread; self-healing (re-check
        # after a framework update overwrites the files) is preserved --
        # only the blocking wait moves off the loop.
        try:
            await asyncio.to_thread(self._patch_memorize_files)
        except Exception:
            pass
        try:
            self._patch_memorize_runtime()
        except Exception:
            pass
        try:
            await asyncio.to_thread(self._ensure_config)
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

        current = RecallWait.execute
        if getattr(current, "_mfb_safe_wrapper", False):
            # Ours (this or a previous plugin version): re-patch only when
            # the plugin version changed (AD).
            if getattr(current, "_mfb_patch_version", "") == _PATCH_VERSION:
                return
            original = getattr(current, "__wrapped__", None) or current
        else:
            if getattr(RecallWait, "_mfb_memory_patched", False):
                return
            original = current

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
        safe_execute._mfb_safe_wrapper = True
        safe_execute._mfb_patch_version = _PATCH_VERSION
        RecallWait.execute = safe_execute
        RecallWait._mfb_memory_patched = True
        RecallWait._mfb_patch_version = _PATCH_VERSION

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

        if module_globals.get("_mfb_timeout_patched", False) and (
            module_globals.get("_mfb_timeout_patch_version", "") == _PATCH_VERSION
        ):
            return

        module_globals["SEARCH_TIMEOUT"] = _resolve_memory_cfg(self.agent)["recall_timeout_s"]
        module_globals["_mfb_timeout_patched"] = True
        module_globals["_mfb_timeout_patch_version"] = _PATCH_VERSION

    def _patch_memorize_files(self):
        """Patch MAX_MSGS_CHARS in memorize files on disk for next process.

        v2.8.5 (wiring#9): the paths were hardcoded to /a0 (container) and
        silently no-oped anywhere else; derive them from this file's own
        location (usr/plugins/<this>/... -> framework root) with /a0 as
        the fallback. Runs in a worker thread (see execute)."""
        try:
            root = os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(
                    os.path.dirname(__file__)))))))
            # __file__ = <root>/usr/plugins/_model_fallback/extensions/python/
            # monologue_start/<this>.py -> 7 dirname() steps up = framework root
        except Exception:
            root = "/a0"
        base = os.path.join(root, "plugins", "_memory", "extensions", "python",
                            "monologue_end")
        files = [
            os.path.join(base, "_50_memorize_fragments.py"),
            os.path.join(base, "_51_memorize_solutions.py"),
        ]
        for f in files:
            try:
                # v2.8.5 (wiring#6): the 50000 target was hardcoded; use
                # the memory_memorize_max_chars knob.
                target = _resolve_memory_cfg(self.agent)["memorize_max_chars"]
                with open(f, 'r') as fh:
                    content = fh.read()
                if 'MAX_MSGS_CHARS = 80000' in content:
                    content = content.replace(
                        'MAX_MSGS_CHARS = 80000', f'MAX_MSGS_CHARS = {target}')
                    with open(f, 'w') as fh:
                        fh.write(content)
            except Exception:
                pass

    def _patch_memorize_runtime(self):
        """Patch memorize methods at runtime to enforce the
        memory_memorize_max_chars limit on the current session."""
        limit = _resolve_memory_cfg(self.agent)["memorize_max_chars"]
        for point, name, suffix in (
            ("monologue_end", "MemorizeMemories", "_50_memorize_fragments"),
            ("monologue_end", "MemorizeSolutions", "_51_memorize_solutions"),
        ):
            cls = _find_framework_class(self.agent, point, name, suffix)
            if cls is None:
                continue
            current = cls.memorize
            if getattr(cls, "_mfb_memorize_patched", False):
                # Ours (this or a previous plugin version): re-patch only
                # when the plugin version changed (AD).
                if getattr(cls, "_mfb_memorize_patch_version", "") == _PATCH_VERSION:
                    continue
                original = getattr(current, "_mfb_orig", None) or current
            else:
                original = current

            def make_safe(orig, _limit=limit):
                async def safe_memorize(self_mem, loop_data, log_item, **kwargs):
                    agent = self_mem.agent
                    if agent:
                        orig_concat = agent.concat_messages
                        def limited_concat(history):
                            text = orig_concat(history)
                            if len(text) > _limit:
                                text = text[-_limit:]
                            return text
                        agent.concat_messages = limited_concat
                        try:
                            await orig(self_mem, loop_data, log_item, **kwargs)
                        finally:
                            # v2.8.5 (wiring#10): restore only if nothing
                            # else swapped the attribute while we were
                            # awaiting. A blind restore would clobber a
                            # wrapper installed meanwhile (re-entrant or
                            # concurrent memorize on the same agent) and
                            # silently un-limit the session.
                            if getattr(agent, "concat_messages", None) is limited_concat:
                                agent.concat_messages = orig_concat
                    else:
                        await orig(self_mem, loop_data, log_item, **kwargs)
                return safe_memorize

            new_memorize = make_safe(original)
            new_memorize._mfb_orig = original
            cls.memorize = new_memorize
            cls._mfb_memorize_patched = True
            cls._mfb_memorize_patch_version = _PATCH_VERSION

    def _ensure_config(self):
        """Ensure the _memory plugin's config.json exists with safe defaults.

        v2.8.5 (wiring#9): path derived from this file's location (see
        _patch_memorize_files) instead of a hardcoded /a0; runs in a
        worker thread (see execute). v2.8.5 (wiring#6): the defaults now
        come from the plugin's memory_* knobs instead of a second
        hardcoded dict that could drift from the YAML."""
        try:
            root = os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(
                    os.path.dirname(__file__)))))))
        except Exception:
            root = "/a0"
        config_path = os.path.join(root, "plugins", "_memory", "config.json")
        if os.path.exists(config_path):
            return
        cfg = _resolve_memory_cfg(self.agent)
        defaults = dict(_DEFAULT_MEMORY_CONFIG)
        defaults["memory_recall_delayed"] = cfg["recall_delayed"]
        defaults["memory_memorize_consolidation"] = cfg["memorize_consolidation"]
        try:
            with open(config_path, 'w') as fh:
                json.dump(defaults, fh, indent=4)
        except Exception:
            pass
