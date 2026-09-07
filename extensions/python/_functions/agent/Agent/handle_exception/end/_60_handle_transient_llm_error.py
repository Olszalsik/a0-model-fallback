"""Defense-in-depth: swallow transient LLM/provider errors so the agent
never dies to a 429 / 5xx / timeout on ANY call path.

Why this exists (v2.8.0): the v2.10/2.11 upstream merge moved the main
agent loop onto ``call_chat_model_turn`` -> ``unified_turn`` ->
``LiteLLMTransport.astream``, which had NO fallback coverage. A shared-pool
OpenRouter 429 (``upstream_429`` on z-ai/glm-5.2:free, Retry-After: 5) was
litellm-retried twice against the SAME model, then the RateLimitError
escaped to ``handle_exception`` with no handler -> HandledException ->
the agent stopped.

The primary fix is the turn-path fallback cascade in fallback.py
(``install_chat_turn_patch``). This extension is the LAST-RESORT net for
errors that still escape (mid-stream re-raises, patch-install failures,
paths not yet covered). It must therefore stay narrow:

  - Only transient provider errors: rate-limit phrases/429, 5xx status,
    connection errors, timeouts. Never code errors, never
    CancelledError, never HandledException.
  - Books the cooldown (shared store) so the next turn routes around the
    failing label even when the cascade couldn't record it.
  - Bounded: at most ``transient_swallow_max`` consecutive swallows per
    agent (plugin config; default 5), with the counter resetting after
    ``transient_swallow_reset_window_s`` (default 300s) of quiet --
    past that the exception is left for _90 so a genuinely broken setup
    still surfaces instead of looping forever. v3.1.1 made both bounds
    configurable so slow free-tier presets (NVIDIA NIM free, OpenRouter
    :free) can tolerate longer transient-error storms.

Ordering: runs BEFORE _70_handle_retry_after_hours (which owns the
RetryAfterHours contract) -- we explicitly skip RetryAfterHours and leave
it untouched for _70.
"""
import asyncio
import time

from helpers.extension import Extension
from helpers.print_style import PrintStyle

DATA_KEY_SWALLOW_COUNT = "_mfb_transient_swallow_count"
DATA_KEY_SWALLOW_AT = "_mfb_transient_swallow_at"
# v3.1.1: the bound is configurable via plugin config so slow free-tier
# presets can tolerate longer transient-error storms. The historic
# hard-coded values (5 / 300s) stay as the defaults -- a missing or
# unreadable knob preserves the original behavior exactly.
_DEFAULT_MAX_CONSECUTIVE_SWALLOWS = 5
_DEFAULT_SWALLOW_RESET_WINDOW_S = 300.0


def _swallow_limits(agent) -> tuple[int, float]:
    """(max_consecutive_swallows, reset_window_s) from plugin config.

    Read at exception time (not import time) so a WebUI config change
    takes effect on the next failure without a restart. Any read
    failure falls back to the historic defaults -- this net must never
    become the bug.
    """
    cfg: dict = {}
    try:
        from helpers import plugins as plugin_helpers

        raw = plugin_helpers.get_plugin_config("_model_fallback", agent)
        if isinstance(raw, dict):
            cfg = raw
    except Exception:
        cfg = {}
    try:
        max_swallows = int(
            cfg.get("transient_swallow_max", _DEFAULT_MAX_CONSECUTIVE_SWALLOWS)
        )
    except Exception:
        max_swallows = _DEFAULT_MAX_CONSECUTIVE_SWALLOWS
    if max_swallows < 1:
        max_swallows = _DEFAULT_MAX_CONSECUTIVE_SWALLOWS
    try:
        reset_window_s = float(
            cfg.get(
                "transient_swallow_reset_window_s",
                _DEFAULT_SWALLOW_RESET_WINDOW_S,
            )
        )
    except Exception:
        reset_window_s = _DEFAULT_SWALLOW_RESET_WINDOW_S
    if reset_window_s <= 0:
        reset_window_s = _DEFAULT_SWALLOW_RESET_WINDOW_S
    return max_swallows, reset_window_s


def _is_transient_llm_error(exc: Exception) -> bool:
    """True for provider-side transient failures worth an automatic retry."""
    if not isinstance(exc, Exception):
        return False  # excludes CancelledError (BaseException in 3.8+)

    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int) and (status_code == 429 or status_code >= 500):
        return True

    # litellm's connection/timeout exception classes (lazy import -- the
    # plugin must keep loading even if litellm import state is odd).
    try:
        import litellm.exceptions as _lexc

        if isinstance(
            exc,
            (
                _lexc.APIConnectionError,
                _lexc.Timeout,
                _lexc.ServiceUnavailableError,
                _lexc.InternalServerError,
            ),
        ):
            return True
    except Exception:
        pass

    # v2.8.5: a builtin TimeoutError / asyncio.TimeoutError (same class on
    # 3.11+) escaping the turn cascade's mid-stream re-raise carries no
    # status_code and a bare str(exc), so none of the checks above matched
    # and the error fell through to _90 -- ONE 20s warm-ceiling timeout on
    # a streaming turn stopped the whole run. The cascade already booked
    # its cooldown + evicted the warm label before re-raising, so treating
    # the shape as transient here is safe (this handler only sees LLM-call
    # failures anyway).
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return True

    # Shared detector from the plugin (phrase-based: "rate limit",
    # "too many requests", "upstream_429", ...).
    try:
        from usr.plugins._model_fallback.models_ext import _is_rate_limited_error

        if _is_rate_limited_error(exc):
            return True
    except Exception:
        pass

    return False


def _extract_model_label(exc: Exception) -> str:
    """Best-effort model label from the exception (litellm sets ``.model``)."""
    model = getattr(exc, "model", None)
    if isinstance(model, str) and model.strip():
        return model.strip()
    return ""


class HandleTransientLLMError(Extension):
    async def execute(self, data: dict = {}, **kwargs):
        if not self.agent:
            return

        exc = data.get("exception")
        if exc is None:
            return

        # Never touch the RetryAfterHours contract -- _70 owns it.
        try:
            from usr.plugins._model_fallback.fallback import RetryAfterHours

            if isinstance(exc, RetryAfterHours):
                return
        except ImportError:
            pass

        if not _is_transient_llm_error(exc):
            return

        # --- Book the cooldown so the next turn routes around this label ---
        try:
            from usr.plugins._model_fallback.fallback import (
                _get_cooldown_store,
                _save_cooldown_store,
                _handle_error_cooldown,
            )

            label = _extract_model_label(exc)
            if label:
                # No ``or {}``: a freshly seeded store is an empty (falsy)
                # dict; ``or {}`` would save an unregistered literal back
                # over the store and wipe the cooldown we just booked.
                store = _get_cooldown_store(self.agent)
                if isinstance(store, dict):
                    _handle_error_cooldown(exc, label, store, self.agent)
                    _save_cooldown_store(self.agent, store)
        except Exception:
            pass

        # --- Bounded swallow counter ----------------------------------------
        # v3.1.1: bounds come from plugin config (free-model presets can
        # raise both); historic defaults preserved on any read failure.
        max_swallows, reset_window_s = _swallow_limits(self.agent)
        now = time.monotonic()
        last_at = float(self.agent.get_data(DATA_KEY_SWALLOW_AT) or 0.0)
        count = int(self.agent.get_data(DATA_KEY_SWALLOW_COUNT) or 0)
        if now - last_at > reset_window_s:
            count = 0
        count += 1
        self.agent.set_data(DATA_KEY_SWALLOW_COUNT, count)
        self.agent.set_data(DATA_KEY_SWALLOW_AT, now)

        if count > max_swallows:
            # Give up quietly -- leave the exception so the critical handler
            # surfaces it (the agent stops rather than spinning silently).
            self.agent.context.log.log(
                "error",
                content=(
                    f"Transient LLM error swallowed {count} time(s) "
                    f"consecutively -- letting it propagate "
                    f"({type(exc).__name__})."
                ),
            )
            return

        message = (
            f"Transient LLM error ({type(exc).__name__}) -- the agent will "
            f"retry automatically ({count}/{max_swallows}). "
            f"Rate-limited or unreachable provider; rotating on the next turn."
        )
        PrintStyle(font_color="yellow", padding=True).print(message)
        try:
            self.agent.context.log.log("warning", content=message)
        except Exception:
            pass

        # Brief pause so a hard-429 label's Retry-After has a chance to
        # elapse before the monologue loop's next turn (bounded -- the
        # cooldown store does the real gating).
        try:
            await asyncio.sleep(3)
        except Exception:
            pass

        # Swallow -- the message loop continues and the turn is retried.
        data["exception"] = None