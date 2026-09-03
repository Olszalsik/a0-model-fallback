"""Handle RetryAfterHours from the fallback plugin's extended retry mode.

Flow when all models fail and extended retry is enabled:
  1. _run_fallback_loop exhausts its attempt budget → raises RetryAfterHours
  2. call_chat_model propagates RetryAfterHours
  3. message_loop except block catches it → handle_exception("message_loop", e)
  4. This extension intercepts RetryAfterHours:
     a. Prints a user-friendly yellow message (not a red error)
     b. Adds the message to chat history once (first time only)
     c. Logs to internal log as warning
     d. Sleeps for the exception's retry_after (clamped [30, 7200]s;
        v2.8.5 -- was a hardcoded 60s that ignored the phase A/B delays)
     e. Swallows the exception (data["exception"] = None)
  5. Message loop exits cleanly, monologue loop continues
  6. Next iteration re-enters the cascade; cooled-down candidates are
     skipped cheaply. On renewed exhaustion it raises RetryAfterHours
     again and this handler sleeps/swallows once more.
  7. Steps 5-6 repeat until cooldown elapses
  8. When cooldown elapses: normal flow resumes, fresh attempt burst

When the user sends a new message after the cooldown:
  - Agent restarts, loads persisted extended-retry state
  - _run_fallback_loop resume check passes → fresh burst
"""
import asyncio

from helpers.extension import Extension
from helpers.print_style import PrintStyle


class HandleRetryAfterHours(Extension):
    async def execute(self, data: dict = {}, **kwargs):
        if not self.agent:
            return

        exc = data.get("exception")
        if exc is None:
            return

        # Lazy import so the plugin loads even if fallback.py has import errors
        try:
            from usr.plugins._model_fallback.fallback import RetryAfterHours
        except ImportError:
            return

        if not isinstance(exc, RetryAfterHours):
            return

        # --- This is a RetryAfterHours from extended retry mode ---
        message = str(exc)

        # Only add to chat history and print once (check if already notified)
        # v2.8.5: read the key via the fallback module's constant instead of
        # a literal -- the constant was renamed (non-underscore prefix so
        # persist_chat stops stripping it) and the old literal now reads a
        # key nothing writes.
        try:
            from usr.plugins._model_fallback import fallback as _fb

            notified_key = _fb.DATA_KEY_EXT_RETRY_NOTIFIED
        except ImportError:
            notified_key = "mfb_ext_retry_notified"
        already_notified = self.agent.get_data(notified_key) or False
        if not already_notified:
            PrintStyle(font_color="yellow", padding=True).print(message)

            # Add the message to chat history so the user sees it in the UI
            try:
                self.agent.hist_add_ai_response(message)
            except Exception:
                pass

            self.agent.set_data(notified_key, True)

        # Always log as warning
        try:
            self.agent.context.log.log(type="warning", content=message)
        except Exception:
            pass

        # Throttle: sleep before the next attempt to prevent tight looping.
        # v2.8.5: honor the retry_after the cascade computed instead of a
        # hardcoded 60s. The phase A (900s) / phase B (3600s) delays from
        # _maybe_raise_retry_after_hours and the turn cascade's
        # Retry-After-derived hint were calculated and then ignored here --
        # every exhaustion re-entered a full cascade pass ~60s later,
        # hammering exhausted providers. Clamped to [30, 7200]s so a bogus
        # header can't wedge the loop for hours; sliced so cancellation
        # (user intervention) is honored promptly.
        try:
            delay = float(getattr(exc, "retry_after", 0) or 0)
        except Exception:
            delay = 0.0
        if delay <= 0:
            delay = 60.0
        delay = max(30.0, min(delay, 7200.0))
        remaining = delay
        while remaining > 0:
            try:
                await asyncio.sleep(min(2.0, remaining))
            except Exception:
                pass
            remaining -= 2.0

        # Swallow the exception — exit this message loop iteration cleanly.
        # The monologue loop will continue to the next iteration, where
        # candidates in cooldown are skipped cheaply and the cascade either
        # recovers or re-raises RetryAfterHours (this handler sleeps again).
        data["exception"] = None
