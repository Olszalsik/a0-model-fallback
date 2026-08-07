"""Handle RetryAfterHours from the fallback plugin's extended retry mode.

Flow when all models fail and extended retry is enabled:
  1. _run_fallback_loop exhausts its attempt budget → raises RetryAfterHours
  2. call_chat_model propagates RetryAfterHours
  3. message_loop except block catches it → handle_exception("message_loop", e)
  4. This extension intercepts RetryAfterHours:
     a. Prints a user-friendly yellow message (not a red error)
     b. Adds the message to chat history once (first time only)
     c. Logs to internal log as warning
     d. Injects a 60-second sleep to prevent tight looping
     e. Swallows the exception (data["exception"] = None)
  5. Message loop exits cleanly, monologue loop continues
  6. Next iteration: call_chat_model → _run_fallback_loop resume check
     raises RetryAfterHours immediately (no API call). This extension
     fires again, sleeps 60s, swallows.
  7. Steps 5-6 repeat until cooldown elapses
  8. When cooldown elapses: normal flow resumes, fresh 60-attempt burst

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
        already_notified = self.agent.get_data("_ext_retry_notified") or False
        if not already_notified:
            PrintStyle(font_color="yellow", padding=True).print(message)

            # Add the message to chat history so the user sees it in the UI
            try:
                self.agent.hist_add_ai_response(message)
            except Exception:
                pass

            self.agent.set_data("_ext_retry_notified", True)

        # Always log as warning
        try:
            self.agent.context.log.log(type="warning", content=message)
        except Exception:
            pass

        # Throttle: sleep 60 seconds to prevent tight looping.
        # During extended retry cooldown, the monologue loop will keep
        # starting message loop iterations. Without this sleep, we'd
        # burn CPU in a tight loop.  With it, we get ~1 iteration/minute,
        # which is negligible overhead.
        try:
            await asyncio.sleep(60)
        except Exception:
            pass

        # Swallow the exception — exit this message loop iteration cleanly.
        # The monologue loop will continue to the next iteration, where
        # call_chat_model → _run_fallback_loop resume check will raise
        # RetryAfterHours immediately (no API call), and this handler fires
        # again. The 60s sleep prevents tight looping.
        data["exception"] = None
