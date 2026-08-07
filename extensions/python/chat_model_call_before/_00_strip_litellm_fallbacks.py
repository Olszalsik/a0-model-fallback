"""Strip LiteLLM fallback keys and provider-invalid keys before unified_call (chat path).

This is the chat-model counterpart of util_model_call_before/_00_strip_litellm_fallbacks.py.
The core framework calls `chat_model_call_before` from agent.py:813 right before
invoking `unified_call`. Without this, keys like TIMEOUT, REQUEST_TIMEOUT,
retry_after, retry_backoff_factor, MAX_FALLBACK_CYCLES, etc. would leak into
acompletion() and the upstream provider (OpenAI / OpenRouter) returns HTTP 400
"Unrecognized key(s) in object".
"""
from helpers.extension import Extension


# A0-only / provider-invalid keys — must never reach acompletion().
# Keep this in sync with util_model_call_before/_00_strip_litellm_fallbacks.py.
_A0_ONLY_KWARGS = frozenset(
    {
        "TIMEOUT",
        "REQUEST_TIMEOUT",
        "FALLBACK_CYCLE_DELAY",
        "MAX_FALLBACK_CYCLES",
        "FALLBACK_ATTEMPT_DELAY",
        "FALLBACK_TIMEOUT_S",
        "FALLBACK_UTILITY_TIMEOUT_S",
        "retry_after",
        "retry_backoff_factor",
        "fallbacks",
        "fallback",
    }
)


class StripLitellmFallbacksChat(Extension):
    def execute(self, call_data: dict = {}, **kwargs):
        model = call_data.get("model")
        if model is None:
            return
        if hasattr(model, "kwargs") and isinstance(model.kwargs, dict):
            for k in _A0_ONLY_KWARGS:
                model.kwargs.pop(k, None)
        for k in _A0_ONLY_KWARGS:
            call_data.pop(k, None)
