"""Strip LiteLLM fallback keys and provider-invalid keys before unified_call.

Why:
  - "fallbacks" / "fallback" would otherwise trigger LiteLLM's own fallback loop,
    conflicting with our custom cycling loop in this plugin.
  - A0-only / provider-invalid keys (TIMEOUT, REQUEST_TIMEOUT, retry_after,
    retry_backoff_factor, MAX_FALLBACK_CYCLES, FALLBACK_CYCLE_DELAY, ...) are
    read by THIS plugin locally and must never reach acompletion() — when they
    do, OpenAI / OpenRouter returns 400 "Unrecognized key(s) in object".

This extension mutates `model.kwargs` in place and also clears the same keys
from `call_data` so nothing slips through.
"""
from helpers.extension import Extension


# Keys that are A0/plugin-only and must not be passed to LiteLLM's acompletion().
# LiteLLM's OpenAI/OpenRouter adapter serializes unknown kwargs into the JSON
# request body, which the upstream API then rejects with HTTP 400.
_A0_ONLY_KWARGS = frozenset(
    {
        # Fallback plugin local config (read by fallback.py)
        "TIMEOUT",
        "REQUEST_TIMEOUT",
        "FALLBACK_CYCLE_DELAY",
        "MAX_FALLBACK_CYCLES",
        "FALLBACK_ATTEMPT_DELAY",
        "FALLBACK_TIMEOUT_S",
        "FALLBACK_UTILITY_TIMEOUT_S",
        "retry_after",
        "retry_backoff_factor",
        # LiteLLM-level keys that we don't want leaking into the request body
        "fallbacks",
        "fallback",
    }
)


def _strip(model, call_data: dict) -> None:
    if model is None:
        return
    if hasattr(model, "kwargs") and isinstance(model.kwargs, dict):
        for k in _A0_ONLY_KWARGS:
            model.kwargs.pop(k, None)
    for k in _A0_ONLY_KWARGS:
        call_data.pop(k, None)


class StripLitellmFallbacksUtil(Extension):
    def execute(self, call_data: dict = {}, **kwargs):
        model = call_data.get("model")
        _strip(model, call_data)
