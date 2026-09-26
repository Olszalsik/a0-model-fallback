"""Force chat-completions mode for selected providers/models (option B).

The newer agent-zero transport defaults to a0_api_mode=responses (LiteLLM
`aresponses` -> `<provider>/v1/responses`). Some OpenAI-compatible providers
return HTTP 500 from /v1/responses while their /v1/chat/completions endpoint
works fine (notably ollama.com for `:cloud` models). Injecting
`a0_api_mode=chat_completions` into the wrapper's kwargs makes the transport
call /v1/chat/completions instead.

This runs AFTER _00_strip_litellm_fallbacks (which cleans A0-only keys).
`a0_api_mode` is intentionally not in the strip list, so it survives and is
consumed by TransportPolicy._pop_mode.

Two triggers (see models_ext.should_force_chat_completions):
  - static (option B): provider in force_chat_completions_providers, or
    model name matches force_chat_completions_patterns.
  - dynamic (option D): model previously 5xx'd on Responses (sticky set).

Sibling of util_model_call_before/_01_force_chat_completions.py.
"""
from helpers.extension import Extension

from usr.plugins.model_fallback.models_ext import (
    should_force_chat_completions,
    force_chat_completions_mode,
)


class ForceChatCompletionsChat(Extension):
    def execute(self, call_data: dict = {}, **kwargs):
        model = call_data.get("model")
        if model is None:
            return
        if should_force_chat_completions(model, self.agent):
            force_chat_completions_mode(model)