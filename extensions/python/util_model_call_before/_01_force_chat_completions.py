"""Force chat-completions mode for selected providers/models (option B, util).

Utility-model counterpart of chat_model_call_before/_01_force_chat_completions.py.
See that file for the full rationale. The utility-model transport also defaults
to /v1/responses and hits the same ollama.com 500, so the same injection applies.
"""
from helpers.extension import Extension

from usr.plugins._model_fallback.models_ext import (
    should_force_chat_completions,
    force_chat_completions_mode,
)


class ForceChatCompletionsUtil(Extension):
    def execute(self, call_data: dict = {}, **kwargs):
        model = call_data.get("model")
        if model is None:
            return
        if should_force_chat_completions(model, self.agent):
            force_chat_completions_mode(model)