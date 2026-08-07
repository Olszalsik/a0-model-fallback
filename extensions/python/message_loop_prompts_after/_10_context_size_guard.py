"""Trim runaway chat history at ``message_loop_prompts_after``.

WHY THIS EXISTS
---------------
A 2.6M-token request to every utility model in the cascade is the
fastest way to brick the agent: every model rejects with
``ContextOverflow``, the cascade cycles, the agent falls into
extended-retry mode, and the cascade is stuck in a 5-minute
``_yielding_sleep`` between cycles. The previous v2.2 patches
(memory_hardening, the memorize 50k char limit) trim the
persistence path but NOT the live ``history_output`` that
``prepare_prompt`` builds at agent.py:602.

WHAT THIS DOES
--------------
At the ``message_loop_prompts_after`` extension point (fires AFTER
``loop_data.history_output = self.history.output()`` at agent.py:575
and BEFORE the prompt is built at agent.py:602), inspect the total
character count of ``loop_data.history_output`` and, if it exceeds
``max_chars`` (default 50000), drop the oldest messages until the
total is within budget.

The trim is conservative: it only kicks in when the budget is
exceeded, never trims below 2 messages (so the LLM always sees the
last user turn + the agent's last reply), and preserves the most
recent N messages (the last user/agent turn is almost always the
relevant one).

OPT-IN
------
The hook is gated by the new ``context_size_guard`` config section
in ``default_config.yaml``. Default ``enabled: false`` because
aggressive trimming can confuse the LLM. When enabled, default
``max_chars: 50000`` and ``min_messages: 2``.

PLUGIN CONTRACT
---------------
- Plugin-only. ``agent.py`` and ``helpers/history.py`` are unchanged.
- Idempotent: a sentinel on the helper prevents the trim from
  firing twice in the same monologue tick.
- Never raises: every operation is wrapped in try/except. A bug in
  the trim never crashes the agent loop.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, TYPE_CHECKING

from helpers.extension import Extension

if TYPE_CHECKING:
    from agent import LoopData  # type: ignore

_log = logging.getLogger("model_fallback.context_trim")

DEFAULTS: Dict[str, Any] = {
    "enabled": False,         # opt-in. Aggressive trim can confuse the LLM.
    "max_chars": 50000,       # ~12.5k tokens, well within 32k context.
    "min_messages": 2,        # never trim below this; the LLM needs the last turn.
    "notice_text": (
        "[note: earlier messages in this conversation were trimmed to keep the "
        "request within the model's context window. Continue from the most recent "
        "turns.]"
    ),
}

_resolved: Dict[str, Any] = {}


def resolve_config(overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    cfg = dict(DEFAULTS)
    if overrides:
        for k, v in overrides.items():
            if v is not None:
                cfg[k] = v
    cfg["enabled"] = bool(cfg.get("enabled", False))
    cfg["max_chars"] = max(1000, int(cfg.get("max_chars") or 50000))
    cfg["min_messages"] = max(1, int(cfg.get("min_messages") or 2))
    return cfg


def set_resolved(cfg: Dict[str, Any]) -> None:
    global _resolved
    _resolved = dict(cfg)


def get_resolved() -> Dict[str, Any]:
    if not _resolved:
        _resolved = dict(DEFAULTS)
    return _resolved


# ---------------------------------------------------------------------------
# Trim logic (pure, testable)
# ---------------------------------------------------------------------------


def _msg_text(msg: Any) -> str:
    """Best-effort extract a string from an OutputMessage.

    OutputMessage is a TypedDict with ``content: MessageContent``.
    MessageContent is itself a dict like ``{"role": "user", "content": "..."}``
    or a plain string. We extract the actual text.
    """
    if msg is None:
        return ""
    content = msg.get("content") if hasattr(msg, "get") else None
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return str(content.get("content", "") or "")
    if isinstance(content, list):
        out = []
        for piece in content:
            if isinstance(piece, str):
                out.append(piece)
            elif isinstance(piece, dict):
                txt = piece.get("text") or piece.get("content") or ""
                if isinstance(txt, str):
                    out.append(txt)
        return "\n".join(out)
    return str(content)


def _msg_length(msg: Any) -> int:
    return len(_msg_text(msg))


def trim_history(
    history: List[Any],
    *,
    max_chars: int,
    min_messages: int,
    notice_text: str,
) -> tuple[List[Any], bool, int]:
    """Trim ``history`` to fit within ``max_chars`` chars.

    Returns ``(new_list, trimmed, dropped_count)``:
    - ``new_list`` is the trimmed list (or the original if no trim
      was needed). It is always in chronological order (oldest first,
      newest last).
    - ``trimmed`` is True iff any messages were dropped.
    - ``dropped_count`` is the number of messages removed from the
      front.

    Algorithm
    ---------
    1. If total chars (including the synthetic notice) fit in
       ``max_chars``, no trim is needed; return the original.
    2. Otherwise, start with the most recent ``min_messages`` (the
       LLM must see the last user/agent turn). Then walk backward
       from the second-most-recent, adding each older message while
       the running total + notice_len stays within ``max_chars``.
    3. The synthetic notice is prepended when a trim happened.
    4. The most recent message is always the last element of the
       returned list.
    """
    if not history:
        return history, False, 0
    notice_len = len(notice_text or "")
    # If we can't even fit the notice alone, the budget is too
    # tight to be useful; return the original rather than produce a
    # broken trim.
    if max_chars <= notice_len:
        return history, False, 0
    if len(history) <= min_messages:
        # Too few to trim; respect min_messages.
        return history, False, 0

    # Take the most recent ``min_messages`` as a guaranteed tail.
    # Walk backward from the element just before the tail, adding
    # older messages one at a time while the budget allows.
    # The invariant: ``kept`` is always in chronological order
    # (oldest first, newest last) so the LLM sees the conversation
    # in the order it happened.
    kept: List[Any] = list(history[-min_messages:])
    kept_len = sum(_msg_length(m) for m in kept)
    # Insert older messages at the FRONT of ``kept``, starting with
    # the second-newest and going to the oldest, while the budget
    # allows. We walk newest-to-oldest of the head because the most
    # recent earlier messages are more likely to be relevant to the
    # current turn than the oldest ones.
    head = list(history[:-min_messages])
    for msg in reversed(head):
        mlen = _msg_length(msg)
        if kept_len + mlen + notice_len > max_chars:
            break
        kept.insert(0, msg)
        kept_len += mlen

    if len(kept) == len(history):
        # No trim happened (everything fit). Return original.
        return history, False, 0

    notice_msg = {
        "ai": False,
        "content": notice_text,
        "metadata": {"_trim_notice": True},
    }
    new_list = [notice_msg] + kept
    dropped = len(history) - len(kept)
    return new_list, True, dropped


# ---------------------------------------------------------------------------
# Extension
# ---------------------------------------------------------------------------


class _Counter:
    """Per-process counters exposed by the stats endpoint."""

    def __init__(self):
        self.trims = 0
        self.messages_dropped = 0
        self.last_kept = 0
        self.last_dropped = 0

    def snapshot(self) -> Dict[str, Any]:
        return {
            "trims": self.trims,
            "messages_dropped": self.messages_dropped,
            "last_kept": self.last_kept,
            "last_dropped": self.last_dropped,
        }


_counter = _Counter()


def get_counter() -> _Counter:
    return _counter


def reset_counter() -> None:
    global _counter
    _counter = _Counter()


def _resolve_runtime_config(agent) -> Dict[str, Any]:
    try:
        from helpers import plugins as plugin_helpers  # type: ignore
        cfg = plugin_helpers.get_plugin_config("_model_fallback", agent) or {}
    except Exception:  # noqa: BLE001
        cfg = {}
    # v2.5 WebUI: top-level ``context_size_guard_enabled`` wins
    # over the nested section's ``enabled``. The piece defaults
    # to OFF, so an explicit True in either place turns it on.
    from usr.plugins._model_fallback.helpers import toggles
    if not toggles.resolve_toggle(cfg, "context_size_guard", default=False):
        return {"enabled": False}
    overrides = cfg.get("context_size_guard") if isinstance(cfg, dict) else None
    if not isinstance(overrides, dict):
        overrides = {}
    return resolve_config(overrides)


class ContextSizeGuard(Extension):
    def execute(self, loop_data=None, **kwargs: Any) -> None:
        # ``loop_data`` is the framework-injected LoopData instance.
        # We accept the type as Any at runtime to avoid an
        # unconditional ``from agent import LoopData`` at module
        # load (which would pull in models.py / sentence_transformers
        # in host envs that don't have them).
        try:
            cfg = _resolve_runtime_config(self.agent)
            if not cfg.get("enabled", False):
                return
            history = getattr(loop_data, "history_output", None) if loop_data else None
            if not history:
                return
            new_list, trimmed, dropped = trim_history(
                history,
                max_chars=int(cfg["max_chars"]),
                min_messages=int(cfg["min_messages"]),
                notice_text=str(cfg.get("notice_text") or DEFAULTS["notice_text"]),
            )
            if trimmed:
                loop_data.history_output = new_list
                _counter.trims += 1
                _counter.messages_dropped += dropped
                _counter.last_kept = len(new_list)
                _counter.last_dropped = dropped
                _log.info(
                    "context_size_guard: dropped %d oldest message(s) to fit under "
                    "%d char budget (kept %d)",
                    dropped, int(cfg["max_chars"]), len(new_list),
                )
        except Exception as exc:  # noqa: BLE001
            # A bug in the trim never crashes the agent loop.
            _log.debug("context_size_guard failed: %s", exc)
