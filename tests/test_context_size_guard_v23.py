"""Tests for the v2.3 context size guard (chat history trim)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(os.environ.get("REPO_ROOT_OVERRIDE") or "/a0")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import the helper directly so tests don't depend on the agent
# loader or the runtime ``agent`` module.
from usr.plugins.model_fallback.extensions.python.message_loop_prompts_after import (  # noqa: E402
    _10_context_size_guard as csg,
)


@pytest.fixture(autouse=True)
def _reset():
    csg.reset_counter()
    yield
    csg.reset_counter()


def test_resolve_config_defaults():
    cfg = csg.resolve_config()
    assert cfg["enabled"] is False
    assert cfg["max_chars"] == 50000
    assert cfg["min_messages"] == 2


def test_resolve_config_overrides():
    cfg = csg.resolve_config({
        "enabled": True, "max_chars": 10000, "min_messages": 3,
    })
    assert cfg["enabled"] is True
    assert cfg["max_chars"] == 10000
    assert cfg["min_messages"] == 3


def test_resolve_config_clamps_min():
    cfg = csg.resolve_config({"max_chars": 10, "min_messages": -5})
    assert cfg["max_chars"] == 1000  # min 1000
    assert cfg["min_messages"] == 1  # min 1


def test_trim_history_no_trim_under_budget():
    msgs = [
        {"ai": False, "content": "hi", "sequence": 1},
        {"ai": True, "content": "hello", "sequence": 2},
    ]
    out, trimmed, dropped = csg.trim_history(
        msgs, max_chars=1000, min_messages=2,
        notice_text="<notice>",
    )
    assert trimmed is False
    assert dropped == 0
    assert out is msgs  # no copy


def test_trim_history_trims_to_budget():
    """100 small messages summing to 100k chars: trim to 5k."""
    msgs = [
        {"ai": (i % 2 == 0), "content": "x" * 1000, "sequence": i}
        for i in range(100)
    ]
    out, trimmed, dropped = csg.trim_history(
        msgs, max_chars=5000, min_messages=2,
        notice_text="<trim-notice>",
    )
    assert trimmed is True
    # We dropped the oldest; kept the newest.
    assert dropped > 0
    # Kept at least min_messages plus the notice.
    assert len(out) >= 3  # notice + 2
    # The notice is the first message.
    assert out[0].get("metadata", {}).get("_trim_notice") is True
    # The most recent message is still the most recent.
    assert out[-1]["sequence"] == 99
    # Total chars (including notice) are now under budget.
    total = sum(csg._msg_length(m) for m in out)
    assert total <= 5000


def test_trim_history_respects_min_messages():
    """When every message is huge, we keep min_messages anyway."""
    msgs = [
        {"ai": True, "content": "x" * 1000, "sequence": i}
        for i in range(5)
    ]
    # Use a budget that fits exactly the notice + the last 2 messages.
    out, trimmed, dropped = csg.trim_history(
        msgs, max_chars=2014, min_messages=2,
        notice_text="<trim-notice>",  # 14 chars
    )
    # We trim to the notice + the 2 most recent.
    assert trimmed is True
    assert len(out) == 3  # notice + 2
    # The most recent message is the last element.
    assert out[-1]["sequence"] == 4
    # The second-most-recent is the second-to-last.
    assert out[-2]["sequence"] == 3


def test_trim_history_empty():
    out, trimmed, dropped = csg.trim_history(
        [], max_chars=100, min_messages=2, notice_text="<n>",
    )
    assert out == []
    assert trimmed is False


def test_trim_history_handles_string_content():
    msgs = [
        {"ai": True, "content": "x" * 100},
        {"ai": False, "content": "y" * 100},
    ]
    out, trimmed, dropped = csg.trim_history(
        msgs, max_chars=1000, min_messages=2, notice_text="<n>",
    )
    # Under budget; no trim.
    assert trimmed is False


def test_msg_text_handles_list_content():
    """Some OutputMessages have list content (multi-part)."""
    msg = {"content": [{"text": "hello"}, {"text": "world"}]}
    assert "hello" in csg._msg_text(msg)
    assert "world" in csg._msg_text(msg)


def test_counter_snapshot_shape():
    csg.reset_counter()
    snap = csg.get_counter().snapshot()
    for key in ("trims", "messages_dropped", "last_kept", "last_dropped"):
        assert key in snap
    assert snap["trims"] == 0


def test_context_size_guard_no_op_when_disabled():
    """The extension class is importable and the resolve_config
    helper returns the right shape when disabled. We don't invoke
    the extension here because that requires a real ``agent`` import.
    """
    from usr.plugins.model_fallback.extensions.python.message_loop_prompts_after._10_context_size_guard import (  # noqa: E501
        ContextSizeGuard,
    )
    cfg = csg.resolve_config({"enabled": False})
    assert cfg["enabled"] is False
    assert ContextSizeGuard is not None
