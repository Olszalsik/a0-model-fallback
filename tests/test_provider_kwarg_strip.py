"""Tests for the provider-specific kwarg stripping added 2026-07-21.

When the cascade's primary model is a0_venice (or any local provider
with `a0_api_mode` in its YAML defaults), the parent's kwargs contain
nested provider-specific keys like `venice_parameters` or
`a0_api_mode`. The cascade's `build_fallback_wrapper` copies the
parent's kwargs into each fallback wrapper, so a Groq fallback would
get a `venice_parameters` kwarg it doesn't understand and reject
with 400. Both `_A0_ONLY_KWARGS` in `fallback.py` and the inline
strip loop in `build_fallback_wrapper` (`models_ext.py`) must include
the provider-specific keys.

These tests verify both lists strip the right keys. They don't
re-build a real wrapper -- they import the lists / re-execute the
strip logic in isolation.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(os.environ.get("REPO_ROOT_OVERRIDE") or "/a0")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_fallback_a0_only_kwargs_contains_venice_parameters():
    """The cascade's strip list must include venice_parameters."""
    from usr.plugins.model_fallback.fallback import _A0_ONLY_KWARGS
    assert "venice_parameters" in _A0_ONLY_KWARGS, (
        "venice_parameters is not in _A0_ONLY_KWARGS; a0_venice's "
        "kwargs will leak into Groq / other providers, causing 400"
    )
    assert "a0_api_mode" in _A0_ONLY_KWARGS, (
        "a0_api_mode is not in _A0_ONLY_KWARGS; local provider's "
        "kwargs will leak into cloud providers, causing 400"
    )


def test_models_ext_strip_loop_contains_venice_parameters():
    """build_fallback_wrapper's inline strip list must mirror fallback.py."""
    from usr.plugins.model_fallback import models_ext
    # Read the strip loop from models_ext.py source. We can't import
    # the constant because it's inline in a function, so we read the
    # source and assert the key string is present.
    src = Path(models_ext.__file__).read_text(encoding="utf-8")
    # Find the inline strip list (search for the comment block)
    marker = "Drop A0-only / provider-invalid keys so they never reach acompletion()"
    idx = src.find(marker)
    assert idx > 0, "Could not find the strip-list comment in models_ext.py"
    block = src[idx:idx + 2000]  # 2KB is plenty for the strip list
    assert '"venice_parameters"' in block, (
        "venice_parameters is not in models_ext.py's strip loop; the "
        "wrapper builder will not strip it from fallback wrappers"
    )
    assert '"a0_api_mode"' in block, (
        "a0_api_mode is not in models_ext.py's strip loop; the wrapper "
        "builder will not strip it from fallback wrappers"
    )


def test_strip_lists_are_in_sync():
    """Both strip lists must contain the same provider-specific keys.

    If one is updated and the other isn't, the bug re-appears in the
    other path. This test guards against the lists drifting.
    """
    from usr.plugins.model_fallback import models_ext
    from usr.plugins.model_fallback.fallback import _A0_ONLY_KWARGS

    # Provider-specific keys that must appear in BOTH lists
    required = ("venice_parameters", "a0_api_mode")
    for key in required:
        assert key in _A0_ONLY_KWARGS, f"{key} missing from _A0_ONLY_KWARGS"

    # Read the inline strip loop from models_ext.py
    src = Path(models_ext.__file__).read_text(encoding="utf-8")
    marker = "Drop A0-only / provider-invalid keys so they never reach acompletion()"
    idx = src.find(marker)
    block = src[idx:idx + 2000]
    for key in required:
        assert f'"{key}"' in block, f"{key} missing from models_ext.py strip loop"
