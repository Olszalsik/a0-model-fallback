"""Tests for malformed-fallback-spec validation (added 2026-07-21).

Defensive: build_fallback_wrapper must raise a clear TypeError when
the spec's `model` key is not a string. Otherwise the cascade
constructs a wrapper whose `.model_name` is a list, and litellm
fails downstream with the confusing "LLM Provider NOT provided"
error -- which doesn't tell the user that their preset is wrong.

Repro (from docker log 2026-07-21):
  Every fallback candidate [1]-[6] returned:
    BadRequestError: litellm.BadRequestError: LLM Provider NOT provided.
    You passed model=[{"model":"nvidia_nim/..."}]
  The cascade silently accepted the list-valued `model` and built a
  wrapper whose `.model_name` was the list. The 30s timeout was
  hiding this; the 90s timeout (raised earlier on 2026-07-21) gave
  the cascade time to hit every candidate and surface the bug.

These tests guard the validation against future refactors that
might remove it, and confirm _get_model_label produces an
informative label (not a Python repr dump) for invalid specs.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(os.environ.get("REPO_ROOT_OVERRIDE") or "/a0")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_build_fallback_wrapper_rejects_list_model_value():
    """A spec whose `model` is a list must raise TypeError, not silently
    construct a wrapper that fails downstream with a confusing litellm
    BadRequestError."""
    import pytest
    from usr.plugins._model_fallback.models_ext import build_fallback_wrapper
    with pytest.raises(TypeError, match="must be a string"):
        build_fallback_wrapper(
            {"model": [{"model": "nvidia_nim/x"}]},
            {"model": "primary", "provider": "openai",
             "api_key": "x", "api_base": ""},
        )


def test_build_fallback_wrapper_rejects_dict_model_value():
    """A spec whose `model` is a dict must also raise TypeError. The
    cascade has no way to know which inner key is the model name in
    general, so any non-string shape is rejected."""
    import pytest
    from usr.plugins._model_fallback.models_ext import build_fallback_wrapper
    with pytest.raises(TypeError, match="must be a string"):
        build_fallback_wrapper(
            {"model": {"oops": "nested"}},
            {"model": "primary", "provider": "openai",
             "api_key": "x", "api_base": ""},
        )


def test_build_fallback_wrapper_accepts_string_model():
    """Sanity check: a properly-shaped spec still works. The new
    validation must not fire on a string-valued `model` key. The
    wrapper construction itself may fail (e.g. litellm not available
    in test env, or provider YAML missing); we only care that the
    TypeError validation doesn't fire."""
    from usr.plugins._model_fallback.models_ext import build_fallback_wrapper
    try:
        build_fallback_wrapper(
            {"model": "nvidia_nim/x"},
            {"model": "primary", "provider": "openai",
             "api_key": "x", "api_base": ""},
        )
    except TypeError as e:
        if "must be a string" in str(e):
            import pytest
            pytest.fail(
                "build_fallback_wrapper rejected a valid string spec "
                f"with the new validation: {e}"
            )
    except Exception:
        # Construction may fail for other reasons (e.g. litellm not
        # available in test env, provider YAML lookup missing). The
        # validation is the only thing we care about.
        pass


def test_get_model_label_handles_list_model():
    """_get_model_label must produce an informative label, not a Python
    repr dump, when spec['model'] is a list. The label is used in
    the per-attempt log line and the cooldown dict; an unparseable
    label would break cooldown dedupe (different malformed specs
    would each have a unique str() output)."""
    from usr.plugins._model_fallback.fallback import _get_model_label
    label = _get_model_label(
        {"model": [{"model": "nvidia_nim/x"}]},
        None,
    )
    assert "invalid spec" in label, f"label should be informative, got: {label!r}"
    assert "list" in label, f"label should mention the type, got: {label!r}"
    # Specifically must NOT contain the raw list repr.
    assert "[{" not in label, f"label leaked list repr: {label!r}"


def test_get_model_label_handles_dict_model():
    """Same as the list case, but for a dict-valued `model` key.
    Defensive: a different malformed shape produces a similarly
    clean label."""
    from usr.plugins._model_fallback.fallback import _get_model_label
    label = _get_model_label(
        {"model": {"oops": "nested"}},
        None,
    )
    assert "invalid spec" in label, f"label should be informative, got: {label!r}"
    assert "dict" in label, f"label should mention the type, got: {label!r}"
