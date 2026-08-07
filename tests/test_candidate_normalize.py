"""Tests for fallback-candidate list normalization (added 2026-07-22).

Repro (from docker log 2026-07-22 21:04 onwards):
  Every utility-model fallback candidate [1]-[6] returned:
    BadRequestError: litellm.BadRequestError: LLM Provider NOT provided.
    You passed model=[{"model":"nvidia_nim/stepfun-ai/step-3.7-flash"}...
  The cascade spent 221s in a full cycle, slept 12s, and on the second
  cycle "recovered" -- but it never actually fell back to a working
  model; it just retried candidate 0 when its quota briefly refreshed.

Root cause:
  The user's preset / agent config passed a malformed `fallbacks` list
  where each entry was itself a list-of-dict (e.g. `[[{...}], [{...}]]`)
  or a list-wrapped dict (`[{"model": "..."}]`) or a dict whose `model`
  key held a non-string value. `_build_candidates` ingested these
  shapes verbatim (the list-comprehension `[x for x in fb if x]`
  preserved the extra wrapping), and downstream `str(spec)` was called
  in a log path that also fed into the litellm call.

Fix:
  New helper `_normalize_spec(spec)` in `fallback.py` flattens nested
  lists, unwraps single-element list wrappers, and unwraps dict-typed
  `model` values. `_build_candidates` runs every entry through
  `_normalize_spec` before adding it to the candidate list, so the
  downstream `build_fallback_wrapper` (in `models_ext.py`) only ever
  sees `str` or `dict` with a `str` `model` key.

These tests pin the new behavior so a future refactor can't silently
regress the cascade into the same stringified-list state.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(os.environ.get("REPO_ROOT_OVERRIDE") or "/a0")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# _normalize_spec: pure-function tests (no fixtures needed)
# ---------------------------------------------------------------------------


def test_normalize_passes_through_string():
    from usr.plugins._model_fallback.fallback import _normalize_spec
    assert _normalize_spec("nvidia_nim/x") == "nvidia_nim/x"


def test_normalize_passes_through_valid_dict():
    from usr.plugins._model_fallback.fallback import _normalize_spec
    spec = {"model": "nvidia_nim/x", "provider": "nvidia_nim"}
    out = _normalize_spec(spec)
    assert out == spec
    # Must not mutate the caller's dict.
    assert spec == {"model": "nvidia_nim/x", "provider": "nvidia_nim"}


def test_normalize_unwraps_single_element_list_of_dict():
    """`[{"model": "nvidia_nim/x"}]` -> `{"model": "nvidia_nim/x"}`.

    This is the exact shape that produced the docker-log BadRequestError.
    """
    from usr.plugins._model_fallback.fallback import _normalize_spec
    bad = [{"model": "nvidia_nim/x"}]
    out = _normalize_spec(bad)
    assert out == {"model": "nvidia_nim/x"}


def test_normalize_unwraps_single_element_list_of_string():
    from usr.plugins._model_fallback.fallback import _normalize_spec
    assert _normalize_spec(["nvidia_nim/x"]) == "nvidia_nim/x"


def test_normalize_flattens_double_nested_list():
    """`[[{...}], [{...}]]` -> 2 separate normalized specs (as a list).

    The outer wrapper is the ingestion bug: the list-comprehension in
    `_build_candidates` preserved `[[{...}], [{...}]]` as a 2-element
    list of single-element lists. `_normalize_spec` returns a flat
    list of well-formed specs; the caller (`_build_candidates`)
    extends the candidate list with them.
    """
    from usr.plugins._model_fallback.fallback import _normalize_spec
    bad = [[{"model": "a/x"}], [{"model": "b/y"}]]
    out = _normalize_spec(bad)
    assert out == [{"model": "a/x"}, {"model": "b/y"}]


def test_normalize_dict_with_dict_model_unwraps_name_key():
    """`{"model": {"name": "nvidia_nim/x"}}` -> `{"model": "nvidia_nim/x"}`.

    The unwrap tries the common inner keys in order: name, id,
    model_name, model.
    """
    from usr.plugins._model_fallback.fallback import _normalize_spec
    bad = {"model": {"name": "nvidia_nim/x", "extra": "ignored"}}
    out = _normalize_spec(bad)
    assert out == {"model": "nvidia_nim/x"}


def test_normalize_dict_with_list_model_takes_first_string():
    """`{"model": ["a/x", "b/y"]}` -> `{"model": "a/x"}`."""
    from usr.plugins._model_fallback.fallback import _normalize_spec
    bad = {"model": ["a/x", "b/y"]}
    out = _normalize_spec(bad)
    assert out == {"model": "a/x"}


def test_normalize_drops_none_model():
    from usr.plugins._model_fallback.fallback import _normalize_spec
    assert _normalize_spec({"model": None}) is None


def test_normalize_drops_empty_string():
    from usr.plugins._model_fallback.fallback import _normalize_spec
    assert _normalize_spec("") is None
    assert _normalize_spec("   ") is None


def test_normalize_drops_empty_list():
    from usr.plugins._model_fallback.fallback import _normalize_spec
    assert _normalize_spec([]) is None


def test_normalize_drops_bool_and_int():
    from usr.plugins._model_fallback.fallback import _normalize_spec
    assert _normalize_spec(True) is None
    assert _normalize_spec(42) is None


def test_normalize_dict_with_unparseable_inner_model():
    """`{"model": {"oops": "x"}}` -> None (no recognized inner key)."""
    from usr.plugins._model_fallback.fallback import _normalize_spec
    assert _normalize_spec({"model": {"oops": "x"}}) is None


# ---------------------------------------------------------------------------
# _build_candidates: end-to-end through every source path
# ---------------------------------------------------------------------------


class _FakePrimary:
    """Minimal stand-in for the real LiteLLM chat wrapper. Only the
    attributes that `_build_candidates` reads are populated."""
    def __init__(self, kwargs=None):
        self.kwargs = kwargs or {}


class _FakeAgent:
    def __init__(self, config=None):
        self.config = config or {}


def test_build_candidates_from_kwargs_flattens_nested_list():
    """The exact docker-log shape: kwargs['fallbacks'] = [[{...}], [{...}]].

    The cascade must end up with a non-list, parseable spec for each
    candidate -- not the stringified list that triggered the cascade of
    BadRequestErrors.
    """
    from usr.plugins._model_fallback.fallback import _build_candidates
    primary = _FakePrimary(kwargs={
        "fallbacks": [[{"model": "a/x"}], [{"model": "b/y"}]],
    })
    agent = _FakeAgent()
    cands = _build_candidates(primary, use_utility_models=True, agent=agent)
    # Index 0 is the primary (None); the rest must be normalized dicts,
    # never lists, never None.
    assert cands[0] is None
    for c in cands[1:]:
        assert not isinstance(c, list), (
            f"candidate still a list after normalization: {c!r}"
        )
        assert not isinstance(c, tuple)
        assert c is not None
        # Must be str or dict with str `model` key.
        if isinstance(c, dict):
            assert isinstance(c.get("model"), str), (
                f"dict candidate has non-string model: {c!r}"
            )


def test_build_candidates_from_agent_config_flattens_list_wrapper():
    from usr.plugins._model_fallback.fallback import _build_candidates
    primary = _FakePrimary(kwargs={})  # no fallbacks in kwargs
    agent = _FakeAgent(config={
        "utility_fallback_models": [{"model": "groq/llama"}, {"model": "venice/gemma"}],
    })
    cands = _build_candidates(primary, use_utility_models=True, agent=agent)
    assert cands[0] is None
    assert cands[1] == {"model": "groq/llama"}
    assert cands[2] == {"model": "venice/gemma"}


def test_build_candidates_from_env_comma_string_unchanged():
    """Comma-separated env var produces bare strings -- the path that
    was already working. The new normalization must not break it."""
    os.environ["A0_UTILITY_FALLBACK_MODELS"] = "a/x, b/y ,c/z"
    try:
        # Re-import to get a fresh module-level `os.environ` view.
        from importlib import reload
        from usr.plugins._model_fallback import fallback as fb_mod
        reload(fb_mod)
        primary = _FakePrimary(kwargs={})
        agent = _FakeAgent()
        cands = fb_mod._build_candidates(
            primary, use_utility_models=True, agent=agent
        )
        assert cands[1:] == ["a/x", "b/y", "c/z"]
    finally:
        del os.environ["A0_UTILITY_FALLBACK_MODELS"]


def test_build_candidates_drops_unparseable_entries_silently():
    """A mixed list with one good spec and one bad spec must return
    only the good one. Better a 1-candidate cascade than a cascade
    where 1/2 candidates fail with BadRequestError."""
    from usr.plugins._model_fallback.fallback import _build_candidates
    primary = _FakePrimary(kwargs={
        "fallbacks": [
            {"model": "good/x"},
            {"model": {"oops": "bad"}},  # unparseable
            "also_good/y",
        ],
    })
    agent = _FakeAgent()
    cands = _build_candidates(primary, use_utility_models=True, agent=agent)
    # All non-primary candidates must be valid str / dict-with-str-model.
    valid = [c for c in cands[1:]]
    assert {"model": "good/x"} in valid
    assert "also_good/y" in valid
    # And the bad one is NOT there.
    for c in valid:
        if isinstance(c, dict):
            assert isinstance(c.get("model"), str), (
                f"unparseable spec survived: {c!r}"
            )


def test_build_candidates_extends_double_nested_list():
    """The exact docker-log ingestion bug. When the user's preset
    accidentally double-nests (`[[{...}], [{...}]]` instead of
    `[{...}, {...}]`), the cascade must end up with a flat
    `[None, spec, spec]` -- not with a 2-element list of single-element
    lists that would then be stringified into
    `model=[{"model": "..."}]` by the litellm call."""
    from usr.plugins._model_fallback.fallback import _build_candidates
    primary = _FakePrimary(kwargs={
        "fallbacks": [[{"model": "a/x"}], [{"model": "b/y"}]],
    })
    agent = _FakeAgent()
    cands = _build_candidates(primary, use_utility_models=True, agent=agent)
    assert cands == [
        None,
        {"model": "a/x"},
        {"model": "b/y"},
    ]


# ---------------------------------------------------------------------------
# _coerce_to_list: JSON-encoded-string fallback list (added 2026-07-22)
# ---------------------------------------------------------------------------
#
# Repro (from docker log 2026-07-22 21:04:52):
#   User's preset YAML stores the fallback list as a single-quoted
#   string:
#     fallbacks: '[{"model":"nvidia_nim/..."},{"model":"a0_venice/..."},
#                  [{"model":"groq/..."},{"model":"ollama_cloud/..."}]]'
#   After YAML parsing the value is a Python `str` of length 268. The
#   previous ingestion code did `fb.split(",")`, splitting the JSON
#   string on commas INSIDE the JSON syntax (between dict keys and
#   inside the nested list) -- producing 6 fragment strings, each of
#   which is a partial JSON snippet like `[{"model":"nvidia_nim/...`.
#   `build_fallback_wrapper` then wrapped each as
#   `{"model": <fragment>}` and litellm received a model argument
#   that LOOKS like a JSON list, returning `BadRequestError: LLM
#   Provider NOT provided. You passed model=[{...}]`.
#
# Fix: new helper `_coerce_to_list` in fallback.py decodes a string
# value that starts with `[` or `{` as JSON, so the downstream
# list/dict code path in `_build_candidates` sees the user's
# intended list of dicts (which `_normalize_spec` then flattens and
# cleans). The 6-fragment smoke is gone, the cascade gets 5 clean
# candidates (nvidia_nim, a0_venice, groq, ollama, plus the primary).


def test_coerce_to_list_passes_through_non_string():
    """Non-string values (list, dict, None) flow through unchanged.
    Only strings are candidates for JSON-decoding; everything else
    should hit the list/dict branch in `_build_candidates` directly."""
    from usr.plugins._model_fallback.fallback import _coerce_to_list
    assert _coerce_to_list(None) is None
    assert _coerce_to_list([{"model": "x"}]) == [{"model": "x"}]
    assert _coerce_to_list({"model": "x"}) == {"model": "x"}
    assert _coerce_to_list(42) == 42


def test_coerce_to_list_passes_through_bare_string():
    """A bare comma-separated model list (env-var convention) must NOT
    be JSON-decoded. `'a, b, c'` is not a JSON list; decoding it would
    raise (or produce a string `'a'` if the bare value `'a'` parses as
    a JSON string). The helper must return the value as-is so the
    downstream comma-split still works."""
    from usr.plugins._model_fallback.fallback import _coerce_to_list
    assert _coerce_to_list("a, b, c") == "a, b, c"
    assert _coerce_to_list("a/x") == "a/x"
    assert _coerce_to_list("") == ""
    assert _coerce_to_list("   ") == "   "


def test_coerce_to_list_passes_through_malformed_json():
    """A string that starts with `[` but is not valid JSON must be
    returned unchanged (so the downstream comma-split path handles it
    as a fallback). Decode failure must NEVER raise."""
    from usr.plugins._model_fallback.fallback import _coerce_to_list
    assert _coerce_to_list("[not valid json") == "[not valid json"
    assert _coerce_to_list("[{model: x}]") == "[{model: x}]"  # unquoted key


def test_coerce_to_list_decodes_list_string():
    """`'[1, 2, 3]'` -> `[1, 2, 3]` (Python list)."""
    from usr.plugins._model_fallback.fallback import _coerce_to_list
    out = _coerce_to_list("[1, 2, 3]")
    assert out == [1, 2, 3]
    assert isinstance(out, list)


def test_coerce_to_list_decodes_dict_string():
    """`'{"k": "v"}'` -> `{"k": "v"}` (Python dict)."""
    from usr.plugins._model_fallback.fallback import _coerce_to_list
    out = _coerce_to_list('{"k": "v"}')
    assert out == {"k": "v"}
    assert isinstance(out, dict)


def test_coerce_to_list_strips_whitespace():
    """A string with leading/trailing whitespace is decoded normally
    (we `.strip()` before checking the first char)."""
    from usr.plugins._model_fallback.fallback import _coerce_to_list
    assert _coerce_to_list('  [{"model": "x"}]  ') == [{"model": "x"}]


def test_build_candidates_decodes_json_string_preset():
    """The exact docker-log user preset: a single-quoted YAML string
    that holds a JSON list. After coercion, `_build_candidates` should
    see 5 well-formed dict candidates (None + 4 cleaned dicts), NOT
    6 fragment strings."""
    from usr.plugins._model_fallback.fallback import _build_candidates
    user_preset = (
        '[{"model":"nvidia_nim/stepfun-ai/step-3.7-flash"},'
        '{"model":"a0_venice/google-gemma-4-26b-a4b-it"},'
        '[{"model": "groq/llama-3.3-70b-versatile",'
        '"api_base":"https://api.groq.com/openai/v1"},'
        '{"model":"ollama_cloud/nemotron-3-super:cloud",'
        '"api_base":"https://ollama.com/v1"}]]'
    )
    primary = _FakePrimary(kwargs={"fallbacks": user_preset})
    agent = _FakeAgent()
    cands = _build_candidates(primary, use_utility_models=True, agent=agent)
    # 4 dict candidates (the inner list was flattened by _normalize_spec)
    assert cands == [
        None,
        {"model": "nvidia_nim/stepfun-ai/step-3.7-flash"},
        {"model": "a0_venice/google-gemma-4-26b-a4b-it"},
        {
            "model": "groq/llama-3.3-70b-versatile",
            "api_base": "https://api.groq.com/openai/v1",
        },
        {
            "model": "ollama_cloud/nemotron-3-super:cloud",
            "api_base": "https://ollama.com/v1",
        },
    ]


def test_build_candidates_decodes_json_string_in_env():
    """The env-var path also goes through `_coerce_to_list`, so a
    `A0_UTILITY_FALLBACK_MODELS` env var that holds a JSON list is
    decoded the same way."""
    from importlib import reload
    from usr.plugins._model_fallback import fallback as fb_mod
    reload(fb_mod)
    os.environ["A0_UTILITY_FALLBACK_MODELS"] = (
        '[{"model":"a/x"},{"model":"b/y"}]'
    )
    try:
        primary = _FakePrimary(kwargs={})  # no kwargs
        agent = _FakeAgent()
        cands = fb_mod._build_candidates(
            primary, use_utility_models=True, agent=agent
        )
        assert cands == [
            None,
            {"model": "a/x"},
            {"model": "b/y"},
        ]
    finally:
        del os.environ["A0_UTILITY_FALLBACK_MODELS"]


def test_build_candidates_decodes_json_string_in_agent_config():
    """The agent-config path also goes through `_coerce_to_list`."""
    from usr.plugins._model_fallback.fallback import _build_candidates
    primary = _FakePrimary(kwargs={})  # no kwargs
    agent = _FakeAgent(config={
        "utility_fallback_models": '[{"model":"a/x"},{"model":"b/y"}]',
    })
    cands = _build_candidates(primary, use_utility_models=True, agent=agent)
    assert cands == [
        None,
        {"model": "a/x"},
        {"model": "b/y"},
    ]


def test_build_candidates_env_comma_string_still_works():
    """Backward-compat: the env-var comma-separated convention must
    still work. The new coercion must not break the path that was
    already working."""
    from importlib import reload
    from usr.plugins._model_fallback import fallback as fb_mod
    reload(fb_mod)
    os.environ["A0_UTILITY_FALLBACK_MODELS"] = "a/x, b/y ,c/z"
    try:
        primary = _FakePrimary(kwargs={})
        agent = _FakeAgent()
        cands = fb_mod._build_candidates(
            primary, use_utility_models=True, agent=agent
        )
        assert cands[1:] == ["a/x", "b/y", "c/z"]
    finally:
        del os.environ["A0_UTILITY_FALLBACK_MODELS"]
