"""
Patched call_utility_model and call_chat_model implementations.

These are assigned onto the Agent class by the plugin's extension
(extensions/python/agent_init/_00_install_fallback_patches.py) so
they survive upstream updates that overwrite agent.py.

Fixes bundled here:
  #1  - DATA_NAME_*_IDX typo (was DATA_NAME_UTILITY_MODEL_IDX)
  #2  - Context overflow -> session cooldown instead of infinite retry
  #3  - Permanent failures (401/402/403/404 + 400-invalid-key) -> session cooldown
  #4  - Inter-model delay configurable via FALLBACK_ATTEMPT_DELAY kwarg
  #5  - DirtyJson.parse_string for JSON array + object parsing
  #6  - Invalid JSON triggers fallback rotation instead of immediate raise
  #7  - 402 (credit depletion) treated as permanent, not transient
  #8  - Double-prefix stripping in build_fallback_wrapper
  #9  - custom_llm_provider passed to LiteLLM acompletion()
  #10 - asyncio.CancelledError caught and treated like timeout
  #11 - ALL exceptions cycle to next fallback instead of crashing agent
  #12 - Cycle counting fixed (separate cycle_count, attempt never reset)
  #13 - Use real Agent API: read fallbacks from primary model.kwargs,
       store rotation index in agent.data with plugin-local keys
  #14 - RetryAfterHours class + extended retry (phase A / phase B)
  #15 - build_fallback_wrapper, extract_retry_after_seconds, resolve_callback
       implemented locally in models_ext.py (no longer rely on models.py)
  #16 - Code-level errors (ImportError, ModuleNotFoundError, etc.) detected
       and treated as permanent -- no pointless cycling
  #17 - 400 invalid-api-key errors detected and treated as permanent
  #18 - Sane upper bounds on cycle_delay (max 60s) and attempt_delay (max 10s)
       to prevent misconfigured model kwargs from causing excessive waits
  #19 - Pre-flight validation: code errors detected before entering the loop
       fail immediately with a clear error instead of cycling
"""

import asyncio
import time
from typing import Any, Callable, Awaitable, List, Tuple, Optional

from helpers import extract_tools, extension, plugins
from helpers.dirty_json import DirtyJson
from helpers.print_style import PrintStyle

# All model-related helpers are now local to the plugin (models_ext.py)
from usr.plugins._model_fallback.models_ext import (
    _is_code_error,
    _is_context_overflow_error,
    _is_invalid_api_key_400,
    _is_permanently_failed_model,
    _is_rate_limited_error,
    extract_retry_after_seconds,
    build_fallback_wrapper,
    resolve_callback,
    # Responses -> chat-completions fallback (options B + D, plugin form)
    should_force_chat_completions,
    force_chat_completions_mode,
    is_responses_server_error,
    mark_responses_5xx_seen,
)


# ---------------------------------------------------------------------------
# Plugin-local agent data keys (Agent class has no DATA_NAME_*_IDX constants)
# ---------------------------------------------------------------------------
#
# Cooldown storage: we keep TWO copies of the cooldown dict.
#   - DATA_KEY_COOLDOWNS is stored in agent.data (persisted to chat.json on
#     container restart / computer sleep). Used as a fallback when the
#     in-memory dict is empty (e.g. on the very first call after restart).
#   - _INMEM_COOLDOWNS is a module-level dict (in-memory only, reset on every
#     Python process restart). This is the authoritative copy and gets
#     rewritten to agent.data via set_data() on every change so the
#     framework can persist it as a backup.
#
# Why two copies: previously the cooldown lived only in agent.data, which
# meant a 24h cooldown survived computer sleep and Docker restarts. That is
# wrong for a laptop user -- if you sleep the laptop at 11pm and wake at 8am,
# you don't want the plugin to still remember that 11pm was a 429. So the
# in-memory dict resets on every run_ui restart (Docker container restart,
# or `supervisorctl restart run_ui`). The persisted copy is only used to
# bootstrap the in-memory dict on first call after a restart.
# ---------------------------------------------------------------------------
DATA_KEY_UTIL_IDX = "_mfb_utility_idx"
DATA_KEY_CHAT_IDX = "_mfb_chat_idx"
DATA_KEY_COOLDOWNS = "_model_cooldowns"
DATA_KEY_COOLDOWN_SEED_AT = "_mfb_cooldown_seed_at"
DATA_KEY_COOLDOWN_LAST_STATUS = "_mfb_cooldown_last_status"
DATA_KEY_EXT_RETRY_NOTIFIED = "_ext_retry_notified"
DATA_KEY_EXT_RETRY_PHASE = "_mfb_ext_retry_phase"   # 0 = A, 1 = B, 2 = off
DATA_KEY_EXT_RETRY_ATTEMPTS = "_mfb_ext_retry_attempts"

# In-memory cooldown store, keyed by (agent_id, model_label).
# Reset on every Python process restart -> resets on every run_ui bounce.
# The agent.data copy is only used to seed this on first use after a restart.
_INMEM_COOLDOWNS: dict = {}

# Persisted cooldowns older than this (in seconds, measured at process
# start) are discarded on the next restart. 10 minutes is enough to
# survive a quick `supervisorctl restart run_ui` while still letting a
# laptop user who slept the machine come back to a clean slate.
_COOLDOWN_SEED_MAX_AGE_S = 600.0

# Cooldown durations by HTTP status class.
# Rationale (2026-06-15 user feedback):
#   - 401/403 (auth/quota, e.g. "User has no quota left"): try again in 5 min.
#     Quota resets are usually hourly or daily, NOT a permanent fail. The
#     previous 24h wait was way too long for a daily-reset quota like a0_venice
#     or for an OpenRouter free-model daily allowance.
#   - 402 (payment required): treat as 1h. User may add credits.
#   - 404 (model not found / deprecated): 24h. Truly gone.
#   - 429 (rate limited): use Retry-After header if present, else 60s.
#   - Context overflow: 24h (the prompt won't shrink on its own).
#   - Other (transient, timeout, 5xx): 5 min.
_DEFAULT_COOLDOWNS_S = {
    401: 300.0,        # 5 min -- auth/quota, daily reset
    402: 3600.0,       # 1 h  -- payment required
    403: 300.0,        # 5 min -- quota/permission
    404: 86400.0,      # 24 h -- model gone
    408: 60.0,         # 1 min -- request timeout
    429: 60.0,         # 1 min default; Retry-After overrides
    500: 120.0,        # 2 min -- transient server error
    502: 60.0,         # 1 min -- bad gateway
    503: 60.0,         # 1 min -- service unavailable
    504: 60.0,         # 1 min -- gateway timeout
}

# When a rate-limit error has no Retry-After header (e.g. NVIDIA NIM's
# plain {"status":429,"title":"Too Many Requests"}), use a short backoff
# so free-tier endpoints that recover in ~30s are not locked out for
# minutes. Tunable via plugin config (rate_limit_no_retry_after_cooldown_s)
# so a user who hits a harder throttle can raise it back up.
#
# History: was 300s (5 min) from the original 2026-06 implementation;
# raised because we assumed upstream recovery was slow. Free-tier
# endpoints in practice recover in 20-40s, so 300s was over-locking
# healthy endpoints after a single shared-agent 429. Lowered to 30s
# (Laci, 2026-07-27, v2.5.2) and exposed as a knob.
_DEFAULT_RATE_LIMIT_NO_RETRY_AFTER_COOLDOWN_S = 30.0

# Module-level state for the cross-agent healthy-label reset (v2.5.2).
# Maps label -> monotonic deadline by which the next cascade iteration
# should treat the label as "recently proven healthy". Other agents'
# cooldowns for that label may be cleared (only-cleared, never
# overwritten) on the cascade's skip-cooldown path.
#
# Keyed by label (string), not (agent_id, label), because it's a
# process-global index of "what works". Reset on every Python process
# restart (same lifecycle as _INMEM_COOLDOWNS).
_INMEM_HEALTHY_LABELS: dict = {}

# Default horizon for the cross-agent healthy-label reset. Short enough
# that a stale "healthy" signal can't override a fresh "broken" cooldown
# from a different agent, but long enough to span at least one cascade
# cycle. Tunable via plugin config (health_horizon_s).
_DEFAULT_HEALTH_HORIZON_S = 60.0

# --- v2.6 warm/cold cascade timeouts --------------------------------------
# Per-label "last successful call" timestamp. Mirrors _INMEM_HEALTHY_LABELS
# but serves a different purpose: healthy labels trigger cross-agent
# cooldown *clears*, warm labels trigger *per-call timeout reduction*. The
# two are kept separate because a stale "warm" signal (a label was warm
# 30s ago) shouldn't clear another agent's fresh "broken" cooldown.
_WARM_LABELS: dict = {}

# Cold timeout is the legacy `fallback_utility_timeout_s` /
# `fallback_timeout_s` (typically 60-300s). Warm timeout is shorter —
# the connection / auth / DNS is already established. Tunable via
# `cascade_warm_timeout_s` (default 20s) and `cascade_warm_window_s`
# (default 600s — after this, treat the label as cold again).
_DEFAULT_CASCADE_WARM_TIMEOUT_S = 20.0
_DEFAULT_CASCADE_WARM_WINDOW_S = 600.0

# v2.6: Phase 3 — Adaptive cycle sleep on stagnation. When the cascade
# accumulates `cycle_stagnation_threshold` consecutive full cycles with
# zero successes (every candidate still in cooldown), we multiply the
# cycle sleep by `cycle_stagnation_factor`. This buys the upstreams
# more time to recover from quota exhaustion without the cascade
# spinning every cycle. The amplified sleep still respects the
# `max_cycle_delay_s` cap, and we emit one log line per stagnation
# transition (not per cycle) to avoid log spam.
_DEFAULT_CYCLE_STAGNATION_FACTOR = 1.5
_DEFAULT_CYCLE_STAGNATION_THRESHOLD = 2


def _resolve_per_call_timeout(
    label: str,
    base_timeout_s: float,
    warm_timeout_s: float,
    warm_window_s: float,
) -> float:
    """Return the per-candidate timeout for `label`. Warm labels get the
    shorter `warm_timeout_s`; cold labels (or those outside the warm
    window) get `base_timeout_s`. The user-set TIMEOUT= model kwarg is
    handled at the call site, not here — this helper is only invoked
    after the kwarg has already been checked (so a user kwarg wins
    regardless of warm state).
    """
    try:
        last_warm_at = _WARM_LABELS.get(label, 0.0)
        if time.monotonic() - last_warm_at < warm_window_s:
            return warm_timeout_s
    except Exception:
        # Defensive: if _WARM_LABELS access fails for any reason, fall
        # back to the legacy cold-timeout. Better to over-wait once than
        # to crash a cascade mid-rotation.
        pass
    return base_timeout_s


def _get_cooldown_store(agent) -> dict:
    """Return the in-memory cooldown dict for an agent, seeding from
    agent.data on first use after a process restart -- BUT only if the
    persisted snapshot is fresh.

    Subsequent mutations of the returned dict are NOT auto-persisted;
    callers must call _save_cooldown_store(agent, store) after changes.
    """
    agent_id = getattr(agent, "context", None)
    agent_id = getattr(agent_id, "id", None) or id(agent)
    key = ("__global__",) if agent_id is None else (str(agent_id),)
    store = _INMEM_COOLDOWNS.get(key)
    if store is None:
        # Seed from agent.data (persisted) ONLY if the snapshot is
        # recent enough to be useful. A laptop user who slept the
        # machine at 11pm and woke at 8am should NOT still be locked
        # out of models that were in cooldown at 11pm -- the upstream
        # has had nine hours to recover. We accept cooldowns younger
        # than _COOLDOWN_SEED_MAX_AGE_S so a quick
        # `supervisorctl restart run_ui` during a transient outage
        # still preserves the cooldown, but a long sleep resets it.
        seed = None
        try:
            seed = agent.get_data(DATA_KEY_COOLDOWNS)
        except Exception:
            seed = None
        seed_at = None
        try:
            seed_at = agent.get_data(DATA_KEY_COOLDOWN_SEED_AT)
        except Exception:
            seed_at = None
        if (
            isinstance(seed, dict)
            and seed
            and isinstance(seed_at, (int, float))
            and (time.time() - float(seed_at)) <= _COOLDOWN_SEED_MAX_AGE_S
        ):
            store = dict(seed)
        else:
            # Either there's no persisted snapshot, or it's stale.
            # Start clean. Persisted copy will be re-saved as soon as
            # we record our first new cooldown.
            store = {}
        _INMEM_COOLDOWNS[key] = store
    return store


def _save_cooldown_store(agent, store: dict) -> None:
    """Persist the in-memory cooldown dict to agent.data (as a backup) and
    to the in-memory store."""
    agent_id = getattr(agent, "context", None)
    agent_id = getattr(agent_id, "id", None) or id(agent)
    key = ("__global__",) if agent_id is None else (str(agent_id),)
    _INMEM_COOLDOWNS[key] = store
    try:
        agent.set_data(DATA_KEY_COOLDOWNS, dict(store))
        # Stamp the seed time so a future restart can tell whether
        # the persisted snapshot is fresh enough to trust.
        agent.set_data(DATA_KEY_COOLDOWN_SEED_AT, time.time())
    except Exception:
        pass  # best-effort persistence


def _cooldown_seconds_for_status(status_code, exc) -> float:
    """Return the cooldown duration for a given exception, in seconds.

    Priority:
      1. Retry-After header from the exception (respects server's own hint)
      2. Per-status default from _DEFAULT_COOLDOWNS_S
      3. 300.0 (5 min) -- conservative default for unknown errors
    """
    retry_after = extract_retry_after_seconds(exc)
    if retry_after and retry_after > 0:
        # Cap at 1 hour to avoid the previous 24h-bug if a server returns
        # an absurd Retry-After.
        return float(min(retry_after, 3600.0))
    if isinstance(status_code, int):
        return _DEFAULT_COOLDOWNS_S.get(status_code, 300.0)
    return 300.0


# Sane upper bounds to prevent misconfigured model kwargs from causing
# excessive waits (e.g. cycle_delay=120s, attempt_delay=30s).
_MAX_CYCLE_DELAY = 60.0
_MAX_ATTEMPT_DELAY = 10.0


# ---------------------------------------------------------------------------
# Exception types
# ---------------------------------------------------------------------------

class RetryAfterHours(Exception):
    """Raised when the fallback loop has exhausted its budget and the plugin
    is entering long-cooldown extended-retry mode.

    The handle_exception extension will swallow this, sleep, and let the
    monologue loop retry. After the cooldown elapses, the loop's resume check
    passes and a fresh burst of attempts is allowed.
    """

    def __init__(self, retry_after: float, message: str = ""):
        self.retry_after = float(retry_after)
        if not message:
            message = (
                f"All model candidates are currently unavailable. "
                f"Retrying in {int(self.retry_after)}s. "
                f"The agent will keep working in the background."
            )
        super().__init__(message)


class CodeError(Exception):
    """Raised when a code-level bug is detected before entering the fallback loop.

    ImportError, ModuleNotFoundError, and similar are bugs in the plugin or
    framework code -- they will never succeed on retry. The caller should fail
    fast instead of cycling through the entire candidate list.
    """

    def __init__(self, original_error: Exception, context: str = ""):
        self.original_error = original_error
        msg = (
            f"Code-level error detected in {context}: "
            f"{type(original_error).__name__}: {original_error}"
        )
        super().__init__(msg)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _resolve_model_idx(agent, use_utility_models: bool) -> int:
    """Get the current model rotation index, safely handling None."""
    key = DATA_KEY_UTIL_IDX if use_utility_models else DATA_KEY_CHAT_IDX
    return int(agent.get_data(key) or 0)


def _set_model_idx(agent, use_utility_models: bool, idx: int) -> None:
    key = DATA_KEY_UTIL_IDX if use_utility_models else DATA_KEY_CHAT_IDX
    agent.set_data(key, int(idx))


def _get_model_label(spec, model_obj) -> str:
    if spec is None:
        return getattr(model_obj, "model_name", "primary")
    if isinstance(spec, dict):
        model_val = spec.get("model", "unknown")
        if not isinstance(model_val, str):
            # Don't dump a Python repr of a list/dict into the log.
            # The real fix (TypeError) fires in build_fallback_wrapper;
            # by the time we hit a call, the cascade has already cycled
            # past this candidate. The label is for the cooldown dict
            # and the summary log -- keep it informative and parseable.
            # Added 2026-07-21 (Laci) when a hand-edited preset nested
            # a list under the `model` key and every cycle's log line
            # was a wall of repr.
            return f"<invalid spec: model is {type(model_val).__name__}>"
        return model_val
    return str(spec)


# ---------------------------------------------------------------------------
# A0-only / provider-invalid kwargs that must never reach LiteLLM's
# acompletion(). The fallback plugin reads some of these locally (e.g.
# FALLBACK_CYCLE_DELAY, MAX_FALLBACK_CYCLES, TIMEOUT) before they would
# otherwise be forwarded to LiteLLM, but LiteLLM's OpenAI/OpenRouter adapter
# then serializes the remaining unknown keys into the JSON request body and
# the upstream provider returns HTTP 400 "Unrecognized key(s) in object".
#
# _strip_a0_only_kwargs() mutates a wrapper's kwargs in place so the wrapper
# is safe to pass to unified_call(). Called from _build_model() so both the
# primary and every fallback wrapper end up clean.
# ---------------------------------------------------------------------------

_A0_ONLY_KWARGS = (
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
    # Some OpenAI-API-compatible providers (OpenRouter, NVIDIA NIM) choke when
    # LiteLLM forwards a stale/unknown "usage" parameter. The OpenAI SDK does
    # not accept "usage" as a top-level chat.completions argument; it expects
    # stream_options={"include_usage": True}. Stripping it here prevents the
    # "AsyncCompletions.create() got an unexpected keyword argument 'usage'" error.
    "usage",
    # Provider-specific nested kwargs injected by _merge_provider_defaults
    # (see conf/model_providers.yaml). These belong to one provider's API
    # contract and must NEVER be forwarded to a different provider's
    # upstream -- e.g. a0_venice's `venice_parameters` is rejected by Groq
    # with 400 "property 'venice_parameters' is unsupported" when it
    # leaks into a groq wrapper. The cascade's build_fallback_wrapper
    # copies parent kwargs into each fallback wrapper, so the primary's
    # provider-specific nested kwargs need to be stripped here.
    # Added 2026-07-21 (Laci).
    "venice_parameters",
    "a0_api_mode",
)


def _strip_a0_only_kwargs(model) -> None:
    """Mutate model.kwargs in place: pop every A0-only / provider-invalid key.

    Safe to call on the primary (which carries FALLBACK_CYCLE_DELAY etc. for
    the plugin to read locally) AFTER the plugin has finished reading them
    -- the order is: plugin reads at the top of the patched function, then
    this strip is called just before the first unified_call.
    """
    if model is None:
        return
    kwargs = getattr(model, "kwargs", None)
    if not isinstance(kwargs, dict):
        return
    for k in _A0_ONLY_KWARGS:
        kwargs.pop(k, None)


def _build_model(spec, model_obj):
    """Build a wrapper for a fallback spec, or return the primary as-is."""
    if spec is None:
        return model_obj
    parent_config = {
        "api_key": (model_obj.kwargs or {}).get("api_key", ""),
        "api_base": (model_obj.kwargs or {}).get("api_base", ""),
        "model": getattr(model_obj, "model_name", ""),
        "provider": getattr(model_obj, "provider", ""),
        "kwargs": getattr(model_obj, "kwargs", {}) or {},
    }
    return build_fallback_wrapper(spec, parent_config)


def _build_candidates(primary, use_utility_models: bool, agent) -> list:
    """Build the candidate list: [None (primary), fallback1, fallback2, ...].

    Source priority (first non-empty wins):
      1. primary model kwargs: {"fallbacks": [...]} or {"fallback": [...]}
      2. agent config: agent.config.get("fallback_models") or ("utility_fallback_models"
         if use_utility_models else "fallback_models")
      3. environment variables: A0_FALLBACK_MODELS / A0_UTILITY_FALLBACK_MODELS
         (comma-separated)
      4. plugin config: _model_fallback_plugin.config.fallbacks
    """
    import os

    fallbacks: list = []

    # 1. From primary model kwargs (most common)
    primary_kwargs = getattr(primary, "kwargs", {}) or {}
    if isinstance(primary_kwargs, dict):
        fb = _coerce_to_list(
            primary_kwargs.get("fallbacks")
        ) or _coerce_to_list(primary_kwargs.get("fallback"))
        if isinstance(fb, (list, tuple)):
            fallbacks = [x for x in fb if x]
        elif isinstance(fb, str) and fb.strip():
            fallbacks = [s.strip() for s in fb.split(",") if s.strip()]

    # 2. From agent config (multiple possible key names)
    if not fallbacks and hasattr(agent, "config") and isinstance(agent.config, dict):
        cfg_keys = (
            ("utility_fallback_models", "fallback_models")
            if use_utility_models
            else ("fallback_models", "chat_model_fallbacks")
        )
        for ckey in cfg_keys:
            cfg_fb = _coerce_to_list(agent.config.get(ckey))
            if isinstance(cfg_fb, (list, tuple)) and cfg_fb:
                fallbacks = [x for x in cfg_fb if x]
                break
            elif isinstance(cfg_fb, str) and cfg_fb.strip():
                fallbacks = [s.strip() for s in cfg_fb.split(",") if s.strip()]
                break

    # 3. From environment variables
    if not fallbacks:
        env_keys = (
            ("A0_UTILITY_FALLBACK_MODELS", "A0_FALLBACK_MODELS")
            if use_utility_models
            else ("A0_FALLBACK_MODELS",)
        )
        for ekey in env_keys:
            env_val = _coerce_to_list(os.environ.get(ekey, ""))
            if isinstance(env_val, (list, tuple)) and env_val:
                fallbacks = [x for x in env_val if x]
                break
            elif isinstance(env_val, str) and env_val.strip():
                fallbacks = [s.strip() for s in env_val.split(",") if s.strip()]
                break

    # 4. From plugin config (global fallback list)
    if not fallbacks:
        plugin_cfg = plugins.get_plugin_config("_model_fallback", agent)
        if isinstance(plugin_cfg, dict):
            cfg_fb = plugin_cfg.get("fallbacks")
            if isinstance(cfg_fb, (list, tuple)) and cfg_fb:
                fallbacks = [x for x in cfg_fb if x]

    normalized: list = []
    for raw in fallbacks:
        out = _normalize_spec(raw)
        if out is None:
            continue
        if isinstance(out, list):
            # A multi-element list of well-formed specs (recovered from
            # a double-nested input) -- extend the result instead of
            # nesting again.
            normalized.extend(out)
        else:
            normalized.append(out)
    return [None] + normalized


def _coerce_to_list(value):
    """If `value` is a JSON-encoded string starting with `[` or `{`, return
    the parsed result. Otherwise return `value` unchanged.

    Some user presets store the fallback list as a single-quoted YAML
    scalar (e.g. `fallbacks: '[{"model": "x"}, ...]'`). After YAML parsing
    that flows into `model_obj.kwargs["fallbacks"]` as a Python `str` of
    length 268 -- the JSON content has not been decoded yet. The naive
    handling of string values in `_build_candidates` is to split on `,`
    (matching the env-var convention), but that splits the JSON-syntax
    string at commas inside dict keys and inside nested list elements,
    producing 6 fragment strings that look like partial JSON snippets.
    Those fragments then reach `build_fallback_wrapper` as bare-string
    model names, get wrapped in `{"model": <fragment>}`, and pass to
    litellm as a model argument that LOOKS like a JSON list -- which
    litellm rejects with `LLM Provider NOT provided. You passed
    model=[{...}]`.

    The user's intent here is clearly a list of dicts (the YAML quoting
    is a common way to write multi-line JSON inside a single YAML
    scalar). Decode it on the ingestion boundary so the downstream
    list/dict code path produces a well-formed candidate list.

    Falls back to returning the original string when JSON parsing fails
    (so a non-JSON string still works as a comma-separated model list
    per the existing env-var convention).

    Added 2026-07-22 (Laci) after the docker log showed
    `model=[{"model":"nvidia_nim/stepfun-ai/step-3.7-flash"}]` reaching
    litellm because the user's preset stored the fallback list as a
    single-quoted YAML string.
    """
    if not isinstance(value, str):
        return value
    s = value.strip()
    if not s or s[0] not in ("[", "{"):
        return value
    try:
        import json as _json_mod
        parsed = _json_mod.loads(s)
    except Exception:
        return value
    return parsed


def _normalize_spec(spec):
    """Coerce a raw fallback-list entry to a `str` or `dict` (with str `model`).

    The candidate list can ingest user presets where the shape is wrong, e.g.:
      - `[[{...}], [{...}]]` (double-nested lists)
      - `[{"model": "..."}]` (single-element list wrapper)
      - `{"model": {"name": "..."}}` (dict-typed `model` value)

    These all reach `_build_model` -> `build_fallback_wrapper` (models_ext.py)
    which expects a bare string or a `{"model": "<str>"}` dict. The downstream
    guard at `models_ext.py:489` already raises TypeError on non-string
    `model`, but if the spec is itself a `list`, the wrapper at
    `models_ext.py:481-484` raises TypeError too -- except when the spec
    reaches a logging / label path first, where `str(spec)` happily produces
    `'[{"model": "..."}]'`, which then gets stringified into a log line
    AND (in some paths) forwarded to litellm as the `model=` argument,
    producing the
        BadRequestError: LLM Provider NOT provided.
        You passed model=[{"model":"nvidia_nim/..."}]
    error that we see in the docker log.

    This helper runs at the ingestion boundary (in `_build_candidates`) so
    every source (kwargs, agent config, env, plugin config) produces the
    same normalized shape. Unparseable entries are dropped (return None).
    For multi-element lists of well-formed specs (e.g. the result of
    recovering from a `[[{...}], [{...}]]` input), it returns a flat list
    of those specs -- `_build_candidates` then `.extend()`s the result
    so the cascade sees a flat `[None, spec, spec, ...]` as intended.

    Added 2026-07-22 (Laci) after the docker log showed every fallback
    candidate returning BadRequestError with a stringified list as the
    `model` argument to acompletion().
    """
    if spec is None:
        return None
    # Unwrap nested list / tuple: `[[{...}], [{...}]]`, `[{...}]`, `("a", "b")`
    if isinstance(spec, (list, tuple)):
        if len(spec) == 0:
            return None
        if len(spec) == 1:
            return _normalize_spec(spec[0])
        # Multi-element: each element must normalize to a str or dict.
        # If any fails, the whole entry is ambiguous (user almost
        # certainly mis-edited the preset), so drop it rather than
        # guessing.
        flat: list = []
        for x in spec:
            n = _normalize_spec(x)
            if n is None or isinstance(n, list):
                return None
            flat.append(n)
        return flat  # caller will extend(), not append()
    # Bare string is fine.
    if isinstance(spec, str):
        s = spec.strip()
        return s if s else None
    # Dict: must have a `model` key with a STRING value.
    if isinstance(spec, dict):
        m = spec.get("model")
        if isinstance(m, str):
            s = m.strip()
            if not s:
                return None
            # Preserve other keys (provider, api_key, api_base) but rebuild
            # the dict so we own the shape and don't mutate user input.
            out = {k: v for k, v in spec.items() if k != "model"}
            out["model"] = s
            return out
        if isinstance(m, dict):
            # `{"model": {"name": "x", ...}}` -- try common inner keys.
            for inner in ("name", "id", "model_name", "model"):
                inner_val = m.get(inner)
                if isinstance(inner_val, str) and inner_val.strip():
                    out = {k: v for k, v in spec.items() if k != "model"}
                    out["model"] = inner_val.strip()
                    return out
            return None
        if isinstance(m, (list, tuple)) and m:
            # `{"model": ["a", "b"]}` -- take the first string.
            for inner in m:
                if isinstance(inner, str) and inner.strip():
                    out = {k: v for k, v in spec.items() if k != "model"}
                    out["model"] = inner.strip()
                    return out
            return None
        # `model` is None / bool / number / etc. -- drop.
        return None
    # Unknown shape (bool, int, etc.) -- drop.
    return None


def _validate_json_response(response: str, call_data: dict):
    """Validate JSON if require_json is set. Uses DirtyJson for array+object support."""
    if call_data.get("require_json"):
        parsed = DirtyJson.parse_string(response.strip())
        if parsed is None:
            raise ValueError("Utility model output is not valid JSON")


def _record_last_status(agent, store: dict, label: str, status_code) -> None:
    """Side-channel: remember the most recent status_code for `label` so
    the early-exit / summary-log code can report a breakdown (e.g. 2x 401,
    1x 403 quota) without having to re-run the model.

    Stored in agent.data under DATA_KEY_COOLDOWN_LAST_STATUS, parallel to
    the cooldown dict. Best-effort -- any failure is swallowed.
    """
    try:
        existing = agent.get_data(DATA_KEY_COOLDOWN_LAST_STATUS) or {}
        if not isinstance(existing, dict):
            existing = {}
        existing[label] = status_code
        agent.set_data(DATA_KEY_COOLDOWN_LAST_STATUS, existing)
    except Exception:
        pass


def _get_last_status(agent) -> dict:
    """Read the parallel last-status dict (best-effort)."""
    try:
        v = agent.get_data(DATA_KEY_COOLDOWN_LAST_STATUS)
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}


def _mark_label_healthy(agent, label: str) -> None:
    """Record that `label` just succeeded for some agent.

    Called from every _succeed() in both cascades. Other agents'
    cooldowns for the same label are eligible to be cleared within
    `health_horizon_s` (default 60s, tunable via plugin config) by
    _maybe_clear_cooldown_for_healthy_label(). Only-cleared,
    never-overwritten: a fresh "broken" cooldown from a different
    agent always wins.

    Added 2026-07-27 (v2.5.2) to fix the multi-agent case where one
    agent's rate-limit cooldown locks out other agents that need the
    same endpoint right now.
    """
    try:
        horizon = float(
            _get_plugin_cfg(agent).get(
                "health_horizon_s", _DEFAULT_HEALTH_HORIZON_S
            )
        )
    except Exception:
        horizon = _DEFAULT_HEALTH_HORIZON_S
    horizon = max(5.0, min(horizon, 600.0))
    _INMEM_HEALTHY_LABELS[label] = time.monotonic() + horizon


def _maybe_clear_cooldown_for_healthy_label(agent, label: str) -> bool:
    """If `label` was marked healthy within `health_horizon_s`, and
    this agent's per-agent cooldown for `label` has already expired
    (stale entry), clear it.

    Returns True if a stale cooldown was cleared. Never shortens an
    active cooldown -- a fresh "broken" signal from any agent always
    wins over a stale "healthy" signal from a different one.

    Called from the skip-cooldown path in both cascades, right before
    the `if cooldown_until and cooldown_until > now: continue` branch,
    so the cascade gets the label back into rotation one tick sooner
    than waiting for the next cycle sleep.
    """
    healthy_until = _INMEM_HEALTHY_LABELS.get(label)
    if healthy_until is None or healthy_until < time.monotonic():
        return False
    store = _get_cooldown_store(agent)
    if label not in store:
        return False
    # Don't shorten an active cooldown. Only clear a stale entry.
    if store[label] > time.monotonic():
        return False
    store.pop(label, None)
    _save_cooldown_store(agent, store)
    return True


# v2.6: Per-provider capacity inference. The cascade uses this to decide
# whether a 429 deserves a cooldown at all (concurrent_paid: no, just
# skip), how long (free_per_minute / unlimited_paid: configured
# `rate_limit_no_retry_after_cooldown_s`, default 30s), and whether
# the primary-skip escalation should apply (concurrent_paid: never).
#
# Inference rules (locked 2026-07-28, see plan §Phase 2):
#   - ``ollama/*`` (exact local provider) → ``concurrent_paid``. Local
#     ollama on localhost genuinely accepts many parallel requests; a
#     429 here is a competing agent occupying a slot — release in seconds.
#   - ``ollama_*`` (e.g. ``ollama_cloud``) → ``free_per_minute``. The
#     ``ollama:cloud`` route is metered; 429 = real throttling.
#   - ``a0_venice/*`` → ``unlimited_paid``. Venice's daily credit window
#     is generous, but a quota-exhaustion 429 still means we're done for
#     now. Conservative 30s cooldown per the user's 2026-07-28 call.
#   - ``omniroute/*`` → ``router`` (added v2.6.2). OmniRoute is a
#     self-healing gateway that re-routes the next request to a healthy
#     upstream tier, so a 429/5xx does NOT mean the model is broken.
#     429 skips cooldown entirely; 5xx/transient gets a short
#     ``router_cooldown_s`` (default 5s). Primary-skip escalation is
#     skipped (see ``_capacity_skips_cooldown``).
#   - Anything else (including 3-segment labels like
#     ``nvidia_nim/stepfun-ai/step-3.7-flash``) → ``free_per_minute``.
#     Safe default: a 429 on an unknown provider means back off and
#     retry; better to over-cooldown than to hammer a metered endpoint.
#
# This is a pure function over the label string. No config block, no
# module-level state. Test: test_capacity_v26.py, test_router_capacity.py.
# ``router`` is added in v2.6.2; the cooldown-skip helper
# ``_capacity_skips_cooldown`` centralizes the "no cooldown / no
# primary-skip escalation" policy for ``concurrent_paid`` + ``router``.

# v2.6.2: providers whose failures do NOT indicate a broken model. A
# self-healing router (OmniRoute) re-routes the next request to a healthy
# upstream, so any real cooldown over-locks it. 429 skips cooldown
# entirely; 5xx/transient gets ``_DEFAULT_ROUTER_COOLDOWN_S``.
# (OpenRouter is NOT a router here: it serves a specific requested model
# and a 429 on ``openrouter/x:free`` is a real free-tier limit, so it
# stays ``free_per_minute``.)
_DEFAULT_ROUTER_PROVIDERS = ("omniroute",)
_DEFAULT_ROUTER_COOLDOWN_S = 5.0


def _classify_capacity(label: str) -> str:
    """Return one of ``concurrent_paid``, ``unlimited_paid``,
    ``router``, ``free_per_minute``. Unknown providers default to
    ``free_per_minute`` (conservative)."""
    provider = label.split("/", 1)[0].lower() if "/" in label else ""
    # Local ollama is truly concurrent (many parallel requests to
    # localhost are fine). Anything matching "ollama*" but not exactly
    # "ollama" (e.g. ollama_cloud) is metered.
    if provider == "ollama":
        return "concurrent_paid"
    if provider.startswith("ollama"):
        return "free_per_minute"
    if provider == "a0_venice":
        return "unlimited_paid"
    # v2.6.2: self-healing routers. OmniRoute fronts 230+ providers with
    # a 4-tier internal fallback (Sub -> Key -> Cheap -> Free). A 429/5xx
    # from the router usually means ONE upstream tier failed — the next
    # call re-routes, so don't lock it out.
    if provider in _DEFAULT_ROUTER_PROVIDERS:
        return "router"
    # nvidia_nim, groq, mistral, cohere, together_ai,
    # together, deepseek, anthropic, google, openai, etc. — all metered
    # or rate-limited at the per-minute granularity.
    return "free_per_minute"


def _capacity_skips_cooldown(label: str) -> bool:
    """True for capacity classes that should NOT get a 429 cooldown and
    should NOT trigger primary-skip escalation: ``concurrent_paid``
    (local ollama — a 429 is a competing agent) and ``router``
    (self-healing gateways — the next call re-routes)."""
    return _classify_capacity(label) in ("concurrent_paid", "router")


def _handle_error_cooldown(e, label, model_cooldowns, agent):
    """Apply cooldown logic to a failed model. Returns True if model was cooldowned.

    Cooldown duration policy (see _DEFAULT_COOLDOWNS_S for full table):
      - 401/403 (auth/quota, daily reset): 5 min
      - 402 (payment required): 1 h
      - 404 (model not found / deprecated): 24 h
      - 400 with `invalid_api_key` / `unauthorized` text: 24 h
        (key won't fix itself, but the user can fix it manually and
        call `clear_cooldowns`).
      - 429 (rate limited): Retry-After if present, else 60 s
      - 5xx / transient: 60-120 s
      - Context overflow: 24 h
      - Unknown: 5 min

    The previous 24-hour blanket cooldown for ALL permanent-looking errors
    was too aggressive: a0_venice's daily quota resets at midnight UTC, and
    OpenRouter's free-model daily allowance resets on its own schedule, so
    waiting 24h meant the user would be locked out until the next day even
    though the upstream was ready again. The new policy keeps the model in
    cooldown only for the duration the upstream actually needs.
    """
    store = _get_cooldown_store(agent)
    status_code = getattr(e, "status_code", None)

    if _is_context_overflow_error(e):
        # Permanent for this prompt; needs a model with bigger context.
        store[label] = time.monotonic() + 86400.0
        _record_last_status(agent, store, label, 400)  # synthetic: 400 overflow
        _save_cooldown_store(agent, store)
        return True

    # Rate-limit errors: detect even when LiteLLM wraps the underlying 429 in
    # InternalServerError / MidStreamFallbackError / APIConnectionError.  When
    # the provider gives no Retry-After header (NVIDIA NIM does this), use a
    # longer default cooldown so we don't retry every minute.
    if _is_rate_limited_error(e):
        if not isinstance(status_code, int):
            status_code = 429
        # v2.6: Phase 2 — per-provider capacity inference. For
        # ``concurrent_paid`` (local ollama), a 429 means a competing
        # agent is holding the slot; release in seconds. For ``router``
        # (omniroute/*), a 429 means one upstream tier
        # failed and the gateway re-routes the next call — also skip.
        # Don't poison the chat cascade or other utility callers with a
        # cooldown — just skip without writing anything to ``store``.
        # The ``_last_skip_log_until`` dedupe (see below) will still
        # rate-limit the "skipping" log line.
        if _capacity_skips_cooldown(label):
            return False
        retry_after = extract_retry_after_seconds(e)
        if retry_after and retry_after > 0:
            dur = float(min(retry_after, 3600.0))
        else:
            # Read the no-Retry-After cooldown from plugin config; fall
            # back to the conservative default. Configurable so the user
            # can raise it back up if a particular provider really does
            # need the longer wait.
            try:
                dur = float(
                    _get_plugin_cfg(agent).get(
                        "rate_limit_no_retry_after_cooldown_s",
                        _DEFAULT_RATE_LIMIT_NO_RETRY_AFTER_COOLDOWN_S,
                    )
                )
            except Exception:
                dur = _DEFAULT_RATE_LIMIT_NO_RETRY_AFTER_COOLDOWN_S
            # Clamp to a sane minimum (10s) so a misconfigured 0 doesn't
            # produce an instant-retry loop, and maximum (3600s) so a
            # typo doesn't lock out an endpoint for a day.
            dur = max(10.0, min(dur, 3600.0))
        store[label] = time.monotonic() + dur
        _record_last_status(agent, store, label, status_code)
        _save_cooldown_store(agent, store)
        try:
            agent.context.log.log(
                "info",
                f"Rate-limited on [{label}]. Cooldown {int(dur)}s.",
            )
        except Exception:
            pass
        return True

    if _is_permanently_failed_model(e):
        # 400-with-invalid-key is genuinely permanent (the user has to
        # change the key manually), so we keep the long cooldown. The
        # status-code-based 401/403/etc. path uses a 5-minute cooldown
        # because the upstream may auto-recover (quota resets, etc.).
        if status_code == 400 or _is_invalid_api_key_400(e):
            dur = 86400.0
        else:
            dur = _cooldown_seconds_for_status(status_code, e)
        store[label] = time.monotonic() + dur
        _record_last_status(agent, store, label, status_code)
        _save_cooldown_store(agent, store)
        if dur >= 3600.0:
            # Log only long cooldowns; the short ones are noise.
            try:
                agent.context.log.log(
                    "info",
                    f"Permanent fail on [{label}]. Cooldown {int(dur)}s.",
                )
            except Exception:
                pass
        return True

    # Transient / unknown error -> short cooldown so we don't hammer a flaky
    # endpoint but also don't lock it out for 24h.
    # v2.6.2: ``router`` labels (omniroute/*) re-route on the
    # next call, so use a tiny ``router_cooldown_s`` (default 5s) — bypassing
    # the 30s floor — to space out a 5xx storm without locking the router
    # out for the 60-120s a normal endpoint would get.
    if _classify_capacity(label) == "router":
        try:
            dur = float(
                _get_plugin_cfg(agent).get(
                    "router_cooldown_s", _DEFAULT_ROUTER_COOLDOWN_S,
                )
            )
        except Exception:
            dur = _DEFAULT_ROUTER_COOLDOWN_S
        dur = max(0.0, min(dur, 60.0))
        if dur > 0.0:
            store[label] = time.monotonic() + dur
            _record_last_status(agent, store, label, status_code)
            _save_cooldown_store(agent, store)
        return True
    dur = _cooldown_seconds_for_status(status_code, e)
    # Don't apply cooldowns shorter than 30s -- they'd be cleared by the
    # attempt_delay anyway, and writing them just adds IO overhead.
    if dur >= 30.0:
        store[label] = time.monotonic() + dur
        _record_last_status(agent, store, label, status_code)
        _save_cooldown_store(agent, store)
    return True


def _format_exception(exc: Exception) -> str:
    """Return a short, safe string representation of an exception."""
    name = type(exc).__name__
    msg = str(exc)[:200].replace("\n", " ")
    return f"{name}: {msg}"


def _get_plugin_cfg(agent) -> dict:
    """Safely load plugin config (handles None, bad types)."""
    cfg = plugins.get_plugin_config("_model_fallback", agent) or {}
    if not isinstance(cfg, dict):
        cfg = {}
    return cfg


def _clamp_delay(value: float, maximum: float, name: str) -> float:
    """Clamp a delay value to a sane maximum, warning if it was excessive."""
    if value > maximum:
        PrintStyle(font_color="orange", padding=True).print(
            f"[_model_fallback] {name}={value}s exceeds sane maximum, "
            f"clamped to {maximum}s. Check your model kwargs."
        )
        return maximum
    return value


# Slice size for the decomposed cycle sleep. Smaller = more frequent yields
# (better for keeping the WebSocket keepalive and other extension hooks
# running on the same event loop), but more `asyncio.sleep` round-trips.
# 2s is a good middle ground: the browser's default WebSocket reconnect
# window is 30-60s, so each 2s tick is invisible to the user.
_YIELDING_SLEEP_SLICE_S: float = 2.0


async def _yielding_sleep(total_seconds: float) -> None:
    """Sleep for ``total_seconds`` while yielding to the event loop frequently.

    The fallback cascade used to do ``await asyncio.sleep(sleep_s)`` once, with
    ``sleep_s`` climbing to 81s / 161s / 300s in continuous-fallback mode. The
    single ``asyncio.sleep`` already yields (it's not ``time.sleep``), but the
    long uninterrupted interval was a contributing factor in observed UI
    freezes during multi-minute utility-model outages:

    * The agent's monologue and the WebSocket dispatcher live on the same
      async loop, so a 161s uninterrupted sleep starves the WebSocket
      pings even though other tasks are technically eligible to run.
    * Some asyncio task schedulers (and the ``nest_asyncio`` patch Agent
      Zero uses in v2.5) can have surprising timing behaviour on
      long single sleeps.

    Decomposing into 2s slices keeps the loop ticking and gives the
    ``memory_hardening`` watchdog + the WebSocket dispatcher a fair
    share of every window. The first slice is shorter so a quick
    cancellation is honoured promptly; the remainder is uniform.

    If ``memory_hardening`` is enabled, hook the helper via
    ``memory_hardening.helpers.coroutine_guard``'s ``on_yield`` callback
    to record ticks. (Optional, no-op if the module is missing.)
    """
    if total_seconds <= 0:
        return
    # Honour cancellation promptly: first slice is at most 0.25s so an
    # outside caller that cancels while we're sleeping wakes up almost
    # immediately.
    first = min(0.25, total_seconds)
    await asyncio.sleep(first)
    remaining = total_seconds - first
    while remaining > 0:
        await asyncio.sleep(min(_YIELDING_SLEEP_SLICE_S, remaining))
        remaining -= _YIELDING_SLEEP_SLICE_S
        # Optional observability: if memory_hardening registered an
        # ``on_long_sleep_tick`` callback, fire it. Imported lazily so
        # the plugin doesn't take a hard dependency on memory_hardening.
        try:
            from usr.plugins.memory_hardening.helpers.coroutine_guard import (  # type: ignore
                on_long_sleep_tick,
            )
            on_long_sleep_tick(remaining)
        except Exception:
            pass





def _maybe_raise_retry_after_hours(agent, cycle_count: int, n: int, total_attempts: int) -> None:
    """Check extended-retry conditions and raise RetryAfterHours if appropriate.

    Behavior:
      - If `extended_retry_enabled` is false: do nothing (caller will raise RuntimeError).
      - If `continuous_fallback` is true: do nothing. The cascade runs forever
        so the agent survives multi-hour provider outages by cycling through
        candidates and waiting for quotas to refresh. See AGENTS.md.
      - If `max_cycles > 0 and cycle_count >= max_cycles`: enter extended retry
        using phase A or B delay, then raise RetryAfterHours.
      - State is persisted in agent.data so subsequent calls can resume.
    """
    cfg = _get_plugin_cfg(agent)
    if bool(cfg.get("continuous_fallback", False)):
        return
    if not cfg.get("extended_retry_enabled", True):
        return

    max_cycles = int(cfg.get("fallback_max_cycles", 4))
    if max_cycles <= 0 or cycle_count < max_cycles:
        return

    phase = int(agent.get_data(DATA_KEY_EXT_RETRY_PHASE) or 0)
    phase_a_delay = float(cfg.get("phase_a_delay_s", 900.0))
    phase_b_delay = float(cfg.get("phase_b_delay_s", 3600.0))

    # Phase A: 2 bursts of `initial_cycle_attempts`, then phase B
    # Phase B: bursts separated by `phase_b_delay`
    ext_attempts = int(agent.get_data(DATA_KEY_EXT_RETRY_ATTEMPTS) or 0)
    initial_burst = int(cfg.get("initial_cycle_attempts", 60))

    # First two phase A bursts, then switch to phase B for subsequent
    if phase == 0 and ext_attempts >= 2 * initial_burst:
        phase = 1
        agent.set_data(DATA_KEY_EXT_RETRY_PHASE, phase)
        ext_attempts = 0

    ext_attempts += 1
    agent.set_data(DATA_KEY_EXT_RETRY_ATTEMPTS, ext_attempts)

    if phase == 0:
        delay = phase_a_delay
    else:
        delay = phase_b_delay

    agent.set_data(DATA_KEY_EXT_RETRY_NOTIFIED, False)
    raise RetryAfterHours(retry_after=delay)


def _emit_fallback_summary(agent, kind: str, candidates: list) -> None:
    """Emit one deduplicated error-level summary of the current cooldown set.

    Called right before raising RetryAfterHours. Reads the parallel
    last-status dict (DATA_KEY_COOLDOWN_LAST_STATUS) to produce a
    breakdown like:
        "Utility fallbacks exhausted: 3 candidates, 3 permanent
         (2x 401 invalid key, 1x 403 quota), 0 transient."

    Best-effort -- any failure is swallowed.
    """
    try:
        statuses = _get_last_status(agent)
        # Restrict to labels we still know about (in case the user
        # reconfigured the candidate list after a failure).
        labels = set()
        for c in candidates:
            labels.add(_get_model_label(c, None) if c is not None else "primary")
        # Group: permanent 401, permanent 403, etc.
        permanent_buckets: dict = {}
        transient_count = 0
        unknown_count = 0
        for label in statuses:
            if label not in labels:
                continue
            sc = statuses[label]
            if isinstance(sc, int) and sc in (401, 402, 403, 404, 400):
                # Add a friendly name to the bucket key
                if sc == 400:
                    bucket = "400 invalid_key"
                elif sc == 401:
                    bucket = "401 invalid_key"
                elif sc == 402:
                    bucket = "402 payment"
                elif sc == 403:
                    bucket = "403 quota/perm"
                elif sc == 404:
                    bucket = "404 not_found"
                else:
                    bucket = f"{sc}"
                permanent_buckets[bucket] = permanent_buckets.get(bucket, 0) + 1
            elif sc is None:
                unknown_count += 1
            else:
                transient_count += 1
        breakdown_parts = [f"{v}x {k}" for k, v in sorted(permanent_buckets.items())]
        breakdown = ", ".join(breakdown_parts) if breakdown_parts else "0 permanent"
        msg = (
            f"{kind.capitalize()} fallbacks exhausted: {len(labels)} candidates, "
            f"{sum(permanent_buckets.values())} permanent ({breakdown}), "
            f"{transient_count} transient, {unknown_count} unknown."
        )
        try:
            agent.context.log.log("error", content=msg)
        except Exception:
            pass
        PrintStyle(font_color="red", padding=True).print(msg)
    except Exception:
        # Last-ditch: don't let the summary itself crash the loop.
        pass


# ---------------------------------------------------------------------------
# Pre-flight validation
# ---------------------------------------------------------------------------

def _preflight_check(model_obj, candidates: list, context: str) -> None:
    """Run a quick sanity check before entering the fallback loop.

    If a code-level error is detected (e.g. missing symbol in models.py),
    fail immediately with a clear error instead of cycling through every
    candidate and burning time.
    """
    # Try building the first non-primary candidate to catch import errors early
    if len(candidates) > 1 and candidates[1] is not None:
        try:
            _build_model(candidates[1], model_obj)
        except Exception as e:
            if _is_code_error(e):
                raise CodeError(e, f"{context} (building fallback candidate)")


# ---------------------------------------------------------------------------
# Patched call_utility_model
# ---------------------------------------------------------------------------

async def _patched_call_utility_model(
    self,
    system: str,
    message: str,
    callback: Callable | None = None,
    background: bool = False,
    require_json: bool = False,
):
    # Used by the v2.5.1 primary-skip helper to label its log lines.
    _is_utility_cascade: bool = True
    model_obj = self.get_utility_model()
    if model_obj is None:
        raise RuntimeError("Agent has no utility model configured")

    # Timeout resolution: plugin config > model kwarg > default (300s).
    # v2.6: the user-set TIMEOUT= model kwarg still wins (legacy
    # contract -- tests assume kwarg overrides everything). The warm/
    # cold per-label decision is applied per-candidate inside
    # _call_utility_model so each fallback candidate gets its own
    # warm-check based on its own label.
    plugin_cfg = _get_plugin_cfg(self)
    default_timeout = float(plugin_cfg.get("fallback_utility_timeout_s", 300))
    warm_timeout_s = float(plugin_cfg.get(
        "cascade_warm_timeout_s", _DEFAULT_CASCADE_WARM_TIMEOUT_S,
    ))
    warm_window_s = float(plugin_cfg.get(
        "cascade_warm_window_s", _DEFAULT_CASCADE_WARM_WINDOW_S,
    ))
    model_kwargs = getattr(model_obj, "kwargs", {}) or {}
    timeout_s: float = float(
        model_kwargs.get("TIMEOUT", model_kwargs.get("timeout", default_timeout))
    )

    candidates: list = _build_candidates(model_obj, use_utility_models=True, agent=self)
    n = len(candidates)
    if n == 0:
        raise RuntimeError("No utility model candidates available")

    # Pre-flight: catch code errors before entering the loop
    _preflight_check(model_obj, candidates, "utility model fallback")

    current_idx: int = _resolve_model_idx(self, use_utility_models=True) % n

    cycle_delay: float = _clamp_delay(
        float(
            model_kwargs.get(
                "FALLBACK_CYCLE_DELAY",
                model_kwargs.get("fallback_cycle_delay",
                                 float(plugin_cfg.get("fallback_cycle_delay", 5.0))),
            )
        ),
        _MAX_CYCLE_DELAY,
        "cycle_delay",
    )
    attempt_delay: float = _clamp_delay(
        float(
            model_kwargs.get(
                "FALLBACK_ATTEMPT_DELAY",
                model_kwargs.get("fallback_attempt_delay",
                                 float(plugin_cfg.get("fallback_attempt_delay", 2.0))),
            )
        ),
        _MAX_ATTEMPT_DELAY,
        "attempt_delay",
    )
    max_cycles: int = int(
        model_kwargs.get(
            "MAX_FALLBACK_CYCLES",
            model_kwargs.get("max_fallback_cycles",
                             int(plugin_cfg.get("fallback_max_cycles", 4))),
        )
    )

    model_cooldowns: dict = _get_cooldown_store(self) or {}
    if not isinstance(model_cooldowns, dict):
        model_cooldowns = {}

    cycle_count = 0
    attempt = 0
    tried_this_cycle = 0  # how many models we actually CALLed (not skipped)
    cycle_permanent_count = 0  # subset of tried_this_cycle that hit a permanent error
    early_exit_enabled: bool = bool(plugin_cfg.get("early_exit_on_all_permanent", True))
    responses_5xx_retry_enabled: bool = bool(
        plugin_cfg.get("responses_5xx_retry_enabled", True)
    )
    # Per-label dedupe of the "in cooldown, skipping (Xs)" log. Maps
    # label -> the cooldown_until value we last logged for it. When the
    # cascade iterates every ~2s through a candidate that's still in
    # cooldown, the same cooldown_until is reused, so we skip the
    # redundant log. A NEW cooldown_until (set after a fresh error)
    # produces one log line per transition, not one per iteration.
    # Without this, a 300s cooldown with n=1 candidate produced
    # ~150 countdown lines (300s -> 0s) per error cycle (Laci,
    # 2026-07-21).
    _last_skip_log_until: dict = {}

    # --- Continuous-fallback state -----------------------------------------
    # When `continuous_fallback` is true, the cascade runs forever (instead
    # of stopping at `max_cycles`) and uses an exponential backoff envelope
    # so that a multi-hour outage doesn't produce thousands of log lines.
    # Reset on every successful call so the cascade returns to its base
    # cadence the moment a quota refreshes.
    #
    # See AGENTS.md "Continuous-fallback mode" and "Backoff envelope" for
    # the full contract. Added 2026-07-19.
    continuous_mode: bool = bool(plugin_cfg.get("continuous_fallback", False))
    max_cycle_delay_s: float = _clamp_delay(
        float(plugin_cfg.get("max_cycle_delay_s", 300.0)),
        3600.0,
        "max_cycle_delay_s",
    )
    backoff_multiplier: float = max(1.0, float(plugin_cfg.get("backoff_multiplier", 2.0)))
    backoff_jitter_s: float = max(0.0, float(plugin_cfg.get("backoff_jitter_s", 2.0)))
    consecutive_full_cycles: int = 0
    fallback_started_at: float = time.monotonic()
    # v2.6: Phase 3 — stagnation counter (closure-local). Counts
    # consecutive full cycles with zero successes. Reset on any success.
    # Used by _compute_cycle_sleep to apply ``cycle_stagnation_factor``
    # when stagnation exceeds ``cycle_stagnation_threshold`` so the
    # cascade backs off further during sustained outages.
    consecutive_no_success_cycles: int = 0
    stagnation_logged_this_outage: bool = False
    cycle_stagnation_factor: float = max(
        1.0, float(plugin_cfg.get(
            "cycle_stagnation_factor", _DEFAULT_CYCLE_STAGNATION_FACTOR,
        ))
    )
    cycle_stagnation_threshold: int = max(
        1, int(plugin_cfg.get(
            "cycle_stagnation_threshold", _DEFAULT_CYCLE_STAGNATION_THRESHOLD,
        ))
    )

    # --- v2.5.1 Primary-skip-after-N-strikes (added 2026-07-23) --------------
    # When the primary model (candidate index 0) fails repeatedly, the
    # cascade used to keep retrying it every 5 min (the default transient
    # cooldown) even though the cost of each retry is 60-90s of timeout. The
    # log on 2026-07-23 22:50 showed "consecutive=2" in a 14+ min stall, all
    # attributed to a hung primary.
    #
    # Mechanism: track consecutive failures of [0] within a single
    # cascade invocation. When the count crosses ``primary_skip_strikes``,
    # extend [0]'s cooldown to ``primary_skip_cooldown_s`` (default 600s =
    # 10 min) so the cascade routes around the primary. The next time the
    # primary's cooldown expires, the cascade tries it again -- if the
    # primary is back, success and the counter resets; if it's still hung,
    # another escalation.
    #
    # The counter resets to 0 on:
    #   - A successful call from ANY candidate (cascade is healthy)
    #   - A successful call from the primary itself (quota recovered)
    primary_skip_enabled: bool = bool(plugin_cfg.get("primary_skip_enabled", True))
    primary_skip_strikes: int = max(
        1, int(plugin_cfg.get("primary_skip_strikes", 2))
    )
    # Default lowered from 600s (10 min) to 120s (2 min). 600s was
    # over-locking the primary when it had recovered in 2 min, which
    # is the common case for short-lived upstream hiccups. 120s still
    # covers one full pass through 6 candidates at 60s timeout each
    # (6 min) and then some, but routes around a hung primary quickly.
    # Tunable via config (primary_skip_cooldown_s).
    primary_skip_cooldown_s: float = max(
        30.0, float(plugin_cfg.get("primary_skip_cooldown_s", 120.0))
    )
    _consecutive_primary_failures: int = 0

    def _maybe_extend_primary_cooldown(reason: str) -> None:
        """If the primary (candidate 0) has failed ``primary_skip_strikes``
        times in a row, push its cooldown out to ``primary_skip_cooldown_s``
        so the cascade stops paying the 60s timeout tax on every cycle.

        Called after every primary failure. The dedupe in the cooldown
        table's skip-log means the new cooldown value produces exactly one
        log line per escalation, not one per cascade iteration.

        v2.6: if a peer agent has recently proven this label healthy
        (within ``health_horizon_s``), reset our local strike counter
        and any stale per-agent cooldown first. This prevents the case
        where one agent's stale primary failures keep escalating while
        another agent's success on the same label would have cleared
        the cooldown entirely.
        """
        nonlocal _consecutive_primary_failures
        if not primary_skip_enabled:
            return
        if idx != 0:
            return
        # v2.6: Phase 2 — per-provider capacity inference. For
        # ``concurrent_paid`` (local ollama) and ``router``
        # (omniroute/*), a primary 429 does not mean the
        # model is broken — the slot frees in seconds (ollama) or the
        # next call re-routes (routers). Never escalate the primary-skip
        # cooldown for these. Skip before the Phase 5 healthy-label
        # check so a peer success doesn't trigger a counter reset that
        # would later be wasted by a 30s escalation.
        if _capacity_skips_cooldown(label):
            return
        # v2.6: Phase 5 short-circuit. _maybe_clear_cooldown_for_healthy_label
        # already enforces the only-cleared-never-overwritten invariant
        # (returns True only if the local cooldown is stale AND the
        # cross-agent healthy index proves the label works). When it
        # returns True we reset our strike counter and return early --
        # no escalation needed, the peer agent proved this works.
        if _maybe_clear_cooldown_for_healthy_label(self, label):
            _consecutive_primary_failures = 0
            return
        _consecutive_primary_failures += 1
        if _consecutive_primary_failures < primary_skip_strikes:
            return
        # Escalate: write a fresh long cooldown for the primary label.
        # Bypasses _handle_error_cooldown (already called) -- we just
        # extend whatever it set.
        prev_until = model_cooldowns.get(label) or 0.0
        target_until = time.monotonic() + primary_skip_cooldown_s
        if target_until <= prev_until:
            return  # existing cooldown is already longer, don't shorten
        model_cooldowns[label] = target_until
        _save_cooldown_store(self, model_cooldowns)
        # ``use_utility_models`` is a parameter of the chat cascade (default
        # False) and is implicit True inside the utility cascade. We
        # disambiguate via the caller-supplied label so the helper stays
        # identical for both cascades.
        kind_label = "Utility" if _is_utility_cascade else "Chat"
        try:
            self.context.log.log(
                "warning",
                f"{kind_label} primary [{label}] has failed "
                f"{_consecutive_primary_failures} time(s) in a row -- "
                f"escalating cooldown to {int(primary_skip_cooldown_s)}s "
                f"(reason: {reason}). Cascade will route around it.",
            )
        except Exception:
            pass

    def _reset_primary_strikes() -> None:
        """Reset the primary-skip counter. Called on any successful call --
        either the primary recovered, or a fallback succeeded and proves
        the cascade is healthy (next primary attempt is no longer a
        'consecutive failure')."""
        nonlocal _consecutive_primary_failures
        if _consecutive_primary_failures > 0:
            _consecutive_primary_failures = 0

    def _compute_cycle_sleep() -> float:
        """Backoff envelope: cycle_delay * multiplier^consecutive_full_cycles,
        capped at max_cycle_delay_s, plus optional jitter to avoid lockstep
        retries when multiple agents share the same provider.

        v2.6: Phase 3 — when ``consecutive_no_success_cycles`` exceeds
        ``cycle_stagnation_threshold`` (every candidate still in
        cooldown), the cascade multiplies the capped sleep by
        ``cycle_stagnation_factor`` to give upstreams more time to
        recover. One log line per stagnation transition (gated by
        ``stagnation_logged_this_outage``), not per cycle. The
        ``max_cycle_delay_s`` cap is reapplied after amplification so we
        never sleep longer than the configured ceiling."""
        import random
        base = cycle_delay * (backoff_multiplier ** min(consecutive_full_cycles, 10))
        capped = min(base, max_cycle_delay_s)
        if (
            consecutive_no_success_cycles >= cycle_stagnation_threshold
            and cycle_stagnation_factor > 1.0
        ):
            amplified = capped * cycle_stagnation_factor
            capped = min(amplified, max_cycle_delay_s)
            if not stagnation_logged_this_outage:
                stagnation_logged_this_outage = True
                try:
                    self.context.log.log(
                        "info",
                        f"Cascade stagnation detected — "
                        f"{consecutive_no_success_cycles} consecutive cycles "
                        f"with 0 successes; amplifying cycle sleep by "
                        f"{cycle_stagnation_factor:.1f}x to {int(capped)}s.",
                    )
                except Exception:
                    pass
        if backoff_jitter_s > 0:
            capped += random.uniform(0.0, backoff_jitter_s)
        return capped

    while True:
        idx = (current_idx + attempt) % n
        label = _get_model_label(candidates[idx], model_obj)

        # Skip models in cooldown
        cooldown_until = model_cooldowns.get(label)
        # v2.5.2: cross-agent healthy-label reset. If another agent's
        # cascade just succeeded on this label within health_horizon_s,
        # clear any stale per-agent cooldown for it. Only fires when
        # the cooldown has already expired (so we never shorten an
        # active cooldown from another agent). See
        # _maybe_clear_cooldown_for_healthy_label.
        if cooldown_until is not None:
            _maybe_clear_cooldown_for_healthy_label(self, label)
            cooldown_until = model_cooldowns.get(label)
        now = time.monotonic()
        if cooldown_until and cooldown_until > now:
            remaining = int(cooldown_until - now)
            # Dedupe: only log when the cooldown value is NEW for this
            # label. A 5-min cooldown (the default for transient / timeout
            # errors via _handle_error_cooldown) would otherwise produce
            # ~150 countdown lines per error cycle, since the cascade
            # iterates every ~2s through the all-skipped path. When the
            # model errors again after this cooldown expires, a fresh
            # cooldown_until is recorded and the log fires once more.
            if _last_skip_log_until.get(label) != cooldown_until:
                _last_skip_log_until[label] = cooldown_until
                self.context.log.log(
                    "info",
                    f"Utility model [{label}] in cooldown, skipping ({remaining}s)",
                )
            # Advance idx so the NEXT iteration considers a different model,
            # but do NOT count this slot as "tried this cycle" -- otherwise
            # the cascade would treat "all models skipped" as "all models
            # attempted" and sleep 60s for no reason. The user explicitly
            # asked for cascade progress even when some models are in
            # cooldown (Laci, 2026-06-15).
            attempt += 1
            continue
        # Cooldown has expired for this label; clear the dedupe marker
        # so the NEXT cooldown window logs its first skip again.
        if _last_skip_log_until.get(label) is not None:
            _last_skip_log_until.pop(label, None)

        # Reset the "tried" counter at the start of each new pass through
        # the candidate list. A pass is complete when we've visited every
        # candidate index exactly once without hitting the `continue` above.
        if attempt > 0 and attempt % n == 0:
            if tried_this_cycle == 0:
                # Every model in this pass was skipped (all in cooldown).
                # Don't count it as a "cycle" -- just wait a bit and try
                # the same set again. The shortest cooldown among them
                # determines when the next real attempt can happen.
                await asyncio.sleep(2.0)
                attempt = 0
                tried_this_cycle = 0
                cycle_permanent_count = 0
                continue

            # Early-exit: if EVERY attempted model in this cycle was a
            # permanent failure, the upstream isn't going to recover in
            # the next 5 seconds -- skip the noisy cycle-and-sleep
            # ritual and go straight to extended-retry mode. This is
            # what the user sees as "agent-zero stopped working": an
            # all-dead candidate set spamming "switching to [N/2]" /
            # "cycling" lines for minutes instead of saying "all
            # models are out, waiting 15 min" once.
            if (
                early_exit_enabled
                and cycle_permanent_count > 0
                and cycle_permanent_count == tried_this_cycle
            ):
                cycle_count += 1
                _emit_fallback_summary(self, "utility", candidates)
                # Continuous mode: never raise RetryAfterHours; the cascade
                # must keep cycling so the agent stays alive across multi-hour
                # rate-limit windows. In legacy mode, raise if applicable.
                if not continuous_mode:
                    _maybe_raise_retry_after_hours(self, cycle_count, n, attempt)
                # If extended-retry is disabled, fall through to the
                # normal cycle log + sleep. In continuous mode, the sleep is
                # the backoff envelope, not the flat cycle_delay.
                consecutive_full_cycles += 1
                sleep_s = _compute_cycle_sleep() if continuous_mode else cycle_delay
                cycle_msg = (
                    f"Utility models cycling (all permanent). "
                    f"Sleeping {sleep_s:.0f}s "
                    f"(cycle {cycle_count}, consecutive={consecutive_full_cycles})..."
                )
                self.context.log.log("warning", content=cycle_msg)
                PrintStyle(font_color="orange", padding=True).print(cycle_msg)
                await _yielding_sleep(sleep_s)
                _set_model_idx(self, use_utility_models=True, idx=0)
                attempt = 0
                tried_this_cycle = 0
                cycle_permanent_count = 0
                continue

            cycle_count += 1
            # Continuous mode: skip the max_cycles cap. The cascade is
            # designed to run forever so a multi-hour provider outage does
            # not kill the agent. See AGENTS.md.
            if not continuous_mode and max_cycles > 0 and cycle_count >= max_cycles:
                # Try extended retry before giving up
                _maybe_raise_retry_after_hours(self, cycle_count, n, attempt)
                break
            consecutive_full_cycles += 1
            # v2.6: Phase 3 — stagnation counter increments on every
            # cycle iteration that didn't produce a success. _succeed
            # resets it. _compute_cycle_sleep applies the stagnation
            # amplification factor once it crosses the threshold.
            consecutive_no_success_cycles += 1
            sleep_s = _compute_cycle_sleep() if continuous_mode else cycle_delay
            cycle_msg = (
                f"Utility models cycling (all attempted). "
                f"Sleeping {sleep_s:.0f}s "
                f"(cycle {cycle_count}, consecutive={consecutive_full_cycles})..."
            )
            self.context.log.log("warning", content=cycle_msg)
            PrintStyle(font_color="orange", padding=True).print(cycle_msg)
            await _yielding_sleep(sleep_s)
            _set_model_idx(self, use_utility_models=True, idx=0)
            attempt = 0
            tried_this_cycle = 0
            cycle_permanent_count = 0
        elif attempt > 0:
            await _yielding_sleep(attempt_delay)

        tried_this_cycle += 1
        current_model = _build_model(candidates[idx], model_obj)

        # Strip A0-only / provider-invalid keys from the wrapper's kwargs right
        # before the first acompletion() so they never reach the request body.
        # The plugin has already read cycle_delay / max_cycles / timeout above.
        _strip_a0_only_kwargs(current_model)

        if attempt > 0:
            warn = f"Utility model switching to [{idx}/{n - 1}]: {label}"
            self.context.log.log("warning", content=warn)
            PrintStyle(font_color="orange", padding=True).print(warn)

        call_data = {
            "model": current_model,
            "system": system,
            "message": message,
            "callback": callback,
            "background": background,
            "require_json": require_json,
        }
        await extension.call_extensions_async(
            "util_model_call_before", self, call_data=call_data
        )

        async def stream_callback(chunk: str, total: str):
            if call_data["callback"]:
                await call_data["callback"](chunk)

        async def _succeed(response):
            # Shared success path: validate JSON, clear cooldown, advance
            # rotation index, clear extended-retry state, emit after-hook.
            _validate_json_response(response, call_data)
            # Success -- clear cooldown for this model
            model_cooldowns.pop(label, None)
            _save_cooldown_store(self, model_cooldowns)
            # v2.5.2: cross-agent healthy-label reset. Mark this label
            # healthy for `health_horizon_s` so other agents' cooldowns
            # for the same label can be cleared on their next cascade
            # iteration. See _mark_label_healthy / _maybe_clear_cooldown_for_healthy_label.
            _mark_label_healthy(self, label)
            # v2.6: warm-label timestamp. Subsequent calls to this label
            # within `cascade_warm_window_s` get the shorter warm timeout.
            # Mirrors the _INMEM_HEALTHY_LABELS pattern at module top.
            _WARM_LABELS[label] = time.monotonic()
            # v2.5.1: a successful call voids any pending primary-skip
            # escalation -- the cascade has proven itself healthy. Both
            # primary-recovered and fallback-succeeded paths count.
            _reset_primary_strikes()
            if idx != current_idx:
                ok = f"Utility model fallback succeeded, now using [{idx}]: {label}"
                self.context.log.log("info", content=ok)
                PrintStyle(font_color="cyan", padding=True).print(ok)
            # Continuous mode: if we were in a long fallback stretch, announce
            # the recovery so the user can correlate with provider quotas.
            nonlocal consecutive_full_cycles, fallback_started_at
            if continuous_mode and consecutive_full_cycles > 0:
                elapsed = int(time.monotonic() - fallback_started_at)
                recovery_msg = (
                    f"Utility model recovered after {consecutive_full_cycles} "
                    f"full cycle(s) and {elapsed}s of fallback -- quota "
                    f"refresh detected on [{idx}]: {label}."
                )
                self.context.log.log("info", content=recovery_msg)
                PrintStyle(font_color="cyan", padding=True).print(recovery_msg)
            consecutive_full_cycles = 0
            # v2.6: Phase 3 — any success clears the stagnation counter
            # and re-arms the one-log-per-outage gate. Until the next
            # zero-success run starts, _compute_cycle_sleep won't
            # amplify again.
            consecutive_no_success_cycles = 0
            stagnation_logged_this_outage = False
            fallback_started_at = time.monotonic()
            _set_model_idx(self, use_utility_models=True, idx=idx)
            # Clear extended-retry state on success
            self.set_data(DATA_KEY_EXT_RETRY_ATTEMPTS, 0)
            self.set_data(DATA_KEY_EXT_RETRY_PHASE, 0)
            self.set_data(DATA_KEY_EXT_RETRY_NOTIFIED, False)
            await extension.call_extensions_async(
                "util_model_call_after",
                self,
                call_data=call_data,
                response=response,
            )
            return response

        # Build the inner coroutine once so we can close() it on timeout.
        # Without this, when ``asyncio.wait_for`` cancels the call the
        # ``OpenAIChatCompletion.acompletion`` coroutine that litellm
        # constructed inside ``unified_call`` is never awaited nor closed;
        # Python logs::
        #   RuntimeWarning: coroutine 'OpenAIChatCompletion.acompletion'
        #     was never awaited
        # which leaks a frame per timeout. Across a multi-hour provider
        # outage in continuous-fallback mode this accumulates and pollutes
        # the log. We can't easily access the inner litellm coroutine, but
        # the OUTER coroutine (the awaitable returned by unified_call) is
        # sufficient: closing it releases its frame and the inner
        # coroutine's frame becomes collectable.
        _inner_coro = call_data["model"].unified_call(
            system_message=call_data["system"],
            user_message=call_data["message"],
            response_callback=(
                stream_callback if call_data["callback"] else None
            ),
            rate_limiter_callback=(
                self.rate_limiter_callback
                if not call_data["background"]
                else None
            ),
            fallbacks=None,
        )

        # v2.6: per-candidate warm/cold timeout resolution. Each fallback
        # candidate gets its own warm-check against its own label. The
        # user-kwarg TIMEOUT= override was already applied at the top of
        # the cascade to compute `timeout_s`; if the user set a kwarg we
        # skip the warm-check (preserves the legacy contract that kwarg
        # overrides everything). Resolved at the per-iteration site so
        # `effective_timeout_s` is in scope for both the inner call and
        # the timeout-warning log line.
        user_kwarg_set = "TIMEOUT" in model_kwargs or "timeout" in model_kwargs
        if user_kwarg_set:
            effective_timeout_s = timeout_s
        else:
            effective_timeout_s = _resolve_per_call_timeout(
                label, timeout_s, warm_timeout_s, warm_window_s,
            )

        async def _call_utility_model():
            # v2.6: on external cancellation (CancelledError from outer
            # guard / container shutdown), skip the close-coroutine
            # cleanup -- the call was healthy and running, the cancel
            # was externally forced. On TimeoutError (wait_for fired at
            # the timeout budget), keep the legacy close cleanup so the
            # frame doesn't leak. This is a smaller, more conservative
            # interpretation of the plan's "don't preempt healthy calls"
            # intent -- the 50% wall-clock gate from the plan doesn't
            # apply because asyncio.wait_for raises at the timeout
            # boundary, not after, so elapsed is always >= timeout_s.
            started_at = time.monotonic()
            try:
                return await asyncio.wait_for(_inner_coro, timeout=effective_timeout_s)
            except asyncio.CancelledError:
                # External cancellation -- don't preempt the inner coro.
                # The outer guard / container shutdown is the source of
                # truth here.
                raise
            except asyncio.TimeoutError:
                # Genuine inner timeout. Inner coro is already cancelled
                # by wait_for; close it explicitly so it doesn't leak a
                # frame + a warning. Calling .close() on an already-
                # running coroutine raises a RuntimeWarning of its own,
                # so only do this on the timeout path (where wait_for is
                # responsible for the cancellation, not us).
                try:
                    if _inner_coro.cr_frame is not None:
                        _inner_coro.close()
                except Exception:
                    pass
                raise

        try:
            response, _reasoning = await _call_utility_model()
            return await _succeed(response)

        except (asyncio.TimeoutError, TimeoutError, asyncio.CancelledError) as e:
            warn = f"Utility model [{idx}] timed out after {int(effective_timeout_s)}s: {label}"
            self.context.log.log("warning", content=warn)
            PrintStyle(font_color="orange", padding=True).print(warn)
            _handle_error_cooldown(e, label, model_cooldowns, self)
            # v2.5.1: track consecutive primary failures. If [0] has just
            # timed out, escalate its cooldown so the cascade routes around
            # it for the next ``primary_skip_cooldown_s`` seconds instead of
            # paying the 60s timeout tax on every cycle.
            _maybe_extend_primary_cooldown(reason="timeout")
            # Timeouts count as transient, not permanent.

        except Exception as e:
            # Code-level errors: fail fast -- don't cycle through all candidates
            if _is_code_error(e):
                err_msg = (
                    f"Utility model [{idx}] CODE ERROR ({type(e).__name__}): "
                    f"{label} - {_format_exception(e)}. "
                    f"This is a code bug, not an API failure -- failing fast."
                )
                self.context.log.log("error", content=err_msg)
                PrintStyle(font_color="red", padding=True).print(err_msg)
                # Mark as permanent so it's not retried, then raise
                _handle_error_cooldown(e, label, model_cooldowns, self)
                raise CodeError(e, f"utility model [{label}]")

            is_json_err = isinstance(e, ValueError) and "valid JSON" in str(e)
            is_overflow = _is_context_overflow_error(e)
            err_type = (
                "Format Error"
                if is_json_err
                else "ContextOverflow"
                if is_overflow
                else f"Error ({type(e).__name__})"
            )
            err_msg = f"Utility model [{idx}] {err_type}: {label} - {_format_exception(e)}"
            self.context.log.log("warning", content=err_msg)
            PrintStyle(font_color="orange", padding=True).print(err_msg)

            # Option D: a 5xx from the Responses path means the endpoint is
            # broken (not a malformed request). Retry ONCE with
            # chat-completions, then mark sticky so all later turns force
            # chat proactively. Guard: only when this call actually used
            # Responses (chat was not already forced). See _patched_call_chat_model.
            if (
                responses_5xx_retry_enabled
                and is_responses_server_error(e)
                and not should_force_chat_completions(current_model, self)
            ):
                mark_responses_5xx_seen(current_model)
                force_chat_completions_mode(current_model)
                self.context.log.log(
                    "warning",
                    content=(
                        f"Utility model [{idx}] Responses 5xx; retrying "
                        f"{label} once with chat-completions."
                    ),
                )
                try:
                    response, _reasoning = await _call_utility_model()
                    return await _succeed(response)
                except Exception:
                    # chat-completions also failed -> fall through to normal
                    # cooldown + advance. Keep the original exception `e`.
                    pass

            _handle_error_cooldown(e, label, model_cooldowns, self)
            # v2.5.1: same primary-skip escalation for the general error
            # path. The helper is a no-op when idx != 0.
            _maybe_extend_primary_cooldown(reason=type(e).__name__)
            if _is_permanently_failed_model(e):
                cycle_permanent_count += 1

        attempt += 1

    # Last-ditch extended-retry check (if cycle_count was below max when we broke)
    _emit_fallback_summary(self, "utility", candidates)
    _maybe_raise_retry_after_hours(self, cycle_count, n, attempt)

    exhausted = (
        f"All utility model candidates exhausted after {cycle_count} cycle(s) "
        f"({attempt} attempt(s))."
    )
    self.context.log.log("error", content=exhausted)
    PrintStyle(font_color="red", padding=True).print(exhausted)
    # Continuous mode: a "break" out of the loop is unreachable because we
    # skip the max_cycles cap. If something else causes a fallthrough (e.g.
    # a future maintainer adds a new break path), do not kill the agent --
    # convert the RuntimeError into a safe error log and return None so the
    # caller can retry. This is a belt-and-suspenders guard for AGENTS.md
    # invariant: "the agent must never die from a transient unavailability
    # of every cloud model in the chain".
    if continuous_mode:
        return None
    raise RuntimeError(exhausted)


# ---------------------------------------------------------------------------
# Patched call_chat_model
# ---------------------------------------------------------------------------

async def _patched_call_chat_model(
    self,
    messages=None,
    response_callback: Callable | None = None,
    reasoning_callback: Callable | None = None,
    background: bool = False,
    explicit_caching: bool = True,
    use_utility_models: bool = False,
    callback: Callable | None = None,
) -> Tuple[str, str]:
    if response_callback or reasoning_callback:
        pass
    else:
        response_callback, reasoning_callback = resolve_callback(callback)

    model_obj = (
        self.get_utility_model() if use_utility_models else self.get_chat_model()
    )
    if model_obj is None:
        raise RuntimeError(
            "Agent has no chat model configured"
            if not use_utility_models
            else "Agent has no utility model configured"
        )

    # Used by the v2.5.1 primary-skip helper to label its log lines.
    # In the chat cascade ``use_utility_models`` is a function parameter
    # (the agent can call chat fallback with utility candidates), so we
    # honor that distinction.
    _is_utility_cascade: bool = bool(use_utility_models)

    # Timeout resolution: plugin config > model kwarg > default (300s)
    plugin_cfg = _get_plugin_cfg(self)
    default_timeout = float(plugin_cfg.get("fallback_timeout_s", 300))
    model_kwargs = getattr(model_obj, "kwargs", {}) or {}
    timeout_s = float(
        model_kwargs.get("TIMEOUT", model_kwargs.get("timeout", default_timeout))
    )
    # v2.6.1 fix: warm/cold per-call timeout reads. The utility cascade reads
    # these at fallback.py:1153-1158; the chat cascade originally omitted them,
    # leaving ``warm_timeout_s``/``warm_window_s`` unbound at the call site
    # (fallback.py:_resolve_per_call_timeout call inside the chat loop). With no
    # user ``TIMEOUT`` kwarg on the chat model (the common case) that was a
    # ``NameError`` raised in the loop body before the try/except, killing the
    # chat call; with a ``TIMEOUT`` kwarg the warm/cold path was silently dead
    # (always cold). Mirror the utility cascade reads so the chat cascade gets
    # the same warm 20s fast-path after a label's first successful call.
    warm_timeout_s = float(plugin_cfg.get(
        "cascade_warm_timeout_s", _DEFAULT_CASCADE_WARM_TIMEOUT_S,
    ))
    warm_window_s = float(plugin_cfg.get(
        "cascade_warm_window_s", _DEFAULT_CASCADE_WARM_WINDOW_S,
    ))

    candidates = _build_candidates(model_obj, use_utility_models=use_utility_models, agent=self)
    n = len(candidates)
    if n == 0:
        raise RuntimeError("No chat model candidates available")

    # Pre-flight: catch code errors before entering the loop
    _preflight_check(model_obj, candidates, "chat model fallback")

    current_idx = _resolve_model_idx(self, use_utility_models) % n

    cycle_delay: float = _clamp_delay(
        float(
            model_kwargs.get(
                "FALLBACK_CYCLE_DELAY",
                model_kwargs.get("fallback_cycle_delay",
                                 float(plugin_cfg.get("fallback_cycle_delay", 5.0))),
            )
        ),
        _MAX_CYCLE_DELAY,
        "cycle_delay",
    )
    attempt_delay: float = _clamp_delay(
        float(
            model_kwargs.get(
                "FALLBACK_ATTEMPT_DELAY",
                model_kwargs.get("fallback_attempt_delay",
                                 float(plugin_cfg.get("fallback_attempt_delay", 2.0))),
            )
        ),
        _MAX_ATTEMPT_DELAY,
        "attempt_delay",
    )
    max_cycles: int = int(
        model_kwargs.get(
            "MAX_FALLBACK_CYCLES",
            model_kwargs.get("max_fallback_cycles",
                             int(plugin_cfg.get("fallback_max_cycles", 4))),
        )
    )

    model_cooldowns: dict = _get_cooldown_store(self) or {}
    if not isinstance(model_cooldowns, dict):
        model_cooldowns = {}

    cycle_count = 0
    attempt = 0
    tried_this_cycle = 0  # how many models we actually CALLed (not skipped)
    cycle_permanent_count = 0
    early_exit_enabled: bool = bool(plugin_cfg.get("early_exit_on_all_permanent", True))
    responses_5xx_retry_enabled: bool = bool(
        plugin_cfg.get("responses_5xx_retry_enabled", True)
    )
    # Per-label dedupe of the "in cooldown, skipping (Xs remaining)" log.
    # Same contract as the utility cascade above -- log once per
    # cooldown window, not once per ~2s iteration. See the comment in
    # _patched_call_utility_model for the rationale (Laci, 2026-07-21).
    _last_skip_log_until: dict = {}

    # --- Continuous-fallback state -----------------------------------------
    # Mirrors _patched_call_utility_model. The chat cascade runs the same
    # backoff envelope so chat and utility stay in lockstep during outages.
    # See AGENTS.md for the contract.
    continuous_mode: bool = bool(plugin_cfg.get("continuous_fallback", False))
    max_cycle_delay_s: float = _clamp_delay(
        float(plugin_cfg.get("max_cycle_delay_s", 300.0)),
        3600.0,
        "max_cycle_delay_s",
    )
    backoff_multiplier: float = max(1.0, float(plugin_cfg.get("backoff_multiplier", 2.0)))
    backoff_jitter_s: float = max(0.0, float(plugin_cfg.get("backoff_jitter_s", 2.0)))
    consecutive_full_cycles: int = 0
    fallback_started_at: float = time.monotonic()
    # v2.6: Phase 3 — stagnation counter (closure-local). Counts
    # consecutive full cycles with zero successes. Reset on any success.
    # Used by _compute_cycle_sleep to apply ``cycle_stagnation_factor``
    # when stagnation exceeds ``cycle_stagnation_threshold`` so the
    # cascade backs off further during sustained outages.
    consecutive_no_success_cycles: int = 0
    stagnation_logged_this_outage: bool = False
    cycle_stagnation_factor: float = max(
        1.0, float(plugin_cfg.get(
            "cycle_stagnation_factor", _DEFAULT_CYCLE_STAGNATION_FACTOR,
        ))
    )
    cycle_stagnation_threshold: int = max(
        1, int(plugin_cfg.get(
            "cycle_stagnation_threshold", _DEFAULT_CYCLE_STAGNATION_THRESHOLD,
        ))
    )

    # --- v2.5.1 Primary-skip-after-N-strikes (added 2026-07-23) --------------
    # When the primary model (candidate index 0) fails repeatedly, the
    # cascade used to keep retrying it every 5 min (the default transient
    # cooldown) even though the cost of each retry is 60-90s of timeout. The
    # log on 2026-07-23 22:50 showed "consecutive=2" in a 14+ min stall, all
    # attributed to a hung primary.
    #
    # Mechanism: track consecutive failures of [0] within a single
    # cascade invocation. When the count crosses ``primary_skip_strikes``,
    # extend [0]'s cooldown to ``primary_skip_cooldown_s`` (default 600s =
    # 10 min) so the cascade routes around the primary. The next time the
    # primary's cooldown expires, the cascade tries it again -- if the
    # primary is back, success and the counter resets; if it's still hung,
    # another escalation.
    #
    # The counter resets to 0 on:
    #   - A successful call from ANY candidate (cascade is healthy)
    #   - A successful call from the primary itself (quota recovered)
    primary_skip_enabled: bool = bool(plugin_cfg.get("primary_skip_enabled", True))
    primary_skip_strikes: int = max(
        1, int(plugin_cfg.get("primary_skip_strikes", 2))
    )
    # Default lowered from 600s (10 min) to 120s (2 min). 600s was
    # over-locking the primary when it had recovered in 2 min, which
    # is the common case for short-lived upstream hiccups. 120s still
    # covers one full pass through 6 candidates at 60s timeout each
    # (6 min) and then some, but routes around a hung primary quickly.
    # Tunable via config (primary_skip_cooldown_s).
    primary_skip_cooldown_s: float = max(
        30.0, float(plugin_cfg.get("primary_skip_cooldown_s", 120.0))
    )
    _consecutive_primary_failures: int = 0

    def _maybe_extend_primary_cooldown(reason: str) -> None:
        """If the primary (candidate 0) has failed ``primary_skip_strikes``
        times in a row, push its cooldown out to ``primary_skip_cooldown_s``
        so the cascade stops paying the 60s timeout tax on every cycle.

        Called after every primary failure. The dedupe in the cooldown
        table's skip-log means the new cooldown value produces exactly one
        log line per escalation, not one per cascade iteration.

        v2.6: if a peer agent has recently proven this label healthy
        (within ``health_horizon_s``), reset our local strike counter
        and any stale per-agent cooldown first. This prevents the case
        where one agent's stale primary failures keep escalating while
        another agent's success on the same label would have cleared
        the cooldown entirely.
        """
        nonlocal _consecutive_primary_failures
        if not primary_skip_enabled:
            return
        if idx != 0:
            return
        # v2.6: Phase 2 — per-provider capacity inference. For
        # ``concurrent_paid`` (local ollama) and ``router``
        # (omniroute/*), a primary 429 does not mean the
        # model is broken — the slot frees in seconds (ollama) or the
        # next call re-routes (routers). Never escalate the primary-skip
        # cooldown for these. Skip before the Phase 5 healthy-label
        # check so a peer success doesn't trigger a counter reset that
        # would later be wasted by a 30s escalation.
        if _capacity_skips_cooldown(label):
            return
        # v2.6: Phase 5 short-circuit. _maybe_clear_cooldown_for_healthy_label
        # already enforces the only-cleared-never-overwritten invariant
        # (returns True only if the local cooldown is stale AND the
        # cross-agent healthy index proves the label works). When it
        # returns True we reset our strike counter and return early --
        # no escalation needed, the peer agent proved this works.
        if _maybe_clear_cooldown_for_healthy_label(self, label):
            _consecutive_primary_failures = 0
            return
        _consecutive_primary_failures += 1
        if _consecutive_primary_failures < primary_skip_strikes:
            return
        # Escalate: write a fresh long cooldown for the primary label.
        # Bypasses _handle_error_cooldown (already called) -- we just
        # extend whatever it set.
        prev_until = model_cooldowns.get(label) or 0.0
        target_until = time.monotonic() + primary_skip_cooldown_s
        if target_until <= prev_until:
            return  # existing cooldown is already longer, don't shorten
        model_cooldowns[label] = target_until
        _save_cooldown_store(self, model_cooldowns)
        # ``use_utility_models`` is a parameter of the chat cascade (default
        # False) and is implicit True inside the utility cascade. We
        # disambiguate via the caller-supplied label so the helper stays
        # identical for both cascades.
        kind_label = "Utility" if _is_utility_cascade else "Chat"
        try:
            self.context.log.log(
                "warning",
                f"{kind_label} primary [{label}] has failed "
                f"{_consecutive_primary_failures} time(s) in a row -- "
                f"escalating cooldown to {int(primary_skip_cooldown_s)}s "
                f"(reason: {reason}). Cascade will route around it.",
            )
        except Exception:
            pass

    def _reset_primary_strikes() -> None:
        """Reset the primary-skip counter. Called on any successful call --
        either the primary recovered, or a fallback succeeded and proves
        the cascade is healthy (next primary attempt is no longer a
        'consecutive failure')."""
        nonlocal _consecutive_primary_failures
        if _consecutive_primary_failures > 0:
            _consecutive_primary_failures = 0

    def _compute_cycle_sleep() -> float:
        """Same envelope as the utility cascade. See _patched_call_utility_model."""
        import random
        base = cycle_delay * (backoff_multiplier ** min(consecutive_full_cycles, 10))
        capped = min(base, max_cycle_delay_s)
        if backoff_jitter_s > 0:
            capped += random.uniform(0.0, backoff_jitter_s)
        return capped

    while True:
        idx = (current_idx + attempt) % n
        label = _get_model_label(candidates[idx], model_obj)

        # Skip models in cooldown
        cooldown_until = model_cooldowns.get(label)
        # v2.5.2: cross-agent healthy-label reset. See
        # _maybe_clear_cooldown_for_healthy_label. Mirrors the utility
        # cascade's skip path above.
        if cooldown_until is not None:
            _maybe_clear_cooldown_for_healthy_label(self, label)
            cooldown_until = model_cooldowns.get(label)
        now = time.monotonic()
        if cooldown_until and cooldown_until > now:
            remaining = int(cooldown_until - now)
            # Dedupe: only log when the cooldown value is NEW for this
            # label. Same contract as the utility cascade.
            if _last_skip_log_until.get(label) != cooldown_until:
                _last_skip_log_until[label] = cooldown_until
                self.context.log.log(
                    "info",
                    f"Chat model [{label}] in cooldown, skipping ({remaining}s remaining)",
                )
            # Advance idx but do NOT count this slot as "tried this cycle"
            # so the cascade progresses through candidates even when some
            # are in cooldown. (Laci, 2026-06-15: previously the loop
            # treated "all models skipped" as "all models attempted" and
            # 60s-slept forever, which is what produced the spam log.)
            attempt += 1
            continue
        # Cooldown has expired for this label; clear the dedupe marker
        # so the NEXT cooldown window logs its first skip again.
        if _last_skip_log_until.get(label) is not None:
            _last_skip_log_until.pop(label, None)

        if attempt > 0 and attempt % n == 0:
            if tried_this_cycle == 0:
                # Every model in this pass was skipped (all in cooldown).
                # Don't count it as a "cycle" -- just wait a bit and try
                # the same set again.
                await asyncio.sleep(2.0)
                attempt = 0
                tried_this_cycle = 0
                cycle_permanent_count = 0
                continue

            # Early-exit: if every attempted model in this cycle was a
            # permanent failure, skip the cycle-and-sleep ritual and go
            # straight to extended-retry mode. See _patched_call_utility_model
            # for the same logic + rationale.
            if (
                early_exit_enabled
                and cycle_permanent_count > 0
                and cycle_permanent_count == tried_this_cycle
            ):
                cycle_count += 1
                _emit_fallback_summary(self, "chat", candidates)
                # Continuous mode: never raise RetryAfterHours; the cascade
                # must keep cycling so the agent stays alive across multi-hour
                # rate-limit windows. In legacy mode, raise if applicable.
                if not continuous_mode:
                    _maybe_raise_retry_after_hours(self, cycle_count, n, attempt)
                consecutive_full_cycles += 1
                sleep_s = _compute_cycle_sleep() if continuous_mode else cycle_delay
                cycle_msg = (
                    f"Chat models cycling (all permanent). "
                    f"Sleeping {sleep_s:.0f}s "
                    f"(cycle {cycle_count}, consecutive={consecutive_full_cycles})..."
                )
                self.context.log.log("warning", content=cycle_msg)
                PrintStyle(font_color="orange", padding=True).print(cycle_msg)
                await _yielding_sleep(sleep_s)
                _set_model_idx(self, use_utility_models, idx=0)
                attempt = 0
                tried_this_cycle = 0
                cycle_permanent_count = 0
                continue

            cycle_count += 1
            # Continuous mode: skip the max_cycles cap. See AGENTS.md.
            if not continuous_mode and max_cycles > 0 and cycle_count >= max_cycles:
                _maybe_raise_retry_after_hours(self, cycle_count, n, attempt)
                break
            consecutive_full_cycles += 1
            # v2.6: Phase 3 — stagnation counter increments on every
            # cycle iteration that didn't produce a success. _succeed
            # resets it. _compute_cycle_sleep applies the stagnation
            # amplification factor once it crosses the threshold.
            consecutive_no_success_cycles += 1
            sleep_s = _compute_cycle_sleep() if continuous_mode else cycle_delay
            cycle_msg = (
                f"Chat models cycling (all attempted). "
                f"Sleeping {sleep_s:.0f}s "
                f"(cycle {cycle_count}, consecutive={consecutive_full_cycles})..."
            )
            self.context.log.log("warning", content=cycle_msg)
            PrintStyle(font_color="orange", padding=True).print(cycle_msg)
            await _yielding_sleep(sleep_s)
            _set_model_idx(self, use_utility_models, idx=0)
            attempt = 0
            tried_this_cycle = 0
            cycle_permanent_count = 0
        elif attempt > 0:
            await _yielding_sleep(attempt_delay)

        tried_this_cycle += 1
        current_model = _build_model(candidates[idx], model_obj)

        # Strip A0-only / provider-invalid keys from the wrapper's kwargs right
        # before the first acompletion() so they never reach the request body.
        # The plugin has already read cycle_delay / max_cycles / timeout above.
        _strip_a0_only_kwargs(current_model)

        if attempt > 0:
            warn = f"Chat model switching to [{idx}/{n - 1}]: {label}"
            self.context.log.log("warning", content=warn)
            PrintStyle(font_color="orange", padding=True).print(warn)

        call_data = {
            "model": current_model,
            "messages": messages,
            "response_callback": response_callback,
            "reasoning_callback": reasoning_callback,
            "background": background,
            "explicit_caching": explicit_caching,
        }
        await extension.call_extensions_async(
            "chat_model_call_before", self, call_data=call_data
        )

        async def _succeed(response, reasoning):
            # Shared success path: validate tool request, clear cooldown,
            # advance rotation index, clear extended-retry state, emit after-hook.
            if not call_data.get("background"):
                tool_request = extract_tools.json_parse_dirty(response)
                if tool_request is not None:
                    await self.validate_tool_request(tool_request)
            # Success -- clear cooldown for this model
            model_cooldowns.pop(label, None)
            _save_cooldown_store(self, model_cooldowns)
            # v2.5.2: cross-agent healthy-label reset. See
            # _mark_label_healthy / _maybe_clear_cooldown_for_healthy_label.
            _mark_label_healthy(self, label)
            # v2.6: warm-label timestamp. Subsequent calls to this label
            # within `cascade_warm_window_s` get the shorter warm timeout.
            # Mirrors the _INMEM_HEALTHY_LABELS pattern at module top.
            _WARM_LABELS[label] = time.monotonic()
            # v2.5.1: a successful call voids any pending primary-skip
            # escalation -- the cascade has proven itself healthy. Both
            # primary-recovered and fallback-succeeded paths count.
            _reset_primary_strikes()
            if idx != current_idx:
                ok = f"Chat model fallback succeeded, now using [{idx}]: {label}"
                self.context.log.log("info", content=ok)
                PrintStyle(font_color="cyan", padding=True).print(ok)
            # Continuous mode: announce recovery from a long fallback stretch.
            nonlocal consecutive_full_cycles, fallback_started_at
            if continuous_mode and consecutive_full_cycles > 0:
                elapsed = int(time.monotonic() - fallback_started_at)
                recovery_msg = (
                    f"Chat model recovered after {consecutive_full_cycles} "
                    f"full cycle(s) and {elapsed}s of fallback -- quota "
                    f"refresh detected on [{idx}]: {label}."
                )
                self.context.log.log("info", content=recovery_msg)
                PrintStyle(font_color="cyan", padding=True).print(recovery_msg)
            consecutive_full_cycles = 0
            # v2.6: Phase 3 — any success clears the stagnation counter
            # and re-arms the one-log-per-outage gate. Until the next
            # zero-success run starts, _compute_cycle_sleep won't
            # amplify again.
            consecutive_no_success_cycles = 0
            stagnation_logged_this_outage = False
            fallback_started_at = time.monotonic()
            # Clear extended-retry state on success
            self.set_data(DATA_KEY_EXT_RETRY_ATTEMPTS, 0)
            self.set_data(DATA_KEY_EXT_RETRY_PHASE, 0)
            self.set_data(DATA_KEY_EXT_RETRY_NOTIFIED, False)
            await extension.call_extensions_async(
                "chat_model_call_after",
                self,
                call_data=call_data,
                response=response,
                reasoning=reasoning,
            )
            return response, reasoning

        # Capture the inner coroutine so we can close() it on timeout. See
        # the matching block in ``_call_utility_model`` for the rationale
        # (litellm's ``OpenAIChatCompletion.acompletion`` leaks an
        # unawaited coroutine on every ``asyncio.wait_for`` cancellation).
        _inner_coro = call_data["model"].unified_call(
            messages=call_data["messages"],
            reasoning_callback=call_data["reasoning_callback"],
            response_callback=call_data["response_callback"],
            rate_limiter_callback=(
                self.rate_limiter_callback
                if not call_data["background"]
                else None
            ),
            explicit_caching=call_data["explicit_caching"],
            fallbacks=None,
        )

        # v2.6: per-candidate warm/cold timeout resolution (mirrors
        # _patched_call_utility_model). Each fallback candidate gets its
        # own warm-check against its own label. The user-kwarg TIMEOUT=
        # override was already applied at the top of the cascade to
        # compute `timeout_s`; if the user set a kwarg we skip the
        # warm-check (preserves the legacy contract that kwarg overrides
        # everything).
        user_kwarg_set = "TIMEOUT" in model_kwargs or "timeout" in model_kwargs
        if user_kwarg_set:
            effective_timeout_s = timeout_s
        else:
            effective_timeout_s = _resolve_per_call_timeout(
                label, timeout_s, warm_timeout_s, warm_window_s,
            )

        async def _call_chat_model():
            # v2.6: same CancelledError-vs-TimeoutError split as the
            # utility cascade. See the comment in _call_utility_model
            # for the rationale (wait_for raises at the timeout
            # boundary, so a 50% gate is meaningless; the actual user
            # intent of "don't preempt healthy calls" is honored by
            # leaving externally-cancelled calls alone).
            started_at = time.monotonic()
            try:
                return await asyncio.wait_for(_inner_coro, timeout=effective_timeout_s)
            except asyncio.CancelledError:
                # External cancellation -- don't preempt the inner coro.
                raise
            except asyncio.TimeoutError:
                # Genuine inner timeout. Inner coro is already cancelled
                # by wait_for; close it explicitly so it doesn't leak a
                # frame + a warning.
                try:
                    if _inner_coro.cr_frame is not None:
                        _inner_coro.close()
                except Exception:
                    pass
                raise

        try:
            response, reasoning = await _call_chat_model()
            return await _succeed(response, reasoning)

        except (asyncio.TimeoutError, TimeoutError, asyncio.CancelledError) as e:
            warn = f"Chat model [{idx}] timed out after {int(effective_timeout_s)}s: {label}"
            self.context.log.log("warning", content=warn)
            PrintStyle(font_color="orange", padding=True).print(warn)
            _handle_error_cooldown(e, label, model_cooldowns, self)
            # Timeouts are transient.

        except Exception as e:
            # Code-level errors: fail fast -- don't cycle through all candidates
            if _is_code_error(e):
                err_msg = (
                    f"Chat model [{idx}] CODE ERROR ({type(e).__name__}): "
                    f"{label} - {_format_exception(e)}. "
                    f"This is a code bug, not an API failure -- failing fast."
                )
                self.context.log.log("error", content=err_msg)
                PrintStyle(font_color="red", padding=True).print(err_msg)
                _handle_error_cooldown(e, label, model_cooldowns, self)
                raise CodeError(e, f"chat model [{label}]")

            is_overflow = _is_context_overflow_error(e)
            err_type = "ContextOverflow" if is_overflow else type(e).__name__
            err_msg = f"Chat model [{idx}] error ({err_type}): {label} - {_format_exception(e)}"
            self.context.log.log("warning", content=err_msg)
            PrintStyle(font_color="orange", padding=True).print(err_msg)

            # Option D: a 5xx from the Responses path means the endpoint is
            # broken (not a malformed request). Retry ONCE with
            # chat-completions, then mark sticky so all later turns force
            # chat proactively (mirrors the core's RESPONSES_UNSUPPORTED_CACHE).
            # Guard: only when this call actually used Responses -- i.e. chat
            # was not already forced (option B static, or a prior 5xx). This
            # avoids a wasted identical retry when chat itself 5xxs.
            if (
                responses_5xx_retry_enabled
                and is_responses_server_error(e)
                and not should_force_chat_completions(current_model, self)
            ):
                mark_responses_5xx_seen(current_model)
                force_chat_completions_mode(current_model)
                self.context.log.log(
                    "warning",
                    content=(
                        f"Chat model [{idx}] Responses 5xx; retrying "
                        f"{label} once with chat-completions."
                    ),
                )
                try:
                    response, reasoning = await _call_chat_model()
                    return await _succeed(response, reasoning)
                except Exception:
                    # chat-completions also failed -> fall through to normal
                    # cooldown + advance. Keep the original exception `e`.
                    pass

            _handle_error_cooldown(e, label, model_cooldowns, self)
            # v2.5.1: same primary-skip escalation for the general error
            # path. The helper is a no-op when idx != 0.
            _maybe_extend_primary_cooldown(reason=type(e).__name__)
            if _is_permanently_failed_model(e):
                cycle_permanent_count += 1

        attempt += 1

    _emit_fallback_summary(self, "chat", candidates)
    _maybe_raise_retry_after_hours(self, cycle_count, n, attempt)

    exhausted = (
        f"All chat model candidates exhausted after {cycle_count} cycle(s) "
        f"({attempt} attempt(s))."
    )
    self.context.log.log("error", content=exhausted)
    PrintStyle(font_color="red", padding=True).print(exhausted)
    # Continuous mode: see the same guard in _patched_call_utility_model.
    if continuous_mode:
        return None, None
    raise RuntimeError(exhausted)
