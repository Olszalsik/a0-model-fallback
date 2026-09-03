"""
Local copies of model-related helpers used by the fallback plugin.

These mirror / extend the functions expected from models.py so the plugin
is fully self-contained and survives upstream updates that overwrite models.py.

Contents:
  - _is_context_overflow_error: detect prompt-too-long errors
  - _is_permanently_failed_model: detect unrecoverable per-session errors
  - extract_retry_after_seconds: parse Retry-After hint from exceptions
  - build_fallback_wrapper: construct a model object for a fallback spec
  - resolve_callback: split a single callback into response + reasoning halves
"""

import re
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

def _is_context_overflow_error(exc: Exception) -> bool:
    """True when the prompt is too long for the model's context window (HTTP 400 variants).

    These errors are permanent for a given prompt — retrying with the same content
    will always fail. The fallback loop applies a session-length cooldown instead.
    """
    status_code = getattr(exc, "status_code", None)
    exc_str = str(exc).lower()

    if isinstance(status_code, int) and status_code == 400:
        if any(
            kw in exc_str
            for kw in (
                "too long",
                "context length",
                "maximum context",
                "prompt is too long",
                "token limit",
                "context_length_exceeded",
                "reduce the length",
            )
        ):
            return True

    if any(
        kw in exc_str
        for kw in (
            "prompt is too long",
            "context_length_exceeded",
            "maximum context length",
            "reduce the length of the messages",
        )
    ):
        return True

    return False


def _is_code_error(exc: Exception) -> bool:
    """True when the exception is a code-level error, not a model API error.

    Code-level errors live in our own code (or in a plugin we import) and will
    not self-resolve by rotating to a different model. Examples:

      - ImportError / ModuleNotFoundError: a dependency is missing
      - NameError / AttributeError: a symbol we expected to find is missing
      - SyntaxError: a .py file is broken
      - TypeError: a function is being called with the wrong shape

    Returning True here lets the fallback loop short-circuit on a developer-side
    bug instead of pointlessly cycling through every configured model.

    Note: this is intentionally narrower than `isinstance(exc, Exception)` --
    we want a stable set of "this is our fault, not the API's" classes.
    """
    return isinstance(
        exc,
        (
            ImportError,
            ModuleNotFoundError,
            NameError,
            AttributeError,
            SyntaxError,
            TypeError,
        ),
    )


# String fragments that, when seen in an exception's message, indicate the
# call was rejected for an *authentication* / *authorization* reason even
# though the HTTP status came back as 400 (which is the case for Groq's
# `BadRequestError - {"error":{"code":"invalid_api_key"}}` and for some
# OpenAI / OpenRouter 400 variants). Without this, the plugin would treat
# a permanently-broken key as a transient error and re-try the same
# failed model every few seconds.
_INVALID_KEY_400_PHRASES = (
    "invalid_api_key",
    "invalid api key",
    "incorrect api key",
    "api key not valid",
    "api key is invalid",
    "no api key",
    "missing api key",
    "authentication",
    "unauthorized",
    "authorization",
    "credential",
    "auth failed",
    "auth error",
    "access denied",
    "forbidden",
)


def _is_invalid_api_key_400(exc: Exception) -> bool:
    """True when status_code is 400 and the body indicates an auth/key problem.

    Groq returns 400 (not 401) for a bad API key. OpenAI / OpenRouter
    sometimes return 400 too. The HTTP layer only sees the 400, so we
    inspect the message body for the well-known auth-failure phrases.
    """
    status_code = getattr(exc, "status_code", None)
    if not (isinstance(status_code, int) and status_code == 400):
        return False
    msg = str(exc).lower()
    return any(phrase in msg for phrase in _INVALID_KEY_400_PHRASES)


def _is_rate_limited_error(exc: Exception) -> bool:
    """True when the provider is explicitly throttling / rate-limiting us.

    LiteLLM and the underlying HTTP clients sometimes wrap a plain HTTP 429
    in InternalServerError, APIConnectionError, MidStreamFallbackError, etc.
    We therefore look at both the ``status_code`` attribute and the raw
    exception message string.  This catches providers like NVIDIA NIM that
    return ``{"status":429,"title":"Too Many Requests"}`` without a
    Retry-After header.
    """
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int) and status_code == 429:
        return True

    msg = str(exc).lower()
    rate_limit_phrases = (
        "too many requests",
        "rate limit",
        "rate-limit",
        "ratelimit",
        "throttled",
        "quota exceeded",
        "capacity exceeded",
        "capacity limit",
        "try again later",
        "overloaded",
        "server is busy",
    )
    if any(phrase in msg for phrase in rate_limit_phrases):
        return True

    # Embedded status-code patterns, e.g. {"status":429}, "code":429, HTTP 429
    import re as _re

    if _re.search(
        r'["\']status["\']?\s*[:=]\s*429\b|'
        r'["\']code["\']?\s*[:=]\s*429\b|'
        r'\bstatus[_-]?code\s*[:=]\s*429\b|'
        r'\bhttp\s+429\b',
        msg,
    ):
        return True

    return False


# String fragments that indicate a deterministic "request too big" rejection
# from the upstream gateway. Distinct from context_overflow (which is about
# the model's context window) -- this is about the HTTP request body itself
# exceeding a transport-level cap (OpenAI's 10 MB default, some providers
# set tighter limits). Retrying with the same payload will always fail, so
# we mark the model as permanently failed for the session and the cascade
# short-circuits via its early-exit path instead of cycling through every
# fallback and waiting on the outer timeout guard.
_PAYLOAD_TOO_LARGE_PHRASES = (
    "request body too large",
    "payload_too_large",
    "request entity too large",
    "maximum allowed:",  # OpenAI's "Maximum allowed: 10 MB"
    "body too large",
    "content too large",
    "request too large",
    "request size exceeded",
    "413 ",  # HTTP 413 status, often emitted in the message
)


def _is_payload_too_large_error(exc: Exception) -> bool:
    """True when the upstream rejected the call because the request body
    exceeded its transport-level cap.

    Distinct from ``_is_context_overflow_error`` (which detects
    context-window rejections). The two are sometimes conflated because
    both are "too big" errors, but the recovery semantics differ:
    context-window overflow may resolve on a different model with a
    wider window, whereas a request-body-too-large rejection is a
    transport-level cap that all candidates with the same payload will
    hit. Cycling through every fallback just stacks the same error N
    times before the outer timeout fires.
    """
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int) and status_code == 413:
        return True
    msg = str(exc).lower()
    return any(phrase in msg for phrase in _PAYLOAD_TOO_LARGE_PHRASES)


def _is_permanently_failed_model(exc: Exception) -> bool:
    """True when this model should be skipped for the rest of the session.

    Covers errors that will never self-resolve: wrong API key, model not
    found / deprecated, credit depletion, prompt permanently exceeds context
    window, hard rate-limiting with no usable Retry-After header, or a
    request body that exceeds the upstream transport cap (the same payload
    will fail against every candidate).
    """
    if _is_context_overflow_error(exc):
        return True

    if _is_payload_too_large_error(exc):
        return True

    if _is_invalid_api_key_400(exc):
        return True

    if _is_rate_limited_error(exc):
        return True

    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        if status_code in (401, 402, 403, 404):
            return True

    return False


# ---------------------------------------------------------------------------
# Responses -> chat-completions fallback (options B + D, plugin form)
# ---------------------------------------------------------------------------
#
# The newer agent-zero transport defaults to a0_api_mode=responses (LiteLLM
# `aresponses` -> `<provider>/v1/responses`). Some OpenAI-compatible providers
# (notably ollama.com for `:cloud` models) return HTTP 500 from /v1/responses
# while their /v1/chat/completions endpoint works fine. The core shipped a
# transport-level patch for this, but that patch lives in a tracked file
# (helpers/litellm_transport.py) and is lost on the next agent-zero update.
#
# These helpers replicate the behavior at the plugin level so it survives
# updates:
#   - Option B (proactive): force chat-completions for providers/models
#     listed in the plugin config (force_chat_completions_providers /
#     force_chat_completions_patterns).
#   - Option D (reactive): if a model still 5xxs on the Responses path, mark
#     it sticky and retry once with chat-completions; all later turns force
#     chat proactively via the sticky set (mirrors the core's
#     RESPONSES_UNSUPPORTED_CACHE).

# Sticky set of model cache-keys that returned a 5xx on /v1/responses.
# Module-level (process-global): a provider that 5xxs for one agent 5xxs for
# all. Reset on every run_ui restart, like _INMEM_COOLDOWNS.
_RESPONSES_5XX_SEEN: set = set()


def model_cache_key(model) -> str:
    """Stable key for a model wrapper: model|provider|api_base.

    Matches the shape of the core's RESPONSES_UNSUPPORTED_CACHE cache_key so
    the two systems (if both present) agree on granularity.
    """
    model_name = str(getattr(model, "model_name", "") or "")
    provider = str(getattr(model, "provider", "") or "")
    api_base = ""
    kwargs = getattr(model, "kwargs", {}) or {}
    if isinstance(kwargs, dict):
        api_base = str(kwargs.get("api_base", "") or "")
    return "|".join((model_name, provider, api_base))


def _force_chat_config(agent) -> tuple:
    """Read the force-chat provider/pattern/api_base lists from plugin config.

    Returns a (providers, patterns, api_bases) tuple of strings. Best-effort:
    any read failure yields empty tuples (option B disabled, option D still
    works).

    `api_bases` is a list of substrings matched against the model's
    `kwargs["api_base"]` (case-insensitive). This is the most reliable matcher
    for OpenAI-compatible providers that all share litellm_provider="openai"
    (ollama_cloud, a0_venice, openrouter, ...): the fallback wrapper built by
    `build_fallback_wrapper` carries `provider="openai"` (the litellm provider),
    NOT the agent-zero service name ("ollama_cloud"), so the provider-name list
    cannot match fallback wrappers -- and model-name patterns are fragile
    across a vendor's changing model catalog. The api_base (e.g.
    "https://ollama.com/v1") is stable, present on both primary and fallback
    wrappers, and uniquely identifies the upstream that 500s on /v1/responses.
    """
    try:
        from helpers import plugins as _plugins

        cfg = _plugins.get_plugin_config("_model_fallback", agent) or {}
        if not isinstance(cfg, dict):
            cfg = {}
        # v2.8.4: get_plugin_config does NOT merge default_config.yaml with
        # config.json (same gotcha the v2.6.7 router-detection fix hit). All
        # three force-chat lists live ONLY in default_config.yaml, so without
        # this merge the static option-B matcher was dead at runtime -- the
        # lists never reached this code unless the user duplicated them into
        # config.json by hand. Default YAML first, live config.json on top.
        try:
            defaults = _plugins.get_default_plugin_config("_model_fallback")
            if isinstance(defaults, dict):
                merged = dict(defaults)
                merged.update(cfg)
                cfg = merged
        except Exception:  # noqa: BLE001
            pass
        providers = cfg.get("force_chat_completions_providers") or []
        patterns = cfg.get("force_chat_completions_patterns") or []
        api_bases = cfg.get("force_chat_completions_api_bases") or []
        prov_t = tuple(providers) if isinstance(providers, (list, tuple)) else ()
        pat_t = tuple(patterns) if isinstance(patterns, (list, tuple)) else ()
        base_t = tuple(api_bases) if isinstance(api_bases, (list, tuple)) else ()
        return prov_t, pat_t, base_t
    except Exception:
        return (), (), ()


def should_force_chat_completions(model, agent) -> bool:
    """True if this model should use chat-completions instead of /v1/responses.

    Reasons:
      1. Dynamic (option D): the model previously returned a 5xx on the
         Responses endpoint (sticky, see _RESPONSES_5xx_SEEN).
      2. Static (option B): provider is in force_chat_completions_providers,
         model name contains a pattern from force_chat_completions_patterns,
         or the model's api_base contains a substring from
         force_chat_completions_api_bases (the reliable matcher -- see
         _force_chat_config for why api_base is preferred over provider name).
    """
    if model_cache_key(model) in _RESPONSES_5XX_SEEN:
        return True
    providers, patterns, api_bases = _force_chat_config(agent)
    provider = str(getattr(model, "provider", "") or "")
    model_name = str(getattr(model, "model_name", "") or "")
    if provider and provider in providers:
        return True
    for pat in patterns:
        if pat and pat in model_name:
            return True
    # api_base match: works for both primary and fallback wrappers, including
    # those whose .provider is the litellm "openai" rather than the service name.
    if api_bases:
        kwargs = getattr(model, "kwargs", None)
        api_base = str((kwargs or {}).get("api_base", "") or "").lower()
        if api_base:
            for base in api_bases:
                b = str(base or "").strip().lower()
                if b and b in api_base:
                    return True
    return False


def force_chat_completions_mode(model) -> None:
    """Inject a0_api_mode=chat_completions into model.kwargs (idempotent).

    `a0_api_mode` is consumed by TransportPolicy._pop_mode and is NOT in the
    plugin's A0-only strip list, so it survives `_strip_a0_only_kwargs` and
    never reaches the LiteLLM request body (no 400 "unrecognized key").
    """
    kwargs = getattr(model, "kwargs", None)
    if not isinstance(kwargs, dict):
        return
    if kwargs.get("a0_api_mode") != "chat_completions":
        kwargs["a0_api_mode"] = "chat_completions"


def is_responses_server_error(exc: Exception) -> bool:
    """True when exc is an HTTP 5xx from the Responses endpoint.

    Excludes 4xx (client errors incl. 429 rate limit) and connection/timeout
    errors with no status_code (the host/network being down affects chat too,
    so falling back would just double-spend). Stickiness is enforced by the
    caller marking _RESPONSES_5XX_SEEN, so this fires at most once per model.
    """
    status = getattr(exc, "status_code", None)
    if not isinstance(status, int):
        return False
    return 500 <= status <= 599


def mark_responses_5xx_seen(model) -> None:
    """Record that `model` returned a 5xx on /v1/responses (sticky)."""
    _RESPONSES_5XX_SEEN.add(model_cache_key(model))


def responses_5xx_seen(model) -> bool:
    """True if `model` was already marked as a Responses 5xx offender."""
    return model_cache_key(model) in _RESPONSES_5XX_SEEN


# ---------------------------------------------------------------------------
# Retry-After parsing
# ---------------------------------------------------------------------------

_RETRY_AFTER_HEADER_RE = re.compile(
    r"retry[-_\s]?after[^\d]{0,8}(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def extract_retry_after_seconds(exc: Exception) -> Optional[float]:
    """Best-effort parse of a Retry-After hint from a LiteLLM / provider exception.

    Returns a non-negative float (seconds) or None if no hint is present.
    Handles both dict-shaped `exc.headers` and string message bodies.
    """
    # 1. Direct header attribute (some LiteLLM exceptions expose this)
    headers = getattr(exc, "headers", None)
    if isinstance(headers, dict):
        for key in ("retry-after", "Retry-After", "RETRY-AFTER", "x-ratelimit-reset"):
            val = headers.get(key)
            if val is None:
                continue
            try:
                seconds = float(val)
                if seconds >= 0:
                    return seconds
            except (TypeError, ValueError):
                pass

    # 2. response attribute (litellm often nests the original response here)
    response = getattr(exc, "response", None)
    if response is not None:
        resp_headers = getattr(response, "headers", None)
        if isinstance(resp_headers, dict):
            for key in ("retry-after", "Retry-After", "RETRY-AFTER"):
                val = resp_headers.get(key)
                if val is None:
                    continue
                try:
                    seconds = float(val)
                    if seconds >= 0:
                        return seconds
                except (TypeError, ValueError):
                    pass

    # 3. Fallback: scrape the message body for `retry-after: N` style
    msg = str(exc)
    match = _RETRY_AFTER_HEADER_RE.search(msg)
    if match:
        try:
            seconds = float(match.group(1))
            if seconds >= 0:
                return seconds
        except (TypeError, ValueError):
            pass

    return None


# ---------------------------------------------------------------------------
# Fallback model wrapper construction
# ---------------------------------------------------------------------------

def build_fallback_wrapper(spec: Any, parent_config: dict):
    """Construct a chat-model wrapper for a fallback `spec`.

    `spec` may be:
      - a string: the new model name (inherits api_key/api_base from parent)
      - a dict:   {"model": "...", "api_key": "...", "api_base": "...",
                   "provider": "...", ...}

    `parent_config` is a dict with at least:
      - "model": the current primary model name
      - "provider": the current primary provider
      - "api_key": inherited by default
      - "api_base": inherited by default

    Returns a model object that exposes `.unified_call(...)` and has
    `.model_name`, `.provider`, `.kwargs` like the primary wrapper.
    """
    # Lazy import to avoid module-load order issues with models.py
    import models

    if isinstance(spec, str):
        spec = {"model": spec}
    elif isinstance(spec, (list, tuple)):
        # Defense in depth: `_build_candidates` in fallback.py now normalizes
        # every spec via `_normalize_spec` before it reaches here, so a list
        # should never arrive. If it does (e.g. a future code path that
        # bypasses _build_candidates), fail loud with a hint about the
        # likely cause so the user knows where to look.
        # Added 2026-07-22 (Laci) after the docker log showed
        # `model=[{"model": "..."}]` reaching litellm.
        raise TypeError(
            f"Fallback spec must be str or dict, got {type(spec).__name__}: "
            f"{spec!r}. This is usually caused by a double-nested list in "
            f"the fallback preset (e.g. `fallbacks: [[{...}]]` instead of "
            f"`fallbacks: [{...}]`). The fix is in `_build_candidates` "
            f"in fallback.py -- this guard is a backstop."
        )
    elif not isinstance(spec, dict):
        raise TypeError(f"Fallback spec must be str or dict, got {type(spec).__name__}")

    new_model_name = spec.get("model")
    if not new_model_name:
        raise ValueError("Fallback spec is missing 'model' key")
    if not isinstance(new_model_name, str):
        # The cascade has no way to recover from a non-string model name
        # (the litellm call would fail downstream with a confusing
        # "LLM Provider NOT provided" error). Fail loud with a message
        # that names the likely cause: a user preset where the `model`
        # key holds a list-of-dict instead of a bare model-name string.
        # Common in hand-edited presets that accidentally nested a list
        # under `model:` instead of putting the entries at the top
        # level of the spec dict.
        # Added 2026-07-21 (Laci) after the docker log showed every
        # fallback candidate returning BadRequestError with a stringified
        # list as the `model` argument to acompletion().
        raise TypeError(
            f"Fallback spec 'model' must be a string, got "
            f"{type(new_model_name).__name__}: {new_model_name!r}. "
            f"Check the fallback list in the agent's preset / WebUI "
            f"and ensure each entry is either a bare model-name "
            f"string or a dict like {{'model': '<model-name>'}}."
        )

    # Strip the explicit provider prefix from the model name. The framework's
    # `get_chat_model` / `LiteLLMChatWrapper` expects a bare model name; the
    # provider is passed separately as the `provider=` arg. If we leave the
    # prefix in `new_model_name` (e.g. "ollama_cloud/gemma4:31b"), the
    # wrapper builds `model_name = f"{provider}/{model}"` which produces
    # "openai/ollama_cloud/gemma4:31b" (double prefix). LiteLLM then parses
    # `openai/` as the provider, strips it from the model name, and sends
    # "ollama_cloud/gemma4:31b" upstream -- which Ollama Cloud rejects with
    # 401 (it doesn't recognize the "ollama_cloud/" prefix in the model
    # name; the bare model is just "gemma4:31b").
    inferred_provider_for_prefix = None
    if "/" in new_model_name:
        inferred_provider_for_prefix, _, stripped = new_model_name.partition("/")
        # Only strip the prefix if it looks like a known provider name
        # (lowercase letters, digits, underscore, hyphen). Otherwise leave
        # alone (e.g. model name "gpt-4o/something" wouldn't be a provider
        # prefix even though it contains a slash).
        if inferred_provider_for_prefix.replace("_", "").replace("-", "").isalnum():
            new_model_name = stripped
            # If the spec didn't carry an explicit "provider", use the one
            # we just stripped -- it's a strong hint.
            if not spec.get("provider"):
                spec["provider"] = inferred_provider_for_prefix

    # Inherit missing fields from parent. We do this BEFORE resolving the
    # final provider, so the spec has a chance to carry the key through
    # transiently. The next block (after we know the final `provider`)
    # will drop the inherited key if the parent and the fallback are
    # different providers -- otherwise the parent's key (e.g. a0_venice)
    # would leak into a wrapper for a different vendor (e.g. nvidia_nim)
    # and the upstream would reject the call as "Incorrect API key".
    # Note: by the time we get here, the spec may have a `provider` key
    # that was set by the prefix-strip block above (e.g. "groq" for
    # "groq/x"). We use that as a hint: don't inherit api_key/api_base
    # from the parent if the spec already has an explicit provider
    # (those values were chosen for that provider, not for the parent).
    spec_has_explicit_provider = bool(spec.get("provider"))
    for k in ("api_key", "api_base"):
        if spec.get(k):
            # Spec already carries this field; user-provided values win.
            continue
        if spec_has_explicit_provider:
            # Spec names a different provider than the parent; the parent's
            # credentials don't apply. Leave the field absent and let
            # get_api_key() resolve from env below.
            continue
        if parent_config.get(k):
            spec[k] = parent_config[k]

    # Determine provider: explicit > auto-derive from spec model name > parent.
    # The parent's provider is only a fallback when neither the spec nor the
    # model name gives a clear hint -- otherwise the parent provider leaks
    # into fallbacks for a different service (e.g. ollama for ollama_cloud).
    if spec.get("provider"):
        provider = spec["provider"]
    elif "/" in new_model_name:
        provider = _infer_provider(new_model_name)
    else:
        provider = parent_config.get("provider") or _infer_provider(new_model_name)

    # NOTE: The previous "drop inherited api_key if parent_provider !=
    # provider" block is now redundant -- the inheritance step above
    # already skips api_key/api_base whenever the spec carries an
    # explicit `provider` (or has one inferred from a model-name prefix).
    # What remains is to resolve the api_key from the env in the case
    # where the spec has none and the parent's key was dropped.

    # Resolve the litellm_provider from the provider registry. The base
    # LiteLLMChatWrapper constructor does NOT call _merge_provider_defaults,
    # so a wrapper built directly with provider="ollama_cloud" leaves
    # litellm_provider at "ollama_cloud" -- which LiteLLM then routes
    # through its Ollama native adapter (appending /api/generate to api_base)
    # instead of the OpenAI-compat adapter the YAML declares.
    try:
        from helpers.providers import get_provider_config
        cfg = get_provider_config("chat", provider)
        if cfg and cfg.get("litellm_provider"):
            litellm_provider = cfg["litellm_provider"].lower()
        else:
            litellm_provider = provider
    except Exception:
        litellm_provider = provider

    # Build kwargs: start with parent's, override with spec's
    parent_kwargs = {}
    # The parent's own kwargs aren't passed in; rebuild from parent_config as a minimum
    # but the caller usually passes only essentials here.
    kwargs = dict(spec)
    # Strip non-LiteLLM keys
    for k in ("model", "provider"):
        kwargs.pop(k, None)
    # Merge parent kwargs (e.g. timeout, custom_llm_provider).
    # CRITICAL: do NOT inherit api_base OR api_key from the parent. The
    # parent's credentials are for the parent's provider. If we leak the
    # parent's api_key into a different-provider wrapper (e.g. the user's
    # openrouter key into a groq wrapper), the upstream rejects the call
    # as "Incorrect API Key" / "invalid_api_key" and the real key for the
    # target vendor (looked up later via `models.get_api_key()`) never
    # gets a chance to be used. The previous fix in lines 298-302 popped
    # the inherited key off `spec`, but the loop below then re-added the
    # parent's key from `parent_config["kwargs"]` via setdefault. This
    # block is the actual defense: skip both api_key and api_base when
    # merging from parent. Same provider => the api_key/api_base the
    # _get_litellm_chat lookup put on the primary is still wrong for the
    # spec (different model on the same vendor), so we still drop them
    # here and let get_api_key() fill them in below.
    parent_kwargs_to_skip = {"api_key", "api_base"}
    for k, v in (parent_config.get("kwargs") or {}).items():
        if k in parent_kwargs_to_skip:
            # Drop the parent's credentials. Either the spec carries its
            # own (already in kwargs from `dict(spec)`), or we'll resolve
            # via get_api_key(litellm_provider) just before constructing
            # the wrapper.
            continue
        kwargs.setdefault(k, v)
    # Drop A0-only / provider-invalid keys so they never reach acompletion().
    # The util_model_call_before / chat_model_call_before extensions also strip,
    # but doing it here keeps the wrapper self-consistent.
    for k in (
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
        # Provider-specific nested kwargs that belong to ONE provider's API
        # contract. They get copied in from the parent above (the parent
        # might be a0_venice), and would be rejected by any other provider
        # (e.g. Groq returns 400 "property 'venice_parameters' is
        # unsupported"). Mirrors _A0_ONLY_KWARGS in fallback.py.
        # Added 2026-07-21 (Laci).
        "venice_parameters",
        "a0_api_mode",
    ):
        kwargs.pop(k, None)

    # Apply provider YAML defaults (api_base, etc.) just like get_chat_model does.
    # This is what fixes the /v1/api/generate bug: the YAML's
    # `ollama_cloud.litellm_provider: openai` and `kwargs.api_base` get applied.
    try:
        from helpers import settings as _settings
        from models import _merge_provider_defaults
        _litellm_provider, kwargs = _merge_provider_defaults("chat", provider, kwargs)
        if _litellm_provider:
            litellm_provider = _litellm_provider
    except Exception:
        pass

    # CRITICAL: resolve the api_key the same way _get_litellm_chat does.
    # The primary's get_chat_model() path goes through _get_litellm_chat,
    # which calls `api_key = kwargs.pop("api_key", None) or get_api_key(provider_name)`.
    # The plugin's wrapper builder previously called LiteLLMChatWrapper(...)
    # DIRECTLY and skipped that lookup, so fallback wrappers were constructed
    # with api_key=None and LiteLLM sent unauthenticated requests upstream.
    # Groq responds to those with `400 invalid_api_key`, Ollama/Codex with
    # `403 no quota left` (a misnomer -- it's actually "no auth, can't check
    # quota"), and so on. The user has all the keys in their .env but the
    # plugin was never reading them. This step fixes that.
    #
    # Order of precedence matches _get_litellm_chat exactly:
    #   1. explicit api_key in the spec (kwargs already has it)
    #   2. models.get_api_key(provider) -- the SERVICE name, e.g. "nvidia_nim"
    #   3. models.get_api_key(litellm_provider) -- the upstream provider,
    #      e.g. "openai" (which nvidia_nim reuses). Fallback only.
    # If the user supplied a key in the spec, that wins.
    #
    # IMPORTANT: previously this used `litellm_provider` (e.g. "openai")
    # only. That's wrong for OpenAI-API-compatible vendors that share the
    # "openai" litellm_provider with the real OpenAI service -- the user's
    # NVIDIA key is stored under API_KEY_NVIDIA_NIM, not API_KEY_OPENAI,
    # and get_api_key("openai") returns "None". The result: every
    # nvidia_nim fallback wrapper was constructed with api_key=None and
    # LiteLLM sent unauthenticated requests upstream, which NVIDIA NIM
    # answers with a generic 500 ("could not resolve response") that
    # looks identical to a real upstream outage. Resolution: try the
    # service name first (matches what _get_litellm_chat does), then
    # fall back to the litellm_provider for the case where the user
    # actually stored the key under the upstream provider's name.
    existing_key = kwargs.get("api_key")
    if not existing_key or existing_key in (None, "", "None", "NA"):
        # Build a small ordered list of lookup names. We try the service
        # name first because that's where users put vendor-specific keys
        # in .env (API_KEY_NVIDIA_NIM, API_KEY_GROQ, etc.). The
        # litellm_provider ("openai" for many of these) is the
        # generic-OpenAI fallback that often returns "None" -- but for
        # some vendors it's the only place the key lives, so we keep it
        # as a last resort.
        lookup_names: list[str] = []
        for candidate in (provider, litellm_provider):
            if candidate and candidate not in lookup_names:
                lookup_names.append(candidate)
        for name in lookup_names:
            try:
                resolved = models.get_api_key(name)
            except Exception:
                continue
            if resolved and resolved not in (None, "", "None", "NA"):
                kwargs["api_key"] = resolved
                break

    try:
        wrapper = models.LiteLLMChatWrapper(
            model=new_model_name,
            provider=litellm_provider,
            **kwargs,
        )
    except TypeError:
        # Older signatures may not accept **kwargs on the constructor;
        # retry with just the essentials.
        wrapper = models.LiteLLMChatWrapper(
            model=new_model_name,
            provider=provider,
            api_key=kwargs.get("api_key", ""),
            api_base=kwargs.get("api_base", ""),
        )
    return wrapper


def _infer_provider(model_name: str) -> str:
    """Best-effort provider inference from a model name string."""
    name = model_name.lower()
    if "/" in name:
        return name.split("/", 1)[0]
    # Common patterns
    if name.startswith("gpt-") or name.startswith("o1") or name.startswith("o3"):
        return "openai"
    if name.startswith("claude"):
        return "anthropic"
    if name.startswith("gemini"):
        return "google"
    if name.startswith("mistral") or name.startswith("mixtral"):
        return "mistral"
    if name.startswith("llama") or name.startswith("meta-llama"):
        return "meta_llama"
    if name.startswith("command"):
        return "cohere"
    return "openai"  # safest default for LiteLLM


# ---------------------------------------------------------------------------
# Callback resolution
# ---------------------------------------------------------------------------

def resolve_callback(callback):
    """Split a combined (chunk, total) callback into (response_cb, reasoning_cb).

    Returns a (response_callback, reasoning_callback) tuple suitable for
    `unified_call`. If `callback` is None, returns (None, None).
    """
    if callback is None:
        return None, None

    async def _response_cb(chunk: str, total: str):
        if callback is not None:
            return await callback(chunk)
        return None

    async def _reasoning_cb(chunk: str, total: str):
        # Best effort: route reasoning to the same callback if it accepts it
        if callback is not None:
            try:
                return await callback(chunk)
            except Exception:
                return None
        return None

    return _response_cb, _reasoning_cb
