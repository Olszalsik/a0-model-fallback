# AGENTS.md — `_model_fallback` plugin

This file documents the **intended behavior** of the `_model_fallback` plugin so
that future maintainers (human or AI) do not regress the cascade into a
"showstopper" design. The plugin exists because the host environment mixes
**quota-limited cloud providers** with **rate-limited gateways**, so a single
provider going dark for hours is normal operating conditions, not a failure.

**If you are about to change `fallback.py`, read this first.**

---

## What this plugin does

`fallback.py` monkey-patches two methods onto the `Agent` class:

| Method | Patched at | Purpose |
|---|---|---|
| `Agent.call_utility_model` | `_patched_call_utility_model`, `fallback.py:694` | Wraps the utility model call (used for memory, summarization, JSON validation, tool sub-tasks). |
| `Agent.call_chat_model` | `_patched_call_chat_model`, `fallback.py:1075+` | Wraps the main chat model call. |

Both wrappers share the same contract: build a candidate list from the active
preset (`_build_candidates`), then loop through candidates until one succeeds.

This plugin also owns six **LLM-error-handling** extensions on top of the
cascade. They live here (not in separate plugins) so a single
`usr/plugins/_model_fallback/.toggle-0` disables everything and a single
`default_config.yaml` holds the knobs:

* v2.2 Piece 1 — outer timeout guard on `call_utility_model` (catches
  hung utility calls that the cascade's own timeout misses).
* v2.2 Piece 2 — TTL cache + circuit breaker on `get_webui_extensions`
  (breaks the 150ms WebUI polling storm).
* v2.3 Piece A — opt-in context-size guard on the live
  `loop_data.history_output` (prevents the all-models-ContextOverflow
  cascade death-spiral).
* v2.4 Piece B — `langchain.prompts` / `langchain.schema` v0 -> v1
  import shim (fixes `ModuleNotFoundError` on langchain v1-only docker
  images; default ON with a no-op install for v0 users).

In v2.5 each piece gained its own top-level boolean on the
WebUI settings page so a future agent-zero update can replace one
of them without dragging the rest along. See the
[WebUI configuration](#v25--webui-configuration-2026-07-20) section.

See the [v2.2](#v2.2--ui--event-loop-resilience-2026-07-20) and
[v2.3+v2.4](#v23v24--llm-error-handling-expansion-2026-07-20)
sections below for the full design.

## Cooldown table (unchanged by the continuous-fallback changes)

Each candidate's `(agent_id, model_label)` pair is held in a per-process
in-memory cooldown dict (`_get_cooldown_store`, `fallback.py:754`). Permanent
errors set long cooldowns; transient errors set short cooldowns:

| Error | Cooldown | Treated as | Rationale |
|---|---|---|---|
| 400 `invalid_api_key` / `unauthorized` | 24 h | permanent | API key is wrong or revoked — won't fix itself |
| 404 model not found / deprecated | 24 h | permanent | Model is gone — won't fix itself |
| Context overflow | 24 h | permanent | Same prompt will overflow again |
| 402 payment required | 1 h | long-retry | May recover when credits are added |
| 401, 403 | 5 min | short-retry | Quota / auth may reset |
| 429 rate limit | Retry-After header (else 60s/5min) | transient | Will recover on its own |
| 408, 500-504 | 60-120 s | transient | Server hiccup |
| Timeout (60s) | (no separate cooldown) | transient | Treated as transient; cycle to next candidate |
| `ImportError`, `ModuleNotFoundError`, `CodeError` | (no cooldown) | fatal | Bug, not API failure — fail fast |

The cooldown dict is intentionally **in-memory only** so a laptop sleep /
container restart resets it. A copy is also persisted to `agent.data` and
used only as a bootstrap read on first call after restart (see the comment
at `fallback.py:65-80`).

## Continuous-fallback mode (the new default)

**Added 2026-07-19 in response to a quota-storm failure on `omniroute`.**

When `continuous_fallback: true` in `config.json` (the new default), the
cascade has three properties that together guarantee the agent stays alive
across multi-hour provider outages:

### 1. The loop never breaks

The `if max_cycles > 0 and cycle_count >= max_cycles: break` checks in
both `_patched_call_utility_model` (`fallback.py:864-867` before the
patch, now wrapped in `if not continuous_mode and ...`) and
`_patched_call_chat_model` (`fallback.py:1148-1150` before the patch)
are skipped. The loop becomes effectively `while True` with a backoff.

`_maybe_raise_retry_after_hours` (`fallback.py:564`) also short-circuits
in continuous mode: it returns `None` immediately so the cascade never
raises `RetryAfterHours`, which is what `_error_retry` only retries once
(it's a `HandledException` consumer, not an infinite-retry consumer —
see `plugins/_error_retry/.../_80_retry_critical_exception.py:13-14`).

### 2. After every cycle, the primary model is retried first

The cascade resets the rotation index to `0` after every cycle via
`_set_model_idx(idx=0)` at `fallback.py:857` (utility) and `fallback.py:1230`
(chat). This means any per-cycle cooldown that has expired during the sleep
is automatically re-tried on the next pass — the primary provider always
gets a free second chance every cycle. This is the whole point of the
system when providers have rate limits that recover.

### 3. The backoff envelope throttles the noise

Flat `cycle_delay: 5s` would produce ~720 log lines per hour of outage
and 2880 per 4 hours. The backoff envelope (`_compute_cycle_sleep`,
defined inside each cascade) ramps the sleep exponentially:

```
sleep = min(cycle_delay * backoff_multiplier ** consecutive_full_cycles,
            max_cycle_delay_s) + jitter
```

With the defaults (`cycle_delay=5`, `backoff_multiplier=2.0`,
`max_cycle_delay_s=300`, `backoff_jitter_s=2.0`), the cadence is:

| Cycle | Sleep | Cumulative time |
|---|---|---|
| 1 | 5 s | 5 s |
| 2 | 10 s | 15 s |
| 3 | 20 s | 35 s |
| 4 | 40 s | 75 s |
| 5 | 80 s | 155 s |
| 6 | 160 s | 315 s |
| 7+ | 300 s (+0-2s jitter) | ~6.3 min and beyond |

So a 4-hour outage produces ~80 log lines instead of ~3000.

When any candidate succeeds, the `consecutive_full_cycles` counter resets
to `0` and a one-time `... recovered after N full cycle(s) ...` log line
fires (`fallback.py:_succeed`, both cascades). This lets you correlate
recovery with provider dashboards (look for the cyan line in the log).

## Configuration knobs

`usr/plugins/_model_fallback/config.json`:

| Key | Default | Effect |
|---|---|---|
| `continuous_fallback` | `true` | If `true`, never stop the cascade. |
| `max_cycle_delay_s` | `300` | Cap on the backoff envelope. |
| `backoff_multiplier` | `2.0` | Exponential growth factor per consecutive full cycle. `1.0` disables growth. |
| `backoff_jitter_s` | `2.0` | Random extra seconds per sleep (0 to jitter). Prevents lockstep retries across agents. |
| `fallback_max_cycles` | `4` | Legacy cap, ignored when `continuous_fallback=true`. |
| `fallback_cycle_delay` | `5` | Base sleep between cycles (also the first sleep). |
| `fallback_attempt_delay` | `2` | Sleep between candidates inside one cycle. |
| `fallback_timeout_s` | `60` | Per-call timeout for the chat cascade. |
| `fallback_utility_timeout_s` | `30` | Per-call timeout for the utility cascade. |
| `extended_retry_enabled` | `false` | Legacy phase-A/phase-B scheduler, ignored in continuous mode. |
| `phase_a_delay_s` | `900` | Legacy phase A delay. |
| `phase_b_delay_s` | `3600` | Legacy phase B delay. |
| `initial_cycle_attempts` | `60` | Legacy phase A burst size. |
| `force_chat_completions_api_bases` | `["ollama.com"]` | Force chat-completions API on these bases. |
| `early_exit_on_all_permanent` | `true` (default in code) | If all attempted models in a cycle hit a permanent error, skip the cycle log. |
| `responses_5xx_retry_enabled` | `true` (default in code) | Retry once with chat-completions on a 5xx from the Responses API. |

## How to tune later

The user said: *"if the cycles are making problems, we can fine tune them
to not cycle that aggressively or something, but I don't want this to
be a showstopper"*. The levers, in order of how invasive each change is:

1. **Raise `max_cycle_delay_s`** (e.g. 600 or 1200) — fewer log lines per
   outage, slower recovery.
2. **Lower `backoff_multiplier`** (e.g. 1.5) — slower ramp, more chances
   to catch a quota refresh.
3. **Raise `fallback_cycle_delay`** (e.g. 15) — fewer cycles, slower
   recovery, but also slower first-attempt on a fresh outage.
4. **Add more fallback candidates** in `presets.yaml` — diversity
   reduces the chance that every provider is in cooldown at once.

## When continuous mode is wrong

There is exactly one documented downside: the
`Utility model [N] timed out after 30s` /
`Chat model [N] timed out after 60s` log lines will repeat at the
backoff cadence (5s → 300s) as long as every provider stays down. The
backoff envelope keeps this at ~80 lines per 4 hours instead of ~3000,
but the noise is still there. If this becomes a problem in practice,
the knobs above are the right place to start — do not turn off
continuous mode and put the agent back in the "die after 4 cycles"
behaviour unless the user explicitly asks for it.

## Why the sleep is decomposed into 2-second slices (Phase 4, 2026-07-19)

`asyncio.sleep` already yields to the event loop, so a single
`await asyncio.sleep(80)` is not blocking. **But** it does ask the
scheduler to come back in 80 seconds, which is fine in isolation and
fragile in practice:

* During a multi-minute sleep the agent's monologue extension
  points, the WebSocket dispatcher on the ASGI loop, and the
  StateMonitor push loop are all technically eligible to run, but
  long single sleeps starve any code that depends on regular
  event-loop "heartbeats" (e.g. the WebSocket ping/pong, the
  `monologue_end` cleanup, the `job_loop` extensions).
* Some asyncio scheduling implementations (and the
  `nest_asyncio` patch Agent Zero uses in v2.5) have surprising
  behaviour on long single sleeps, particularly for tasks
  that themselves await asyncio primitives.

`_yielding_sleep(s)` (added 2026-07-19 in
`fallback.py:_yielding_sleep`) decomposes the cycle sleep into a
0.25s first slice (prompt cancellation) followed by 2s slices for the
remainder. Every slice invokes
`usr.plugins.memory_hardening.helpers.coroutine_guard.on_long_sleep_tick`
so the dashboard can confirm the loop is alive.

## Why the timeout closes the inner coroutine (Phase 4, 2026-07-19)

The cycle's per-call timeout is implemented as
`asyncio.wait_for(coro, timeout=...)`. When the timeout fires,
`asyncio.wait_for` cancels the outer task, but the litellm transport
(`helpers/litellm_transport.py:239`) has by that point constructed a
chain of inner coroutines (`OpenAIChatCompletion.acompletion` etc.)
that, on the cancellation path, are NEVER awaited. Python then logs:

    RuntimeWarning: coroutine 'OpenAIChatCompletion.acompletion'
        was never awaited

and the inner coroutine's frame, with its references to the request
payload, the response iterator, the httpx transport, and the openai
client, stays on the heap until garbage-collected by the next
collection cycle. Across a 4-minute outage with 5 timeouts, that's
5 half-open HTTP connections pinning the openai client's connection
pool.

The 2026-07-19 patch captures the inner coroutine explicitly and
calls `.close()` on it inside the timeout handler. This releases
the frame immediately and eliminates the warning. The close path
uses a conservative allowlist of coroutine name prefixes
(`OpenAIChatCompletion`, `OpenAIResponses`, `ChatCompletion`) so
we can never accidentally close a framework coroutine.

The plumbing lives in
`usr.plugins.memory_hardening.helpers.coroutine_guard.close_inner_coro`.
The `_model_fallback` plugin imports it lazily, so the cascade
continues to work when `memory_hardening` is disabled (the import
catches `Exception` and no-ops).

## Cross-references

- Plugin config: `usr/plugins/_model_fallback/config.json`
- Patch entrypoint: `usr/plugins/_model_fallback/extensions/python/agent_init/_00_install_fallback_patches.py`
- v2.2 timeout guard: `extensions/python/agent_init/_10_install_utility_timeout_patch.py` + `helpers/utility_timeout.py`
- v2.2 extensions cache: `extensions/python/_functions/run_ui/init_a0/start/_10_install_extensions_cache.py` + `helpers/webui_extensions_cache.py`
- v2.3 context-size guard: `extensions/python/message_loop_prompts_after/_10_context_size_guard.py` (the `trim_history` function is pure and testable)
- v2.4 langchain v1 shim: `extensions/python/agent_init/_00_install_langchain_shim.py` + `helpers/langchain_compat.py`
- Stats endpoint: `api/stats.py` (`GET /api/plugins/_model_fallback/stats`)
- Preset that triggers the most cycling: `usr/plugins/_model_config/presets.yaml:1-32` (Default preset, `utility: gemma4:31b` on `omniroute`)
- Single-call timeout source: `usr/plugins/omniroute/default_config.yaml:32` (`timeout_seconds: 30`)
- WebSocket heartbeat tuning (companion fix): `usr/.env:158-159`
- Coroutine guard consumer: `usr/plugins/memory_hardening/AGENTS.md` and `usr/plugins/memory_hardening/helpers/coroutine_guard.py`

---

## v2.2 — UI resilience (2026-07-20)

Originally three new pieces added in response to a WebSocket-disconnect
investigation (see docker log: agent idle, 8 minutes of
"WebSocket disconnected" with no recovery). The standalone
event-loop housekeeping loop (Piece 2 in the original design)
was removed in v2.5; the outer timeout guard (Piece 1) and the
WebUI extensions cache (Piece 3 in the original design, now
Piece 2) are still here. All remaining pieces are independent
toggles in `default_config.yaml`; the existing `fallback.py`
cascade is unchanged.

### Piece 1 — Outer timeout guard on `call_utility_model`

**Files:** `helpers/utility_timeout.py`,
`extensions/python/agent_init/_10_install_utility_timeout_patch.py`.

`Agent.call_utility_model` is monkey-patched **on top of the cascade** at
`agent_init` time. The cascade is installed by `_00_*`; this guard is
`_10_*` (the framework sorts extensions by module name, so the cascade is
in place by the time we run). The guard wraps the (now-cascaded) call in
`asyncio.wait_for(..., timeout=...)`. On `asyncio.TimeoutError` it:

1. Best-effort closes the inner litellm coroutine chain via
   `usr.plugins.memory_hardening.helpers.coroutine_guard.close_inner_coro`
   (lazy import; no-op if the plugin is disabled).
2. Raises `helpers.errors.RepairableException("utility_model_timeout")` so
   the agent's monologue exception handler catches it and tells the LLM
   the utility call failed (the LLM can then rephrase or skip). The
   cascade itself sees the propagated exception.

**Defaults:** `default_timeout_s: 30` (matches `misformat_guard.cascade.timeout_s`),
`max_wait_s: 120` (hard cap; raise for very slow ollama CPU models),
`jitter_s: 1.0` (random spread so concurrent agents don't lockstep),
`close_inner_on_timeout: true`.

**Why not use the implicit `start` extension point?**
`Agent.call_utility_model` is `@extension.extensible`, so a `start` hook
exists. But the framework's `_run_async` skips the original function when
a start hook sets `data["result"]` to a coroutine (it does not await it
in the start branch). That makes the `start` hook unable to wrap the
function body — it would replace it. The monkey-patch approach is the
only one that preserves the cascade + the inner `util_model_call_before`
extension point.

**Migration if upstream changes:** if a future agent-zero version
refactors `call_utility_model` to lose the `@extensible` decorator or
moves the cascade out of the monkey-patch, the wrapper here still works
as long as the function is still a regular `async def` on `Agent`. If
the function is renamed, update `_install()` in
`_10_install_utility_timeout_patch.py` accordingly. The `_utility_timeout_patched`
sentinel prevents double-install during refactors.

### Piece 2 — `get_webui_extensions` TTL cache + circuit breaker

**Files:** `helpers/webui_extensions_cache.py`,
`extensions/python/_functions/run_ui/init_a0/start/_10_install_extensions_cache.py`.

The `get_webui_extensions` helper is monkey-patched at `init_a0/start`
with a 2-second TTL cache. The cache is busted by an extension watchdog
we register (same roots as the framework's own watchdog, so plugin
enable/disable/file-edit is reflected immediately — no stale UI).

The circuit breaker is the second defensive layer. When
`get_webui_extensions` itself raises more than
`circuit_breaker_threshold` (default 5) times in
`circuit_breaker_window_s` (default 10s), the breaker opens and the
wrapper returns `[]` immediately without touching the filesystem, for
`circuit_breaker_recovery_s` (default 30s). This keeps the WebUI
responsive when the framework's FS layer is wedged (e.g. on a slow
network mount).

**Why we patch the helper and not the endpoint:** `ApiHandler.process`
is **not** `@extension.extensible` in v2.5 (verified in
`helpers/api.py:33-90`). The endpoint's only real work is
`extension.get_webui_extensions(...)`, so caching the helper gives a
near-instant response with no framework changes. The patch is sentinel-
guarded so re-running `init_a0` (e.g. test harness, plugin reload)
preserves the original reference — the wrapper stack stays at exactly
one layer.

**Migration if upstream changes:** if a future agent-zero version adds
a built-in TTL cache to `get_webui_extensions`, set
`webui_extensions_cache.enabled: false` and delete
`helpers/webui_extensions_cache.py` + the `init_a0/start` hook.
The watchdog listener that busted the cache on file-watch events
is registered in the same `init_a0/start` extension (the
`init_a0/end` extension is gone in v2.5 together with the
housekeeping loop).

### Stats endpoint

`GET /api/plugins/_model_fallback/stats` returns:

```json
{
  "version": "2.5.0",
  "utility_timeout": {
    "calls_total": 89,
    "timeouts_total": 3,
    "max_observed_wait_s": 67.2,
    "last_timeout_at": 1753000000.0,
    "last_timeout_model": "gemma4:31b",
    "close_inner_attempted": 3,
    "close_inner_succeeded": 3
  },
  "extensions_cache": {
    "hits": 1842,
    "misses": 312,
    "errors": 0,
    "circuit_opened_at": 0.0,
    "circuit_open_count": 0,
    "circuit_short_circuits": 0,
    "last_error": "",
    "busts": 7,
    "circuit_open": false,
    "hit_rate": 0.855
  },
  "context_size_guard": {
    "trims": 4,
    "messages_dropped": 18,
    "last_kept": 12,
    "last_dropped": 6
  },
  "langchain_compat": {
    "installed": true,
    "shims": {
      "langchain.prompts": true,
      "langchain.schema": true
    }
  }
}
```

A simple WebUI tile that polls this every 10s and renders three
traffic-light cells would let the user see at a glance whether the
resilience layer is healthy. (Tile not included in v2.4; the
endpoint is enough for a future tile to consume.)

### How to disable each piece independently

| Piece | Config key | Default | Disable by |
|---|---|---|---|
| Utility timeout guard | `utility_timeout_guard.enabled` | `true` | set to `false` |
| Extensions cache | `webui_extensions_cache.enabled` | `true` | set to `false` (cache becomes pass-through) |
| Circuit breaker | `webui_extensions_cache.circuit_breaker_enabled` | `true` | set to `false` (cache still works, breaker open) |
| Context size guard (v2.3) | `context_size_guard.enabled` | `false` | set to `true` to enable (opt-in) |
| LangChain v1 shim (v2.4) | `langchain_compat.enabled` | `true` | set to `false` (v0 users get a no-op install, no impact) |

The `housekeeping.enabled` and `housekeeping.second_pulse_path_enabled`
keys are no-ops in v2.5 (the loop is gone); they are still recognised
so a hand-edited config does not raise on save.

Disabling any one piece does not affect the others. The framework
re-reads the config on the next call (no restart required for
`utility_timeout_guard`, `webui_extensions_cache`,
`context_size_guard`, and `langchain_compat`).

### Why this is in `_model_fallback` and not its own plugin

The user preferred to keep all four pieces in the existing plugin so
a single `.toggle-0` disables everything and a single config file
holds the knobs. The pieces fall into two groups:

* **Cascade-coupled** (timeout guard, context-size guard): these
  touch the LLM-call path directly and need to know the cascade's
  internals. Splitting them out would require duplicating config
  plumbing.
* **Cascade-adjacent** (extensions cache, langchain shim): these
  run before or after the cascade and are loosely coupled, but the
  user preferred single-plugin symmetry
  over per-piece independence.

If a future maintainer wants to split, the migration is
straightforward: each piece is a single helper + a single
extension hook; nothing shares state across pieces.

### Verification

* `python scripts/scan_v22_to_v25.py` — must stay at 0 LIVE findings.
* `python scripts/scan_plugin_structure.py` — must stay at 0 findings
  (or only the pre-existing non-fatal ones).
* `usr/plugins/_model_fallback/tests/test_resilience_v22.py` —
  unit tests for the v2.2 layer (cache circuit breaker, timeout
  guard's `close_inner_coro` lazy import). Housekeeping tests
  were removed in v2.5.
* `usr/plugins/_model_fallback/tests/test_context_size_guard_v23.py` —
  11 unit tests for the v2.3 context-size guard.
* `usr/plugins/_model_fallback/tests/test_langchain_compat_v24.py` —
  8 unit tests for the v2.4 langchain v0 -> v1 shim.
* Manual: `/api/plugins/_model_fallback/stats` returns sensible
  numbers after a 2-minute idle period; `langchain_compat.installed`
  is `true` on a v1-only docker image.


## v2.3+v2.4 — LLM-error-handling expansion (2026-07-20)

Two version bumps, two independent fixes (the third, the
second WebSocket pulse path, was removed in v2.5 together with
the housekeeping loop that hosted it). All under the same "this
is an LLM-side error the cascade can't recover from" umbrella.
None of them is a network or quota problem; the cascade is
unchanged. Each fix is opt-in (or default-on with a no-op
install path for envs that don't need it) and is plugin-only —
no official file is modified.

### Piece A — Context size guard (v2.3, opt-in)

The "all 6 utility models fail with ContextOverflow, cascade
cycles for 5 minutes" failure mode the user reported. The
memorize 50k-char patch in v2.2 (and `memory_hardening`) trim the
persistence path, but they do NOT trim
`loop_data.history_output` — the LIVE list `prepare_prompt` reads
at agent.py:602 to build the LLM prompt. A 2.6M-token history
sends every utility model into ContextOverflow, which feeds the
cascade's extended-retry mode, which sleeps for 5 minutes between
cycles.

The new hook at
`extensions/python/message_loop_prompts_after/_10_context_size_guard.py`
fires AFTER the framework sets `loop_data.history_output` (agent.py:575)
and BEFORE the prompt is built (agent.py:602). It:

1. Sums the char count of every `OutputMessage` in
   `loop_data.history_output`.
2. If the total exceeds `max_chars` (default 50000, ~12.5k
   tokens, well within a 32k context), drops the oldest messages
   until the total fits.
3. Always preserves the most recent `min_messages` (default 2) —
   the LLM must see the last user turn + last agent reply.
4. Prepends a synthetic notice message so the LLM knows older
   context was trimmed.
5. Updates the `context_size_guard` counter so the WebUI tile
   shows how often trims have fired.

Configuration (`default_config.yaml:context_size_guard`):
```yaml
context_size_guard:
  enabled: false        # default off; aggressive trim can confuse the LLM
  max_chars: 50000      # ~12.5k tokens
  min_messages: 2       # never trim below this
```

Disable by setting `enabled: false`; the hook becomes a one-line
no-op. The cascade's existing `ContextOverflow` recovery path is
unchanged — the trim is purely additive.

### Piece B (REMOVED in v2.5) — Second WebSocket pulse path

> The second `socketio.emit` WebSocket pulse path was removed in
> v2.5 together with the housekeeping loop that hosted it. The
> `second_pulse_path_enabled` key is still accepted by
> `helpers/toggles.py` for back-compat with hand-edited configs
> but the runtime ignores it.
>
> If a future agent-zero version ships an equivalent fallback path
> for engine.io polling-transport clients, you can leave the key
> at its default (`false`) and let the upstream one do the work.

### Piece C — LangChain v0 -> v1 import compatibility shim (v2.4, default ON)

Merged from the now-deleted standalone `_langchain_compat`
plugin. The user preferred to keep all LLM-error-handling fixes
in `_model_fallback` so a single toggle disables everything.

The bug: LangChain v1 moved several submodules to `langchain_core`
(`langchain.prompts` -> `langchain_core.prompts`,
`langchain.schema` -> `langchain_core.messages`). Docker images
that only ship langchain v1 raise
`ModuleNotFoundError: No module named 'langchain.prompts'` when
core code at `helpers/call_llm.py:2` does a v0-style import. The
cascade can't recover from a ModuleNotFoundError because the
import happens BEFORE any model call — this is an LLM-stack
interface error, not a network or quota error, so it belongs in
this plugin.

The shim is in `helpers/langchain_compat.py` and is installed at
`agent_init` by
`extensions/python/agent_init/_00_install_langchain_shim.py`. It:

1. Resolves the v1 source modules
   (`langchain_core.prompts`, `langchain_core.messages`).
2. Registers them under the v0 paths
   (`langchain.prompts`, `langchain.schema`) in `sys.modules`,
   tagged with a sentinel attribute so uninstall removes only
   shims WE installed.
3. Skips any legacy name that already has a real module in
   `sys.modules` (a v0 user with the actual `langchain.prompts`
   keeps theirs; the shim is a no-op for them).
4. Marks the process as "shim installed" so subsequent
   `agent_init` calls (one per agent) skip the work.

Configuration (`default_config.yaml:langchain_compat`):
```yaml
langchain_compat:
  enabled: true         # default ON; v0 users get a no-op install
```

Disable by setting `enabled: false`; the agent_init hook
becomes a one-line no-op. The shim also uninstalls cleanly via
`hooks.py:uninstall()` (which calls `langchain_compat.uninstall_shim`).

### v2.5 update — process-start install (was: agent_init only)

In v2.4 the shim only fired on the first `agent_init` extension.
On a v1-only docker image, that is too late: any utility-model
call that runs BEFORE the first agent is created (e.g. during
framework bootstrap or a pre-agent memory warmup) hits
`helpers/call_llm.py:2`'s v0 import and raises
`ModuleNotFoundError`. The cascade can't recover from that error
because the import happens before any model call.

v2.5 adds a **second** install path in `hooks.py:install()` that
runs at **process startup** — before any agent, before any
utility-model call, before any extension loader. The agent_init
extension is kept as a defensive backup for reloader scenarios.

The two paths cooperate safely:
* `install()` is idempotent via `already_installed_in_process()`.
  If the shim is already in `sys.modules`, both paths skip.
* The process-start install reads the same
  `langchain_compat_enabled` toggle (`config.json` > `default_config.yaml`
  > default ON) so the user's OFF choice still wins.
* `uninstall()` already calls `uninstall_shim()` and
  `clear_installed_marker()`, so disabling the plugin cleans
  up the shim from `sys.modules` entirely.

**Workaround for upstream bug:** `helpers/call_llm.py:2` is a
core file and is not modified by this plugin. The fix is a
`sys.modules` shim that intercepts the v0 import name and
redirects it to the v1 source module. When the upstream
maintainer patches `helpers/call_llm.py` to import from
`langchain_core.prompts` directly, this shim becomes a no-op
(the install path skips any legacy name that already resolves
in `sys.modules`). No action required from you after such an
update — leave the toggle ON or turn it OFF; both are safe.

**Verification after a process start** (no agent needed):
```python
import sys
# either:
assert "langchain.prompts" in sys.modules
# or, more precisely:
from usr.plugins._model_fallback.helpers import langchain_compat
assert langchain_compat.already_installed_in_process()
```

### v2.3+v2.4 tests

* `usr/plugins/_model_fallback/tests/test_context_size_guard_v23.py` —
  11 tests for Piece A (config resolve, trim under/over budget,
  `min_messages` invariant, empty history, list/string content,
  counter snapshot, disabled-by-default contract).
* `usr/plugins/_model_fallback/tests/test_langchain_compat_v24.py` —
  8 tests for the langchain v0 -> v1 shim (install, idempotency,
  process marker, user-module protection, uninstall, status
  shape, fallback target, location-in-_model_fallback invariant).
* `usr/plugins/_model_fallback/tests/test_resilience_v22.py` —
  tests for the v2.2 layer (cache circuit breaker, timeout
  guard's `close_inner_coro` lazy import). The two-path
  WebSocket pulse tests were removed in v2.5 together with
  the housekeeping module that hosted them.

### Migration (if upstream agent-zero adds an equivalent)

* If upstream adds a `loop_data.history_output` size limit
  before the prompt is built: delete
  `extensions/python/message_loop_prompts_after/_10_context_size_guard.py`
  and remove the `context_size_guard` block from
  `default_config.yaml`.
* (v2.5) The second WebSocket pulse path was removed in this
  branch; the corresponding `helpers/housekeeping.py:_try_ws_pulse`
  two-path code is no longer in the tree.
* If upstream agent-zero updates `helpers/call_llm.py` to use
  the v1 langchain paths, OR ships a built-in shim: delete
  `helpers/langchain_compat.py` and
  `extensions/python/agent_init/_00_install_langchain_shim.py`,
  then remove the `langchain_compat` block from
  `default_config.yaml` and the related code from `hooks.py`.

---

## v2.5 — WebUI configuration (2026-07-20)

User-facing additions in v2.5:

1. Six **per-feature toggles** on the WebUI settings page so each
   resilience piece can be turned OFF independently as upstream
   agent-zero releases equivalents.
2. A **user manual** (`webui/help.html`) reachable from a book
   icon next to the "Resilience features" section heading. The
   manual covers all six features, the migration story, and a
   "when to disable" guide.
3. An **advanced settings** disclosure on the same page that
   lists the per-feature inner knobs (timeouts, intervals,
   thresholds) for users who want to fine-tune the defaults.

### Per-feature toggles (top-level keys on `context.settings`)

| Setting key | Default | What it controls |
|---|---|---|
| `utility_timeout_guard_enabled` | true | Outer `asyncio.wait_for` on `Agent.call_utility_model` (`max_wait_s` default 120). |
| `housekeeping_enabled` | false (v2.5) | **No-op in v2.5** — the standalone loop was removed. The key is still accepted for back-compat. |
| `second_pulse_path_enabled` | false (v2.5) | **No-op in v2.5** — the second `socketio.emit` path was removed with the loop. The key is still accepted for back-compat. |
| `webui_extensions_cache_enabled` | true | 2s TTL cache + circuit breaker on `get_webui_extensions`. |
| `context_size_guard_enabled` | false | Opt-in history trim on `ContextOverflow` spirals. |
| `langchain_compat_enabled` | true | `langchain.prompts` / `langchain.schema` v0 -> v1 import shim. |

The top-level key is the **WebUI's source of truth** — when it
is present its value wins, even if the legacy nested section
(`<piece>.enabled`) has the opposite value. This is so a user
toggling OFF in the WebUI always takes effect on the next
server start. When the top-level key is absent, the runtime
falls back to the nested section so hand-edited
`config.json` files that pre-date v2.5 keep working.

Resolution rules (also in `helpers/toggles.py`):

```
1. cfg[<piece>_enabled]  (if key is present — even if False)
2. cfg[<piece>]["enabled"]  (if nested section present)
3. per-piece built-in default
```

The settings round-trip through
`helpers.plugins.get_plugin_config`, which respects the
framework's per-project / per-agent / global resolution chain.
A `config.json` saved by the WebUI in the project directory is
the persisted form.

### Where the toggles live in the WebUI

`usr/plugins/_model_fallback/webui/config.html` injects a new
"Resilience features" section between the existing "Fallback
cycles" and "Quick presets" sections. Each toggle is a
`text-input`-free checkbox with a `version-badge` (v2.2 / v2.3
/ v2.4) and a one-paragraph description. The settings bind via
the framework's standard `x-model="context.settings.<key>"`
pattern; `webui/fallback-store.js` exposes
`isEnabled(settings, piece)` and
`isSecondPulsePathEnabled(settings)` for the `:checked`
binding so the displayed state matches the resolved
top-level-or-nested rule.

### Restart requirements

| Toggle | Takes effect on |
|---|---|
| `utility_timeout_guard_enabled` | next `agent_init` (next agent created, or server restart) |
| `housekeeping_enabled` | **No-op in v2.5** — the loop is gone, so restart requirement is moot. |
| `second_pulse_path_enabled` | **No-op in v2.5** — the second pulse path is gone. |
| `webui_extensions_cache_enabled` | next `init_a0` (server restart) |
| `context_size_guard_enabled` | next agent turn (the hook re-resolves per call) |
| `langchain_compat_enabled` | next `agent_init` (next agent created, or server restart) |

### When to disable a feature (summary)

The full per-feature "when to disable" guide is in
`webui/help.html` (linked from the settings page). Quick
summary:

* **Utility-model timeout guard** — disable only if you have a
  model with a known slow first-byte that you want to wait out
  past `max_wait_s` (raise the cap instead, or disable).
* **Housekeeping** — *no longer a feature in v2.5*; the toggle
  is a no-op. If a future agent-zero version ships an equivalent
  idle loop, leave the key at its default and disable the
  upstream one instead.
* **Second pulse path** — *no longer a feature in v2.5*; the
  toggle is a no-op.
* **WebUI extensions cache** — disable during plugin
  development if the 2s TTL hides file edits. Re-enable when
  done.
* **Context-size guard** — disable if the trim confuses the
  LLM (it forgets early instructions). Try a larger model
  window or shorter conversation instead.
* **LangChain shim** — disable only on langchain v0.x if the
  shim causes surprising behavior (it shouldn't; the shim is
  a no-op when the v0 module is already in `sys.modules`).

### v2.5 implementation files

* `helpers/toggles.py` — the per-piece top-level toggle
  resolver. Single source of truth for the resolution order
  (top-level -> nested -> default). Imported by every
  extension's `_resolve_config` to short-circuit when the
  toggle is OFF.
* `webui/fallback-store.js` — extends the existing Alpine
  store with `isEnabled` / `isSecondPulsePathEnabled` /
  `applyDefaults` helpers that mirror the Python resolution
  rules exactly. The store keeps the existing
  `getDefaults` / `normalizeSettings` / `openHelp` surface
  intact.
* `webui/config.html` — adds the "Resilience features"
  section (6 toggles + advanced disclosure) and a help icon
  that opens `help.html` via the existing `$store.modelFallback.openHelp()`.
* `webui/help.html` — rewritten as a 7-section user manual
  (Overview, Cascade, Features, Toggles, When to disable,
  Migration, Observability). Each feature has its own
  sub-section with "symptom it fixes / what it does / default"
  prose.
* `default_config.yaml` — adds the four top-level
  `*_enabled` keys (plus the `housekeeping_enabled` and
  `second_pulse_path_enabled` no-op shims) with the
  conservative defaults. The nested sections stay for
  back-compat with hand-edited configs.
* `plugin.yaml` — bumped to 2.5.0, description mentions the
  per-feature WebUI toggles + user manual.
* `api/stats.py` — version string bumped to "2.5.0".

### v2.5 tests

* `usr/plugins/_model_fallback/tests/test_webui_toggles_v25.py` —
  covers:
  * Per-piece top-level toggle resolution (top-level wins over
    nested; nested falls through to default; explicit default
    override; garbage config returns default; back-compat
    with hand-edited config).
  * Second-pulse-path sub-toggle (top-level, nested fallback,
    top-level wins, default ON) — `toggles.is_second_pulse_path_enabled`
    is a no-op back-compat shim in v2.5, but the precedence
    contract is still asserted.
  * Each remaining extension's `_resolve_config` returns
    `{"enabled": False}` early when the toggle is OFF.
  * The v2.5 housekeeping tests (housekeeping module,
    two-path WS pulse) were removed together with the module.

The full plugin suite is N/N. Any pre-existing failure in
`test_resilience_v22.py` is unrelated to the v2.5 housekeeping
removal.

### Migration (v2.5 toggles -> upstream equivalents)

If a future agent-zero release ships its own equivalent of
one of the v2.2-v2.4 features:

1. Open the plugin settings page.
2. Toggle OFF the matching `*_enabled` switch.
3. Restart the server (or wait for the next `agent_init` /
   `init_a0` per the table above).
4. Confirm the upstream equivalent is active (e.g. the new
   keepalive ping in the server logs, or the new cache in the
   `get_webui_extensions` response headers).
5. If the migration is permanent, the corresponding helper
   file can be deleted:
   * `helpers/utility_timeout.py`
   * `helpers/webui_extensions_cache.py`
   * `helpers/langchain_compat.py`
   * `extensions/python/message_loop_prompts_after/_10_context_size_guard.py`

(`helpers/housekeeping.py` is already gone in v2.5; the
`init_a0/end/_20_start_housekeeping_loop` and
`job_loop/_10_idle_housekeeping_safety_net` extensions are
already gone as well.)

The toggles, the `helpers/toggles.py` module, and the WebUI
"Resilience features" section can all be deleted when the
upstream equivalents land. The `webui/help.html` and
`webui/fallback-store.js` would lose the toggle rows but the
rest of the page (timeout / cycles / presets) is independent
of the v2.2-v2.4 features.

## v2.6 — Adaptive fallback cascade (2026-07-28)

v2.6 makes the cascade *adaptive*: it distinguishes a warm provider from a
cold one, a "busy" 429 from a "broken" one, a sustained outage from a
transient blip, and it stops preempting healthy calls. Five phases, all in
`fallback.py` (Phase 4 touches the inner cascade only; the outer
`utility_timeout.guarded_call` is unchanged). Added 2026-07-28; v2.6.1
(2026-08-07) fixes a chat-cascade wiring gap in Phase 1.

### Phase 1 — Warm/cold per-call timeouts

A label that succeeded within `cascade_warm_window_s` (default 600s) is
"warm" and gets the short `cascade_warm_timeout_s` (default 20s) instead of
the cold default (`fallback_utility_timeout_s` for utility,
`fallback_timeout_s` for chat, typically 90-300s). Cloud providers return
warm calls in ~1.2s (a0_venice) or 2-5s (nvidia_nim); a flat 90-300s ceiling
on a warm call wastes minutes per candidate when several are hung.

- `_WARM_LABELS: dict[str, float]` (module-level, `fallback.py:162`) maps
  `label -> time.monotonic()` of the last successful call. Set in `_succeed`
  (utility `:1583`, chat `:2199`).
- `_resolve_per_call_timeout(label, base, warm, window)` (`:184`): returns
  `warm` if `time.monotonic() - _WARM_LABELS.get(label, 0.0) < window`, else
  `base`. Strict `<`, so a label exactly at the window boundary is cold.
  Defensive `try/except` falls back to `base`.
- A user-set `TIMEOUT=` model kwarg wins (legacy contract): the call site
  checks `user_kwarg_set = "TIMEOUT" in model_kwargs or "timeout" in
  model_kwargs` first (`:1660` utility, `:2264` chat) and skips the warm
  logic entirely.
- `_WARM_LABELS` is module-level, so the warm signal is shared across agents
  in the same process (same lifecycle as `_INMEM_HEALTHY_LABELS`). Reset on
  process restart.

### Phase 2 — Per-provider capacity inference

`_classify_capacity(label)` (`fallback.py:775`) returns one of:

- `concurrent_paid` — local `ollama/*` (exact provider). Local ollama
  genuinely accepts many parallel requests to localhost; a 429 means a
  competing agent is holding a slot. **No cooldown is written** —
  `_handle_error_cooldown` returns `False` (`:841`) — and the primary-skip
  escalation is skipped (`_maybe_extend_primary_cooldown` returns early,
  `:1325` utility / `:1996` chat).
- `unlimited_paid` — `a0_venice/*`. Quota-exhaustion 429; uses the configured
  cooldown (default 30s).
- `free_per_minute` — everything else, including `ollama_*` (e.g.
  `ollama_cloud`, which is metered), `nvidia_nim`, `openrouter`, `groq`,
  `mistral`, `cohere`, `together`, etc. Uses the configured cooldown.

This is a pure function over the label string (the part before the first
`/`). There is **no `provider_profiles` config block** — inference was
chosen over configuration so the user does not maintain a label-to-provider
mapping; the plan's `provider_profiles` block was dropped in favor of this.
Wired at three sites: `_handle_error_cooldown` rate-limit branch (`:841`)
and `_maybe_extend_primary_cooldown` in both cascades (`:1325`, `:1996`).

### Phase 3 — Adaptive cycle sleep on stagnation

When `consecutive_no_success_cycles >= cycle_stagnation_threshold` (default
2) — every candidate still in cooldown and the cascade keeps paying the
cycle-sleep tax for no benefit — `_compute_cycle_sleep` multiplies the
capped sleep by `cycle_stagnation_factor` (default 1.5), then re-caps at
`max_cycle_delay_s` (default 300s). One log line per outage transition
(gated by closure-local `stagnation_logged_this_outage`), not per cycle. The
counter increments on every full cycle with no success (`:1522` utility /
`:2141` chat) and resets on any success (`:1609` utility / `:2224` chat),
which also re-arms the one-log-per-outage gate.

The backoff envelope is unchanged: `cycle_delay * backoff_multiplier **
min(consecutive_full_cycles, 10)`, capped at `max_cycle_delay_s`, plus
optional `backoff_jitter_s` spread. Stagnation amplification is applied
after the cap and re-capped.

### Phase 4 — Don't preempt healthy calls

The inner cascade's per-candidate `wait_for` (`_call_utility_model`
`:1668-1699`, `_call_chat_model` `:2272-2294`) distinguishes:

- `asyncio.CancelledError` (external cancellation from the outer guard or
  container shutdown) — just `raise`. **No close-coroutine cleanup.** The
  call was healthy and running; the cancel was externally forced. Closing
  the inner coro preemptively would just add a `RuntimeWarning` with no
  benefit.
- `asyncio.TimeoutError` (wait_for fired at the timeout budget) —
  `_inner_coro.close()` then `raise`. Legacy frame-leak prevention; the call
  was genuinely hung.

The outer guard `helpers/utility_timeout.guarded_call` is **unchanged** —
Phase 4 only touches the inner cascade. The plan's "don't preempt past 50%
wall-clock" gate was abandoned during implementation because
`asyncio.wait_for` raises at the timeout boundary (not after), so `elapsed`
is always `>= timeout_s` at the catch — a 50% gate would be meaningless. The
CancelledError-vs-TimeoutError split honors the same intent more
conservatively.

### Phase 5 — Primary-skip healthy-label reset

`_maybe_extend_primary_cooldown` (utility `:1297`, chat `:1968`) gains a
short-circuit: after the `concurrent_paid` skip (Phase 2), it calls
`_maybe_clear_cooldown_for_healthy_label(self, label)`. If that returns
`True` — a peer agent proved the label healthy within `health_horizon_s`
(default 60s) AND this agent's local cooldown is stale (the
only-cleared-never-overwritten invariant from v2.5.2) — it resets
`_consecutive_primary_failures = 0` and returns early, skipping the
escalation. This prevents the case where one agent's stale primary failures
keep escalating while another agent's success on the same label would have
cleared the cooldown entirely.

The helpers (`_INMEM_HEALTHY_LABELS`, `_mark_label_healthy`,
`_maybe_clear_cooldown_for_healthy_label`) are v2.5.2; Phase 5 is the wiring
of the existing healthy-label reset into the primary-skip escalation path,
for the primary candidate only (`idx == 0`).

### v2.6.1 — Chat-cascade warm-read fix (2026-08-07)

The chat cascade (`_patched_call_chat_model`) referenced
`warm_timeout_s`/`warm_window_s` at its per-candidate warm resolution
(`:2269`) but **never read them from config**. The utility cascade reads
them at `fallback.py:1153-1158`; the chat cascade's setup at `:1838-1843`
read `fallback_timeout_s` and `timeout_s` but omitted the two warm reads.
The other chat-cascade mirrors (`_compute_cycle_sleep`,
`_maybe_extend_primary_cooldown`, the success-path `_WARM_LABELS` write)
were all copied correctly — only the warm config reads were dropped.

Consequence: with no user `TIMEOUT` kwarg on the chat model (the common
case), the warm-path branch executed with `warm_timeout_s`/`warm_window_s`
unbound -> `NameError` raised in the loop body before the try/except,
killing the chat call. With a `TIMEOUT` kwarg present, the warm logic was
skipped -> no crash, but Phase 1 warm/cold was silently dead for chat
(always the cold timeout).

Fix: added the two config reads after `fallback.py:1843`, mirroring
`1153-1158`. Regression test `tests/test_chat_warm_wiring_v26.py` is a
compile-time check (`_patched_call_chat_model.__code__.co_varnames` must
contain `warm_timeout_s`/`warm_window_s`; pre-fix it did not). Verified the
test fails on the pre-fix backup and passes post-fix. Backup:
`fallback.py.before-v2.6.1-chat-warm.bak`.

### v2.6.2 — Router capacity class (2026-08-08)

**Problem.** OmniRoute (`omniroute/*`) is a local Docker gateway that routes
one request across 230+ upstream LLM providers with a 4-tier internal
fallback (Sub → Key → Cheap → Free). `_classify_capacity` was classifying
`omniroute/*` as `free_per_minute` (the conservative default for unknown
providers), so a 429/5xx from the gateway got the same cooldown treatment
as a single free-tier endpoint: a propagated `Retry-After` (≤ 3600s), a
401/403 (300s), or a no-`Retry-After` 429 (`rate_limit_no_retry_after_cooldown_s`).
But OmniRoute is **self-healing** — a 429/5xx usually means one upstream
tier failed and the gateway re-routes the *next* request to a healthy tier.
Any real cooldown over-locks it; the user saw the router get needlessly
stuck in cooldown mode when it could have served the next call.

**Fix — a new `router` capacity class.** `_classify_capacity` now returns
`"router"` for labels whose provider is in the hardcoded set
`_DEFAULT_ROUTER_PROVIDERS = ("omniroute",)` (case-insensitive, checked
before the final `free_per_minute` fallthrough). A new helper
`_capacity_skips_cooldown(label)` returns
`_classify_capacity(label) in ("concurrent_paid", "router")` and replaces
the three previous `== "concurrent_paid"` skip checks (the 429 branch in
`_handle_error_cooldown`, and the two `_maybe_extend_primary_cooldown`
primary-skip sites). So for a `router` label:

- a **429** writes **no cooldown** and returns `False` (the gateway re-routes
  on the next cascade pass — mirrors the existing `concurrent_paid` /
  local-ollama skip);
- a **5xx/transient** error writes a short `router_cooldown_s` (read from
  config, default 5s, clamped to `[0, 60]` and bypassing the 30s floor) via a
  new branch in `_handle_error_cooldown`'s transient tail — enough to space
  out a 5xx storm without locking the router out for the 60-120s a normal
  endpoint would get; `0` retries immediately;
- the **primary-skip escalation** is skipped (a router failing is not a
  signal to escalate a 2-min cooldown onto the primary — the primary itself
  is fine; the router just re-routes).

**Why `omniroute` only, not `openrouter`.** OpenRouter serves a *specific*
requested model — a 429 on `openrouter/x:free` is a real free-tier rate
limit on that model, not a self-healing re-route. It stays
`free_per_minute` (the existing `test_capacity_v26.py` assertions at lines
142 and 252 encode this and still pass). OmniRoute's 4-tier auto-fallback is
the distinguishing property: the gateway, not the caller, picks the
upstream per request. A provider enters the `router` class only when the
gateway is the unit that self-heals. If a future gateway needs the same
treatment, add it to `_DEFAULT_ROUTER_PROVIDERS` (a config-driven
`router_providers` list is intentionally deferred — the hardcoded set keeps
`_classify_capacity` a pure function over the label).

**Config.** New top-level knob in `default_config.yaml`:

| Knob | Default | Effect |
|---|---|---|
| `router_cooldown_s` | 5 | 5xx/transient cooldown for `router` labels; clamped `[0, 60]`; `0` retries immediately |

**Tests.** `tests/test_router_capacity.py` — 7 tests: `omniroute/*`
classifies as `router` (incl. 3-segment labels + case-insensitivity);
`openrouter/*` stays `free_per_minute` (over-classification guard);
`_capacity_skips_cooldown` True for `ollama/*` + `omniroute/*`, False
otherwise; a fake 429 on `omniroute/auto` writes no cooldown (returns
`False`); a fake 500 on `omniroute/auto` writes `router_cooldown_s` (≤ 60s,
not the 120s default); and a 500 on `nvidia_nim/*` still gets the normal
120s transient cooldown (the router short-cooldown does not leak). Full
suite (159 tests) green. Backup: `fallback.py.before-v2.6.2-router.bak`.

### v2.6.3 — Router warm-timeout exemption (2026-08-08)

**Problem.** The v2.6 cascade gives a "warm" label (used within
`cascade_warm_window_s`, default 600s) a short per-call ceiling of
`cascade_warm_timeout_s` (default 20s). That ceiling assumes a fast warm
cloud call (~1-5s). OmniRoute (`omniroute/*`) is a self-healing gateway
that routes one request across 230+ upstream tiers, often *free* coding
providers; a routed free call can take far longer than 20s even when the
gateway is "warm". With the "free coding fast" presets (`2 Agent` →
`auto/coding:free`, `Drága` → `auto/best-coding-fast`) using OmniRoute
as the *utility* model, the 20s warm ceiling was timing the utility
call out after exactly 20s.

**Fix.** `_resolve_per_call_timeout` (fallback.py) is now capacity-aware:
`router`-class labels (the A1 `router` capacity class — `omniroute/*`)
skip the warm fast-path and always get the cold `base_timeout_s` ceiling
(`fallback_utility_timeout_s` / `fallback_timeout_s`, default 300s).
Fast routed calls still return fast (the ceiling only bites on slow
calls); slow routed free-tier calls get the headroom they need. The
user-set `TIMEOUT=` model kwarg still wins (legacy contract, unchanged).

**Why not raise `cascade_warm_timeout_s` globally.** That would weaken
the fast-path for every fast warm provider (a0_venice ~1.2s, nvidia_nim
2-5s), letting hung warm calls waste minutes. A capacity-targeted
exemption keeps the 20s fast-path for providers it fits and gives
routers the cold ceiling.

**Tests.** `tests/test_router_capacity.py` — +3 tests (10 total): a warm
`omniroute/auto` returns the cold base (300s, not 20s); a warm
non-router (`a0_venice/*`) still returns the 20s warm timeout (fast-path
intact); a cold label returns the base (unchanged). Full suite (162
tests) green. Backup: `fallback.py.before-warm-router.bak`.

### v2.6 configuration knobs

| Knob | Default | Phase | Effect |
|---|---|---|---|
| `cascade_warm_timeout_s` | 20 | 1 | warm-call timeout ceiling (router labels exempt since v2.6.3 — use the cold base) |
| `cascade_warm_window_s` | 600 | 1 | how long a label stays "warm"; 0 disables warm/cold |
| `cycle_stagnation_factor` | 1.5 | 3 | sleep multiplier at stagnation; 1.0 disables |
| `cycle_stagnation_threshold` | 2 | 3 | consecutive zero-success cycles to trigger amplification |
| `max_cycle_delay_s` | 300 | 3 | hard cap on cycle sleep (re-applied after amplification) |
| `rate_limit_no_retry_after_cooldown_s` | 30 | 2 | cooldown for no-Retry-After 429s (not `concurrent_paid`) |
| `health_horizon_s` | 60 | 5 | cross-agent healthy-label reset window |
| `primary_skip_enabled` | true | 5 | master toggle for primary-skip escalation (incl. Phase 5) |
| `primary_skip_strikes` | 2 | 5 | consecutive primary failures to escalate |
| `primary_skip_cooldown_s` | 120 | 5 | escalation cooldown for a hung primary |
| `router_cooldown_s` | 5 | 2 | 5xx/transient cooldown for `router` labels (v2.6.2); clamped `[0,60]`; `0` retries now |

Phase 2 (`_classify_capacity`) and Phase 4 (the CancelledError-vs-TimeoutError
split) have **no config toggle** — they are inherent behaviors of the cascade,
not opt-in features. To revert Phase 2 for a specific provider, change the
label prefix; to revert Phase 4, restore the legacy cleanup-on-cancel branch.

### v2.6 implementation files

- `fallback.py` — all five phases (warm resolution, capacity inference,
  stagnation, no-preempt, primary-skip reset) in both
  `_patched_call_utility_model` and `_patched_call_chat_model`.
- `default_config.yaml` — the v2.6 knobs (`cascade_warm_*`,
  `cycle_stagnation_*`).
- `helpers/utility_timeout.py` — the outer guard; **unchanged** by Phase 4
  (Phase 4 touches the inner cascade only).
- Tests: `test_warm_timeout_v26.py`, `test_capacity_v26.py`,
  `test_adaptive_sleep_v26.py`, `test_no_preempt_v26.py`,
  `test_primary_skip_v26.py`, `test_chat_warm_wiring_v26.py` (v2.6.1),
  `test_router_capacity.py` (v2.6.2).

### When to disable a v2.6 feature

- **Warm/cold (Phase 1):** `cascade_warm_window_s: 0` in config —
  `_resolve_per_call_timeout` always returns the cold base (the window check
  `time.monotonic() - last < 0` is always False).
- **Stagnation (Phase 3):** `cycle_stagnation_factor: 1.0` — the
  amplification branch is skipped.
- **Primary-skip (Phase 5):** `primary_skip_enabled: false` — disables the
  whole primary-skip escalation including the Phase 5 strike reset.
  (`health_horizon_s` is clamped to a minimum of 5s so it cannot fully
  disable the v2.5.2 cross-agent reset on its own; that reset also runs
  independently on the skip-cooldown path, not only inside Phase 5.)
- **Capacity inference (Phase 2) / no-preempt (Phase 4):** no toggle —
  inherent behavior.
- **Router 5xx spacing (v2.6.2):** `router_cooldown_s: 0` — a 5xx on an
  `omniroute/*` label retries immediately (no spacing). The 429-skip and
  primary-skip exemption for `router` labels are inherent (there is no knob
  to put a router into normal cooldown treatment — change the label prefix
  or remove it from `_DEFAULT_ROUTER_PROVIDERS`).
