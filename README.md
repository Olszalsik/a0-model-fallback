# Model Fallback System (`_model_fallback`)

An [agent-zero](https://github.com/agent0ai/agent-zero) plugin that turns a single
model call into a resilient, multi-candidate cascade with rate-limit evasion,
permanent-failure detection, intelligent cooldown management, and an adaptive
warm/cold timeout strategy — all without modifying any official agent-zero file.

When your primary model rate-limits, times out, auth-fails, or 500s on the
Responses endpoint, the cascade rotates through your fallback candidates and
keeps the agent running. When *every* candidate is rate-limited, it enters an
extended-retry mode paced for real-world reset windows (minutes to hours) instead
of hard-stopping. It also hardens the surrounding pieces that break under load:
the utility-model outer timeout, the memory plugin's recall task, and the
langchain v0→v1 import gap.

**Version:** 2.6.6 · **Self-contained:** no official agent-zero file is modified ·
**Configurable:** per-project and per-agent

---

## How it works

The plugin installs two monkey-patches at agent init
(`extensions/python/agent_init/_00_install_fallback_patches.py`):

- `Agent.call_utility_model` → `_patched_call_utility_model`
- `Agent.call_chat_model`    → `_patched_call_chat_model`

Each patched method builds an ordered list of candidate models from your
`FALLBACK` model chain (via `helpers`/`build_fallback_wrapper`), then walks them
in cycle. On a transient error it writes a cooldown for that label and moves to
the next candidate; on a permanent error (auth, quota, model-not-found) it marks
the candidate dead for the cycle. The cooldown store is in-memory and shared
across agents in the same process, so one agent's rate-limit doesn't silently
re-fire on another. Everything else the plugin does — the outer timeout guard,
the memory patches, the langchain shim, the per-feature toggles — is layered on
top of that core cascade.

No official agent-zero source file is edited. All behavior is installed through
agent-zero's extension hooks (`agent_init`, `*_model_call_before`,
`monologue_start`, `message_loop_prompts_after`, `handle_exception`) plus the
two `Agent.*` monkey-patches.

---

## Features

| Version | Feature |
|---|---|
| **v2.6.6** | **Migrated the server-side WebUI extensions cache to `ui_loader_optimizer` v3.5.0.** The 2s TTL cache + non-lossy circuit breaker on `get_webui_extensions` (and its `webui_extensions_cache_enabled` toggle) moved to the UI Loader Optimizer plugin, which already owned the complementary client-side `fetch` coalescing — the two layers now live in one plugin. This plugin retains the utility-timeout guard, context-size guard, langchain shim, and the cascade. Removed the `webui_extensions_cache` helper, its `run_ui/init_a0/start` install hook, the `_EXTENSIONS_CACHE` counter group, and the `extensions_cache` stats block; fixed the stale hard-coded stats-endpoint version (`2.5.0` → `2.6.6`). |
| **v2.6.5** | **Settings-panel hardening + icon.** (1) The resilience toggle switches now actually save — the panel previously bound each switch with both `x-model` and `:checked`, an Alpine anti-pattern where the one-way `:checked` fought the two-way `x-model`, so clicks never committed and every switch reopened OFF. Switches use `x-model` only and missing keys are backfilled to their effective default on open. (2) The number fields (timeouts / cycles / delays) now save the *clamped* value: previously the box displayed the clamp via `:value` while `x-model` held the raw typed value, so Save persisted e.g. `5` for a timeout that displayed `30`. The `:value` binding is replaced with an `@change` clamp that writes the bounded value back to the same key. Added a plugin-card thumbnail. Rewrote the in-app user manual. *(The non-lossy WebUI cache breaker and the `ttl_s: 0` fix shipped in this row were migrated to `ui_loader_optimizer` in v2.6.6.)* |
| **v2.6.4** | **Warm-label eviction on timeout** — when a warm call times out, the warm label is evicted so the retry uses the cold ceiling instead of looping at the short warm timeout. Label-agnostic (covers `omniroute/*` too); closes the 20s warm-loop lag. *(Shipped in code at commit `f3f6f53`; the `plugin.yaml` manifest version had lagged at 2.6.3 and is caught up by 2.6.5.)* |
| **v2.6.3** | **Router warm-timeout exemption** — `router`-class labels (`omniroute/*`) skip the aggressive 20s warm fast-path and use the cold base ceiling, because a self-healing gateway routing free/slow upstream tiers is not a fast warm cloud call. The 20s warm ceiling was timing out the OmniRoute utility model (`auto/coding:free`, `auto/best-coding-fast`) in the "free coding fast" presets. |
| **v2.6.2** | **Router capacity class** — `omniroute/*` (self-healing gateways) get no cooldown on 429 and a tiny `router_cooldown_s` (default 5s) on 5xx, and are exempt from primary-skip escalation, so the cascade retries the router within seconds instead of locking it out for minutes. OpenRouter stays `free_per_minute`. |
| **v2.6** | **Adaptive fallback cascade** — per-candidate warm/cold timeouts, per-provider capacity inference, stagnation backoff, no-preempt of cancelled-but-healthy calls, cross-agent healthy-label reset. See [below](#the-v26-adaptive-cascade). |
| v2.5.2 | Rate-limit cooldown tuning (no-`Retry-After` 429s) + cross-agent healthy-label reset. |
| v2.5.1 | Primary-skip-after-N-strikes: escalate the primary's cooldown after 2 consecutive failures so the cascade routes around a hung primary instead of paying the full timeout tax every cycle. |
| v2.5 | Per-feature WebUI toggles + user manual; each piece can be turned OFF independently as upstream agent-zero adds equivalents. |
| v2.4 | LangChain v0→v1 import compatibility shim (default ON) so `helpers/call_llm.py`'s v0-style imports resolve on langchain v1-only Docker images. |
| v2.3 | Opt-in chat-history trim when the prompt exceeds `max_chars` (kills the "every utility model returns ContextOverflow" death spiral). |
| v2.2 | Outer timeout guard on `call_utility_model` + TTL cache + circuit breaker on the `get_webui_extensions` polling storm. *(The cache + breaker migrated to `ui_loader_optimizer` in v2.6.6; the timeout guard remains here.)* |
| core | Multi-cycle fallback, rate-limit evasion, permanent-failure detection, cooldown management, memory-plugin resilience, Responses→chat-completions fallback. |

---

## The v2.6 adaptive cascade

Five phases, all wired into **both** the utility and chat cascades:

1. **Per-candidate warm/cold timeouts.** The first call to a label (within
   `cascade_warm_window_s`) gets the cold default (`fallback_*_timeout_s`).
   Subsequent calls within the window get the short warm ceiling
   (`cascade_warm_timeout_s`). Cloud providers return warm calls in 1–5 s; a flat
   90–300 s timeout on a warm call wastes minutes per candidate when several are
   hung. A user-set `TIMEOUT=` kwarg on the model still wins (legacy contract).
   *(v2.6.1 fixed the chat cascade, which shipped the warm call-site but missed
   the config reads — a `NameError` or silently-dead warm path.)*

2. **Per-provider capacity inference** (`_classify_capacity`). Distinguishes
   `concurrent-paid` (local `ollama/*` — a 429 is a competing agent, so no
   cooldown is written and the primary-skip escalation is skipped), `router`
   (`omniroute/*` — a self-healing gateway that re-routes the next call, so a
   429 skips cooldown and a 5xx gets a tiny `router_cooldown_s`; v2.6.2),
   `unlimited-paid` (`a0_venice/*`), and `free-per-minute` (everything else).
   Busy ≠ broken.

3. **Stagnation backoff.** When `cycle_stagnation_threshold` consecutive full
   cycles complete with zero successes (every model still in cooldown), the
   cycle sleep is multiplied by `cycle_stagnation_factor` to give upstreams time
   to recover from quota exhaustion. One log line per outage, not per cycle.

4. **No-preempt of healthy calls.** A `CancelledError` (externally cancelled,
   e.g. the user stopped the run) is split from a `TimeoutError` in the inner
   cascade so an externally-cancelled but otherwise-healthy call is not preempted
   into a cooldown. The outer guard is unchanged.

5. **Cross-agent healthy-label reset.** When any agent succeeds on a label, that
   label is marked healthy for `health_horizon_s`. Other agents' cooldowns for the
   same label become eligible to be cleared (only-cleared, never overwritten) on
   their next cycle — so one agent's rate-limit doesn't lock out a peer that
   needs the same endpoint right now. Wired into the primary-skip escalation so a
   peer's recent success resets stale local strikes.

Full design and migration notes: see `AGENTS.md` → **"v2.6 — Adaptive fallback
cascade"**.

---

## Installation

This is an agent-zero plugin. Drop the directory into your agent-zero plugin
folder:

```
<agent-zero>/plugins/_model_fallback/        # shared / upstream-style
# or
<agent-zero>/usr/plugins/_model_fallback/     # user-local (not overwritten by updates)
```

Then enable it from the agent-zero WebUI (Settings → Plugins) or by ensuring no
`.disabled` marker exists in the plugin directory. The plugin self-registers via
its `plugin.yaml` and installs its patches at agent init — no manual wiring.

> Requires agent-zero v2.5+ (uses the v2.5 extension-hook + WebUI contract).

---

## Configuration

The plugin ships defaults in `default_config.yaml`. Per-install overrides go in
`config.json` (generated/edited via the WebUI or by hand) — **`config.json` is
personal and is excluded from this repo by `.gitignore`.** Both per-project
(`per_project_config: true`) and per-agent (`per_agent_config: true`) config are
supported; the settings surface lives under the `agent` and `developer` sections.

### Key knobs

| Key | Default | Purpose |
|---|---|---|
| `fallback_max_cycles` | `4` | Full passes through all candidates before giving up. |
| `fallback_timeout_s` / `fallback_utility_timeout_s` | `300` / `300` | Cold per-call timeout (chat / utility). |
| `cascade_warm_timeout_s` | `20` | Short ceiling for a label that succeeded within the warm window. |
| `cascade_warm_window_s` | `600` | How long a label stays "warm" after a success. `0` disables. |
| `early_exit_on_all_permanent` | `true` | Stop cycling once every candidate is permanently broken. |
| `extended_retry_enabled` | `true` | Infinite paced retry for long resets (Ollama 5 h limit, daily credit reset). |
| `phase_a_delay_s` / `phase_b_delay_s` | `900` / `3600` | Extended-retry pacing (15 min / 1 h cap). |
| `primary_skip_enabled` | `true` | Escalate primary cooldown after N consecutive failures. |
| `primary_skip_strikes` | `2` | Consecutive primary failures before escalation. |
| `primary_skip_cooldown_s` | `120` | Escalated primary cooldown. |
| `rate_limit_no_retry_after_cooldown_s` | `30` | Cooldown for a 429 with no `Retry-After`. Clamped [10, 3600]. |
| `router_cooldown_s` | `5` | 5xx/transient cooldown for `router` labels (`omniroute/*`). Clamped [0, 60]; `0` retries now. (v2.6.2) |
| `health_horizon_s` | `60` | Cross-agent healthy-label horizon. `0` disables. |
| `cycle_stagnation_factor` / `cycle_stagnation_threshold` | `1.5` / `2` | Stagnation backoff. `factor=1.0` disables. |
| `force_chat_completions_api_bases` | `["ollama.com"]` | Proactively force `/v1/chat/completions` for these upstreams (Responses 5xx). |
| `responses_5xx_retry_enabled` | `true` | Reactively retry a Responses 5xx once on chat-completions, then sticky. |
| `memory_recall_timeout_s` | `90` | Memory plugin recall timeout (core default 30). |
| `utility_timeout_guard_enabled` | `true` | Outer timeout on `call_utility_model`. |
| `context_size_guard_enabled` | `false` | Opt-in history trim on ContextOverflow spiral. |
| `langchain_compat_enabled` | `true` | v0→v1 import shim. |

### Forcing chat-completions for OpenAI-compatible providers

Some providers (notably `ollama.com` for `:cloud` models) 500 on `/v1/responses`
while `/v1/chat/completions` works. The plugin handles this two ways, both without
editing tracked files:

- **Proactive (Option B):** force chat-completions for any provider in
  `force_chat_completions_providers`, any model name matching
  `force_chat_completions_patterns`, or any model whose `api_base` contains a
  substring in `force_chat_completions_api_bases`. (The `api_base` matcher is the
  reliable one for fallback wrappers, which carry `provider="openai"`.)
- **Reactive (Option D):** if `responses_5xx_retry_enabled`, a model that still
  5xxs on Responses is retried once on chat-completions, then marked sticky.

---

## WebUI

The plugin ships a config panel and a user manual:

- `webui/config.html` — per-feature toggles (each piece can be turned OFF).
- `webui/help.html` — the user manual (also served as the plugin's help page).
- `webui/fallback-store.js` — the settings store front-end (toggle resolution + backfill).
- `webui/thumbnail.png` — the plugin-card icon (generated by `scripts/gen_model_fallback_thumbnail.py`).

The WebUI binds to the top-level flat keys (e.g.
`utility_timeout_guard_enabled`) with `x-model` only; the nested per-piece
`enabled` flags are still honoured for back-compat. Missing top-level keys are
backfilled to their resolved default on open (v2.6.5), so a default-ON feature
displays ON even when the loaded config predates the toggle.

> **WebUI extensions cache moved (v2.6.6):** the server-side TTL cache +
> circuit breaker on `get_webui_extensions` used to live here as the
> `webui_extensions_cache_enabled` toggle. It migrated to the **UI Loader
> Optimizer** plugin (v3.5.0), which already owned the complementary
> client-side `fetch` coalescing — the two layers now live in one plugin. See
> `help.html` §7.

---

## Testing

Tests live in `tests/` and cover each version's surface — adaptive sleep,
capacity classification, warm-timeout wiring, no-preempt, primary-skip, the
langchain shim, spec validation, cooldown dedupe, etc. Run from the agent-zero
repo root:

```bash
# The plugin imports via `usr.plugins._model_fallback`, so set the repo root
# (tests default to the container path /a0):
REPO_ROOT_OVERRIDE="$(pwd)" pytest usr/plugins/_model_fallback/tests/ -v
```

The v2.6.1 regression test (`tests/test_chat_warm_wiring_v26.py`) is a
compile-time guard that the chat cascade binds `warm_timeout_s`/`warm_window_s`
and still calls `_resolve_per_call_timeout` — it fails on the pre-v2.6.1 code
and passes after the fix.

---

## Self-contained guarantee

No official agent-zero file is modified. The plugin is installed entirely
through agent-zero's extension hooks and two `Agent.*` monkey-patches applied at
agent init. Removing the plugin (or disabling it via a `.disabled` marker)
restores stock behavior. v2.5's per-feature toggles let you turn each piece OFF
independently as upstream agent-zero adds an equivalent, so the plugin degrades
gracefully rather than fighting the framework.

---

## Status

Private testing repo. The plugin will be contributed to the agent-zero plugin
store once it has been tested thoroughly; a plugin-store manifest/compatibility
check will be run at that time. See `AGENTS.md` for the full per-version design
notes and migration guidance.