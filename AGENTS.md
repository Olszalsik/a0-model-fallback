# AGENTS.md — `model_fallback` plugin

This file documents the **intended behavior** of the `model_fallback` plugin so
that future maintainers (human or AI) do not regress the cascade into a
"showstopper" design. The plugin exists because the host environment mixes
**quota-limited cloud providers** with **rate-limited gateways**, so a single
provider going dark for hours is normal operating conditions, not a failure.

**If you are about to change `fallback.py`, read this first.**

---

## v3.4.0 — quota exhaustion is not a hard stopper

Production report: ollama.com returned
`RateLimitError: ... you (olszalsik) have reached your session usage limit,
upgrade for higher limits ... or add usage credits ...` and the agent died
with `RetryAfterHours: All model candidates are currently unavailable.
Retrying in 60s.` plus a Chat Shepherd `Internal Server Error`.

**The provider never said when the window resets.** ollama.com sends that 429
with no `Retry-After` and no reset timestamp. There is nothing to parse, so
the plugin must supply the policy — that is the whole point of this section.

### The three bugs that produced the report

1. **No distinction between a transient 429 and a spent quota.** Both matched
   `_is_rate_limited_error`, so an exhausted *plan* took the transient path
   and booked `rate_limit_no_retry_after_cooldown_s` (30s). The cascade then
   re-paid the full round-trip — litellm retries twice on its own — once a
   minute against an endpoint guaranteed to refuse until the window rolled
   over. New: `models_ext.detect_quota_scope` / `_is_quota_exhaustion_error`
   classify the window (`session`/`daily`/`weekly`/`monthly`/`account`) from
   the message text; `_quota_cooldown_seconds` (fallback.py) turns that into
   a duration, preferring any provider reset hint over the guess.
2. **Reset headers were nearly invisible.** `extract_retry_after_seconds`
   checked a hardcoded 4-tuple of capitalisations on `exc.headers` only, and
   did `float(val)`. It therefore missed `x-ratelimit-reset-requests: 8.64s`
   (a *duration with a unit*, not a number), every other capitalisation, and
   every header present only on `exc.response.headers`. It also had no
   HTTP-date support. Now: case-insensitive walk of both header dicts, unit
   parsing, HTTP-date, and `x-ratelimit-reset` disambiguated epoch-vs-delta by
   magnitude (>1e9 s cannot be a plausible delta).
3. **The message read as a hard failure.** A quota window is a self-healing
   condition, but the raise sites emitted the "All model candidates are
   currently unavailable" wording. `RetryAfterHours` now takes a `reason`;
   `reason="quota_<scope>"` produces "quota exhausted … will resume in ~N min
   when the window resets". `_70` also raises its sleep ceiling from 7200s to
   43200s **for quota reasons only** — the old clamp silently truncated the
   longer quota waits.

### Invariants — do not regress these

- **Quota exhaustion is a COOLDOWN policy, never a "dead model" verdict.** The
  label stays in the rotation and is re-probed once its window is believed
  expired. It must never be classified permanent-for-rotation
  (`_is_permanent_for_rotation` stays False for it) and must never
  dead-mark (`_mark_label_dead`).
- **A provider reset hint always beats our guessed scope duration**, and
  `quota_cooldown_max_s` always caps the result. An over-long guess costs
  latency on that one label; an unbounded one would park it for a week.
- **The 30s transient default is unchanged** for genuine throttles
  ("Too many Requests", "server is busy"). A shared-pool burst that clears in
  seconds must not be locked out for 15 minutes.
- **Both `RetryAfterHours` raise sites are quota-aware** — the turn path's
  `n <= 1` branch and the all-candidates-exhausted branch. They are the two
  frames in the reported traceback.

Config: `quota_scope_cooldown_s` (per-scope overrides) and
`quota_cooldown_max_s`. Tests: `tests/test_quota_exhaustion_v340.py`.

> Note: `tests/test_capacity_v26.py::test_handle_error_cooldown_30s_for_unlimited_paid`
> used the message `"Quota exhausted"`, which is a genuine quota phrase and is
> now (correctly) routed to the quota cooldown. Its fixture was changed to
> `"Too many requests"` so it tests the transient path it documents.

### Review finding — the `cfg0` NameError (fixed, do not reintroduce)

The first cut of the `n <= 1` quota branch read plugin config inside a
`try/except` and then dereferenced `cfg0` **outside** it:

```python
try:
    cfg0 = _get_plugin_cfg(self)
    ...
except Exception:
    extended_on, continuous_on = True, False   # cfg0 never bound
...
float(cfg0.get("quota_cooldown_max_s", ...))   # NameError
```

A config-read failure therefore raised `NameError` inside the one code path
whose entire purpose is to keep the agent alive. Confirmed by execution, not
inspection. `cfg0` is now pre-initialised (`cfg0: dict = {}`) and the cap read
has its own `try/except`. Pinned by
`test_quota_branch_survives_config_read_failure` +
`test_quota_delay_is_defined_when_config_is_unavailable`.

**Generalisation for this file:** a `try/except` that assigns a config dict and
falls back to a literal must bind the literal *before* the `try`, and every
later use of that name needs its own guard. The plugin already carries
several `except: cfg = {}` shapes — each one is a latent `NameError` if a
later line assumes success.

### Classification matrix (verified, not assumed)

| case | status | rate-limited | quota | scope | dead-marked |
| --- | --- | --- | --- | --- | --- |
| ollama session limit | 429 | yes | yes | `session` | no |
| OpenAI insufficient_quota | 429 | yes | yes | `account` | no |
| generic quota text | 403 | yes | yes | `account` | yes (pre-existing) |
| credit balance low | 402 | no | yes | `account` | yes (pre-existing) |
| model gone | 404 | no | no | `""` | yes (pre-existing) |
| transient throttle | 429 | yes | no | `""` | no |
| server error | 500 | no | no | `""` | no |

The 403/402 rows are pre-existing behaviour and are deliberately left alone:
`_is_dead_for_all_agents` treats 401/402/403 as shared, expiring dead-marks,
which is correct because the credentials are shared across agents. Only the
429 rows (your actual case) are guaranteed to stay in rotation.

---

## v3.3.0 — merged guards (`asyncio_guard`, `import_guard`)

Two standalone plugins were folded in as `helpers/asyncio_guard.py` and
`helpers/import_guard.py`, installed from
`extensions/python/startup_migration/_00_install_early_guards.py`. Toggles:
`asyncio_read_ready_guard_enabled`, `extension_import_guard_enabled` (both
default ON).

### Why `startup_migration` and not `hooks.py`

This is the load-bearing decision; do not "simplify" it back.

A plugin's `hooks.py` is imported **only** by
`helpers.plugins.call_plugin_hook`, which is lazy and on-demand. Nothing
iterates enabled plugins calling `install()` during startup — `install()`
fires only when a plugin is installed/updated through
`plugins/_plugin_installer`. `always_enabled: true` affects
`get_enabled_plugins()` (toggle state) and does **not** import `hooks.py`.

Consequences that motivated the merge:

- The old `_asyncio_guard` applied its patch at *import time of its own
  `hooks.py`*, so it was dead code on every restart. Its docstring's claim
  that "load_plugins runs before VectorDB init" was wrong:
  `helpers/plugins.py::load_plugins` only reads `plugin.yaml` metadata.
- This host runs Python 3.12.10, where
  `_SelectorSocketTransport._read_ready` is a bare `self._read_ready_cb()`,
  so the gh-115514 doom-loop was genuinely live.
- The old `_extension_import_guard` installed from `agent_init`, which is
  itself loaded by the very `load_classes_from_folder` it patches — its own
  cold-cache enumeration ran unpatched.

`startup_migration` is dispatched from `initialize.initialize_migration()`
in `run()` before the web server is prepared and before any agent exists,
and resolves plugin folders with `agent=None`. It is the earliest hook that
reliably fires, and installing there also protects the `agent_init` load.

### `asyncio_guard` — no shape heuristics

Deciding "is this interpreter already patched?" must be **behavioural**:
`_needs_guard(cls)` calls the real `_read_ready` on a bare instance with
`_read_ready_cb = None` and reports whether it raised. Two earlier
heuristics were wrong and are documented in the module:

- `len(code.co_names) > 1` — a *fixed* body compiles to
  `co_names == ('_read_ready_cb',)`, identical to the unguarded body.
- a jump-opcode allowlist — CPython 3.12 *inverts* the guard and emits
  `POP_JUMP_IF_NOT_NONE`, and `dis.hasjabs` / `dis.hasjrel` are empty on
  3.12, so the allowlist missed a fixed interpreter.

`uninstall()` is intentionally a no-op: removing a process-wide safety
monkeypatch from a live process is riskier than leaving it.

### `import_guard` — shim, do not mirror

The old implementation re-implemented `load_classes_from_folder` and
shipped an `inspect.getsource` marker check to warn about drift. This one
keeps upstream authoritative and swaps only `modules.import_module` for the
duration of the call, returning an **empty** `ModuleType` on failure
(`inspect.getmembers(None, isclass)` is *not* empty — it yields
`__class__ -> NoneType`, which would leak into results for a permissive
`base_class`).

Contract for anyone editing `helpers/import_guard.py`:

- `_original_import` must be the **import function**, not the loader. The
  shim calls it with a single file path; assigning the loader there
  produces `TypeError: load_classes_from_folder() missing 2 required
  positional arguments` (this was a real bug caught by the tests).
- Any function that *assigns* `_original_import` or `_depth` needs
  `global` — the plugin has been bitten by this class of bug before
  (`helpers/utility_timeout.get_resolved`, the v2.6.9 `nonlocal` crash).
- The swap is re-entrancy safe: depth counter under `_swap_lock` plus a
  thread-local error sink. A naive save/restore lets one thread restore
  another's shim, permanently patching `import_module` process-wide.
- Only `Exception` is swallowed; `KeyboardInterrupt`, `SystemExit` and
  `asyncio.CancelledError` (BaseException on 3.8+) must still propagate.
- `hooks.uninstall()` restores the upstream loader. The asyncio guard
  deliberately stays installed.

### Other fixes in v3.3.0

`hooks.clear_cooldowns` referenced `_fb` in a second `try` block while it
was only bound inside the first; on import failure the `NameError` was
swallowed by its own `except Exception`, so the persisted cooldown key was
silently left behind. The blocks are now independent.

---

## What this plugin does

`fallback.py` monkey-patches three methods onto the `Agent` class:

| Method | Patched at | Purpose |
|---|---|---|
| `Agent.call_utility_model` | `_patched_call_utility_model` (thin adapter → `_run_rotation_cascade`), `fallback.py:2074+` | Wraps the utility model call (used for memory, summarization, JSON validation, tool sub-tasks). |
| `Agent.call_chat_model` | `_patched_call_chat_model` (thin adapter → `_run_rotation_cascade`), `fallback.py:2062+` | Wraps the main chat model call (legacy path; still used by `_email_integration` / `_document_query`). |
| `Agent.call_chat_model_turn` | `_patched_call_chat_model_turn` + `install_chat_turn_patch`, installed from the same `agent_init/_00_install_fallback_patches.py` | **v2.8.0** — the turn path. Since the v2.10/2.11 upstream merge the MAIN agent loop calls this (monologue → `unified_turn` → `LiteLLMTransport.astream`), not `call_chat_model`. Without this patch the chat cascade is dead code on every main-loop call and a single 429 kills the agent. |

Both wrappers share the same contract: build a candidate list from the active
preset (`_build_candidates`), then loop through candidates until one succeeds.

### v3.1.0 — latency-adaptive timeouts (2026-09-03)

Roadmap item 1. User decision: **ON by default, conservative** (clamped,
tightens only after enough samples, kill-switch knob). Full suite
287/287 (+18 tests in `test_latency_adaptive_v310.py`).

**Problem.** The warm/cold two-tier timeout
(`_resolve_per_call_timeout`) only knows two sizes: the 20s warm
ceiling and the full cold base (300s default). A COLD label with
historically fast calls keeps paying the full cold timeout on a hung
connection before the cascade rotates — the timeout tax the cascade
pays most often.

**Mechanism.** New `helpers/latency.py` — process-global, thread-safe
`dict[label -> list[float]]` of successful-call durations (rolling,
capped at `latency_max_samples`, default 20). Samples are recorded ONLY
on success — a timed-out call is censored data (we only know it
exceeded the budget) and is never used to size anything. Recording
points: engine `_call_inner` right after the successful `wait_for`
(covers BOTH call sites — the primary attempt and the Responses-5xx
chat-completions retry) and the turn cascade's success block (next to
the `_WARM_LABELS` write; the turn body previously had no `started_at`).

Sizing: inside `_resolve_per_call_timeout`, in the block AFTER the warm
check and BEFORE the final `return base_timeout_s` (inside the existing
try — any failure falls back to base). `p95 × latency_p95_margin`
(nearest-rank p95), clamped to `[latency_floor_s, base_timeout_s]` — it
can only SHRINK the cold wait, never grow it beyond the configured base
and never below the floor. Below `latency_min_samples` (default 5) or
with `latency_adaptive_enabled: false` → plain base (true no-op: the
knob also gates recording, so a disabled plugin accumulates no state).

**Precedence preserved (do not regress).** Router labels (warm-up
budgets incl. `router_cold_call_timeout_s`) and `unlimited_paid`
(no-ceiling) return BEFORE the adaptive block; the warm fast-path
returns before it too. The user `TIMEOUT=` kwarg never reaches the
helper. Routers and paid models are unaffected by samples.

**Clear-on-timeout (conservative self-correction).** A genuine timeout
drops the label's samples entirely (`latency.clear_label`) — the sizing
that just fired was too tight, so the next resolution falls back to the
full base until fresh samples rebuild. Sites: engine TimeoutError
branch, engine timeout-shaped branch (`_is_timeout_shaped`), turn
cascade — **gated on `_is_timeout_shaped(e)` because the turn except
block catches every non-code error (429s, 5xx) and clearing samples on
those would discard good latency data** — and the outer utility-timeout
guard (`helpers/utility_timeout.py`).

**Observability.** `GET /stats` gained a `latency_adaptive` block
(per-label samples/p50/p95/last). No new event kind (log lines + stats
only). `latency.reset()` registered in `hooks.uninstall`.

Knobs: `latency_adaptive_enabled` (true) · `latency_min_samples` (5) ·
`latency_p95_margin` (1.5) · `latency_floor_s` (45.0) ·
`latency_max_samples` (20).

### v3.1.1 — free-model tolerance tuning (2026-09-06)

User report: on the all-free preset (`nvidia_nim/moonshotai/kimi-k3` chat +
`openrouter/z-ai/glm-5.2:free` fallback) the agent kept getting interrupted
mid-run with `Transient LLM error (TimeoutError) -- the agent will retry
automatically (N/5)` and eventually stopped.

**Diagnosis.** Free pools are slow AND rate-limited: a healthy kimi-k3
thinking turn streams for 60 s+ and OpenRouter `:free` answers with
`upstream_429` / slow TTFT. Budgets tuned for fast paid cloud calls killed
those healthy-but-slow calls (turn base 180 s, adaptive floor 45 s, warm
ceiling 20 s, outer utility guard 60 s), the cascade rotated into the other
free model, which failed too, and the escaped TimeoutError landed in the
`_60` net. Five consecutive swallows inside the 300 s window stopped the run.

**Changes.**
* `config.json` (user-tuned free-model profile): `fallback_timeout_s`
  180→300, `fallback_utility_timeout_s` 60→120, `cascade_warm_timeout_s`
  20→45, `latency_floor_s` 45→90, `latency_p95_margin` 1.5→2.0,
  `utility_timeout_guard.default_timeout_s` 60→120 and `max_wait_s`
  180→240 (outer guard must exceed the inner cascade's largest legitimate
  per-candidate wait), plus `transient_swallow_max: 8` and
  `transient_swallow_reset_window_s: 600`.
* `_60_handle_transient_llm_error.py`: the swallow bound is now
  configurable — `transient_swallow_max` (default 5) and
  `transient_swallow_reset_window_s` (default 300 s), resolved from plugin
  config at exception time; read/garbage failures keep the historic
  behavior. Both knobs documented in `default_config.yaml` (live via the
  `_get_plugin_cfg` YAML-under-config merge).
* Tests: `test_safety_net_uses_configured_swallow_cap` and
  `test_safety_net_bad_config_falls_back_to_defaults` in
  `tests/test_turn_cascade_v28.py`; the existing cap test now resolves the
  bound via `_swallow_limits` instead of the removed module constant.
* No cascade/engine behavior change: same precedence order, same cooldown
  table — only the budgets and the safety-net bound got looser.

### v3.2.0 — adaptive generation budgets: never punish a thinking agent (2026-09-06)

User report (2026-09-06 docker log, AFTER v3.1.1 tuning): chat turns still
die with `TimeoutError on [0] after 60s cooldown` → `All chat model
candidates exhausted` → `All model candidates are currently unavailable.
Retrying in 60s` while the model was mid-answer (long essay / deep
thinking). Static budgets — no matter how well tuned — trade two failure
modes against each other: raise them and a hung connection stalls the
agent for minutes; lower them and healthy slow generations get killed and
then PUNISHED with a cooldown for the crime of thinking longer than the
budget. A fixed number cannot win both.

**Design goal.** Separate "hung" from "working" per label and make the
per-call budget ADAPTIVE: grow when a label proves it needs longer
(likely still generating), decay back when successes are fast again, keep
an absolute wall so a genuinely wedged connection is still detected.

**Mechanism (phases, in implementation order).**

1. **Grow-on-timeout (learn).** New process-global
   `_GEN_BUDGET_OVERRIDES: dict[label -> float]` in `fallback.py`
   (lifecycle identical to `_WARM_LABELS`: in-memory, cleared by
   `clear_all_cooldowns`, reset by `hooks.uninstall`). At every cascade
   TimeoutError branch (engine wait_for timeout + engine timeout-shaped +
   turn path — the same sites that call `_evict_warm_on_timeout` and book
   the timeout cooldown): when `adaptive_gen_budget_enabled`, raise the
   override to
   `min(max(prev, elapsed) * gen_budget_growth_factor, gen_budget_max_s)`.
   When the cascade's cold base (`timeout_s`) is known and the result
   would not exceed it, the grow is SKIPPED (and any stale override
   removed): the stock base already covers the need — a warm-ceiling
   timeout on a 300s-base label is fixed by the warm eviction, not by a
   budget override. The outer utility guard's grow call passes no base
   (it doesn't know the cascade base; growing is the conservative
   choice). Skip on `asyncio.CancelledError` (external cancel ≠ slow
   model) and on deterministic upstream errors (they fast-fail, not time
   out). One INFO line + `events.record_event("gen_budget_grown", ...)`
   per growth.

2. **Apply (effective budget).** Wrap `_resolve_per_call_timeout`: after
   the existing precedence chain (router warm-up / `unlimited_paid` /
   warm fast-path / latency-adaptive shrink / base) returns `budget`,
   return `min(max(budget, override), gen_budget_max_s)`. The override
   RAISES any capacity class's ceiling — long answers happen on routers
   and paid models too. The user's explicit `TIMEOUT=` kwarg still wins
   (it never reaches this helper; unchanged).

3. **Decay-on-success (un-learn).** Where successes mark
   `_WARM_LABELS[label]` (engine `_call_inner` post-`wait_for` and the
   turn success block — the same sites latency samples are recorded):
   decay the override `* gen_budget_decay_factor`, floored at
   `elapsed * gen_budget_success_margin` (never below what a real
   generation just needed), and DELETE the override once BOTH the decayed
   value AND the evidence floor sit at/below the cold base timeout — the
   floor alone must not trigger deletion, because a recent long success
   is exactly the evidence the override exists to protect.

4. **Outer utility-guard parity.** `helpers/utility_timeout.guarded_call`
   raises its budget with the same learned override AND lifts its
   `max_wait_s` cap to `max(max_wait_s, override)` — otherwise a long
   utility generation dies at the outer cap even when the inner cascade
   would now allow it. Lazy `from ... import fallback` inside the
   function (existing pattern, no import cycle).

5. **Config knobs** (`default_config.yaml`, resolved via
   `_get_plugin_cfg`): `adaptive_gen_budget_enabled` (true, kill-switch —
   gates grow AND apply AND decay), `gen_budget_max_s` (600 — the
   absolute wall; a hung call is still detected within it),
   `gen_budget_growth_factor` (1.5), `gen_budget_decay_factor` (0.9),
   `gen_budget_success_margin` (1.3).

6. **Optional follow-up (v3.2.1).** Persist overrides next to the
   cooldown store (`mfb_gen_budgets` agent.data key) with the same
   fresh-seed policy as the cooldowns, so a quick `run_ui` restart during
   a long-answer workload does not re-learn from scratch. Skip unless
   restarts prove costly in practice.

**Why not stream-aware idle timeouts instead?** The transport
(`LiteLLMTransport.astream`) lives in core `models.py`; wiring chunk-level
progress into the plugin's `wait_for` would either touch core files
(forbidden by the plugin contract) or re-implement the transport here.
The budget-learning approach needs zero core changes and composes with the
existing warm/cold/latency-adaptive precedence. If core ever exposes a
progress callback, an idle-timeout can be layered on top as a refinement.

**Tests.** `tests/test_adaptive_gen_budget_v320.py` (mirror
`test_latency_adaptive_v310.py` style): grow caps at `gen_budget_max_s`;
CancelledError / deterministic errors do not grow; apply overrides every
capacity class; decay floors at `elapsed * margin` and deletes below
base; kill-switch disables all three; `reset()` clears state. Full
plugin suite must stay green.

**Verification.** `pytest usr/plugins/model_fallback/tests/ -q` green;
both AST scanners (`tmp/scan_unbound_locals.py`,
`tmp/scan_undefined_names.py`) clean; live check: one long-generation
turn logs the single grow event, subsequent turns complete without the
60s-cooldown retry loop, and a genuinely hung endpoint still rotates
within `gen_budget_max_s`.

### v3.0.0 — cascade unification (2026-09-03)
### v3.0.0 — cascade unification (2026-09-03)

Roadmap item 2. **`fallback.py` net −590 lines** (4,364 → 3,774). Full
suite 269/269 (+16 characterization tests in `test_cascade_v300.py`).

**Problem.** The utility (`_patched_call_utility_model`) and chat
(`_patched_call_chat_model`) cascades were line-for-line copies differing
only in ~10 parameters (log prefixes, summary kind, timeout cfg key,
`use_utility_models` flag, hook names, call_data keys, unified_call
kwargs, success validation, return shape). Historical bugs were repeatedly
"fixed in one cascade, missed in the other": the pass-boundary placement
(v2.8.5), the missing `nonlocal` (stagnation counter), the v2.6.1 chat
warm-read gap, the v2.8.4 primary-escalation gap in the chat timeout
branch. Every shared fix cost double review.

**Mechanism.** ONE engine `_run_rotation_cascade(self, spec, model_obj)`
(fallback.py:2102+) — the verbatim former utility-cascade body with the
divergences parameterized by a `_CascadeSpec` (fallback.py:2074+). The
two public functions remain as ~100-line thin adapters that keep the
EXACT signatures (hard imports in
`usr\extensions\python\_functions\a0\agent\Agent\{call_utility_model,call_chat_model}\start\_00_fallback_override.py`
depend on the names) and build the spec's closures:

- `call_data_builder(current_model)` → the per-candidate call_data dict
- `inner_coro_builder(call_data)` → the fresh `unified_call` coroutine
  (**SYNC builder returning the coroutine** — an `async def` here returns
  a bare coroutine object instead and the engine's `await` fails; both
  the initial call and the Responses-5xx retry rebuild go through it)
- `success_prepare(call_data, response, reasoning)` → utility JSON
  validation vs chat tool-request validation
- `error_msg_fn(e, idx, label)` → the differing log-line shapes
  (utility "Format Error"/`{err_type}:` vs chat `error ({err_type}):`)
- `after_hook_fire(call_data, response, reasoning)` → utility's hook
  contract does NOT receive `reasoning`; chat's does

Preserved divergences (spec flags): `persist_idx_on_success` (utility's
`_succeed` advances the rotation index via `_set_model_idx`; chat never
did), `skip_remaining_suffix` (chat's cooldown-skip log appends
" remaining"), `stagnation_prefix` ("Cascade" vs "Chat cascade"),
`cycling_prefix`. The engine always returns `(response, reasoning)`;
continuous-mode exhaustion returns `(None, None)` (the utility adapter
unwraps to `None`, preserving its old contract).

**In scope / NOT in scope.** The turn cascade (`_patched_call_chat_model_turn`)
is structurally different (single pass, streaming guards, persisted
counters) and was deliberately NOT unified — its remaining duplication is
small and the streaming/persistence behavior of the main agent loop is
not gambleable without live verification. Engine + spec sit BEFORE the
turn banner so the turn-slice tests (`src.index("async def
_patched_call_chat_model_turn")` windows) are unaffected.

**Latent bug fixed (found by the v2.9.1 audit, plan-approved):** the
empty-pass spin called `random.uniform` with NO `import random` in scope
in BOTH loops — `NameError` on the first all-skipped pass when
`backoff_jitter_s > 0` (the default is 2.0, so any real all-skipped
roster would have crashed the cascade; the `_compute_cycle_sleep`
closure had its own local import, which masked the bug there). Fix:
module-level `import random`.

**Safety net.** `tests/test_cascade_v300.py` — 16 characterization tests,
parametrized utility+chat, written and verified GREEN against the
pre-refactor v2.9.1 code and left UNCHANGED through the refactor. They
pin the external contract: return shapes, hook names, rotation,
cooldown booking, skip logging, exhaustion, continuous-mode sentinel,
CancelledError propagation, fail-fast code errors, all-skipped spin.

**Test updates required by design** (assertions follow the code, none
weakened): `test_chat_warm_wiring_v26.py` co_varnames/co_names moved to
the engine (+ new adapter-delegation test); `test_v286_flagged_fixes.py`
`if _is_permanent_for_rotation(e):` 2→1, `primary_strike_decay_s` >=4
→ >=3; `test_events_log.py` event-site counts 3→2 (engine + turn).

### v2.9.1 — routing event log (2026-09-03)

Observability (roadmap item 1). Full suite 253/253 (+17 tests in
`test_events_log.py`).

**Problem.** Routing state lived in three disconnected places: counters
(`/stats`), context log lines (die with the context), and in-memory dicts.
Answering "what happened to model routing in the last hour" required
reconstructing from all three.

**Mechanism** (`helpers/events.py`): a bounded in-memory ring buffer
(`deque(maxlen=500)`), thread-safe, `record_event` never raises and
coerces every field to JSON-safe values. No persistence by design —
routing decisions are transient state and a restart resets them anyway.

Event kinds (all recorded by fallback.py / recovery_probe.py):

| Kind | Meaning | Recorded at |
|---|---|---|
| `cooldown_booked` | candidate entered/extended cooldown | thin wrapper around `_handle_error_cooldown_impl` — detects a booking by comparing the store entry before/after, so ALL booking branches are covered without touching them (dur=0 no-book outcomes record nothing) |
| `cooldown_cleared_early` | recovery probe succeeded (v2.9.0) | `recovery_probe._clear_cooldown_early` |
| `cooldown_cleared_by_success` | a LIVE call succeeded on a label that was in cooldown (real recovery — distinct from probe recovery) | engine + turn success paths (v3.0.0: the unified engine holds the utility+chat copy) |
| `primary_skip_escalated` | primary-skip escalation wrote a long cooldown for candidate 0 | engine + turn `_maybe_extend_primary_cooldown` closures (v3.0.0: utility+chat share one) |
| `label_dead` | cross-agent dead-mark written | `_mark_label_dead` |
| `cooldowns_cleared` | user-initiated clear (button / hooks API) | `clear_all_cooldowns` (carries `count` + `cross_context`) |
| `cascade_exhausted` | all candidates failed, RetryAfterHours imminent | `_emit_fallback_summary` (field `cascade_kind`, NOT `kind` — `kind` is `record_event`'s own positional param; passing `kind=kind` raises `TypeError` which the summary's blanket `except` swallows → silently no event. GOTCHA.) |

**Endpoint** (`api/events.py`, route `POST /api/plugins/model_fallback/events`):
returns `{"version", "kinds", "count", "events": [...]}` newest-first.
Optional filters: `limit` (1..500, default 100), `kind`, `context`,
`label`. Auth+CSRF inherited defaults (same as `/stats`). Runtime only —
`hooks.reset_fallback_settings` clears the buffer alongside the other
reset state.

### v2.9.0 — background recovery probes (2026-09-03)

New capability (first proactive feature — the plugin previously acted only
AFTER a failure). Full suite 236/236 (+14 tests in
`test_recovery_probes.py`).

**Problem.** A cooled-down label was re-probed only by real agent traffic:
the first post-cooldown call paid the full timeout/429 again. If a
provider recovered 5 minutes into a 10-minute cooldown, every cascade in
every agent kept routing around a healthy label for the rest of the TTL.

**Mechanism** (`helpers/recovery_probe.py`, wired from `_handle_error_cooldown`):

- **Registration**: the 8 probe-able `_handle_error_cooldown` call sites
  (utility timeout / litellm-timeout / general, chat ditto, turn
  n<=1 + general) pass `probe_model=<candidate wrapper>` — the wrapper
  that just failed, carrying api_key/api_base/provider kwargs, so a probe
  replays the exact same transport via `model.unified_call(...)`. Code-error
  fail-fasts and the `_60` safety net leave it None (no probe: a code bug
  would fail every probe).
- **Sweep**: one background asyncio task (started lazily on first
  registration; idempotent; no running loop = no-op) sweeps every
  `recovery_probe_interval_s` (45s default, min 10). Per target: label not
  in cooldown → drop; remaining cooldown < `recovery_probe_min_cooldown_s`
  (120s) → skip (real traffic re-probes soon anyway); dead-marked label →
  skip (recovery needs a user action, e.g. a fixed key; probing a 404
  every 45s is pure waste). Otherwise ONE cheap call
  (`system_message="You are a health check.", user_message="ping"`,
  no streaming/rate-limiter callbacks) under
  `recovery_probe_timeout_s` (15s, clamped 5-60).
- **Success** clears the cooldown EARLY — in place on the live store
  object (v2.8.4 invariant), persisted, `_mark_label_healthy` cross-agent,
  info log, target dropped. **Failure** changes nothing but counters; the
  original expiry stands (a probe never books or extends a cooldown).
- **Burn bound**: max `recovery_probe_max_targets_per_cycle` (2) probes
  per sweep.
- **STRONG ref to the wrapper is load-bearing**: per-call wrappers are
  usually unreachable after the booking iteration, so a weakref would be
  collected before the first sweep and the feature would be a silent
  no-op (caught by the test suite only because FakeModels are kept alive
  by the test frame — in production the wrapper ref must be strong).
  Registry is bounded by distinct (context, label) pairs. The AGENT is a
  weakref: once the context is gone, probing is pointless.
- **Lifecycle**: `hooks.uninstall()` calls `recovery_probe.shutdown_probes()`
  (cancels task, drops registry). Counters + `loop_alive` exposed via
  `helpers.recovery_probe.snapshot()` in `api/stats.py` as
  `recovery_probes` (`loop_alive: false` right after a restart is normal —
  the task starts with the first booked cooldown).
- Knobs (all in default_config.yaml, read through `_get_plugin_cfg`):
  `recovery_probe_enabled`, `recovery_probe_interval_s`,
  `recovery_probe_min_cooldown_s`, `recovery_probe_timeout_s`,
  `recovery_probe_max_targets_per_cycle`. Disable with
  `recovery_probe_enabled: false`.

### v2.8.6 — round-3 flagged fixes (2026-09-03)

The five findings the v2.8.5 pass flagged but did not fix (user-approved).
Full suite 222/222 (+9 new tests in `test_v286_flagged_fixes.py`).

- **(X) 429 is no longer "rotation-permanent".** New helper
  `_is_permanent_for_rotation(e)` (fallback.py, after `_is_timeout_shaped`):
  permanent-shaped AND NOT rate-limited. Used at the three ROTATION sites:
  `cycle_permanent_count += 1` in both cascades and the turn path's n<=1
  fail-fast gate. Before: a 429 hitting the turn gate raised the RAW
  `RateLimitError`, bypassing the structured `RetryAfterHours` raise below
  it, and the summary lines printed contradictory
  "permanent failure" + "rate limited" pairs. `_handle_error_cooldown`'s
  own `_is_permanently_failed_model` check is UNCHANGED — there a 429 has
  already returned via the rate-limit branch, and cooldown policy correctly
  treats quota errors like permanent ones (book + move on).
- **(Z) Strike counters decay.** New knob `primary_strike_decay_s`
  (default 300s, 0 = pre-v2.8.6 never-expire). Utility + chat cascades:
  closure-local `_last_primary_failure_at` (added to the `nonlocal` in
  `_maybe_extend_primary_cooldown` and `_reset_primary_strikes`); before
  the increment, strikes older than the window are discarded. Turn path:
  the persisted counter (`mfb_turn_primary_fails`) now has a persisted
  monotonic timestamp next to it (`DATA_KEY_TURN_PRIMARY_FAILS_AT` =
  `mfb_turn_primary_fails_at`); the increment site reads the knob lazily
  and decays before counting. `hooks.reset_fallback_settings` resets the
  timestamp with the counter.
- **(V) Per-method, version-stamped install guards.**
  `_00_install_fallback_patches.execute` no longer early-returns on
  utility's sentinel before checking chat — a partial state (utility
  patched, chat not) is now repairable, and each method re-installs
  independently when its `_fallback_patched_version` stamp differs from
  `get_plugin_meta("model_fallback").version` (same `_plugin_version()`
  pattern as the v2.8.5 AD fix in `_10`). `install_chat_turn_patch` takes
  an optional `version=`: a stale turn cascade is re-assigned WITHOUT
  overwriting the captured original (module global
  `_ORIGINAL_CALL_CHAT_MODEL_TURN` is only written on first install).
- **(N) Full uninstall chain.** `_00` captures the TRUE originals on the
  Agent class (`Agent._mfb_original_call_utility_model` /
  `..._call_chat_model`) on first install, and gained a module-level
  `uninstall()`: unwrap the `_10` utility-timeout wrapper FIRST (it wraps
  the cascade), then restore the true originals, then the turn original
  from the fallback global. `hooks.uninstall()` calls the chain (falls
  back to the old `_10`-only restore if `_00` predates v2.8.6). Previously
  a plugin disable left the plugin's own cascades installed on the class.
- **(wiring#11) Cross-context cooldown clear.**
  `clear_all_cooldowns(agent, *, cross_context=False)` — with the flag it
  wipes EVERY context's in-memory store (the dead/warm indexes were
  already process-global); `hooks.clear_cooldowns(agent, cross_context=)`
  passes it through. Only the calling agent's persisted copy is wiped —
  peer in-memory dicts ARE the live store, so that is sufficient. A
  per-context clear could not reach sub-agents / other chats sharing a
  broken credential before.

### v2.8.5 — third-pass audit fixes (2026-09-03)

Deep-dive pass over the cascades. Findings fixed (all verified in source
before fixing; full suite 213/213 after):

**CRITICAL / HIGH — cascade mechanics**

- **Event-loop freeze on an all-skipped roster** (both utility + chat
  cascades). The pass-boundary check (`attempt % n == 0`, empty-pass spin +
  cycle accounting) sat BELOW the dead/cooldown `continue` checks, so it
  only ran when the landing candidate was live. All-skipped roster → every
  iteration exited via a skip `continue` with NO `await` → tight loop that
  froze the event loop until the shortest dead TTL (a 24h dead mark → hang
  until restart). A cooled STARTING candidate also meant the boundary never
  fired and cycle accounting / RetryAfterHours never engaged. The boundary
  block moved ABOVE the skip checks; skips still `continue` instantly.
- **Dead 5xx retry** (both cascades). The retry-after-5xx branch awaited a
  coroutine object it had ALREADY awaited (and consumed) in the failed
  attempt — `RuntimeError: cannot reuse already awaited coroutine`. Both
  now build a fresh coroutine inside the retry try.
- **Turn kill path** (`call_chat_model_turn`). The turn cascade resolved
  per-candidate timeouts with the warm fast-path ACTIVE: a main-loop turn
  legitimately streams for 15–90 s, so the 20 s warm ceiling timed out
  healthy turns, `_60` saw a bare TimeoutError (no status, no litellm
  class) and let it reach `_90` → agent stopped. Turn path now passes
  `allow_warm=False`; `_resolve_per_call_timeout` grew the kwarg; `_60`
  treats a bare `TimeoutError`/`asyncio.TimeoutError` as transient.
- **n≤1 quota-death path** (turn cascade). With no fallback candidates a
  sustained outage surfaced the raw error every turn; `_60` swallowed it ≤5
  times then the agent stopped. The single-attempt path now books the
  normal cooldown and raises `RetryAfterHours(retry_after=...)` (owned by
  `_70`) unless the failure is permanent or output already streamed, and
  re-raises as-is when extended retry AND continuous mode are both off.
- **`clear_all_cooldowns` defeated recovery** — it now also clears
  `_INMEM_DEAD_LABELS` and `_WARM_LABELS` and resets the last-status dict.
- **Turn strike-counter persistence** — the turn cascade is a single pass,
  so the closure-local primary-skip counter maxed at 1 and the v2.8.4
  escalation was dead. Counter now persists via `DATA_KEY_TURN_PRIMARY_FAILS`
  across turns.
- **`_60` api_base registry** — `_LABEL_API_BASES` (populated by
  `_resolve_per_call_timeout`, consumed by `_handle_error_cooldown`) so
  safety-net bookings classify router/free/paid correctly instead of
  mis-cooling a gateway 30 s or dead-marking it 24 h.
- **`_maybe_raise_retry_after_hours` max_cycles** — callers' kwarg-resolved
  cycle cap now reaches the gate instead of a hardcoded value.

**MEDIUM**

- **Persistence was dead code** (all `_mfb_*` data keys). `persist_chat.py`
  strips agent.data keys starting with `_` (:194/:223, restore at :311), so
  cooldown seeding / extended-retry phase / cascade position never actually
  persisted. All DATA_KEY_* constants renamed to a non-underscore
  `_mfb_*` prefix; `_70` and hooks use the constants instead of literals.
- **Utility timeout guard toggle-off** — `_install` returned False on
  `enabled: false` BEFORE refreshing the resolved config, so a live wrapper
  kept guarding until restart. Refresh now happens in the disabled branch.
- **Utility guard budget vs cold routers** — the outer guard's flat 60 s
  budget fired first on every utility call to a COLD router (150 s
  warm-up budget), killed the call, and booked nothing (starvation loop).
  `_resolve_config` injects `_router_cold_s`; `guarded_call` raises its
  budget to `max(default, router cold)` still capped by `max_wait_s`, and
  a guard timeout now books the cooldown + evicts the warm label (needs
  `agent=`, passed by the wrapper).
- **`_70` honors `retry_after`** — the phase A (900 s) / phase B (3600 s)
  delays the cascades computed were ignored; the handler slept a hardcoded
  60 s and re-entered a full cascade pass. Now clamps the exception's
  `retry_after` to [30, 7200] s, sliced.
- **Hooks** — `uninstall()` made SYNC (the framework runs async hooks via
  `asyncio.run`, which raises inside the plugin-delete API's running loop
  → 500 → cleanup silently skipped); `get_fallback_settings` reads the
  merged plugin config instead of dead agent-data keys;
  `reset_fallback_settings` clears the REAL extended-retry keys.
- **Stats phantom counters** — the context-size-guard counter lived in the
  extension module (synthetic module, not in sys.modules), so api/stats.py
  importing that path created a SECOND instance with fresh zeros and the
  WebUI tile reported phantom zeros forever. Counter moved to
  `helpers/stats.py` (`context_guard_*` accessors); the extension keeps
  `get_counter`/`reset_counter` wrappers for the tests.
- **Turn shadow UnboundLocalError hazard** — `prev_shadow` captured outside
  the try so the finally can never hit UnboundLocalError.

**LOW**

- Warm fast-path entry is evicted for any label that books a cooldown
  (AA) — closes the 20 s-fail → cooldown → 20 s-fail loop. Router /
  concurrent_paid 429s (no cooldown booked) keep the warm ceiling.
- Expired `_INMEM_HEALTHY_LABELS` entries are popped (AB).
- Chat cascade `_compute_cycle_sleep` gained the Phase-3 stagnation
  amplification (+ `nonlocal` flag) the utility cascade already had (Y);
  the all-skipped spin in BOTH cascades now mirrors the router probe cap
  (`router_max_cycle_delay_s`).
- memory_* knobs are wired (wiring#6): `memory_recall_timeout_s`,
  `memory_memorize_max_chars`, `memory_recall_delayed`,
  `memory_memorize_consolidation` were documented but every consumer
  hardcoded its value.
- context-size guard config merge (wiring#7): `_resolve_runtime_config`
  now merges default_config.yaml under config.json (same gap the utility
  guard fixed in v2.8.3).
- memory recall patches disk I/O offloaded to `asyncio.to_thread` and
  paths derived from the file location instead of hardcoded `/a0`
  (wiring#9); concat_messages swap restore is re-entrancy-safe (wiring#10);
  `mark_installed` only fires when at least one shim actually installed
  (wiring#12); install guards are version-stamped so a plugin UPDATE
  re-applies changed wrappers instead of leaving stale closures live (AD).

### v2.8.0 — turn-path cascade + transient-error safety net (2026-09-01)

Root cause that motivated this: `usr/settings.json` → litellm
`num_retries: 2` retried a shared-pool OpenRouter 429 (`z-ai/glm-5.2:free`,
`upstream_429`, `Retry-After: 5`) against the SAME model, the resulting
`litellm.RateLimitError` escaped to `handle_exception` (which only handled
`RetryAfterHours`), `_90_handle_critical_exception` wrapped it as
`HandledException`, and the agent stopped.

Two additions fix it:

1. **Turn cascade** (`_patched_call_chat_model_turn`). Wraps the captured
   `@extensible` original (so its own `chat_model_call_before/after` hooks and
   Responses-state handling still run) and does ONE pass over
   `_build_candidates(...)` per invocation: skips dead / cooled-down labels
   (shared cooldown store + cross-agent dead-label index), applies
   `_handle_error_cooldown` and `_evict_warm_on_timeout` per failure, honors
   the provider's `Retry-After`, and re-raises immediately if any response
   chunk already streamed to the UI (rotating would duplicate partial output).
   On exhaustion it raises `RetryAfterHours` — the existing
   `handle_exception/end/_70` extension swallows it (60 s sleep) and the
   monologue loop re-enters the cascade, which now skips the cooled-down
   labels. Net effect: same continuous-mode survivability as the chat
   cascade, but interventions stay responsive between passes.
   **Deliberate difference from the chat cascade:** no full-cycle loop inside
   the call. Do not "fix" this by adding a `while True` — it would block
   interventions for the whole outage.
   Candidate-model injection works by shadowing `self.get_chat_model` with an
   instance attribute for the duration of each inner call (restored in
   `finally` by deleting the instance attr — never assign a restored bound
   copy, or plugin reloads stack layers).

2. **Safety net** (`extensions/python/_functions/agent/Agent/handle_exception/
   end/_60_handle_transient_llm_error.py`). Last-resort swallow of transient
   provider errors (429, 5xx, connection, timeout) that still escape any
   cascade. Books the cooldown (best-effort label from `exc.model`), sleeps
   3 s, and clears `data["exception"]` — bounded at 5 consecutive swallows
   (state resets after 5 quiet minutes) so a genuinely broken setup still
   surfaces. v3.1.1: the bound is configurable via `transient_swallow_max`
   / `transient_swallow_reset_window_s` — see the v3.1.1 section.
   Runs BEFORE `_70` and must never touch `RetryAfterHours`.

Both mechanisms share the cooldown store with the chat/utility cascades, so a
429 learned on any path protects every other path on the next turn.

### v2.8.1 — stale-install resilience + `or {}` cooldown fix (2026-09-01)

Two follow-up fixes to v2.8.0, from live-container failures:

1. **Installer must never kill `Agent.__init__`.** A stale `__pycache__`/.pyc
   on a slow bind mount left the container running a pre-v2.8.0 `fallback.py`
   while the updated `_00_install_fallback_patches.py` loaded — the
   `from ... import install_chat_turn_patch` raised `ImportError` inside
   `Agent.__init__`, so **no new chat could be created**. The installer now
   imports the module and resolves `getattr(fb, name, None)` for every patch
   symbol, logs a clear error/warning instead of raising, and installs whatever
   subset exists. Never convert these back to from-imports.
2. **`or {}` cooldown-store wipe.** `_get_cooldown_store(agent)` seeds a fresh
   EMPTY dict on first use; an empty dict is falsy, so
   `model_cooldowns = _get_cooldown_store(self) or {}` bound an UNREGISTERED
   literal in all three cascades. Cooldowns written via `_handle_error_cooldown`
   landed in the registered store, but a fallback success then
   `_save_cooldown_store`'d the empty literal back over it — the first
   cooldown after a restart silently vanished. All three cascades now use the
   registered store object directly (isinstance-guard only). The turn-path
   cascade shipped with this bug in v2.8.0; the utility/chat cascades carried
   it latently since the shared cooldown store was introduced.

Test-suite note (same class as the langchain suite-pollution fix):
`tests/test_candidate_normalize.py` `reload(fb_mod)` REBINDS all fallback
module globals mid-suite. Anything resolved via import-time from-imports
(`_INMEM_COOLDOWNS`, `RetryAfterHours`, ...) is a stale object afterwards.
New tests must read module state through `fallback.<name>` at call time.

### v2.8.4 — round-2 flagged fixes: format-slip cooldown, dead config paths, turn primary-skip (2026-09-03)

Second pass (fixing the items the v2.8.3 audit flagged-not-fixed):

- **F8 — malformed-JSON no longer books 300s.** A `require_json` utility
  response DirtyJson can't parse raised `ValueError("... not valid JSON")`
  from `_succeed` into the generic `except Exception` → `status_code=None`
  → the 300s unknown-error cooldown. Two slips ~5 min apart effectively
  rotated away from a healthy model. New `_is_format_error()` detects
  parse/format-shaped exceptions; `_handle_error_cooldown` books
  `format_error_cooldown_s` (default **20s**, new knob in
  default_config.yaml) for them, bypassing the 30s minimum floor (that
  floor exists for real endpoint errors; a format slip should be retryable
  next cycle). `0` disables booking (pure rotation). Also honored in
  `_cooldown_seconds_for_status` for any other call path.
- **`force_chat_completions_providers` was dead at runtime.**
  `_force_chat_config` (models_ext.py) read only
  `get_plugin_config`, which does NOT merge default_config.yaml with
  config.json (same gotcha as the v2.6.7 router-detection fix) — and all
  three force-chat lists live only in the YAML. Now merged (defaults under
  live config).
- **Turn-path primary-skip escalation wired.** The turn cascade counted
  `_consecutive_primary_failures` (and reset on success) but never
  escalated — the counter was write-only. `_maybe_extend_primary_cooldown`
  now exists in the turn cascade too (knobs read lazily; does NOT
  increment — the caller increments for idx==0; same router/concurrent
  exemptions and only-cleared healthy-label check as chat).
- **`_strip_a0_only_kwargs(is_primary=)` guard.** At idx==0 the candidate
  IS the live primary model object, and popping `venice_parameters` off it
  silently disabled the primary's own Venice features on every call. New
  `_PROVIDER_SPECIFIC_KWARGS` survive on the primary, still stripped from
  fallback wrappers. (`usage` stays always-stripped — the OpenAI SDK
  rejects it as a top-level arg.)
- **api/stats.py `version`** now read from plugin.yaml via
  `plugins.get_plugin_meta` (was hardcoded "2.6.8").

Tests: `tests/test_v284_flagged_fixes.py`.

### v2.8.4 follow-up — second-pass audit fixes (same day)

- **`_get_plugin_cfg` now merges default_config.yaml under config.json**
  (mtime-keyed cache in `_get_merged_defaults`). This was the ROOT CAUSE of
  several flagged items: the `format_error_cooldown_s` knob, the
  turn-helper's `primary_skip_*` knobs, and every other YAML-only knob were
  dead config unless hand-copied into config.json. models_ext's own merge
  stays (it can't call fallback.py — synthetic-module import direction).
- Turn helper `primary_skip_cooldown_s` default corrected 600 → 120 (was
  contradicting the YAML/chat/utility value of 120 — a turn-path chat
  primary would have been escalated to 10 min while the utility path
  escalated the same label to 2 min).
- Chat cascade's dedicated timeout branch now calls
  `_maybe_extend_primary_cooldown(reason="timeout")` (pre-existing gap:
  the utility cascade did this since v2.5.1; the chat branch never did).
- `clear_all_cooldowns` mutates the EXISTING in-memory store in place
  instead of swapping a literal dict — a live cascade held the old object
  and would have written stale cooldowns back over a mid-turn "Clear".
- litellm.Timeout no longer books its cooldown twice in the chat/utility
  generic handlers (`timeout_cooldown_booked` flag).

### v2.8.3 — audit fixes: timeout parity, config-merge, spin backoff (2026-09-03)

Full-audit pass (independent code review of fallback.py + all extensions;
203/203 tests). Nine findings, seven fixed:

- **Utility guard ran at 30s, not the YAML's 60s (F1, HIGH).**
  `get_plugin_config` returns config.json WITHOUT merging
  `default_config.yaml`, config.json has no nested
  `utility_timeout_guard:` section, and `DEFAULTS` still held the stale
  30s/120s values — so the 2026-07-23 raise to 60/180 never took effect
  at runtime and the outer guard killed cold-start cascades at ~31s
  (the known "cascade cold-start timeout" symptom). Fixed three ways:
  `_resolve_config` now merges `get_default_plugin_config()` under
  config.json; `DEFAULTS` synced to 60/180; nested block added to
  config.json.
- **Stale closure on the utility guard (F6).** `wrapped` passed
  `config_overrides=cfg` — the dict frozen at install time — so config
  changes never reached `guarded_call` until process restart. Now
  `config_overrides=None` (reads the refreshed `get_resolved()`).
- **Turn cascade ignored the user `TIMEOUT=` kwarg (F2, HIGH).** The
  chat/utility cascades skip the warm fast-path when the model kwargs
  carry `TIMEOUT=`/`timeout=` ("user-set TIMEOUT= still wins"); the
  v2.8.0 turn cascade omitted the guard, so a warm label capped an
  explicit 300s TIMEOUT at the 20s warm ceiling on the MAIN agent loop,
  then booked a cooldown and rotated. Guard mirrored.
- **`CancelledError` swallowed by both cascade outer handlers (F4).**
  `except (TimeoutError, ..., CancelledError)` treated external
  cancellation (user intervention, the outer guard's wait_for) as a
  model timeout — booked a 300s cooldown on a healthy label, rotated on,
  and let the outer `wait_for` keep awaiting the suppressed cancel. Both
  cascades now re-raise `CancelledError` before the timeout tuple (the
  turn cascade already did).
- **`litellm.Timeout` is not a `TimeoutError` (F5).** Its MRO runs
  `Timeout → APITimeoutError → APIConnectionError → APIError`, so a
  provider-side pure timeout that escaped `unified_call` before the
  cascade's `wait_for` hit the 300s unknown-error default AND kept its
  warm label — recreating the 20s warm-ceiling loop fix A (v2.6.4)
  closed, for provider-side timeouts. New `_is_timeout_shaped()`
  recognizes it: `_cooldown_seconds_for_status` books the short
  `timeout_cooldown_s` and both cascades' generic `except` branch now
  evicts the warm label for timeout-shaped errors.
- **Context-size guard toggle was a silent no-op (F3).** config.json
  carries only `context_size_guard_enabled: true`; with no nested
  section and no YAML merge, `resolve_config({})` → `enabled: False`
  from DEFAULTS while the WebUI toggle said ON. The toggle now wins
  over the nested section (same precedence rule as the utility guard).
- **All-skipped spin livelock (F7).** When every candidate was
  cooldown-skipped or dead-marked, both cascades slept a flat 2.0s
  forever — no backoff, no exit even in legacy mode (a 24h cross-agent
  404 mark = 2s spin for a day). Empty passes now escalate with
  `cycle_stagnation_factor` (capped at `max_cycle_delay_s`, jitter
  applied); legacy mode exhausts after 3 empty passes via
  `_maybe_raise_retry_after_hours` + `RuntimeError`, matching the
  documented legacy contract. Continuous mode keeps cycling (liveness
  invariant) but with the escalated sleeps.
- **hooks.py cooldown API was dead (F9).** `clear_cooldowns` wiped the
  legacy `_model_cooldowns` data key while the live
  `_INMEM_COOLDOWNS` store kept running; `get_cooldowns` read a key
  nothing writes; reset used the wrong ext-retry key names and stale
  9000/43200 defaults. Now operates on the live store via new public
  helpers `fallback.snapshot_cooldowns()` / `fallback.clear_all_cooldowns()`
  and the real `DATA_KEY_EXT_RETRY_*` names; defaults match config.json
  (900/3600).
- **Hardening (v2.8.1 residual):** `_00_install_fallback_patches.py` now
  guards the `import fallback` itself — `call_extensions_sync` does not
  catch exceptions, so an import-time SyntaxError/ImportError in the
  module chain still killed `Agent.__init__` for every new chat.

Flagged, NOT fixed (policy/feature calls, per "flag not autofix"):
- F8: a malformed-JSON response books the 300s unknown-error cooldown on
  a healthy model (two slips ≈ rotate away from a working model for
  5 min). Deliberate rotation; the *duration* is the questionable part.
- Side-findings: `api/stats.py` hardcodes `"version": "2.6.8"`;
  `force_chat_completions_providers` is dead at runtime (only the
  `api_bases` matcher is wired); the turn path tracks
  `_consecutive_primary_failures` but never escalates (primary-skip not
  wired on the turn path); `_strip_a0_only_kwargs` runs on idx=0 too and
  permanently pops `venice_parameters` off the live primary model object.

Tests: `tests/test_v283_audit_fixes.py` (9 tests: litellm.Timeout
classification, turn-kwarg tripwire, DEFAULTS sync, YAML merge reach,
toggle-wins, hooks live-store API). Suite 203/203. Also fixed
`test_utility_timeout_floor.py` hardcoding `/a0` (now repo-relative, so
the 4 floor tests run on host checkouts too).

### v2.8.2 — memory recall patches must bind the FRAMEWORK classes (2026-09-01)

`extensions/python/monologue_start/_10_memory_recall_patches.py` had the
same defect class as the v2.8.1 stale install, but the opposite direction:
it imported core extension classes via CANONICAL dotted paths
(`plugins._memory.extensions..._91_recall_wait`), while the framework loads
extension files via `helpers.modules.import_module` — a SYNTHETIC module
named after the file basename, never registered in `sys.modules`. The
canonical import therefore created a phantom module + class the dispatcher
never instantiates. Every runtime patch here was a silent no-op live:

- `RecallWait.execute` wrap never installed (the 30s recall `TimeoutError`
  still killed the agent loop — 2026-08-27 crash tracebacks show no
  `safe_execute` frame),
- `SEARCH_TIMEOUT = 90` never applied (framework module kept 30),
- `MemorizeMemories`/`MemorizeSolutions` wraps never installed.

The fix resolves classes through `helpers.extension._get_extension_classes`
— the exact cached list `call_extensions_async` iterates — matched by class
name + `__module__.endswith(basename)`. `SEARCH_TIMEOUT` is patched via a
method's `__globals__` (the synthetic module dict; the module itself is not
in `sys.modules`). Two behavioral corrections while there:

- `RecallWait.execute` is now WRAPPED (calls the original), not replaced —
  the old `safe_execute` reimplemented the pre-v2.11 body and would have
  silently dropped upstream's recall-result application had it ever bound.
- The wrapper re-raises `CancelledError` (shutdown cancellation must not be
  swallowed) and swallows only `TimeoutError`/`Exception`.

Tests: `tests/test_memory_recall_patches_v282.py` (5/5). Gotcha: build fake
module globals with `exec()` — a nested `def`'s `__globals__` is the TEST
module's dict, so the `_mfb_timeout_patched` flag leaks across tests and
silently disarms the later assertions.

The same phantom-class bug existed in `memory_hardening` v0.5.2
(recall-wait guard + recall method patch) — fixed there in parallel (v0.5.3,
`helpers/extension_class.py` shared resolver).

This plugin also owns six **LLM-error-handling** extensions on top of the
cascade. They live here (not in separate plugins) so a single
`usr/plugins/model_fallback/.toggle-0` disables everything and a single
`default_config.yaml` holds the knobs:

* v2.2 Piece 1 — outer timeout guard on `call_utility_model` (catches
  hung utility calls that the cascade's own timeout misses).
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

`usr/plugins/model_fallback/config.json`:

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
The `model_fallback` plugin imports it lazily, so the cascade
continues to work when `memory_hardening` is disabled (the import
catches `Exception` and no-ops).

## Cross-references

- Plugin config: `usr/plugins/model_fallback/config.json`
- Patch entrypoint: `usr/plugins/model_fallback/extensions/python/agent_init/_00_install_fallback_patches.py`
- v2.2 timeout guard: `extensions/python/agent_init/_10_install_utility_timeout_patch.py` + `helpers/utility_timeout.py`
- v2.3 context-size guard: `extensions/python/message_loop_prompts_after/_10_context_size_guard.py` (the `trim_history` function is pure and testable)
- v2.4 langchain v1 shim: `extensions/python/agent_init/_00_install_langchain_shim.py` + `helpers/langchain_compat.py`
- Stats endpoint: `api/stats.py` (`GET /api/plugins/model_fallback/stats`)
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
was removed in v2.5; the WebUI extensions cache (Piece 3 in the
original design) was migrated to the `ui_loader_optimizer` plugin
in v2.6.6 (it now owns both the server-side cache and the
client-side fetch coalescing). Only the outer timeout guard
(Piece 1) remains here. All remaining pieces are independent
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

### Piece 2 — `get_webui_extensions` TTL cache + circuit breaker (migrated v2.6.6)

> **Migrated to `ui_loader_optimizer` v3.5.0.** The 2s TTL cache +
> non-lossy circuit breaker on `get_webui_extensions` (and its
> `webui_extensions_cache_enabled` toggle, the
> `helpers/webui_extensions_cache.py` module, and the
> `run_ui/init_a0/start/_10_install_extensions_cache.py` hook) moved to
> the UI Loader Optimizer plugin, which already owned the complementary
> client-side `fetch` coalescing. The two layers now live in one plugin
> and compose multiplicatively. The self-contained cache module there
> owns its own counters + `snapshot()`; this plugin's `helpers/stats.py`
> no longer carries an `_EXTENSIONS_CACHE` group. See
> `usr/plugins/ui_loader_optimizer/AGENTS.md` for the current design.
>
> Historical note: the original implementation here was a *lossy* breaker
> (returned `[]` on repeated FS errors); the non-lossy stale-while-error
> behavior was added in v2.6.5 and traveled with the migration. The
> earlier docstring's claim that the cache "auto-busts on plugin
> enable/disable/file-edit" was inaccurate — freshness was always
> governed by the 2s TTL; `bust()` was a manual/test hook only.

### Stats endpoint

`GET /api/plugins/model_fallback/stats` returns:

```json
{
  "version": "2.6.6",
  "utility_timeout": {
    "calls_total": 89,
    "timeouts_total": 3,
    "max_observed_wait_s": 67.2,
    "last_timeout_at": 1753000000.0,
    "last_timeout_model": "gemma4:31b",
    "close_inner_attempted": 3,
    "close_inner_succeeded": 3
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
| Context size guard (v2.3) | `context_size_guard.enabled` | `false` | set to `true` to enable (opt-in) |
| LangChain v1 shim (v2.4) | `langchain_compat.enabled` | `true` | set to `false` (v0 users get a no-op install, no impact) |

The `housekeeping.enabled` and `housekeeping.second_pulse_path_enabled`
keys are no-ops in v2.5 (the loop is gone); they are still recognised
so a hand-edited config does not raise on save.

Disabling any one piece does not affect the others. The framework
re-reads the config on the next call (no restart required for
`utility_timeout_guard`, `context_size_guard`, and `langchain_compat`).

### Why this is in `model_fallback` and not its own plugin

The user preferred to keep all three pieces in the existing plugin so
a single `.toggle-0` disables everything and a single config file
holds the knobs. The pieces fall into two groups:

* **Cascade-coupled** (timeout guard, context-size guard): these
  touch the LLM-call path directly and need to know the cascade's
  internals. Splitting them out would require duplicating config
  plumbing.
* **Cascade-adjacent** (langchain shim): runs before or after the
  cascade and is loosely coupled, but the user preferred
  single-plugin symmetry over per-piece independence. (The WebUI
  extensions cache was the fourth piece until v2.6.6, when it
  migrated to `ui_loader_optimizer` — see the Piece 2 note above.)

If a future maintainer wants to split, the migration is
straightforward: each piece is a single helper + a single
extension hook; nothing shares state across pieces.

### Verification

* `python scripts/scan_v22_to_v25.py` — must stay at 0 LIVE findings.
* `python scripts/scan_plugin_structure.py` — must stay at 0 findings
  (or only the pre-existing non-fatal ones).
* `usr/plugins/model_fallback/tests/test_resilience_v22.py` —
  unit tests for the v2.2 layer (timeout guard's `close_inner_coro`
  lazy import; the cache circuit-breaker tests moved to
  `ui_loader_optimizer/tests/test_extension_cache.py` in v2.6.6).
  Housekeeping tests were removed in v2.5.
* `usr/plugins/model_fallback/tests/test_context_size_guard_v23.py` —
  11 unit tests for the v2.3 context-size guard.
* `usr/plugins/model_fallback/tests/test_langchain_compat_v24.py` —
  8 unit tests for the v2.4 langchain v0 -> v1 shim.
* Manual: `/api/plugins/model_fallback/stats` returns sensible
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
in `model_fallback` so a single toggle disables everything.

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
from usr.plugins.model_fallback.helpers import langchain_compat
assert langchain_compat.already_installed_in_process()
```

### v2.3+v2.4 tests

* `usr/plugins/model_fallback/tests/test_context_size_guard_v23.py` —
  11 tests for Piece A (config resolve, trim under/over budget,
  `min_messages` invariant, empty history, list/string content,
  counter snapshot, disabled-by-default contract).
* `usr/plugins/model_fallback/tests/test_langchain_compat_v24.py` —
  8 tests for the langchain v0 -> v1 shim (install, idempotency,
  process marker, user-module protection, uninstall, status
  shape, fallback target, location-in-model_fallback invariant).
* `usr/plugins/model_fallback/tests/test_resilience_v22.py` —
  tests for the v2.2 layer (timeout guard's `close_inner_coro`
  lazy import; the cache circuit-breaker tests moved to
  `ui_loader_optimizer/tests/test_extension_cache.py` in v2.6.6).
  The two-path WebSocket pulse tests were removed in v2.5 together
  with the housekeeping module that hosted them.

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

`usr/plugins/model_fallback/webui/config.html` injects a new
"Resilience features" section between the existing "Fallback
cycles" and "Quick presets" sections. Each toggle is a
`text-input`-free checkbox with a `version-badge` (v2.2 / v2.3
/ v2.4) and a one-paragraph description. The settings bind via
the framework's standard `x-model="context.settings.<key>"`
pattern ONLY. **v2.6.5:** the panel previously paired `x-model`
with a one-way `:checked="$store.modelFallback.isEnabled(...)"`
binding — an Alpine anti-pattern where `:checked` fought
`x-model` so a click never committed and every switch reopened
OFF. The `:checked` binding (and the `isEnabled` /
`isSecondPulsePathEnabled` helpers it used) were removed; the
switches now rely on `x-model` alone, and `backfillToggles` (see
the implementation-files list below) writes the resolved
top-level-or-nested boolean onto `context.settings` at panel
open so a default-ON feature whose key is absent from the loaded
config still displays ON. The number fields likewise bind
`x-model` to the raw key and clamp on `@change` via
`clampField` (replacing the old `:value`-to-normalized-snapshot
binding that displayed the clamp but saved the raw value).

### Restart requirements

| Toggle | Takes effect on |
|---|---|
| `utility_timeout_guard_enabled` | next `agent_init` (next agent created, or server restart) |
| `housekeeping_enabled` | **No-op in v2.5** — the loop is gone, so restart requirement is moot. |
| `second_pulse_path_enabled` | **No-op in v2.5** — the second pulse path is gone. |
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
* **WebUI extensions cache** — *migrated to `ui_loader_optimizer`
  in v2.6.6*; the `webui_extensions_cache_enabled` key is no
  longer read here. To disable the 2s TTL cache during plugin
  development (if it hides file edits), toggle
  `extensions_cache_enabled` in the UI Loader Optimizer config
  and re-enable when done.
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
  store with helpers that mirror the Python resolution rules:
  `backfillToggles(settings)` (writes the resolved boolean
  for any MISSING `*_enabled` key so `x-model` reads the
  effective state), `clampField(settings, key, min, max,
  fallback, asInt)` (clamps a number field in place on
  `@change`), plus `getDefaults` / `applyDefaults` /
  `openHelp` / `ensureLoaded`. v2.6.5 removed the now-dead
  `isEnabled` / `isSecondPulsePathEnabled` (only the removed
  `:checked` binding used them) and `normalizeSettings` (only
  the removed `:value` binding used it).
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

* `usr/plugins/model_fallback/tests/test_webui_toggles_v25.py` —
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
   * `helpers/langchain_compat.py`
   * `extensions/python/message_loop_prompts_after/_10_context_size_guard.py`

   (`helpers/webui_extensions_cache.py` was already removed in
   v2.6.6 when the cache migrated to `ui_loader_optimizer`.)

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

### v2.6.7 — Router detection + router-class wait tuning (2026-08-11)

**Problem (the "300 second wait").** The v2.6.2 router capacity class
detected routers by **label prefix only**: a candidate was `router` iff
its cascade label started with `omniroute/`. But the user's OmniRoute
provider is registered with `litellm_provider: openai`
(`conf/model_providers.yaml`), so its chat/utility model_name arrives as
`openai/auto/best-coding` — prefix `openai`, **not** `omniroute/`. The
primary was mis-classified as `free_per_minute`, and a single `asyncio`
timeout on it set a **300 s** cooldown (the `free_per_minute` timeout
cooldown). OmniRoute is a self-healing gateway that re-routes the next
request to a healthy tier, so that 5-min lockout stretched a seconds-long
upstream blip into a 400-700 s recovery — exactly the "modular fallback
interrupting omniroute" symptom. The cascade was doing its job (don't
hammer a hung endpoint) but on the **wrong** capacity class.

**Root cause is detection, not the wait policy.** The wait policy for
`free_per_minute` is correct for a metered endpoint; it's wrong for a
self-healing gateway. Four fixes, all gated on correct router detection:

* **Fix A1 (label prefix, configurable).** `_classify_capacity` now also
  matches configurable `router_label_prefixes` (default `["omniroute"]`;
  add e.g. `"openai/auto"` to classify the aliased primary by label).
* **Fix A2 (api_base, durable).** The provider config injects
  `api_base: http://host.docker.internal:8080/v1` onto the wrapper kwargs
  (`models.py:_merge_provider_defaults`); `_get_candidate_api_base`
  surfaces it and `_classify_capacity` matches it against
  `router_api_bases` (default 5 gateway host spellings on `:8080`).
  This is the reliable matcher — it's stable regardless of which
  `litellm_provider` prefix the label carries. A1 OR A2 wins.
* **Fix B (timeout -> no cooldown).** Once router-class, a **pure**
  `asyncio.TimeoutError` (no HTTP status) sets no cooldown
  (`router_timeout_no_cooldown`, default `true` →
  `router_timeout_cooldown_s`, default `0.0`) — the cascade retries the
  gateway on the very next cycle instead of locking it out for 300 s.
  Actual 5xx still gets `router_cooldown_s` (default 5 s) to space a
  storm.
* **Fix C (cycle backoff cap).** Continuous fallback + exponential
  backoff compounds the wait: a router primary in a stuck cycle backs
  off up to `max_cycle_delay_s` (user default 300 s). When the primary
  is router-class, the cap is overridden by `router_max_cycle_delay_s`
  (default 30 s) so the cascade re-probes within half a minute instead
  of every 5 min.
* **Fix D (per-call timeout headroom knob).** Router-class candidates are
  already exempt from the 20 s warm fast-path (v2.6.3) and use the cold
  base. `router_call_timeout_s` (default `0.0` = base unchanged) lets
  the user give the self-healing gateway **more** headroom than the cold
  base so litellm's internal 3× retry + the core 2×1.5 s retry can
  recover a transient blip before the cascade declares a timeout.

**Expected outcome.** OmniRoute primary reclassified as `router` → 429
no cooldown, timeout → 0 s (retry next cycle), primary-skip exempt,
cycle backoff capped at 30 s, cold timeout 90 s unchanged → recoveries
drop from 400-700 s to < 60 s. The gateway's own 4-tier internal
fallback keeps working because the cascade no longer locks it out.

**Activation (important).** `plugins.get_plugin_config` does **not**
merge `default_config.yaml` when a `config.json` exists — it's one or
the other. So the new `default_config.yaml` keys only load for installs
with no `config.json`. For installs that do have a `config.json` (this
one), the fixes still activate because every helper falls back to a
module-level `_DEFAULT_*` constant and every `cfg.get(...)` has a safe
inline default (`True` / `0.0` / `_DEFAULT_ROUTER_MAX_CYCLE_DELAY_S`).
**`config.json` does not need to be edited**; the user's sacred
`config.json` is left untouched. To override a default (e.g. add
`"openai/auto"` to label prefixes, or raise `router_call_timeout_s` for
extra headroom), add the key to `config.json` or via the WebUI.

**Files.** `fallback.py` — `_DEFAULT_ROUTER_API_BASES`,
`_DEFAULT_ROUTER_MAX_CYCLE_DELAY_S`, `_router_label_prefixes`,
`_router_api_bases`, `_is_router_label_prefix`, `_is_router_api_base`,
`_get_candidate_api_base`, and the `api_base` param on
`_classify_capacity` / `_capacity_skips_cooldown` / `_handle_error_cooldown`
/ `_resolve_per_call_timeout`; `cand_api_base` threaded through both
cascades' call sites; `primary_is_router` cap in both `_compute_cycle_sleep`
functions. `default_config.yaml` — six new documented knobs.

### v2.6.8 — Paid-model warm-path drop-off + router 4xx lockout (2026-08-19)

**Problem (the "drops off after one request" stall).** A working paid
utility model — `a0_venice` (Agent Zero API), capacity `unlimited_paid` —
succeeded on its first (cold, 90s) call, was marked "warm", and then its
**second** call was forced onto the 20s warm fast-path. A long utility
prompt legitimately needs more than 20s, so the call hit a pure
`asyncio.TimeoutError` (no HTTP status). That pure timeout fell through
to the conservative **300 s** unknown-error cooldown → a 5-minute stall
on a model that was perfectly healthy. Symptom: "I have credit and it's
working, but it drops off after one request and sits in cooldown."
The same dynamic stalled `omniroute/auto/coding:free` when cycling.

**Root cause: the warm ceiling + the 300s timeout cooldown, applied to a
capacity class that should never use either.** Three fixes:

* **Fix 1 — pure-timeout 45s cooldown** (`_cooldown_seconds_for_status`).
  A pure `TimeoutError` (no status code) now cools ~45s
  (`timeout_cooldown_s`, default 45) instead of 300s. A slow call is not a
  broken model. Per-HTTP-status cooldowns (`_DEFAULT_COOLDOWNS_S` for 429,
  500, etc.) are unchanged; Retry-After still wins.
* **Fix 2 — `unlimited_paid` warm-path exemption** (`_resolve_per_call_timeout`).
  Paid models legitimately run long utility prompts, so they always get the
  full cold `base_timeout_s` — never the 20s warm ceiling. Only
  `free_per_minute` labels keep the aggressive warm fast-path. This was the
  decisive fix: the 2nd call no longer times out at 20s, so it never reaches
  the cooldown path that caused the drop-off.
* **Fix 3 (Fix B) — router 4xx exemption** (`_handle_error_cooldown`).
  `_is_permanently_failed_model` is True for 401/402/403/404, so an
  `omniroute/*` 401/403/404 used to hit the permanent-fail block → a 300s
  (401/403) or 24h (404) cooldown. The permanent-fail block now skips
  `router`-class labels, so a router 401/403/404 falls through to the 5s
  `router_cooldown_s` block — the gateway re-routes the next call, so a
  minutes-long lockout per outage was wrong. Non-router labels keep the
  permanent 401/403/404 cooldown (sanity-tested).

**Activation.** Same gotcha as v2.6.7: `plugins.get_plugin_config` does
not merge `default_config.yaml` when a `config.json` exists. The new
`timeout_cooldown_s` key only loads for installs with no `config.json`;
for this install the fix still activates because
`_cooldown_seconds_for_status` falls back to the module-level
`_DEFAULT_TIMEOUT_COOLDOWN_S = 45.0` constant. `config.json` is left
untouched.

**Files.** `fallback.py` — `_DEFAULT_TIMEOUT_COOLDOWN_S` +
pure-timeout branch in `_cooldown_seconds_for_status`; the
`unlimited_paid` early-return in `_resolve_per_call_timeout`; the
`_classify_capacity(...) != "router"` guard on the permanent-fail block
in `_handle_error_cooldown`. `tests/test_router_capacity.py` — updated
`test_resolve_per_call_timeout_non_router_warm_uses_warm` to use
`ollama_cloud/*` (free_per_minute, since `a0_venice` is now exempt); added
`test_resolve_per_call_timeout_unlimited_paid_warm_uses_base`,
`test_handle_error_401_short_cooldown_for_router`,
`test_handle_error_401_permanent_cooldown_for_non_router`, and the
`_FakeAuthError` fixture. `default_config.yaml` — new documented
`timeout_cooldown_s` knob. Manifest/stats/README bumped 2.6.6 → 2.6.8.
Suite: 160/160. Commit `e91f63d` → Olszalsik/a0-model-fallback main.

### v2.6.8 addendum — a0_venice api_base classification (2026-08-20)

**Problem (Fix 2 didn't actually reach `a0_venice`).** Fix 2 above exempts
`unlimited_paid` from the warm fast-path, and `_classify_capacity` returns
`unlimited_paid` for the `a0_venice` provider segment (line 1030). But the
a0_venice provider is registered with `litellm_provider: openai` (see
`conf/model_providers.yaml`), so its cascade label arrives as
`openai/deepseek-v4-flash` — the `a0_venice` segment check never matches and
the label fell through to `free_per_minute`. So Fix 2's exemption silently
missed the exact model it was written for, and the "drops off after one
request" symptom persisted. This is the symmetric case of the v2.6.7
omniroute mis-classification (same root cause: `litellm_provider: openai`
aliasing), and the fix is the same shape: classify by the durable
`api_base` signal.

**Fix — `unlimited_paid_api_bases` matcher** (`_classify_capacity`). After
the router api_base check, before the `free_per_minute` fallthrough, a
candidate whose `api_base` contains a substring in `unlimited_paid_api_bases`
(default `["llm.agent-zero.ai"]`, the a0_venice endpoint) is classified
`unlimited_paid`. The `api_base` is threaded end-to-end: both cascade call
sites pass `cand_api_base` → `_resolve_per_call_timeout(api_base=...)` →
`_classify_capacity(label, agent, api_base)`. With `a0_venice` now correctly
`unlimited_paid`, Fix 2's warm-path exemption reaches it → the 2nd call gets
the full cold `base_timeout_s` → no 20s timeout → no 45s cooldown → the
model no longer drops off after one request.

**Activation.** Same gotcha as v2.6.7/v2.6.8: `get_plugin_config` does not
merge `default_config.yaml` when a `config.json` exists. The new
`unlimited_paid_api_bases` key only loads for installs with no `config.json`;
for this install the fix activates via the module-level
`_DEFAULT_UNLIMITED_PAID_API_BASES = ("llm.agent-zero.ai",)` constant.
`config.json` is left untouched.

**Files.** `fallback.py` — `_DEFAULT_UNLIMITED_PAID_API_BASES` constant +
`_unlimited_paid_api_bases(agent)` + `_is_unlimited_paid_api_base(api_base,
agent)` helpers (after `_DEFAULT_ROUTER_MAX_CYCLE_DELAY_S`); the
`_is_unlimited_paid_api_base` call in `_classify_capacity`. The router and
unlimited_paid api_base matchers are intentionally ordered
router-then-unlimited_paid (a gateway api_base wins over a paid-endpoint
api_base if both ever matched; in practice they're disjoint hosts).
`default_config.yaml` — new documented `unlimited_paid_api_bases` knob.
`tests/test_router_capacity.py` — added
`test_classify_a0_venice_api_base_unlimited_paid` (the core regression:
`openai/deepseek-v4-flash` without api_base → `free_per_minute`, with
`llm.agent-zero.ai` → `unlimited_paid`; incl. case-insensitive + substring +
specificity guards) and
`test_resolve_per_call_timeout_a0_venice_api_base_warm_uses_base`
(end-to-end: warm `openai/deepseek-v4-flash` + a0_venice api_base → 300s cold
base, not 20s warm). Suite: 162/162.

> The accompanying UI fix for the empty fallback-provider dropdown lives
> in the `_model_config` plugin (`model-config-store.js` `ensureLoaded`
> re-fetch + `model-field.html` fallback-select `x-effect`), commit
> `648dd7e` → Olszalsik/a0-model-config main. That was a display bug, not
> the cycling cause.

### v2.6.9 — stagnation closure `nonlocal` fix (2026-08-25)

**Problem (crash during a sustained outage).** The v2.6 Phase 3 stagnation
code crashed with `UnboundLocalError: cannot access local variable
'stagnation_logged_this_outage'` exactly when the agent was already
slow — during a sustained outage where every candidate is in cooldown
and `consecutive_no_success_cycles` crosses `cycle_stagnation_threshold`
in continuous mode. The crash fired inside `history.compress()`
(via `summarize_messages` → `call_utility_model` → the utility cascade),
killing history compression on top of the underlying outage.

**Root cause — closure scoping.** `stagnation_logged_this_outage` is
declared in the outer cascade function (~line 1561) as the one-log-per-
outage gate. The nested `_compute_cycle_sleep` both *reads* it
(`if not stagnation_logged_this_outage:`) and *writes* it (`= True`).
In Python, **any** assignment in a function body makes that name local to
the *entire* function, so the read above the assignment raises
`UnboundLocalError` — the outer declaration is ignored, no `nonlocal`
was declared. Trigger requires `continuous_mode` + stagnation
threshold reached + `cycle_stagnation_factor > 1.0`, which is why it
only surfaced mid-outage, not in normal operation or the test suite.

**Second, silent layer (both cascades).** The `_succeed` nested functions
reset `consecutive_no_success_cycles = 0` and
`stagnation_logged_this_outage = False` on recovery, but their
`nonlocal` lines only declared `consecutive_full_cycles,
fallback_started_at` — so those resets wrote to throwaway locals. In the
utility cascade this means the stagnation log-gate never re-armed and
the counter never truly reset after a recovery. In the chat cascade the
vars aren't consumed by `_compute_cycle_sleep` (no stagnation block
there), so it was harmless dead code — but the same latent trap.

**Fix.** Added `nonlocal stagnation_logged_this_outage` inside the
utility `_compute_cycle_sleep`; added
`nonlocal consecutive_no_success_cycles, stagnation_logged_this_outage`
to both `_succeed` functions (utility + chat). No behavioral change to
the algorithm, only to which scope the gate/counter live in.

**Why the 162/162 suite missed it.** `tests/test_adaptive_sleep_v26.py`
drives a `_compute_replica` helper that re-implements the envelope with a
passed-in `log_emitted: list` and `.append()` — it never assigns to a
closure gate variable, so it structurally cannot reproduce the
read-then-write-without-`nonlocal` trap. The real closure was untested.
A future regression test should drive the actual
`_patched_call_utility_model` closure through a stagnation cycle.

**Files.** `fallback.py` only — three `nonlocal` additions, no algorithm
or config change. `py_compile` clean; 9/9 stagnation tests pass; full
suite 158/162 on the Windows host (the 4 fails are the pre-existing
container-path `/a0/.../default_config.yaml` `FileNotFoundError`, unrelated;
they pass inside the container). Note this fix only stops the *crash*
during an outage; the underlying provider outage is upstream and is
routed around by the existing cooldown/router-detection logic once the
crash stops masking it.

### v2.7.0 — Router cold-call warm-up budget + cross-agent dead-mark exemption (2026-08-31)

**Problem (the "omniroute mislabeled unavailable" stall under
multi-agent load).** Three concurrent agents with
`omniroute/auto/coding:free` in the utility chain burn the gateway's free
pool fast. Two compounding failures followed. (1) **Every warm-up call
died at exactly 60 s.** After pool exhaustion the gateway must re-route
to a fresh upstream, and that warm-up (probing rate-limited/dead slugs
before landing on a working one) routinely exceeds the user's 60 s
utility cold base (`fallback_utility_timeout_s: 60`). The router branch
in `_resolve_per_call_timeout` had no cold/warm awareness: it returned
`max(base, router_call_timeout_s=0)` = 60 s unconditionally, so the
gateway never got the headroom to finish re-routing — one
`timed out after 60s` per cycle, then the cascade cycled away.
(2) **Worse, the gateway got cross-agent dead-marked.** The gateway
surfaces its upstreams' failures: one pooled upstream's 401/403/404 (dead
free slugs are common on openrouter) arrives as a gateway 4xx. The
`_mark_label_dead` call in the router block of `_handle_error_cooldown`
then put the `omniroute/<combo>` label into `_INMEM_DEAD_LABELS` with the
per-status dead window — **24 h for 404** — blocking the label for every
agent in the process even though the gateway itself was healthy and had
eleven other upstreams. Symptom: "the fallback plugin labels omniroute
cooldown/broken, but in reality it still has 32 free models."

**Root cause: two router-specific policies that fit a metered endpoint
but not a pooled self-healing gateway.** A timeout at the base budget is
a warm-up-in-progress, not a broken endpoint; and a gateway-surfaced
upstream 4xx is one upstream's state, not shared endpoint death (the
gateway re-routes the next call and its own connection health —
`exhausted_connection` exclusions, rate-limit backoffs — is the
authority for upstream choice).

**Fix 1 — cold-state warm-up budget (`_resolve_per_call_timeout`).**
New knob `router_cold_call_timeout_s` (module default
`_DEFAULT_ROUTER_COLD_CALL_TIMEOUT_S = 150.0`). For a router-class label
that is COLD (no entry in `_WARM_LABELS` within the effective warm
window — the same warm model the warm fast-path uses), return
`max(router_budget, router_cold_call_timeout_s)` where
`router_budget = max(base, router_call_timeout_s)` (Fix D unchanged).
Warm router calls keep the fast base, composing with Fix D. `<= 0`
disables. This is the inverse direction of the existing warm/cold
system: warm/cold *shortens* healthy cloud calls; cold-budget *extends*
a gateway's first call after pool churn.

**Fix 2 — router dead-mark exemption (`_mark_label_dead`).** The
function self-gates: a router-class label (via
`_classify_capacity(label, agent, api_base)` — both the label-prefix
(A1) and api_base (A2) detection paths) returns before the
`_is_dead_for_all_agents` check, so a gateway-surfaced upstream
401/402/403/404 can no longer poison the shared blocklist. `api_base` is
threaded from both call sites inside `_handle_error_cooldown` (it was
already a parameter there since v2.6.7). The router block's call is
removed outright (the whole block is router-class by construction);
non-router labels keep the shared dead-mark unchanged. Per-agent
cooldowns for router labels are unchanged in every branch — only the
cross-agent mark is skipped.

**Activation.** Same gotcha as v2.6.7/v2.6.8: `get_plugin_config` does
not merge `default_config.yaml` when a `config.json` exists. The new
`router_cold_call_timeout_s` key only loads for installs with no
`config.json`; for this install the fix activates via the module-level
`_DEFAULT_ROUTER_COLD_CALL_TIMEOUT_S = 150.0` constant and the inline
default in `_resolve_per_call_timeout`. `config.json` is left untouched.

**Files.** `fallback.py` — `_DEFAULT_ROUTER_COLD_CALL_TIMEOUT_S`
constant (after `_DEFAULT_ROUTER_MAX_CYCLE_DELAY_S`); the rewritten
`capacity == "router"` branch of `_resolve_per_call_timeout`; the
`api_base` param + router guard on `_mark_label_dead`; both
`_handle_error_cooldown` call-site updates (permanent-fail call passes
`api_base`; the router block no longer marks dead).
`default_config.yaml` — new documented `router_cold_call_timeout_s: 150.0`
knob in the router block. `tests/test_router_capacity.py` — six new tests:
cold→150, warm→base(60), knob-0 disable, cold composes with
`router_call_timeout_s`, dead-mark skips routers (label-prefix path),
dead-mark skips routers (api_base path). `plugin.yaml`/`README.md`
bumped 2.6.8 → 2.7.0. Suite: 179/179 on the Windows host
(`REPO_ROOT_OVERRIDE="$(pwd -W)"`).

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
| `router_label_prefixes` | `["omniroute"]` | 2.6.7 | extra label prefixes classified as `router` (A1); OR with `router_api_bases` |
| `router_api_bases` | 5× `*:8080` | 2.6.7 | api_base substrings classified as `router` (A2, durable); OR with prefixes |
| `router_timeout_no_cooldown` | true | 2.6.7 | pure `asyncio.TimeoutError` on a router sets no cooldown (Fix B) |
| `router_timeout_cooldown_s` | 0.0 | 2.6.7 | timeout cooldown when the above is false (Fix B) |
| `router_max_cycle_delay_s` | 30 | 2.6.7 | cycle-backoff cap when the primary is router-class (Fix C); clamped `[0,3600]` |
| `router_call_timeout_s` | 0.0 | 2.6.7 | per-call timeout headroom for router-class; `0` = cold base unchanged (Fix D) |
| `router_cold_call_timeout_s` | 150 | 2.7.0 | per-call timeout for a COLD router call (no success within the warm window): `max(base, router_call_timeout_s, this)`; covers gateway warm-up after free-pool exhaustion; `<= 0` disables |
| `timeout_cooldown_s` | 45 | 2.6.8 | cooldown for a *pure* timeout (no HTTP status) on a non-router label; replaces the 300s unknown-error cooldown. Per-status cooldowns unchanged |
| `unlimited_paid_api_bases` | `["llm.agent-zero.ai"]` | 2.6.8 | api_base substrings classified `unlimited_paid` (durable); catches the a0_venice primary whose `litellm_provider: openai` makes its label `openai/deepseek-v4-flash`, so the warm-path exemption (Fix 2) reaches it. `[]` disables (→ `free_per_minute`) |

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
- **v2.6.8 pure-timeout cooldown:** `timeout_cooldown_s: 300` restores the
  legacy 300s unknown-error cooldown for pure timeouts. The
  `unlimited_paid` warm-path exemption (Fix 2) and the router 4xx exemption
  (Fix 3) have **no knob** — they are inherent capacity-class policies.
  To make a paid model use the warm fast-path again, it would have to be
  reclassified as `free_per_minute` (which is wrong for a paid endpoint).
- **v2.6.8 a0_venice api_base matcher:** `unlimited_paid_api_bases: []`
  disables the durable api_base classification — an `openai/...`-aliased
  paid endpoint then falls back to `free_per_minute` (the original
  drop-off symptom returns). Add a host substring here for any other paid
  endpoint that runs through an `openai`-aliased `litellm_provider`.

---

## v3.4.1 (audit) — comprehensive technical audit, 2026-09-28

READ-ONLY audit of the whole plugin (fallback.py 4,437 lines in two halves,
models_ext.py, hooks.py, helpers/, extensions/, api/, webui/, config) against
framework `helpers/api.py`, `agent.py`, `models.py`, `helpers/plugins.py`.
Method: four parallel audit passes, then every high/medium finding re-verified
by direct code inspection (several against the live framework classes). All
line numbers are against the v3.4.0 working tree (uncommitted at audit time).
**Fix pass: every actionable finding below was FIXED the same day (see the
"v3.4.1 fix pass" ledger at the end of this section); 355/355 tests re-run
green after the fix pass.** Deferred items are marked in the ledger.
Verified-clean areas are listed at the end.

### Critical / high

1. **Retry-After header walk is dead for real litellm exceptions** —
   `models_ext.py:612-615`. `_header_candidates` does
   `if not isinstance(headers, dict): continue`, but `httpx.Headers`
   subclasses `collections.abc.Mapping`, NOT `dict` (verified: MRO
   `Headers → MutableMapping → Mapping → Collection`; `isinstance(h, dict)`
   is False). litellm/openai exceptions carry `.headers` /
   `.response.headers` as `httpx.Headers`, so for exactly the exceptions the
   helper was written for it yields nothing. Silently disabled: the
   Retry-After priority in `_cooldown_seconds_for_status` (fallback.py:1075),
   the provider-hint-wins contract of `_quota_cooldown_seconds`
   (fallback.py:274), and the extended-retry hint path. Only the message-text
   regex scrape (models_ext.py:692+) still works. Fix: `isinstance(headers,
   Mapping)` or `dict(headers)`. **Fixing this one first also improves
   finding 3 below (hints currently never arrive at all).**

2. **User `TIMEOUT=` kwarg is voided — strip-before-check ordering, and the
   strip is persistent** — `fallback.py:3186→3295` (engine), `4150→4175`
   (turn). `model_kwargs = getattr(model_obj, "kwargs", {})` (2644/3910) is
   the LIVE `ModelConfig.kwargs` dict (plain dataclass field, models.py).
   `_strip_a0_only_kwargs(current_model, is_primary=(idx == 0))` pops
   `"TIMEOUT"` from that shared dict, then `user_kwarg_set` is computed
   AFTER the strip — always False on idx==0, so the warm/cold/latency
   resolution overrides the user's kwarg on every attempt, reinstating the
   20s warm-ceiling failure mode fix A (v2.6.4) was for. Worse: because the
   strip persists across invocations, the kwarg is erased from the model
   config entirely after the first call. Fix: snapshot the kwargs flag once
   at the top of the cascade, before any strip.

3. **Provider reset hint LOSES to the scope guess (`max` instead of
   hint-wins)** — `fallback.py:274-276`. `dur = max(dur, float(hint))`
   contradicts the v3.4.0 invariant ("a provider reset hint always beats our
   guessed scope duration") and both the docstring (:247-252) and
   default_config.yaml. A `retry-after: 60` on a `session` scope (900s
   default) parks a healthy-again label 900s. Fix: `dur = float(hint)` when
   hint > 0 (keep the `quota_cooldown_max_s` clamp). Compounded by the
   greedy phrase list: Groq's free-tier transient 429 ("... Usage limit ...
   Please try again in 2m3.16s", models_ext.py:199) classifies as quota
   `account` → 3600s for a ~2-min throttle; with the header walk fixed, the
   hint would at least be visible to `max()`.

4. **kwargs-configured `fallbacks` are erased after the first invocation**
   — `fallback.py:1224` ("fallbacks"/"fallback" in `_A0_ONLY_KWARGS`) +
   `1281-1288` (strip pops them, even `is_primary=True` — the
   `_PROVIDER_SPECIFIC_KWARGS` keep-list only exempts
   venice_parameters/extra_body) + `3186` (strip on the live primary dict).
   `_build_candidates` source #1 (documented "most common", 1318-1326) reads
   that same live dict, so on the next call the list is gone; source #2 is
   dead (finding 5), so rotation silently falls through to env/plugin
   config — or to a single candidate. Latent today (candidates come from
   plugin config here), high impact for any kwargs-based setup. Fix: never
   strip from the live primary's dict, or rebuild/restore after the read.

5. **Agent-config fallback source #2 is dead code** — `fallback.py:1328-1341`.
   `isinstance(agent.config, dict)` is always False: `AgentConfig` is a
   dataclass (agent.py:327-331: mcp_servers/profile/knowledge_subdirs/
   additional). Any preset defining `fallback_models` /
   `utility_fallback_models` / `chat_model_fallbacks` at agent-config level
   is silently ignored. Fix: read from `agent.config.additional` (or the
   framework's serialized config) or drop source #2 from the docs.

### Medium

6. **Blanket `except` in `_quota_cooldown_seconds` converts an internal
   error into a 1-hour account-scope booking** — `fallback.py:282-283`.
   Contract says "Returns None when e is not a quota exhaustion", but any
   internal exception returns `_DEFAULT_QUOTA_SCOPE_COOLDOWNS_S["account"]`
   (3600s) — a plain transient 429 that trips an internal error is parked
   1h instead of 30s. Same `except: return literal` class as the documented
   `cfg0` NameError lesson: bind the fallback BEFORE the try, and make the
   except return `None` here.

7. **Turn path never sets the Responses-5xx sticky mark** — the engine books
   it at `fallback.py:3463-3465` (the ONLY `mark_responses_5xx_seen` call
   site in the repo); the turn path (4150-4300) only READS
   `should_force_chat_completions` (4165). A main-loop model 5xxing on
   `/v1/responses` is never marked sticky, never switched to
   chat-completions, and re-pays the 5xx every turn. The comment at
   4152-4158 describes an intent the code does not implement.

8. **Router pure-timeout check misses litellm.Timeout** — `fallback.py:2194`.
   Raw `isinstance(e, (asyncio.TimeoutError, TimeoutError))` while
   `_is_timeout_shaped` (994-1011, used at 1080/3432/4227) exists precisely
   for litellm's `Timeout → APITimeoutError → APIConnectionError` family. A
   litellm.Timeout on a router label books `router_cooldown_s` (5s→60s)
   instead of the `router_timeout_cooldown_s=0` self-healing intent (v2.6.7
   Fix B). Fix: `status_code is None and _is_timeout_shaped(e)`.

9. **`_get_plugin_cfg(None)` reads an unscoped cache entry** —
   `fallback.py:1080, 1089-1098` (`_cooldown_seconds_for_status`). The
   config resolver's cache key includes project/agent-profile/context id
   only when agent is not None (helpers/config_defaults.py:134-159), so
   project- or profile-scoped overrides of `timeout_cooldown_s` and
   `format_error_cooldown_s` are silently ignored in these two policy paths.
   Fix: thread the agent down from `_handle_error_cooldown_impl` (which has
   it).

10. **Recovery-probe sweep never drops expired-but-present entries** —
    `helpers/recovery_probe.py:208-215`. `until` is a monotonic deadline;
    once it passes with no cascade having popped the entry,
    `(until - now) < min_cooldown` is true with a negative remainder →
    `continue` forever: never popped, never probed, registry grows for the
    process lifetime. `registered_at` (recorded at :133) is never read —
    the intended TTL is missing. Fix: treat `until <= now` as not-in-cooldown.

11. **Recovery probes: no per-target backoff → head-of-line blocking** —
    `recovery_probe.py:225-253`. A failing label is re-probed every 45s
    sweep, max 2 targets/cycle; a quota-exhausted label with a 12h
    cooldown eats both probe slots for 12h, starving shorter-cooldown
    labels that would actually recover. Fix: exponential per-target
    backoff or order targets by remaining cooldown.

12. **Third config call site still shallow-merges** —
    `extensions/python/agent_init/_10_install_utility_timeout_patch.py:56-66`
    (and the context-size guard extension, :268-270). `dict(defaults) +
    update(cfg)` replaces a whole nested section: a config.json carrying a
    partial `utility_timeout_guard:` section silently drops the YAML
    siblings (`enabled`/`max_wait_s`/`close_inner_on_timeout`). This is the
    exact bug class `helpers/config_defaults.py` (v3.4.0 WIP) documents and
    solves — route both call sites through `resolve_config`. The
    `utility_timeout` guard also has stale literals 30/120 instead of
    DEFAULTS 60/180 (helpers/utility_timeout.py:96-97), which makes an
    explicit `0` unusable.

13. **`/stats` is documented GET but the handler is POST-only** —
    `api/stats.py:3` says `GET .../stats` (help.html §10 too), but `Stats`
    never overrides `get_methods()` and the framework default is
    `["POST"]` with a hard 405 at dispatch (helpers/api.py:51-52, 255).
    Any GET consumer (curl, dashboard tile, uptime probe) gets 405. Fix:
    `def get_methods(self): return ["GET", "POST"]` (and consider Events),
    or correct both docs.

14. **Hardcoded `"utility"` cascade kind in the shared engine** —
    `fallback.py:3061` (all-permanent early-exit) and `3511` (post-break
    exhaustion). When the CHAT cascade hits these paths the summary reads
    "Utility fallbacks exhausted" and the event is recorded with
    `cascade_kind="utility"`, corrupting the event timeline's kind filter.
    The adjacent exhausted message correctly uses `spec.kind` (3513).
    Fix: pass `spec.kind` at both sites.

### Low

15. `fallback.py:281` — `max(60.0, min(dur, cap))`: a user
    `quota_cooldown_max_s` below 60 is silently overridden by the floor.
    Clamp to `[min(60, cap), cap]`.
16. `fallback.py:1055-1057` — `_is_format_error` substring match
    ("format"/"parse"/"json" anywhere in the message) misclassifies
    unrelated ValueErrors (e.g. datetime format errors) as model format
    slips → 20s cooldown instead of 300s unknown. Restrict to the
    response-parse path's exception types.
17. `fallback.py:576-578` (+ 587/659/697) — `cfg.get("gen_budget_max_s")
    or _DEFAULT...` coerces an explicit 0 to the 600s default; the three
    `if max_s > 0 else <uncapped>` branches are dead code. Use
    `is not None` semantics; document whether 0 means uncapped.
18. `fallback.py:3815-3905` — the turn cascade's n≤1 path calls `original`
    with NO cooldown-skip check and NO chat-completions forcing: a booked
    cooldown is re-paid on every RetryAfterHours wake, and a Responses-5xx
    primary with no fallbacks configured keeps 5xxing forever.
19. `fallback.py:4222` — the turn path's except block calls
    `_evict_warm_on_timeout(e, label)` UNCONDITIONALLY (the helper itself
    only gates CancelledError, 742-765), while the engine keeps the warm
    entry for router/concurrent_paid 429s (2063-2064 rationale). One
    router 429 on the main loop forces the next turn onto the full cold
    budget. Gate it behind `_is_timeout_shaped(e)`.
20. `fallback.py:3041` — the all-skipped spin path uses a raw
    `asyncio.sleep(spin_s)` up to 300s, the one long sleep that defeats
    `_yielding_sleep`'s event-loop starvation protection — precisely the
    multi-minute all-dead-outage case in continuous mode.
21. `models_ext.py:401` — `_RESPONSES_5XX_SEEN` has no disable-time reset
    (hooks.uninstall resets cooldowns/warm/budgets/latency/stats but not
    this sticky set); after plugin disable/re-enable, stale force-mode
    marks survive for process lifetime. Add `reset()` + call from
    uninstall.
22. `models_ext.py:366-368` — `_is_permanently_failed_model` still returns
    True for ANY rate-limited error, contradicting the module's own v3.4.0
    contract comment (161-164). No live bug today (the cooldown branch
    returns first; rotation uses `_is_permanent_for_rotation`), but any new
    caller re-introduces raw-429-permanent. Fold 429 out of that helper.
23. `models_ext.py:592-598` — `_parse_duration_seconds` digit-sums ISO
    reset timestamps: `"2026-09-28T10:00:00Z"` → 2026+09+28+10 ≈ 2064s of
    plausible-looking garbage cooldown. Reject values containing
    `-`/`:`/`T` or parse ISO dates explicitly.
24. `hooks.py:270-277` — display default `initial_cycle_attempts: 60`
    contradicts default_config.yaml's 20 (config.json masks it); the six
    int/float/bool coercions run OUTSIDE the try (a string-typed
    config.json value raises out of the settings API). Align + move inside.
25. `fallback.py:775-777` — `("__global__",)` store key is unreachable
    (`or id(agent)` makes agent_id never None; agent=None callers share one
    store keyed `id(None)`), and `id(agent)` can alias a GC-reused agent
    instance.
26. `fallback.py:3090-3093` vs `3511-3519` — with the default config
    (extended retry on), standard exhaustion raises RetryAfterHours at
    3092 and the summary/`cascade_exhausted` event at 3511 is only
    reachable in the degraded no-extended-retry path: the default-config
    exhaustion never emits the event the timeline keys on. Emit before the
    raise (the early-exit branch already does).
27. `fallback.py:1526-1541` — `_record_last_status`'s `store` parameter is
    dead (never read by the body; all 4 call sites pass it). Drop it.
28. Config/docs drift: config.json:26 `webui_extensions_cache_enabled` is
    dead (nothing reads it since the v2.6.6 migration); README.md:16 says
    "Version: 2.7.0" vs plugin.yaml 3.4.0 and omits the turn-path cascade +
    v3.4.0 work; help.html/webui/config.json disagree on `max_wait_s`
    defaults (180 prose vs 120 webui vs 420 config); webui/fallback-store.js
    `getDefaults()` returns dead key names (`max_cycles`, `timeout_s` —
    real keys are `fallback_max_cycles` etc.; latent because applyDefaults()
    is never called — delete before anyone wires it up).

### Optimization

29. `fallback.py:1301-1379` — candidate list is rebuilt from scratch (4
    sources + normalize) on every patched call; memoize keyed on
    (primary model_name, kwargs-fallbacks identity, config mtime).
30. `api/events.py:49-50` — double `events.snapshot(...)` per request
    (once for count, once for payload); snapshot once.
31. `extensions/python/message_loop_prompts_after/_10_context_size_guard.py:253-273`
    — per-tick uncached `get_plugin_config` + `get_default_plugin_config`
    reads, bypassing the plugin's 1s-TTL `config_defaults.resolve_config`;
    per-turn stat/read work on this host's fragile 9p mount. Same class:
    `monologue_start/_10_memory_recall_patches.py:223-257` re-reads both
    framework memorize files EVERY monologue with no version-stamp
    short-circuit (its sibling patches all short-circuit).
32. `fallback.py:3208-3215, 4301-4308` — unconditional
    `_save_cooldown_store` (two `agent.set_data` writes) on every success
    even when nothing was popped. Skip when `_popped_until is None`.
33. `fallback.py:3942-3943, 4121-4144, 4343-4388` — turn-path log dedupe
    maps (`_last_skip_log_until` etc.) are per-invocation, so a 60s-cadence
    overnight RetryAfterHours wake re-logs every candidate's skip line +
    error-level summary each pass (~n+2 lines/min for hours). Key the
    dedupe on agent data like the cooldown store.
34. `fallback.py:4160-4163` — redundant per-candidate local import of
    `force_chat_completions_mode` / `should_force_chat_completions`
    (already module-level at 107-108). Also `2396-2455`: unused
    `n`/`total_attempts` params on `_maybe_raise_retry_after_hours`; the
    second call at 3512 is unreachable in practice.
35. `fallback.py:2378-2390` — `_yielding_sleep` re-imports
    `on_long_sleep_tick` every 2s slice (hoist once per sleep) and can pass
    a negative `remaining` on the final tick (clamp to 0).
36. `helpers/config_defaults.py:189-210` — `resolve_config` returns the
    SHARED cached dict by reference within its TTL; one mutating caller
    poisons the cache for all readers (latent; no current mutators). Return
    a copy or document the read-only contract. Also `load_defaults`
    (113-117): when the YAML is absent the cache-hit condition forces a
    re-parse every resolve — add a miss sentinel.
37. `helpers/recovery_probe.py:247-254` — probes run
    `model.unified_call` which applies the rate limiter unconditionally
    (models.py:332-352), contradicting "a probe must not touch the agent's
    rate limiter"; at 2 probes/45s small, but during a multi-label outage
    probes compete with real traffic for the shared budget.
38. `models_ext.py:810-820` — `build_fallback_wrapper` mutates the
    caller's `spec` dict in place (bakes parent api_key/api_base in);
    safe today only because `_normalize_spec` rebuilds the dict. Add
    `spec = dict(spec)` at the top.

### Cross-checks resolved

- **Cooldown-store identity** (wrapper-vs-impl): the cascade passes the live
  `_get_cooldown_store(self)` dict (2695/3939) and the impl books into the
  same object (2041) — the `cooldown_booked` comparison is sound. Only the
  corrupt-state `{}` substitution path (2696-2697/3940-3941) diverges; not a
  live bug.
- **Strip ordering vs `_build_candidates`**: candidates are built BEFORE the
  first strip, so a single invocation is unaffected — but see finding 4 for
  the cross-invocation erasure.

### Verified clean

No empty-POST config-wipe exposure (webui uses the framework save flow);
API handlers follow the `class X(ApiHandler): async process` contract; the
synthetic-module gotcha is handled correctly (no sys.modules assumptions);
the v2.8.1 `or {}` store-wipe guards are in place (2695-2697, 3939-3941);
`clear_all_cooldowns` mutates in place; v3.3.0 cooldown persist/restore
round-trips monotonic↔wall clock correctly; `_quota_cooldown_seconds`
clamps internally so the turn path's `max(60, quota_dur)` cannot exceed the
cap; the engine's fresh-coroutine retry is real (v2.8.5 fix); config keys
used by the engine all exist in default_config.yaml; the `_70`
handle_exception extension clamps `retry_after` to [30, 7200]s; quota
errors are never classified permanent-for-rotation; wrapper builder
isolates parent credentials cross-provider; all `DATA_KEY_*` names match
between hooks.py and fallback.py; probe store comparisons are consistently
monotonic-based including the v3.3.0 persist/restore.

### v3.4.1 fix pass — ledger (2026-09-28, same day as the audit)

Findings above were numbered against the v3.4.0 tree; this ledger maps each
number to its disposition. `models_ext.py` + `hooks.py` + `recovery_probe.py`
+ `config_defaults.py` + `utility_timeout.py` + `events.py` + the api/ +
extensions/ + webui/ files are hot-loaded where the framework allows, but
`fallback.py` is imported once per process — **the fallback.py fixes below
require a container restart to go live** (deferred: non-interference with the
running instance). plugin.yaml bumped 3.4.0 → 3.4.1.

**Fixed:**

- #1 — `models_ext._header_candidates`: `isinstance(headers, dict)` →
  `collections.abc.Mapping` (httpx.Headers subclasses Mapping, not dict).
  Regression surface: `test_retry_after_header_variants` only ever tested
  plain dicts — a httpx.Headers case would have caught this dead subsystem.
- #2/#4 — strip erasing user kwargs: `_strip_a0_only_kwargs(is_primary=True)`
  now STASHES the popped A0-only kwargs on the live primary
  (`model._a0_stripped_kwargs_stash`); `_restore_stripped_primary_kwargs()`
  re-adds them at the top of `_run_rotation_cascade` and
  `_patched_call_chat_model_turn` (before the `model_kwargs` read). The
  `user_kwarg_set` timeout flag is computed once at each cascade top and the
  two per-iteration recomputes (engine + turn) are gone.
- #5 — `_build_candidates` source #2 (dead `isinstance(agent.config, dict)`):
  reads `AgentConfig.additional` now (dict-shaped configs still honoured).
- #6 — `_quota_cooldown_seconds` blanket except returns None (was 3600s).
- #7 — turn path now records the Responses-5xx sticky mark +
  `force_chat_completions_mode` (no one-shot retry — streams may be running —
  but later turns stop re-paying the 5xx); same mark added to the n≤1 path.
- #8 — router pure-timeout check uses `_is_timeout_shaped(e)` (litellm.Timeout
  was missed by the raw asyncio/TimeoutError isinstance).
- #9 — `_cooldown_seconds_for_status` takes `agent`; both `_get_plugin_cfg`
  reads are scoped (all 3 call sites updated).
- #10/#11 — recovery probe sweep pops expired (`until <= now`) targets and
  applies per-target exponential backoff (60s→2^n, cap 15min) on failed
  probes (reset when the target is re-registered).
- #12 — both extension `_resolve_config` sites (utility-timeout patch +
  context-size guard) route through `helpers.config_defaults.resolve_config`
  (deep merge, `use_cache=False` to keep per-call test stubs authoritative);
  `utility_timeout.resolve_config` uses `is not None` semantics against the
  DEFAULTS (the stale 30/120 literals are gone from `resolve_config` AND from
  `guarded_call`'s re-read).
- #13 — `api/stats.py` overrides `get_methods()` → `["GET", "POST"]` (the
  docstring promised GET; the framework default is POST-only → 405).
- #14 — both hardcoded `"utility"` `_emit_fallback_summary` calls in the
  engine use `spec.kind`.
- #15 — `_quota_cooldown_seconds` floor clamps to `[min(60, cap), cap]`.
- #16 — `_is_format_error` narrowed to response-parse-shaped substrings.
- #17 — `gen_budget_max_s` reads honour explicit 0 via `is not None`
  (`_gen_budget_max_s_from_cfg` helper; 3 read sites).
- #18 — turn n≤1 path: RetryAfterHours carries at least the booked
  cooldown's remaining time; Responses-5xx marks sticky (see #7).
- #19 — `_evict_warm_on_timeout` gated behind `_is_timeout_shaped(e)` on all
  three formerly-unconditional sites (turn idx-loop, turn n≤1, engine
  litellm.Timeout branch was already gated).
- #20 — the all-skipped spin path uses `_yielding_sleep` (was raw
  `asyncio.sleep` up to 300s — the one long sleep defeating the yielding
  protection).
- #21 — `models_ext.reset_responses_5xx_seen()` + called from
  `hooks.uninstall`.
- #22 — behavior UNCHANGED (pinned by design): `test_quota_exhaustion_v340`
  pins 429 = cooldown-permanent but NOT rotation-permanent; added a NOTE
  docstring on `_is_permanently_failed_model` warning new callers to use
  `_is_permanent_for_rotation`.
- #23 — `_parse_duration_seconds` rejects ISO timestamps / clock-like text.
- #24 — `hooks.get_fallback_settings` coercions degrade instead of raising;
  display default aligned to the YAML (20).
- #26 — verified already-correct: the engine break path falls through to the
  post-loop summary + `_maybe_raise_retry_after_hours` (3645ff) and the turn
  exhaustion emits summary + exhausted message before its raise.
- #27 — `_record_last_status` dead `store` param dropped (8 call sites).
- #28 — config.json dead `webui_extensions_cache_enabled` removed; README
  version header 2.7.0 → 3.4.1; `fallback-store.js getDefaults()` keys fixed
  to the real `fallback_*` names. `max_wait_s` prose verified CONSISTENT
  (help.html 180 == default_config.yaml 180; config.json 420 is a user
  value, not a doc bug).
- #30 — api/events.py snapshots once (count and payload from one read).
- #31 (first half) — context-size guard per-tick reads now go through the
  cached resolver (see #12).
- #32 — success paths skip `_save_cooldown_store` when the pop was a no-op.
- #34 — redundant per-candidate local import removed (turn path); unused
  `n`/`total_attempts` params dropped from `_maybe_raise_retry_after_hours`
  (4 call sites updated); the max_cycles `break` no longer double-fires the
  extended-retry budget before the post-loop call (ext_attempts was
  incremented twice per exhaustion).
- #35 — `_yielding_sleep` resolves the `on_long_sleep_tick` hook once per
  sleep and clamps the final tick to the actual remainder.
- #36 — `resolve_config` returns a copy (cache-hit + fresh paths) and
  `load_defaults` caches the absent-YAML miss via a sentinel flag.
- #38 — `build_fallback_wrapper` defensive-copies the caller's spec dict.
- plugin.yaml version 3.4.0 → 3.4.1.

**Deferred (deliberate, not forgotten):**

- #29 (candidate-list memoization) — risky for the rotation-index semantics;
  needs its own design pass.
- #33 (turn-path per-invocation log dedupe maps) — needs an agent-data key
  design like the cooldown store; behavior is noisy but correct.
- #31 (second half) — `monologue_start/_10_memory_recall_patches.py` still
  re-reads the framework memorize files every monologue (no version-stamp
  short-circuit); same fix shape as its siblings, separate pass.
- #37 (probe rate-limiter bypass) — needs a framework `models.py` change;
  the plugin must not modify framework files.
- #25 (`("__global__",)` store key unreachable + `id(agent)` GC aliasing) —
  latent; the aliasing risk needs an agent-id scheme decision, not a patch.
