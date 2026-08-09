import { createStore } from "/js/AlpineStore.js";

// The WebUI binds to flat top-level booleans on
// ``context.settings`` (e.g. ``utility_timeout_guard_enabled``).
// The runtime helpers in ``usr/plugins/_model_fallback/helpers/toggles.py``
// read the top-level key first and fall back to the nested
// ``<piece>.enabled`` for back-compat with hand-edited configs.
//
// The resolution rules below must match ``toggles.py`` exactly:
// when a top-level key is present its value wins, even if False.
// When the top-level key is absent we look at the nested section,
// and when neither is present we use the per-piece default.
//
// v2.6.5 toggle-save fix: the config panel binds each switch with
// ``x-model`` ONLY (the previous ``x-model`` + ``:checked`` pair was an
// Alpine anti-pattern — the one-way ``:checked`` fought the two-way
// ``x-model``, so a click never committed to ``context.settings`` and
// Save posted stale values; every switch reopened OFF). Because
// ``x-model`` reads the raw top-level key, we must guarantee that key
// EXISTS on ``context.settings`` with the resolved boolean — otherwise a
// default-ON feature whose key is absent from the loaded config would
// read ``undefined`` and display OFF. ``backfillToggles`` does that:
// it fills only MISSING keys (explicit True/False is never overwritten),
// mirroring ``toggles.resolve_toggle`` so the displayed state matches
// the runtime state.

const TOGGLE_DEFAULTS = {
  utility_timeout_guard: true,   // ON by default; the cascade alone is not enough for slow ollama CPU
  context_size_guard: false,     // OFF; opt-in because aggressive trimming can confuse the LLM
  langchain_compat: true,        // ON; the shim is a no-op on langchain v0.x
};

const TOP_LEVEL_KEYS = {
  utility_timeout_guard: "utility_timeout_guard_enabled",
  context_size_guard: "context_size_guard_enabled",
  langchain_compat: "langchain_compat_enabled",
};

function resolveToggle(settings, piece) {
  const topKey = TOP_LEVEL_KEYS[piece];
  if (settings && topKey in settings) {
    return !!settings[topKey];
  }
  const nested = settings && settings[piece];
  if (nested && typeof nested === "object" && "enabled" in nested) {
    return !!nested.enabled;
  }
  return TOGGLE_DEFAULTS[piece];
}

export const store = createStore("modelFallback", {
  _loaded: false,
  showHelp: false,

  init() {},

  async ensureLoaded() {
    if (this._loaded) return;
    this._loaded = true;
  },

  async openHelp() {
    this.showHelp = true;
    await window.openModal?.('/plugins/_model_fallback/webui/help.html');
  },

  getDefaults() {
    return {
      max_cycles: 4,
      cycle_delay: 5.0,
      attempt_delay: 2.0,
      timeout_s: 300,
      utility_timeout_s: 300,
      // Per-feature toggles (v2.5). The nested defaults below
      // are the ones the runtime reads when the top-level keys
      // are absent. They keep the existing hand-edited config
      // contract intact.
      //
      // housekeeping_enabled and second_pulse_path_enabled are
      // no longer in this list: the corresponding extension and
      // helper were removed in v2.5. Settings that still carry
      // those keys are accepted on save (the toggle resolver
      // honours them) but the runtime ignores them.
      utility_timeout_guard_enabled: TOGGLE_DEFAULTS.utility_timeout_guard,
      context_size_guard_enabled: TOGGLE_DEFAULTS.context_size_guard,
      langchain_compat_enabled: TOGGLE_DEFAULTS.langchain_compat,
    };
  },

  // v2.6.5: ensure every UI toggle key exists on ``settings`` as a real
  // boolean before the panel binds, so ``x-model`` reads the effective
  // on/off state instead of ``undefined``. Only MISSING keys are filled
  // — an explicit False (user turned a feature OFF and saved) is kept.
  // Called from config.html x-init after the framework has loaded
  // ``context.settings``.
  backfillToggles(settings) {
    if (!settings || typeof settings !== "object") return;
    for (const piece of Object.keys(TOGGLE_DEFAULTS)) {
      const topKey = TOP_LEVEL_KEYS[piece];
      if (!(topKey in settings)) {
        settings[topKey] = resolveToggle(settings, piece);
      }
    }
  },

  // v2.6.5: number inputs bind ``x-model`` to the RAW key (e.g.
  // ``fallback_timeout_s``). The previous panel ALSO bound ``:value`` to
  // a normalized/clamped snapshot of the field so the box DISPLAYED the
  // clamped value while ``x-model`` held the raw one — Save persisted the
  // raw value (e.g. typing 5 showed 30 but saved 5). Now the box shows what
  // ``x-model`` holds (the raw value, as typed) and ``@change`` clamps the
  // same key in place on blur/Enter, so the value that gets saved is the
  // clamped one. Bounds: timeout_s / utility_timeout_s 30–600 (int),
  // attempt_delay 0–30, max_cycles 1–10 (int), cycle_delay 0–120.
  clampField(settings, key, min, max, fallback, asInt = false) {
    if (!settings || typeof settings !== "object") return;
    let v = parseFloat(settings[key]);
    if (!isFinite(v)) v = fallback;
    v = Math.max(min, Math.min(max, v));
    if (asInt) v = Math.round(v);
    settings[key] = v;
  },

  // The save flow is the framework's default (POST /plugins with
  // action: save_config); we don't need a custom save() because
  // the framework already persists ``context.settings`` on Save.
  // This helper is here for future plugin-local actions
  // (e.g. "Reset to defaults") that may need it.
  applyDefaults(settings) {
    const d = this.getDefaults();
    return Object.assign({}, d, settings || {});
  },
});
