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

const TOGGLE_DEFAULTS = {
  utility_timeout_guard: true,   // ON by default; the cascade alone is not enough for slow ollama CPU
  webui_extensions_cache: true,  // ON; the polling storm is the dominant WebUI freeze cause
  context_size_guard: false,     // OFF; opt-in because aggressive trimming can confuse the LLM
  langchain_compat: true,        // ON; the shim is a no-op on langchain v0.x
};

const TOP_LEVEL_KEYS = {
  utility_timeout_guard: "utility_timeout_guard_enabled",
  webui_extensions_cache: "webui_extensions_cache_enabled",
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
      webui_extensions_cache_enabled: TOGGLE_DEFAULTS.webui_extensions_cache,
      context_size_guard_enabled: TOGGLE_DEFAULTS.context_size_guard,
      langchain_compat_enabled: TOGGLE_DEFAULTS.langchain_compat,
    };
  },

  normalizeSettings(settings) {
    return {
      max_cycles: Math.max(1, Math.min(10, parseInt(settings?.fallback_max_cycles) || 4)),
      cycle_delay: Math.max(0, Math.min(120, parseFloat(settings?.fallback_cycle_delay) || 5.0)),
      attempt_delay: Math.max(0, Math.min(30, parseFloat(settings?.fallback_attempt_delay) || 2.0)),
      timeout_s: Math.max(30, Math.min(600, parseInt(settings?.fallback_timeout_s) || 300)),
      utility_timeout_s: Math.max(30, Math.min(600, parseInt(settings?.fallback_utility_timeout_s) || 300)),
    };
  },

  isEnabled(settings, piece) {
    return resolveToggle(settings, piece);
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
