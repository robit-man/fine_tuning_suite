(function installCallQueue(root, factory) {
  "use strict";
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  root.OmniCallQueue = api;
}(typeof globalThis !== "undefined" ? globalThis : this, function callQueueFactory() {
  "use strict";

  const DEFAULTS = Object.freeze({
    sampleRate: 48_000,
    maxSeconds: 45,
    gapMs: 120,
  });

  function createState(options = {}) {
    const config = { ...DEFAULTS, ...options };
    if (!(config.sampleRate > 0) || !(config.maxSeconds > 0) || config.gapMs < 0) {
      throw new Error("Invalid call queue configuration");
    }
    return {
      config,
      chunks: [],
      sampleCount: 0,
      activeDurationMs: 0,
      segmentCount: 0,
      truncated: false,
    };
  }

  function trimToLimit(state) {
    const limit = Math.max(1, Math.floor(state.config.sampleRate * state.config.maxSeconds));
    let overflow = state.sampleCount - limit;
    while (overflow > 0 && state.chunks.length) {
      const first = state.chunks[0];
      if (first.length <= overflow) {
        state.chunks.shift();
        state.sampleCount -= first.length;
        overflow -= first.length;
      } else {
        state.chunks[0] = first.slice(overflow);
        state.sampleCount -= overflow;
        overflow = 0;
      }
      state.truncated = true;
    }
    state.activeDurationMs = Math.min(
      state.activeDurationMs,
      state.config.maxSeconds * 1000,
    );
  }

  function enqueue(state, chunks, activeDurationMs = 0) {
    const accepted = Array.from(chunks || []).filter(chunk => (
      chunk instanceof Float32Array && chunk.length > 0
    ));
    if (!accepted.length) return stats(state);

    if (state.segmentCount > 0 && state.config.gapMs > 0) {
      const gapSamples = Math.max(
        1,
        Math.round(state.config.sampleRate * state.config.gapMs / 1000),
      );
      state.chunks.push(new Float32Array(gapSamples));
      state.sampleCount += gapSamples;
    }
    for (const chunk of accepted) {
      state.chunks.push(chunk);
      state.sampleCount += chunk.length;
    }
    state.segmentCount += 1;
    state.activeDurationMs += Math.max(0, Number(activeDurationMs) || 0);
    trimToLimit(state);
    return stats(state);
  }

  function stats(state) {
    return {
      segmentCount: state.segmentCount,
      sampleCount: state.sampleCount,
      activeDurationMs: state.activeDurationMs,
      capturedDurationMs: state.sampleCount / state.config.sampleRate * 1000,
      truncated: state.truncated,
    };
  }

  function hasPending(state) {
    return state.sampleCount > 0 && state.segmentCount > 0;
  }

  function clear(state) {
    state.chunks = [];
    state.sampleCount = 0;
    state.activeDurationMs = 0;
    state.segmentCount = 0;
    state.truncated = false;
  }

  function take(state) {
    const result = { chunks: state.chunks, ...stats(state) };
    clear(state);
    return result;
  }

  return { DEFAULTS, createState, enqueue, stats, hasPending, clear, take };
}));
