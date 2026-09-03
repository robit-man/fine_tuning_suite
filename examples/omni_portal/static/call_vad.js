(function installCallVad(root, factory) {
  "use strict";
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  root.OmniCallVad = api;
}(typeof globalThis !== "undefined" ? globalThis : this, function callVadFactory() {
  "use strict";

  const DEFAULTS = Object.freeze({
    calibrationMs: 800,
    calibrationEscapeThreshold: 0.030,
    calibrationEscapeMultiplier: 3.25,
    startThreshold: 0.014,
    releaseThreshold: 0.008,
    noiseMultiplier: 2.75,
    releaseMultiplier: 1.6,
    startConfirmMs: 200,
    silenceMs: 760,
    minActiveMs: 400,
    preRollFrames: 10,
    initialNoiseFloor: 0.003,
  });

  function createState(startedAt = 0, overrides = {}) {
    const config = { ...DEFAULTS, ...overrides };
    return {
      config,
      readyAt: startedAt + config.calibrationMs,
      noiseFloor: config.initialNoiseFloor,
      preRoll: [],
      candidateFrames: [],
      candidateStartedAt: null,
      frames: [],
      speaking: false,
      activeMs: 0,
      lastActiveAt: 0,
    };
  }

  function resetState(state, now = 0, { calibrate = false } = {}) {
    state.readyAt = now + (calibrate ? state.config.calibrationMs : 0);
    state.preRoll = [];
    state.candidateFrames = [];
    state.candidateStartedAt = null;
    state.frames = [];
    state.speaking = false;
    state.activeMs = 0;
    state.lastActiveAt = 0;
  }

  function updateNoiseFloor(state, level, weight = 0.035) {
    state.noiseFloor = Math.max(
      0.0005,
      state.noiseFloor * (1 - weight) + level * weight,
    );
  }

  function processFrame(state, { level, samples, now, frameMs }) {
    const config = state.config;
    if (now < state.readyAt) {
      const calibrationEscapeThreshold = Math.max(
        config.calibrationEscapeThreshold,
        state.noiseFloor * config.calibrationEscapeMultiplier,
      );
      if (level >= calibrationEscapeThreshold) {
        state.readyAt = now;
      } else {
        updateNoiseFloor(state, level, 0.12);
        state.preRoll = [];
        return {
          event: "calibrating",
          active: false,
          calibrationEscapeThreshold,
        };
      }
    }

    const startThreshold = Math.max(
      config.startThreshold,
      state.noiseFloor * config.noiseMultiplier,
    );
    const releaseThreshold = Math.max(
      config.releaseThreshold,
      state.noiseFloor * config.releaseMultiplier,
    );

    if (!state.speaking) {
      state.preRoll.push(samples);
      if (state.preRoll.length > config.preRollFrames) state.preRoll.shift();
      if (level < startThreshold) {
        updateNoiseFloor(state, level);
        state.candidateFrames = [];
        state.candidateStartedAt = null;
        return { event: "idle", active: false, startThreshold };
      }
      if (state.candidateStartedAt === null) {
        state.candidateStartedAt = now;
        state.candidateFrames = state.preRoll.splice(0);
      } else {
        state.candidateFrames.push(samples);
      }
      if (now - state.candidateStartedAt < config.startConfirmMs) {
        return { event: "candidate", active: false, startThreshold };
      }
      state.speaking = true;
      state.frames = state.candidateFrames.splice(0);
      state.activeMs = Math.max(frameMs, now - state.candidateStartedAt + frameMs);
      state.lastActiveAt = now;
      return { event: "start", active: true, startThreshold, releaseThreshold };
    }

    state.frames.push(samples);
    if (level >= releaseThreshold) {
      state.lastActiveAt = now;
      state.activeMs += frameMs;
    }
    if (now - state.lastActiveAt < config.silenceMs) {
      return { event: "active", active: true, releaseThreshold };
    }

    const utterance = {
      chunks: state.frames.splice(0),
      activeDurationMs: state.activeMs,
    };
    const accepted = utterance.activeDurationMs >= config.minActiveMs;
    resetState(state, now);
    return {
      event: accepted ? "utterance" : "rejected",
      active: false,
      utterance: accepted ? utterance : null,
    };
  }

  return { DEFAULTS, createState, processFrame, resetState };
}));
