#!/usr/bin/env node

import assert from "node:assert/strict";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { createState, processFrame } = require("./static/call_vad.js");
const frameMs = 20;

function run(levels) {
  const state = createState(0);
  const events = [];
  let now = 0;
  for (const level of levels) {
    events.push(processFrame(state, {
      level,
      samples: new Float32Array([level]),
      now,
      frameMs,
    }));
    now += frameMs;
  }
  return events;
}

const frames = (milliseconds, level) => Array(Math.ceil(milliseconds / frameMs)).fill(level);
const calibratedNoise = frames(1000, 0.006);
const trailingSilence = frames(900, 0.006);
const remoteRequests = levels => run(levels).filter(event => event.event === "utterance").length;

assert.equal(remoteRequests([...calibratedNoise, ...frames(2500, 0.007)]), 0);
assert.equal(remoteRequests([...calibratedNoise, ...frames(40, 0.18), ...trailingSilence]), 0);
assert.equal(remoteRequests([...frames(1000, 0.025), ...frames(2500, 0.026)]), 0);
assert.equal(remoteRequests([...calibratedNoise, ...frames(720, 0.085), ...trailingSilence]), 1);
assert.equal(remoteRequests([...calibratedNoise, ...frames(680, 0.065), ...trailingSilence]), 1);
assert.equal(remoteRequests([...calibratedNoise, ...frames(420, 0.018), ...trailingSilence]), 1);
assert.equal(remoteRequests([
  ...calibratedNoise,
  ...frames(720, 0.085),
  ...trailingSilence,
  ...frames(680, 0.065),
  ...trailingSilence,
]), 2);

console.log(JSON.stringify({
  status: "passed",
  remote_requests: {
    calibrated_quiet: 0,
    transient_click: 0,
    elevated_room_noise: 0,
    sustained_speech: 1,
    sustained_alarm: 1,
    quiet_speech: 1,
    continued_speech_segments: 2,
  },
}, null, 2));
