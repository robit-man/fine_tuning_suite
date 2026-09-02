#!/usr/bin/env node

import assert from "node:assert/strict";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const queue = require("./static/call_queue.js");

const pending = queue.createState({ sampleRate: 1000, maxSeconds: 2, gapMs: 100 });
queue.enqueue(pending, [new Float32Array(200).fill(0.1)], 180);
queue.enqueue(pending, [new Float32Array(300).fill(0.2)], 260);
queue.enqueue(pending, [new Float32Array(400).fill(0.3)], 350);

assert.equal(queue.stats(pending).segmentCount, 3);
assert.equal(queue.stats(pending).sampleCount, 1100);
const consolidated = queue.take(pending);
assert.equal(consolidated.segmentCount, 3);
assert.equal(consolidated.activeDurationMs, 790);
assert.equal(consolidated.chunks.length, 5);
assert.equal(queue.hasPending(pending), false);

const bounded = queue.createState({ sampleRate: 10, maxSeconds: 1, gapMs: 0 });
queue.enqueue(bounded, [new Float32Array(8).fill(1)], 800);
queue.enqueue(bounded, [new Float32Array(8).fill(2)], 800);
const latestWindow = queue.take(bounded);
assert.equal(latestWindow.sampleCount, 10);
assert.equal(latestWindow.truncated, true);
assert.equal(latestWindow.chunks.at(-1).at(-1), 2);

console.log(JSON.stringify({
  status: "passed",
  consolidated_segments: consolidated.segmentCount,
  consolidated_samples: consolidated.sampleCount,
  bounded_samples: latestWindow.sampleCount,
  single_flight_contract: "one active inference plus one bounded pending turn",
}, null, 2));
