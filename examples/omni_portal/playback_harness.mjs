#!/usr/bin/env node

import assert from "node:assert/strict";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { supersedeBefore, canStart, retentionState } = require("./static/call_playback.js");

const oldTurn = { sequence: 0, discardReply: false };
const call = {
  turns: new Set([oldTurn]),
  playbackTurn: null,
  nextSequence: 1,
  vadActive: false,
};

assert.equal(canStart(call, oldTurn), true);

// A new utterance has begun before the old turn reaches audio_start.
const bargeIn = supersedeBefore(call, call.nextSequence);
assert.equal(bargeIn.turns.length, 1);
assert.equal(oldTurn.discardReply, true);
assert.equal(canStart(call, oldTurn), false);

const playingTurn = { sequence: 1, discardReply: false };
call.turns.add(playingTurn);
call.nextSequence = 2;
call.playbackTurn = playingTurn;
assert.equal(canStart(call, playingTurn), true);

// Submitting a newer turn revokes ownership from audio that is already playing.
const newerTurn = { sequence: 2, discardReply: false };
const replaced = supersedeBefore(call, newerTurn.sequence);
call.turns.add(newerTurn);
call.nextSequence = 3;
assert.equal(replaced.playbackInterrupted, true);
assert.equal(call.playbackTurn, null);
assert.equal(playingTurn.discardReply, true);
assert.equal(canStart(call, playingTurn), false);
assert.equal(canStart(call, newerTurn), true);

// Active speech always wins over a reply, including the newest pending reply.
call.vadActive = true;
assert.equal(canStart(call, newerTurn), false);

assert.deepEqual(retentionState({ content: "already streamed", languageSettled: true }), {
  preserve: true,
  interrupted: false,
});
assert.deepEqual(retentionState({ content: "partial", languageSettled: false }), {
  preserve: true,
  interrupted: true,
});
assert.deepEqual(retentionState({ content: "", thinking: "", toolTrace: [] }), {
  preserve: false,
  interrupted: false,
});

console.log(JSON.stringify({
  status: "passed",
  stale_pending_audio_suppressed: true,
  active_playback_interrupted: true,
  newest_turn_owns_playback: true,
  settled_transcript_retained: true,
  partial_transcript_retained_and_marked: true,
}, null, 2));
