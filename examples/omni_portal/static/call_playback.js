(function installCallPlayback(root, factory) {
  "use strict";
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  root.OmniCallPlayback = api;
}(typeof globalThis !== "undefined" ? globalThis : this, function callPlaybackFactory() {
  "use strict";

  function supersedeBefore(call, sequence) {
    const turns = [];
    for (const turn of call.turns) {
      if (turn.sequence >= sequence) continue;
      turn.discardReply = true;
      turns.push(turn);
    }
    const playbackInterrupted = Boolean(
      call.playbackTurn && call.playbackTurn.sequence < sequence,
    );
    if (playbackInterrupted) call.playbackTurn = null;
    return { turns, playbackInterrupted };
  }

  function canStart(call, turn) {
    return Boolean(
      !turn.discardReply
      && !call.vadActive
      && turn.sequence === call.nextSequence - 1,
    );
  }

  return { supersedeBefore, canStart };
}));
