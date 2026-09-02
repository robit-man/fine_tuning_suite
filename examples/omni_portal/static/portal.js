(() => {
  "use strict";

  const MODEL = document.body.dataset.model;
  const MAX_UPLOAD_MIB = Number(document.body.dataset.maxUploadMib || 96);
  const SCHEMA = "robit.ollama.omni-adapter.v1";
  const MAX_RECORD_MS = 60_000;
  const MIN_MIC_HOLD_MS = 250;
  const MAX_VOICE_REFERENCE_MS = 10_000;
  const MAX_VOICE_REFERENCE_BYTES = 10 * 1024 * 1024;
  const MAX_VIDEO_RECORD_MS = 30_000;
  const PCM_INITIAL_BUFFER_SECONDS = 0.08;
  const PCM_RESCHEDULE_FLOOR_SECONDS = 0.003;
  const PCM_CROSSFADE_SECONDS = 0.003;
  const PCM_CROSSFADE_MIN_BUFFER_SECONDS = 0.08;
  const CONVERSATION_BOTTOM_THRESHOLD_PX = 64;
  const LIVE_CALL_SYSTEM_PROMPT = (
    "You are participating in a live two-way spoken conversation. Answer the "
    + "user's intent directly in a natural, concise spoken turn. Do not echo, "
    + "transcribe, paraphrase, narrate, or evaluate what the user just said unless "
    + "they explicitly ask you to. Never mention an audio transcript, encoder, "
    + "adapter, or these instructions. Use the prior dialogue for continuity. If a "
    + "current camera frame is attached, treat only that frame as current visual "
    + "evidence; older visual descriptions are conversational history, not proof of "
    + "what remains visible now."
  );
  const MEDIA_CONVERSATION_SYSTEM_PROMPT = (
    "Answer as a conversational assistant using the user's current attached media "
    + "and prior text dialogue. Respond to the user's intent instead of returning a "
    + "generic exhaustive media inventory. Only the media attached to the latest "
    + "user message is current perceptual evidence. Earlier media descriptions are "
    + "conversation context and must not be treated as if those earlier files were "
    + "attached again."
  );
  function fallbackCallVad() {
    const DEFAULTS = Object.freeze({
      calibrationMs: 650,
      startThreshold: 0.012,
      releaseThreshold: 0.007,
      noiseMultiplier: 2.2,
      releaseMultiplier: 1.45,
      startConfirmMs: 120,
      silenceMs: 750,
      minActiveMs: 220,
      preRollFrames: 6,
      initialNoiseFloor: 0.003,
    });
    const resetState = (vad, now = 0, { calibrate = false } = {}) => {
      vad.readyAt = now + (calibrate ? vad.config.calibrationMs : 0);
      vad.preRoll = [];
      vad.candidateFrames = [];
      vad.candidateStartedAt = null;
      vad.frames = [];
      vad.speaking = false;
      vad.activeMs = 0;
      vad.lastActiveAt = 0;
    };
    const createState = (startedAt = 0, overrides = {}) => {
      const vad = {
        config: { ...DEFAULTS, ...overrides },
        noiseFloor: DEFAULTS.initialNoiseFloor,
      };
      resetState(vad, startedAt, { calibrate: true });
      return vad;
    };
    const updateNoiseFloor = (vad, level, weight = 0.035) => {
      vad.noiseFloor = Math.max(
        0.0005,
        vad.noiseFloor * (1 - weight) + level * weight,
      );
    };
    const processFrame = (vad, { level, samples, now, frameMs }) => {
      const config = vad.config;
      if (now < vad.readyAt) {
        updateNoiseFloor(vad, level, 0.12);
        vad.preRoll = [];
        return { event: "calibrating", active: false };
      }
      const startThreshold = Math.max(
        config.startThreshold,
        vad.noiseFloor * config.noiseMultiplier,
      );
      const releaseThreshold = Math.max(
        config.releaseThreshold,
        vad.noiseFloor * config.releaseMultiplier,
      );
      if (!vad.speaking) {
        vad.preRoll.push(samples);
        if (vad.preRoll.length > config.preRollFrames) vad.preRoll.shift();
        if (level < startThreshold) {
          updateNoiseFloor(vad, level);
          vad.candidateFrames = [];
          vad.candidateStartedAt = null;
          return { event: "idle", active: false, startThreshold };
        }
        if (vad.candidateStartedAt === null) {
          vad.candidateStartedAt = now;
          vad.candidateFrames = vad.preRoll.splice(0);
        } else {
          vad.candidateFrames.push(samples);
        }
        if (now - vad.candidateStartedAt < config.startConfirmMs) {
          return { event: "candidate", active: false, startThreshold };
        }
        vad.speaking = true;
        vad.frames = vad.candidateFrames.splice(0);
        vad.activeMs = Math.max(frameMs, now - vad.candidateStartedAt + frameMs);
        vad.lastActiveAt = now;
        return { event: "start", active: true, startThreshold, releaseThreshold };
      }
      vad.frames.push(samples);
      if (level >= releaseThreshold) {
        vad.lastActiveAt = now;
        vad.activeMs += frameMs;
      }
      if (now - vad.lastActiveAt < config.silenceMs) {
        return { event: "active", active: true, releaseThreshold };
      }
      const utterance = {
        chunks: vad.frames.splice(0),
        activeDurationMs: vad.activeMs,
      };
      const accepted = utterance.activeDurationMs >= config.minActiveMs;
      resetState(vad, now);
      return {
        event: accepted ? "utterance" : "rejected",
        active: false,
        utterance: accepted ? utterance : null,
      };
    };
    return { DEFAULTS, createState, processFrame, resetState };
  }
  const callVad = window.OmniCallVad || fallbackCallVad();
  const callPlayback = window.OmniCallPlayback;
  if (!callPlayback) throw new Error("Call playback ownership helper failed to load");
  const BARGE_VAD_OPTIONS = {
    calibrationMs: 0,
    startThreshold: 0.05,
    releaseThreshold: 0.025,
    noiseMultiplier: 3.5,
    releaseMultiplier: 1.8,
    startConfirmMs: 360,
    minActiveMs: 400,
  };
  const LIMITS = {
    audio: 30 * 1024 * 1024,
    image: 18 * 1024 * 1024,
    video: Math.min(68, Math.max(8, MAX_UPLOAD_MIB - 24)) * 1024 * 1024,
    document: 24 * 1024 * 1024,
  };

  const elements = {
    headerStatus: document.getElementById("header-status"),
    statusText: document.getElementById("status-text"),
    activeUsers: document.getElementById("active-users"),
    activeUserCount: document.getElementById("active-user-count"),
    conversation: document.getElementById("conversation"),
    scrollLatest: document.getElementById("scroll-latest-button"),
    template: document.getElementById("message-template"),
    prompt: document.getElementById("prompt"),
    attachments: document.getElementById("attachments"),
    mediaInput: document.getElementById("media-input"),
    micButton: document.getElementById("mic-button"),
    cameraButton: document.getElementById("camera-button"),
    cameraPreview: document.getElementById("camera-preview"),
    cameraVideo: document.getElementById("camera-video"),
    voiceButton: document.getElementById("voice-button"),
    voiceDialog: document.getElementById("voice-dialog"),
    voiceClose: document.getElementById("voice-close"),
    voiceDone: document.getElementById("voice-done"),
    voiceReset: document.getElementById("voice-reset"),
    voiceCloneEnabled: document.getElementById("voice-clone-enabled"),
    voiceCloneToggle: document.getElementById("voice-clone-toggle"),
    voiceModeRow: document.getElementById("voice-mode-row"),
    voicePresetToggle: document.getElementById("voice-preset-toggle"),
    voicePresetCurrent: document.getElementById("voice-preset-current"),
    voicePresetOptions: document.getElementById("voice-preset-options"),
    voiceReferenceControls: document.getElementById("voice-reference-controls"),
    voiceReferenceInput: document.getElementById("voice-reference-input"),
    voiceReferenceAudio: document.getElementById("voice-reference-audio"),
    voiceReferenceStatus: document.getElementById("voice-reference-status"),
    voiceReferenceClear: document.getElementById("voice-reference-clear"),
    voiceRecord: document.getElementById("voice-record"),
    voiceLanguage: document.getElementById("voice-language"),
    voiceTemperature: document.getElementById("voice-temperature"),
    voiceTemperatureValue: document.getElementById("voice-temperature-value"),
    voiceTopP: document.getElementById("voice-top-p"),
    voiceTopPValue: document.getElementById("voice-top-p-value"),
    voiceTopK: document.getElementById("voice-top-k"),
    voiceSeed: document.getElementById("voice-seed"),
    voiceMaxFrames: document.getElementById("voice-max-frames"),
    speak: document.getElementById("speak-toggle"),
    think: document.getElementById("think-toggle"),
    send: document.getElementById("send-button"),
    composer: document.querySelector(".composer"),
    composerStatus: document.getElementById("composer-status"),
    clear: document.getElementById("clear-button"),
    callButton: document.getElementById("call-button"),
    waveform: document.getElementById("waveform"),
    waveformCanvas: document.getElementById("waveform-canvas"),
    recordingTime: document.getElementById("recording-time"),
  };

  const state = {
    token: "",
    attachments: [],
    history: [],
    messages: [],
    recording: null,
    holdingMic: false,
    micHoldStartedAt: 0,
    discardMicRecording: false,
    recordTimer: null,
    recordClock: null,
    playbackContext: null,
    playbackSource: null,
    playbackElement: null,
    playbackEpoch: 0,
    streamController: null,
    requestController: null,
    requestSequence: 0,
    composerHintTimer: null,
    scrollFrame: null,
    autoFollowConversation: true,
    lastConversationScrollTop: 0,
    conversationScrollGesture: false,
    conversationScrollGestureTimer: null,
    cacheScope: `${location.origin}:${MODEL}:${document.body.dataset.sessionScope || ""}`,
    cacheReady: false,
    cacheSuppress: false,
    cacheDeleted: false,
    cacheTimer: null,
    cacheWrite: Promise.resolve(),
    cacheErrorReported: false,
    call: null,
    camera: null,
    voice: {
      initialized: false,
      serverReference: false,
      reference: null,
      referenceUrl: null,
      presets: [],
      selectedPreset: "",
      presetMenuOpen: false,
      recording: null,
      recordTimer: null,
      defaults: {
        language: "en", temperature: 0.7, topK: 40, topP: 0.9, seed: 42, maxFrames: 512,
      },
    },
  };

  function accessToken() {
    const fragment = new URLSearchParams(location.hash.replace(/^#/, ""));
    const supplied = fragment.get("access");
    if (supplied) sessionStorage.setItem("omni_access", supplied);
    return supplied || sessionStorage.getItem("omni_access") || "";
  }

  function authHeaders(extra = {}) {
    return { Authorization: `Bearer ${state.token}`, ...extra };
  }

  function reportDiagnostic(event, fields = {}) {
    if (!state.token) return;
    fetch("/api/diagnostics", {
      method: "POST",
      headers: authHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify({ event, ...fields }),
      keepalive: true,
    }).catch(() => {});
  }

  function clearSessionDiagnostics() {
    if (!state.token) return;
    fetch("/api/diagnostics", {
      method: "DELETE",
      headers: authHeaders(),
      keepalive: true,
    }).catch(() => {});
  }

  function mediaCacheValue(item, { pending = false } = {}) {
    const value = {
      kind: String(item.kind || ""),
      name: String(item.name || ""),
      mime: String(item.mime || "application/octet-stream"),
      bytes: Number(item.bytes || 0),
      source: String(item.source || "upload"),
    };
    if (pending || value.kind === "video" || value.kind === "image") {
      value.data = String(item.data || "");
    }
    return value;
  }

  function hydrateMediaValue(item) {
    const value = mediaCacheValue(item, { pending: true });
    value.previewUrl = value.kind === "video" && value.data
      ? URL.createObjectURL(base64ToBlob(value.data, value.mime))
      : null;
    return value;
  }

  function browserSessionSnapshot() {
    return {
      schema: "robit.omni.browser-session.v1",
      savedAt: Date.now(),
      history: state.history
        .filter(item => item && ["user", "assistant"].includes(item.role))
        .map(item => ({ role: item.role, content: String(item.content || "") })),
      messages: state.messages
        .filter(record => record.node.isConnected)
        .map(record => ({
          role: record.role,
          content: String(record.content || ""),
          thinking: String(record.thinking || ""),
          audioObservation: String(record.audioObservation || ""),
          soundOnly: Boolean(record.soundOnly),
          generationMetrics: record.generationMetrics ? { ...record.generationMetrics } : null,
          audio: record.audio && record.audio.data ? { ...record.audio } : null,
          media: (record.media || []).map(item => mediaCacheValue(item)),
          error: Boolean(record.error),
        })),
      attachments: state.attachments.map(item => mediaCacheValue(item, { pending: true })),
      draft: elements.prompt.value,
    };
  }

  function enqueueCacheOperation(operation) {
    state.cacheWrite = state.cacheWrite
      .catch(() => {})
      .then(operation)
      .catch(error => {
        if (state.cacheErrorReported) return;
        state.cacheErrorReported = true;
        console.warn("Browser session cache unavailable", error);
      });
    return state.cacheWrite;
  }

  function scheduleBrowserSessionSave(delay = 350) {
    if (!state.cacheReady || state.cacheSuppress || !window.OmniSessionCache) return;
    state.cacheDeleted = false;
    if (state.cacheTimer !== null) clearTimeout(state.cacheTimer);
    state.cacheTimer = setTimeout(() => {
      state.cacheTimer = null;
      const snapshot = browserSessionSnapshot();
      enqueueCacheOperation(() => window.OmniSessionCache.save(state.cacheScope, snapshot));
    }, delay);
  }

  function persistBrowserSessionOnLeave() {
    if (
      !state.cacheReady || state.cacheSuppress || state.cacheDeleted
      || !window.OmniSessionCache
    ) return;
    if (state.cacheTimer !== null) {
      clearTimeout(state.cacheTimer);
      state.cacheTimer = null;
    }
    const snapshot = browserSessionSnapshot();
    enqueueCacheOperation(async () => {
      await window.OmniSessionCache.save(state.cacheScope, snapshot);
      await window.OmniSessionCache.markLeft(state.cacheScope);
    });
  }

  function clearBrowserSessionCache() {
    if (state.cacheTimer !== null) {
      clearTimeout(state.cacheTimer);
      state.cacheTimer = null;
    }
    if (!window.OmniSessionCache) return Promise.resolve();
    return enqueueCacheOperation(() => window.OmniSessionCache.clear(state.cacheScope));
  }

  async function restoreBrowserSession() {
    if (!window.OmniSessionCache || !document.body.dataset.sessionScope) {
      state.cacheReady = true;
      return;
    }
    let snapshot = null;
    try {
      snapshot = await window.OmniSessionCache.load(state.cacheScope);
    } catch (error) {
      console.warn("Could not restore browser session", error);
    }
    state.cacheSuppress = true;
    if (snapshot && snapshot.schema === "robit.omni.browser-session.v1") {
      state.cacheDeleted = false;
      state.history = Array.isArray(snapshot.history)
        ? snapshot.history
          .filter(item => item && ["user", "assistant"].includes(item.role))
          .map(item => ({ role: item.role, content: String(item.content || "") }))
        : [];
      for (const item of Array.isArray(snapshot.messages) ? snapshot.messages : []) {
        if (!item || !["user", "assistant"].includes(item.role)) continue;
        addMessage({
          role: item.role,
          content: String(item.content || ""),
          thinking: String(item.thinking || ""),
          audioObservation: String(item.audioObservation || ""),
          soundOnly: Boolean(item.soundOnly),
          generationMetrics: item.generationMetrics || null,
          audio: item.audio && item.audio.data ? item.audio : null,
          media: (Array.isArray(item.media) ? item.media : []).map(hydrateMediaValue),
          error: Boolean(item.error),
          autoplayAudio: false,
        });
      }
      state.attachments = (Array.isArray(snapshot.attachments) ? snapshot.attachments : [])
        .filter(item => item && item.data)
        .map(hydrateMediaValue);
      renderAttachments();
      elements.prompt.value = String(snapshot.draft || "").slice(0, 12_000);
      resizePrompt();
      setComposerStatus("Session restored");
    }
    state.cacheSuppress = false;
    state.cacheReady = true;
    await window.OmniSessionCache.touch(state.cacheScope).catch(() => false);
    scrollConversationToBottom({ smooth: false });
  }

  function setComposerStatus(text, error = false) {
    elements.composerStatus.textContent = text;
    elements.composerStatus.classList.toggle("error", error);
  }

  function transientComposerStatus(text, durationMs = 1400) {
    clearTimeout(state.composerHintTimer);
    setComposerStatus(text);
    state.composerHintTimer = setTimeout(() => {
      state.composerHintTimer = null;
      if (elements.composerStatus.textContent === text) setComposerStatus("");
    }, durationMs);
  }

  function conversationIsAtBottom() {
    return (
      elements.conversation.scrollHeight
      - elements.conversation.clientHeight
      - elements.conversation.scrollTop
    ) <= CONVERSATION_BOTTOM_THRESHOLD_PX;
  }

  function updateScrollLatestButton() {
    const hasOverflow = (
      elements.conversation.scrollHeight
      > elements.conversation.clientHeight + CONVERSATION_BOTTOM_THRESHOLD_PX
    );
    if (conversationIsAtBottom()) state.autoFollowConversation = true;
    elements.scrollLatest.hidden = state.autoFollowConversation || !hasOverflow;
  }

  function handleConversationScroll() {
    const currentTop = elements.conversation.scrollTop;
    const movedUp = currentTop < state.lastConversationScrollTop - 1;
    const atBottom = conversationIsAtBottom();
    if (atBottom) state.autoFollowConversation = true;
    else if (state.conversationScrollGesture && movedUp) state.autoFollowConversation = false;
    state.lastConversationScrollTop = currentTop;
    updateScrollLatestButton();
  }

  function beginConversationScrollGesture() {
    if (state.conversationScrollGestureTimer !== null) {
      clearTimeout(state.conversationScrollGestureTimer);
      state.conversationScrollGestureTimer = null;
    }
    state.conversationScrollGesture = true;
  }

  function endConversationScrollGesture() {
    if (state.conversationScrollGestureTimer !== null) {
      clearTimeout(state.conversationScrollGestureTimer);
    }
    state.conversationScrollGestureTimer = setTimeout(() => {
      state.conversationScrollGesture = false;
      state.conversationScrollGestureTimer = null;
    }, 500);
  }

  function scrollConversationToBottom({ smooth = true, force = false } = {}) {
    if (force) {
      state.autoFollowConversation = true;
      state.conversationScrollGesture = false;
    }
    if (!state.autoFollowConversation && conversationIsAtBottom()) {
      state.autoFollowConversation = true;
    }
    if (!state.autoFollowConversation) {
      updateScrollLatestButton();
      return;
    }
    if (state.scrollFrame !== null) cancelAnimationFrame(state.scrollFrame);
    if (!smooth) {
      // Streaming deltas must pin immediately. A CSS/smooth animation restarted
      // on every token can remain permanently behind the newest content.
      elements.conversation.scrollTop = elements.conversation.scrollHeight;
      state.lastConversationScrollTop = elements.conversation.scrollTop;
    }
    state.scrollFrame = requestAnimationFrame(() => {
      state.scrollFrame = null;
      if (smooth) {
        elements.conversation.scrollTo({
          top: elements.conversation.scrollHeight,
          behavior: "smooth",
        });
      } else {
        // Run once more after Markdown, media, or audio controls finish layout.
        elements.conversation.scrollTop = elements.conversation.scrollHeight;
        state.lastConversationScrollTop = elements.conversation.scrollTop;
      }
      updateScrollLatestButton();
    });
  }

  function resumeConversationAutoFollow() {
    state.autoFollowConversation = true;
    scrollConversationToBottom({ smooth: false, force: true });
  }

  const layoutResizeObserver = typeof window.ResizeObserver === "function"
    ? new window.ResizeObserver(() => {
      scrollConversationToBottom({ smooth: false });
      updateScrollLatestButton();
    })
    : null;
  if (layoutResizeObserver) {
    layoutResizeObserver.observe(elements.conversation);
    layoutResizeObserver.observe(elements.composer);
  }

  function revealMessage(record) {
    const wasHidden = record.node.hidden;
    record.node.hidden = false;
    if (wasHidden) scrollConversationToBottom();
  }

  function setVadActive(call, active) {
    if (!call) return;
    call.vadActive = Boolean(active);
    if (state.call === call) {
      elements.waveform.classList.toggle("vad-active", call.vadActive);
    }
  }

  function appendInlineMarkdown(parent, text) {
    const pattern = /(\*\*[^*\n]+\*\*|`[^`\n]+`|\*[^*\n]+\*|\[[^\]\n]+\]\([^)\s]+\))/g;
    let offset = 0;
    for (const match of text.matchAll(pattern)) {
      if (match.index > offset) parent.appendChild(document.createTextNode(text.slice(offset, match.index)));
      const token = match[0];
      let node;
      if (token.startsWith("**")) {
        node = document.createElement("strong");
        node.textContent = token.slice(2, -2);
      } else if (token.startsWith("`")) {
        node = document.createElement("code");
        node.textContent = token.slice(1, -1);
      } else if (token.startsWith("*")) {
        node = document.createElement("em");
        node.textContent = token.slice(1, -1);
      } else {
        const split = token.lastIndexOf("](");
        const label = token.slice(1, split);
        const href = token.slice(split + 2, -1);
        try {
          const url = new URL(href, location.origin);
          if (!['http:', 'https:'].includes(url.protocol)) throw new Error("unsafe link");
          node = document.createElement("a");
          node.href = url.href;
          node.target = "_blank";
          node.rel = "noopener noreferrer";
          node.textContent = label;
        } catch (_error) {
          node = document.createTextNode(token);
        }
      }
      parent.appendChild(node);
      offset = match.index + token.length;
    }
    if (offset < text.length) parent.appendChild(document.createTextNode(text.slice(offset)));
  }

  function markdownCharacterIsEscaped(text, index) {
    let slashes = 0;
    for (let cursor = index - 1; cursor >= 0 && text[cursor] === "\\"; cursor -= 1) slashes += 1;
    return slashes % 2 === 1;
  }

  function splitMarkdownTableRow(line) {
    let source = String(line || "").trim();
    if (source.startsWith("|")) source = source.slice(1);
    if (source.endsWith("|") && !markdownCharacterIsEscaped(source, source.length - 1)) {
      source = source.slice(0, -1);
    }
    const cells = [];
    let cell = "";
    for (let index = 0; index < source.length; index += 1) {
      if (source[index] === "|" && !markdownCharacterIsEscaped(source, index)) {
        cells.push(cell.trim().replace(/\\\|/g, "|"));
        cell = "";
      } else {
        cell += source[index];
      }
    }
    cells.push(cell.trim().replace(/\\\|/g, "|"));
    return cells;
  }

  function markdownTableSpec(lines, index) {
    if (index + 1 >= lines.length || !lines[index].includes("|") || !lines[index + 1].includes("|")) return null;
    const header = splitMarkdownTableRow(lines[index]);
    const delimiters = splitMarkdownTableRow(lines[index + 1]);
    if (!header.length || header.length !== delimiters.length
      || delimiters.some(cell => !/^:?-+:?$/.test(cell))) return null;
    return {
      header,
      alignments: delimiters.map((cell) => {
        if (cell.startsWith(":") && cell.endsWith(":")) return "center";
        if (cell.endsWith(":")) return "right";
        if (cell.startsWith(":")) return "left";
        return "left";
      }),
    };
  }

  function appendMarkdownTableCell(row, tagName, text, alignment) {
    const cell = document.createElement(tagName);
    cell.classList.add(`align-${alignment}`);
    if (tagName === "th") cell.scope = "col";
    appendInlineMarkdown(cell, text);
    row.appendChild(cell);
  }

  function renderMarkdownTable(parent, lines, index, spec) {
    const wrapper = document.createElement("div");
    wrapper.className = "markdown-table-wrap";
    const table = document.createElement("table");
    table.className = "markdown-table";
    const head = document.createElement("thead");
    const headerRow = document.createElement("tr");
    spec.header.forEach((cell, cellIndex) => {
      appendMarkdownTableCell(headerRow, "th", cell, spec.alignments[cellIndex]);
    });
    head.appendChild(headerRow);
    table.appendChild(head);

    const body = document.createElement("tbody");
    index += 2;
    while (index < lines.length && lines[index].trim() && lines[index].includes("|")
      && !/^(#{1,3})\s+|^```|^[-*]\s+|^\d+\.\s+|^>\s+/.test(lines[index])) {
      const values = splitMarkdownTableRow(lines[index]).slice(0, spec.header.length);
      while (values.length < spec.header.length) values.push("");
      const row = document.createElement("tr");
      values.forEach((cell, cellIndex) => {
        appendMarkdownTableCell(row, "td", cell, spec.alignments[cellIndex]);
      });
      body.appendChild(row);
      index += 1;
    }
    if (body.children.length) table.appendChild(body);
    wrapper.appendChild(table);
    parent.appendChild(wrapper);
    return index;
  }

  function renderMarkdown(parent, markdown) {
    parent.replaceChildren();
    const lines = String(markdown || "").replace(/\r\n?/g, "\n").split("\n");
    for (let index = 0; index < lines.length;) {
      const line = lines[index];
      if (!line.trim()) {
        index += 1;
        continue;
      }
      if (line.startsWith("```")) {
        const language = line.slice(3).trim();
        const codeLines = [];
        index += 1;
        while (index < lines.length && !lines[index].startsWith("```")) {
          codeLines.push(lines[index]);
          index += 1;
        }
        if (index < lines.length) index += 1;
        const pre = document.createElement("pre");
        const code = document.createElement("code");
        if (language) code.dataset.language = language;
        code.textContent = codeLines.join("\n");
        pre.appendChild(code);
        parent.appendChild(pre);
        continue;
      }
      const tableSpec = markdownTableSpec(lines, index);
      if (tableSpec) {
        index = renderMarkdownTable(parent, lines, index, tableSpec);
        continue;
      }
      const heading = /^(#{1,3})\s+(.+)$/.exec(line);
      if (heading) {
        const node = document.createElement(`h${heading[1].length}`);
        appendInlineMarkdown(node, heading[2]);
        parent.appendChild(node);
        index += 1;
        continue;
      }
      const listMatch = /^(?:[-*]\s+|\d+\.\s+)/.exec(line);
      if (listMatch) {
        const ordered = /^\d/.test(line);
        const list = document.createElement(ordered ? "ol" : "ul");
        while (index < lines.length) {
          const current = ordered
            ? /^\d+\.\s+(.+)$/.exec(lines[index])
            : /^[-*]\s+(.+)$/.exec(lines[index]);
          if (!current) break;
          const item = document.createElement("li");
          appendInlineMarkdown(item, current[1]);
          list.appendChild(item);
          index += 1;
        }
        parent.appendChild(list);
        continue;
      }
      if (line.startsWith("> ")) {
        const quote = document.createElement("blockquote");
        const parts = [];
        while (index < lines.length && lines[index].startsWith("> ")) {
          parts.push(lines[index].slice(2));
          index += 1;
        }
        appendInlineMarkdown(quote, parts.join("\n"));
        parent.appendChild(quote);
        continue;
      }
      const paragraphLines = [line];
      index += 1;
      while (index < lines.length && lines[index].trim()
        && !/^(#{1,3})\s+|^```|^[-*]\s+|^\d+\.\s+|^>\s+/.test(lines[index])
        && !markdownTableSpec(lines, index)) {
        paragraphLines.push(lines[index]);
        index += 1;
      }
      const paragraph = document.createElement("p");
      appendInlineMarkdown(paragraph, paragraphLines.join("\n"));
      parent.appendChild(paragraph);
    }
  }

  function normalizedGenerationMetrics(value) {
    if (!value || typeof value !== "object") return null;
    const parsedTime = Date.parse(String(value.createdAt || ""));
    const tokensPerSecond = value.tokensPerSecond == null
      ? Number.NaN
      : Number(value.tokensPerSecond);
    const evalCount = value.evalCount == null ? Number.NaN : Number(value.evalCount);
    return {
      createdAt: Number.isFinite(parsedTime)
        ? new Date(parsedTime).toISOString()
        : new Date().toISOString(),
      tokensPerSecond: Number.isFinite(tokensPerSecond) && tokensPerSecond >= 0
        ? tokensPerSecond
        : null,
      evalCount: Number.isFinite(evalCount) && evalCount >= 0 ? Math.round(evalCount) : null,
    };
  }

  function generationMetricsFromResponse(response) {
    const evalCount = Number(response && response.eval_count);
    const evalDuration = Number(response && response.eval_duration);
    const tokensPerSecond = (
      Number.isFinite(evalCount) && evalCount >= 0
      && Number.isFinite(evalDuration) && evalDuration > 0
    ) ? evalCount / (evalDuration / 1_000_000_000) : null;
    return normalizedGenerationMetrics({
      createdAt: response && response.created_at,
      tokensPerSecond,
      evalCount,
    });
  }

  function renderGenerationMetrics(record) {
    const node = record.node.querySelector(".message-generation-metrics");
    const metrics = record.generationMetrics;
    if (!metrics) {
      node.hidden = true;
      node.textContent = "";
      node.removeAttribute("title");
      return;
    }
    const time = new Intl.DateTimeFormat([], {
      hour: "numeric",
      minute: "2-digit",
      second: "2-digit",
    }).format(new Date(metrics.createdAt));
    const parts = [];
    if (Number.isFinite(metrics.tokensPerSecond)) {
      parts.push(`${metrics.tokensPerSecond.toFixed(1)} tok/s`);
    }
    parts.push(time);
    node.textContent = parts.join(" · ");
    node.hidden = false;
    node.title = Number.isFinite(metrics.evalCount)
      ? `${metrics.evalCount} generated tokens · ${metrics.createdAt}`
      : `Generated ${metrics.createdAt}`;
  }

  function updateMessage(record, {
    content,
    thinking,
    audioObservation,
    soundOnly,
    generationMetrics,
    audio,
    streaming = false,
    autoplayAudio = true,
  }) {
    const contentNode = record.node.querySelector(".message-content");
    if (content !== undefined) {
      record.content = String(content || "");
      if (record.role === "assistant" && !record.error) renderMarkdown(contentNode, content || "");
      else contentNode.textContent = content || "";
    }
    const thinkingBox = record.node.querySelector(".thinking-output");
    const thinkingNode = record.node.querySelector(".thinking-content");
    if (thinking !== undefined) {
      record.thinking = String(thinking || "");
      thinkingBox.hidden = !thinking;
      if (thinking) renderMarkdown(thinkingNode, thinking);
      else thinkingNode.replaceChildren();
    }
    const audioObservationBox = record.node.querySelector(".audio-observation-output");
    const audioObservationNode = record.node.querySelector(".audio-observation-content");
    if (audioObservation !== undefined) {
      record.audioObservation = String(audioObservation || "");
      audioObservationBox.hidden = !audioObservation;
      if (audioObservation) renderMarkdown(audioObservationNode, audioObservation);
      else audioObservationNode.replaceChildren();
    }
    if (soundOnly !== undefined) {
      record.soundOnly = Boolean(soundOnly);
      record.node.classList.toggle("sound-only", record.soundOnly);
    }
    if (generationMetrics !== undefined) {
      record.generationMetrics = normalizedGenerationMetrics(generationMetrics);
    }
    record.node.classList.toggle("streaming", streaming);
    if (audio && audio.data && !record.node.querySelector(".audio-output audio")) {
      record.audio = { ...audio };
      const playback = attachAudioPlayer(record.node, audio, autoplayAudio);
      if (autoplayAudio) record.playback = playback;
    }
    record.streaming = Boolean(streaming);
    renderGenerationMetrics(record);
    const footer = record.node.querySelector(".message-footer");
    footer.hidden = (
      record.role !== "assistant"
      || record.error
      || record.streaming
      || !record.content
    );
    if (!streaming) scheduleBrowserSessionSave();
    scrollConversationToBottom({ smooth: !streaming });
    return record;
  }

  function audioEvidenceHistory(transcript, audioObservation, fallback) {
    const speech = String(transcript || "").trim();
    const sounds = String(audioObservation || "").trim();
    const parts = [];
    if (speech) parts.push(speech);
    if (sounds) parts.push(`[Sounds heard: ${sounds}]`);
    return parts.join("\n") || fallback;
  }

  function loopingVideo(item, { ownsUrl = false } = {}) {
    if (item.mime === "image/gif") {
      const image = document.createElement("img");
      image.className = "looping-video";
      image.alt = "Looping preview of the sent animated GIF";
      image.src = item.previewUrl;
      if (ownsUrl) image.dataset.objectUrl = item.previewUrl;
      return image;
    }
    const video = document.createElement("video");
    video.className = "looping-video";
    video.autoplay = true;
    video.loop = true;
    video.muted = true;
    video.playsInline = true;
    video.preload = "metadata";
    video.disablePictureInPicture = true;
    video.setAttribute("aria-label", "Looping preview of the sent video clip");
    video.src = item.previewUrl;
    if (ownsUrl) video.dataset.objectUrl = item.previewUrl;
    video.addEventListener("loadeddata", () => video.play().catch(() => {}), { once: true });
    return video;
  }

  function appendMessageMedia(node, media) {
    const videos = (media || []).filter(item => item.kind === "video" && item.previewUrl);
    const documents = (media || []).filter(item => item.kind === "document");
    if (!videos.length && !documents.length) return;
    const gallery = document.createElement("div");
    gallery.className = "message-media";
    for (const item of videos) {
      const preview = document.createElement("div");
      preview.className = "message-video-preview";
      preview.appendChild(loopingVideo(item, { ownsUrl: true }));
      const badge = document.createElement("span");
      badge.textContent = item.mime === "image/gif" ? "GIF · LOOP" : "VIDEO · LOOP";
      preview.appendChild(badge);
      gallery.appendChild(preview);
    }
    for (const item of documents) {
      const preview = document.createElement("div");
      preview.className = "message-document-preview";
      const label = document.createElement("span");
      label.textContent = "DOCUMENT";
      const name = document.createElement("strong");
      name.textContent = item.name;
      preview.append(label, name);
      gallery.appendChild(preview);
    }
    node.appendChild(gallery);
  }

  function fallbackCopyText(text) {
    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.readOnly = true;
    textarea.setAttribute("aria-hidden", "true");
    textarea.style.position = "fixed";
    textarea.style.inset = "-1000px auto auto -1000px";
    textarea.style.opacity = "0";
    document.body.appendChild(textarea);
    textarea.select();
    textarea.setSelectionRange(0, textarea.value.length);
    const copied = document.execCommand("copy");
    textarea.remove();
    if (!copied) throw new Error("browser rejected clipboard copy");
  }

  async function copyAssistantMarkdown(record, button) {
    const markdown = String(record.content || "");
    if (!markdown) return;
    try {
      if (navigator.clipboard && typeof navigator.clipboard.writeText === "function") {
        await navigator.clipboard.writeText(markdown);
      } else {
        fallbackCopyText(markdown);
      }
      button.classList.add("copied");
      button.setAttribute("aria-label", "Copied raw Markdown");
      button.title = "Copied";
      setTimeout(() => {
        if (!button.isConnected) return;
        button.classList.remove("copied");
        button.setAttribute("aria-label", "Copy raw Markdown reply");
        button.title = "Copy raw Markdown";
      }, 1200);
    } catch (_error) {
      transientComposerStatus("Could not copy reply");
    }
  }

  function addMessage({
    role,
    content,
    thinking,
    audioObservation,
    soundOnly = false,
    generationMetrics,
    audio,
    media = [],
    error = false,
    streaming = false,
    autoplayAudio = true,
  }) {
    const node = elements.template.content.firstElementChild.cloneNode(true);
    node.classList.add(role === "user" ? "user" : "assistant");
    if (error) node.classList.add("error");
    elements.conversation.appendChild(node);
    if (layoutResizeObserver) layoutResizeObserver.observe(node);
    const record = {
      node,
      role,
      error,
      content: "",
      thinking: "",
      audioObservation: "",
      soundOnly: false,
      generationMetrics: null,
      audio: null,
      media,
      streaming,
      playback: Promise.resolve(),
    };
    const copyButton = node.querySelector(".message-copy-button");
    copyButton.addEventListener("click", () => {
      copyAssistantMarkdown(record, copyButton);
    });
    state.messages.push(record);
    updateMessage(record, {
      content,
      thinking,
      audioObservation,
      soundOnly,
      generationMetrics,
      audio,
      streaming,
      autoplayAudio,
    });
    appendMessageMedia(node, media);
    scrollConversationToBottom();
    return record;
  }

  function removeMessage(record) {
    if (!record) return;
    for (const media of record.node.querySelectorAll("[data-object-url]")) {
      URL.revokeObjectURL(media.dataset.objectUrl);
    }
    if (layoutResizeObserver) layoutResizeObserver.unobserve(record.node);
    const index = state.messages.indexOf(record);
    if (index >= 0) state.messages.splice(index, 1);
    record.node.remove();
    scheduleBrowserSessionSave();
  }

  function base64ToBlob(data, mime) {
    const binary = atob(data);
    const chunks = [];
    for (let offset = 0; offset < binary.length; offset += 32_768) {
      const slice = binary.slice(offset, offset + 32_768);
      const bytes = new Uint8Array(slice.length);
      for (let index = 0; index < slice.length; index += 1) bytes[index] = slice.charCodeAt(index);
      chunks.push(bytes);
    }
    return new Blob(chunks, { type: mime || "audio/wav" });
  }

  async function playWithUnlockedContext(envelope, epoch) {
    const context = state.playbackContext;
    if (!context) return false;
    if (context.state === "suspended") await context.resume();
    if (context.state !== "running") return false;
    const encoded = base64ToBlob(envelope.data, envelope.mime_type);
    const decoded = await context.decodeAudioData(await encoded.arrayBuffer());
    if (state.playbackEpoch !== epoch) return false;
    if (state.playbackSource) {
      try {
        state.playbackSource.stop();
      } catch (_error) {
        // The previous source already ended.
      }
    }
    const source = context.createBufferSource();
    source.buffer = decoded;
    source.connect(context.destination);
    state.playbackSource = source;
    return new Promise(resolve => {
      source.addEventListener("ended", () => {
        if (state.playbackSource === source) state.playbackSource = null;
        resolve(true);
      }, { once: true });
      source.start();
    });
  }

  function stopCurrentPlayback() {
    state.playbackEpoch += 1;
    const controller = state.streamController;
    if (controller) {
      state.streamController = null;
      controller.cancelled = true;
      for (const source of controller.sources) {
        try {
          source.stop();
        } catch (_error) {
          // A scheduled PCM buffer may already have ended.
        }
      }
      controller.sources.clear();
      controller.resolve(false);
    }
    if (state.playbackSource) {
      try {
        state.playbackSource.stop();
      } catch (_error) {
        // Playback ended before interruption was detected.
      }
      state.playbackSource = null;
    }
    if (state.playbackElement) {
      state.playbackElement.pause();
      state.playbackElement.currentTime = 0;
      state.playbackElement = null;
    }
  }

  function playHtmlAudio(audio, epoch) {
    if (state.playbackEpoch !== epoch) return Promise.resolve(false);
    state.playbackElement = audio;
    return audio.play().then(() => new Promise(resolve => {
      const done = () => {
        if (state.playbackElement === audio) state.playbackElement = null;
        resolve(true);
      };
      audio.addEventListener("ended", done, { once: true });
      audio.addEventListener("error", done, { once: true });
    }));
  }

  function attachAudioPlayer(node, envelope, autoplay = true) {
    const box = node.querySelector(".audio-output");
    box.hidden = false;
    const audio = document.createElement("audio");
    audio.controls = true;
    audio.playsInline = true;
    audio.preload = "metadata";
    const url = URL.createObjectURL(base64ToBlob(envelope.data, envelope.mime_type));
    audio.src = url;
    audio.dataset.objectUrl = url;
    box.appendChild(audio);
    if (!autoplay) return Promise.resolve(false);
    const epoch = state.playbackEpoch;
    return playWithUnlockedContext(envelope, epoch)
      .then(playing => {
        if (state.playbackEpoch !== epoch) return false;
        if (!playing) return playHtmlAudio(audio, epoch);
        return playing;
      })
      .catch(() => {
        return playHtmlAudio(audio, epoch).catch(() => false);
      });
  }

  function beginPcmPlayback() {
    const context = state.playbackContext;
    if (!context || context.state !== "running") return null;
    if (state.streamController) stopCurrentPlayback();
    let resolve;
    const promise = new Promise(done => { resolve = done; });
    const controller = {
      epoch: state.playbackEpoch,
      context,
      nextTime: context.currentTime + PCM_INITIAL_BUFFER_SECONDS,
      lastGain: null,
      lastDuration: 0,
      sources: new Set(),
      ended: false,
      cancelled: false,
      resolve,
      promise,
    };
    state.streamController = controller;
    return controller;
  }

  function maybeFinishPcmPlayback(controller) {
    if (!controller.ended || controller.sources.size) return;
    if (state.streamController === controller) state.streamController = null;
    controller.resolve(!controller.cancelled);
  }

  function queuePcmPlayback(controller, encoded) {
    if (!controller || controller.cancelled || state.playbackEpoch !== controller.epoch) return;
    const binary = atob(encoded);
    if (!binary.length || binary.length % 2) throw new Error("TTS stream returned partial PCM samples");
    const samples = new Float32Array(binary.length / 2);
    for (let index = 0; index < samples.length; index += 1) {
      let value = binary.charCodeAt(index * 2) | (binary.charCodeAt(index * 2 + 1) << 8);
      if (value & 0x8000) value -= 0x10000;
      samples[index] = value / (value < 0 ? 32768 : 32767);
    }
    const buffer = controller.context.createBuffer(1, samples.length, 24000);
    buffer.copyToChannel(samples, 0);
    const source = controller.context.createBufferSource();
    const gain = controller.context.createGain();
    source.buffer = buffer;
    source.connect(gain);
    gain.connect(controller.context.destination);
    controller.sources.add(source);
    source.addEventListener("ended", () => {
      controller.sources.delete(source);
      maybeFinishPcmPlayback(controller);
    }, { once: true });
    const playbackFloor = (
      controller.context.currentTime + PCM_RESCHEDULE_FLOOR_SECONDS
    );
    const crossfade = Math.min(
      PCM_CROSSFADE_SECONDS,
      buffer.duration / 4,
      controller.lastDuration / 4,
    );
    const canCrossfade = (
      controller.lastGain
      && controller.lastDuration >= PCM_CROSSFADE_MIN_BUFFER_SECONDS
      && buffer.duration >= PCM_CROSSFADE_MIN_BUFFER_SECONDS
      && controller.nextTime - crossfade >= playbackFloor
    );
    const overlap = canCrossfade ? crossfade : 0;
    const startAt = Math.max(controller.nextTime - overlap, playbackFloor);
    if (overlap > 0) {
      controller.lastGain.gain.setValueAtTime(1, startAt);
      controller.lastGain.gain.linearRampToValueAtTime(0, startAt + overlap);
      gain.gain.setValueAtTime(0, startAt);
      gain.gain.linearRampToValueAtTime(1, startAt + overlap);
    } else {
      const fadeIn = Math.min(PCM_CROSSFADE_SECONDS, buffer.duration / 4);
      gain.gain.setValueAtTime(0, startAt);
      gain.gain.linearRampToValueAtTime(1, startAt + fadeIn);
    }
    source.start(startAt);
    controller.nextTime = startAt + buffer.duration;
    controller.lastGain = gain;
    controller.lastDuration = buffer.duration;
  }

  function endPcmPlayback(controller) {
    if (!controller) return;
    controller.ended = true;
    maybeFinishPcmPlayback(controller);
  }

  async function unlockPlayback() {
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    if (!AudioContext) return;
    if (!state.playbackContext) state.playbackContext = new AudioContext();
    if (state.playbackContext.state === "suspended") await state.playbackContext.resume().catch(() => {});
  }

  async function streamChat(payload, { signal, onEvent } = {}) {
    const startedAt = performance.now();
    const timings = {};
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: authHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify({ ...payload, stream: true }),
      signal,
    });
    timings.response_headers_ms = performance.now() - startedAt;
    const requestId = response.headers.get("X-Omni-Request-ID") || "";
    if (!response.ok) {
      const data = await response.json().catch(() => ({ error: `HTTP ${response.status}` }));
      throw new Error(data.error || `HTTP ${response.status}`);
    }
    if (!response.body) throw new Error("Streaming response has no body");

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffered = "";
    let finalResponse = null;

    const consume = line => {
      if (!line.trim()) return;
      let event;
      try {
        event = JSON.parse(line);
      } catch (_error) {
        throw new Error("Model returned an invalid stream event");
      }
      const elapsed = performance.now() - startedAt;
      if (timings.first_event_ms === undefined) timings.first_event_ms = elapsed;
      if (event.type === "delta" && timings.first_text_ms === undefined) {
        const message = event.message || {};
        if (message.content || message.thinking) timings.first_text_ms = elapsed;
      }
      if (event.type === "stage" && event.stage === "tts"
        && timings.tts_stage_ms === undefined) timings.tts_stage_ms = elapsed;
      if (event.type === "audio_start" && timings.audio_start_ms === undefined) {
        timings.audio_start_ms = elapsed;
      }
      if (event.type === "audio_delta" && timings.first_audio_delta_ms === undefined) {
        timings.first_audio_delta_ms = elapsed;
      }
      if (event.type === "error") throw new Error(event.error || "Streaming inference failed");
      if (event.type === "final") finalResponse = event.response || null;
      if (onEvent) onEvent(event);
    };

    for (;;) {
      const { value, done } = await reader.read();
      buffered += decoder.decode(value || new Uint8Array(), { stream: !done });
      let newline;
      while ((newline = buffered.indexOf("\n")) >= 0) {
        consume(buffered.slice(0, newline));
        buffered = buffered.slice(newline + 1);
      }
      if (done) break;
    }
    if (buffered.trim()) consume(buffered);
    if (!finalResponse) throw new Error("Model stream ended without a final response");
    timings.complete_ms = performance.now() - startedAt;
    reportDiagnostic("client_stream_timing", {
      request_id: requestId,
      ...timings,
    });
    return finalResponse;
  }

  function reasoningEnabled() {
    return elements.think.getAttribute("aria-pressed") === "true";
  }

  async function refreshStatus() {
    if (!state.token) {
      elements.headerStatus.className = "connection offline";
      elements.statusText.textContent = "Access missing";
      return;
    }
    try {
      const response = await fetch("/api/status", { headers: authHeaders() });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      elements.headerStatus.className = `connection ${data.ok ? "online" : "offline"}`;
      elements.statusText.textContent = data.ok ? "Online" : "Unavailable";
      updateActivity(data.requests);
      if (!state.voice.initialized && data.voice_profile) {
        const profile = data.voice_profile;
        state.voice.serverReference = Boolean(profile.speaker_reference);
        state.voice.presets = Array.isArray(profile.presets) ? profile.presets : [];
        state.voice.selectedPreset = String(
          (state.voice.presets.find(preset => preset.default) || state.voice.presets[0] || {}).id || "",
        );
        state.voice.defaults = {
          language: profile.language || "en",
          temperature: Number(profile.temperature ?? 0.7),
          topK: Number(profile.top_k ?? 40),
          topP: Number(profile.top_p ?? 0.9),
          seed: Number(profile.seed ?? 42),
          maxFrames: Number(profile.max_frames ?? 512),
        };
        renderVoicePresets();
        applyVoiceDefaults();
        state.voice.initialized = true;
      }
    } catch (_error) {
      elements.headerStatus.className = "connection offline";
      elements.statusText.textContent = "Offline";
    }
  }

  function updateActivity(activity) {
    if (!elements.activeUsers || !elements.activeUserCount) return;
    const users = Math.max(0, Number((activity || {}).users) || 0);
    const inflight = Math.max(0, Number((activity || {}).inflight) || 0);
    elements.activeUserCount.textContent = String(users);
    elements.activeUsers.setAttribute(
      "aria-label",
      `${users} active ${users === 1 ? "user" : "users"}, ${inflight} in-flight requests`,
    );
  }

  async function refreshActivity() {
    if (!state.token) return;
    try {
      const response = await fetch("/api/activity", { headers: authHeaders() });
      if (!response.ok) return;
      updateActivity(await response.json());
    } catch (_error) {
      // The health poll owns the visible online/offline state.
    }
  }

  function humanBytes(bytes) {
    if (bytes < 1024 * 1024) return `${Math.max(1, Math.ceil(bytes / 1024))} KiB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
  }

  function inferKind(file) {
    const mime = (file.type || "").toLowerCase();
    const name = file.name.toLowerCase();
    if (mime === "audio/wav" || name.endsWith(".wav")) return "audio";
    if (mime === "image/gif" || name.endsWith(".gif")) return "video";
    if (mime.startsWith("image/")) return "image";
    if (mime === "video/mp4" || mime === "video/webm") return "video";
    if (
      mime === "application/pdf"
      || mime === "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
      || mime.startsWith("text/")
      || mime === "application/json"
      || mime === "application/xml"
      || mime.includes("yaml")
      || mime.includes("toml")
      || /\.(pdf|docx|txt|md|markdown|csv|tsv|log|json|jsonl|html?|xml|ya?ml|toml|ini|cfg|rst|sql|py|m?js|cjs|tsx?|jsx|css|sh|ps1|java|go|rs|c|h|cpp|hpp)$/.test(name)
    ) return "document";
    throw new Error("Choose supported audio, image, video/GIF, PDF, DOCX, or UTF-8 text/code");
  }

  function renderAttachments() {
    elements.attachments.replaceChildren();
    elements.attachments.hidden = state.attachments.length === 0;
    for (const [index, item] of state.attachments.entries()) {
      const node = document.createElement("div");
      node.className = "attachment";
      const preview = document.createElement("div");
      preview.className = "attachment-preview";
      if (item.kind === "image") {
        const image = document.createElement("img");
        image.alt = "";
        image.src = `data:${item.mime};base64,${item.data}`;
        preview.appendChild(image);
      } else if (item.kind === "video" && item.previewUrl) {
        preview.appendChild(loopingVideo(item));
      } else {
        preview.textContent = item.kind === "audio" ? "WAV" : "DOC";
      }
      const body = document.createElement("div");
      body.className = "attachment-body";
      const name = document.createElement("div");
      name.className = "attachment-name";
      name.textContent = item.name;
      const meta = document.createElement("div");
      meta.className = "attachment-meta";
      meta.textContent = `${item.kind} · ${humanBytes(item.bytes)}`;
      body.append(name, meta);
      if (item.kind === "audio") {
        const audio = document.createElement("audio");
        audio.controls = true;
        audio.playsInline = true;
        audio.preload = "metadata";
        audio.src = `data:audio/wav;base64,${item.data}`;
        body.appendChild(audio);
      }
      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "icon-button attachment-remove";
      remove.textContent = "×";
      remove.setAttribute("aria-label", `Remove ${item.name}`);
      remove.addEventListener("click", () => {
        if (item.previewUrl) URL.revokeObjectURL(item.previewUrl);
        state.attachments.splice(index, 1);
        renderAttachments();
        scheduleBrowserSessionSave();
      });
      node.append(preview, body, remove);
      elements.attachments.appendChild(node);
    }
  }

  function fileDataUrl(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result));
      reader.onerror = () => reject(reader.error || new Error("Could not read file"));
      reader.readAsDataURL(file);
    });
  }

  async function addFile(file, forcedKind = null, source = "upload") {
    if (!file) return;
    const kind = forcedKind || inferKind(file);
    if (file.size > LIMITS[kind]) {
      throw new Error(`${kind} is ${humanBytes(file.size)}; limit is ${humanBytes(LIMITS[kind])}`);
    }
    const dataUrl = await fileDataUrl(file);
    const comma = dataUrl.indexOf(",");
    if (comma < 0) throw new Error("File could not be encoded");
    const item = {
      kind,
      name: file.name || `microphone-${Date.now()}.wav`,
      mime: kind === "audio"
        ? "audio/wav"
        : (kind === "video" && file.name.toLowerCase().endsWith(".gif")
          ? "image/gif"
          : (file.type || "application/octet-stream")),
      data: dataUrl.slice(comma + 1),
      bytes: file.size,
      source,
      previewUrl: kind === "video" ? URL.createObjectURL(file) : null,
    };
    if (source === "camera") {
      for (const previous of state.attachments.filter(value => value.source === "camera")) {
        if (previous.previewUrl) URL.revokeObjectURL(previous.previewUrl);
      }
      state.attachments = state.attachments.filter(value => value.source !== "camera");
    }
    state.attachments.push(item);
    renderAttachments();
    scheduleBrowserSessionSave();
  }

  function mergeSamples(chunks) {
    const length = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
    const result = new Float32Array(length);
    let offset = 0;
    for (const chunk of chunks) {
      result.set(chunk, offset);
      offset += chunk.length;
    }
    return result;
  }

  function downsample(samples, sourceRate, targetRate = 16000) {
    if (sourceRate === targetRate) return samples;
    const ratio = sourceRate / targetRate;
    const output = new Float32Array(Math.floor(samples.length / ratio));
    for (let index = 0; index < output.length; index += 1) {
      const start = Math.floor(index * ratio);
      const end = Math.max(start + 1, Math.floor((index + 1) * ratio));
      let total = 0;
      for (let cursor = start; cursor < end && cursor < samples.length; cursor += 1) total += samples[cursor];
      output[index] = total / (end - start);
    }
    return output;
  }

  function pcmWav(samples, sampleRate = 16000) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);
    const write = (offset, text) => {
      for (let index = 0; index < text.length; index += 1) view.setUint8(offset + index, text.charCodeAt(index));
    };
    write(0, "RIFF");
    view.setUint32(4, 36 + samples.length * 2, true);
    write(8, "WAVE");
    write(12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    write(36, "data");
    view.setUint32(40, samples.length * 2, true);
    let offset = 44;
    for (const sample of samples) {
      const clamped = Math.max(-1, Math.min(1, sample));
      view.setInt16(offset, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true);
      offset += 2;
    }
    return new Blob([buffer], { type: "audio/wav" });
  }

  function drawWaveform() {
    const capture = state.recording || state.call;
    if (!capture) return;
    const canvas = elements.waveformCanvas;
    const rect = canvas.getBoundingClientRect();
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.floor(rect.width * ratio));
    canvas.height = Math.max(1, Math.floor(rect.height * ratio));
    const context = canvas.getContext("2d");
    context.scale(ratio, ratio);
    const samples = new Uint8Array(capture.analyser.fftSize);
    capture.analyser.getByteTimeDomainData(samples);
    context.clearRect(0, 0, rect.width, rect.height);
    context.strokeStyle = state.call
      ? (capture.vadActive ? "rgba(134,239,172,.98)" : "rgba(134,239,172,.26)")
      : "#fb7185";
    context.lineWidth = 1.5;
    context.beginPath();
    for (let index = 0; index < samples.length; index += 1) {
      const x = index * rect.width / (samples.length - 1);
      const y = samples[index] / 255 * rect.height;
      if (index === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    }
    context.stroke();
    capture.animationFrame = requestAnimationFrame(drawWaveform);
  }

  function updateRecordClock() {
    if (!state.recording) return;
    const seconds = Math.floor((Date.now() - state.recording.started) / 1000);
    const minutes = Math.floor(seconds / 60);
    elements.recordingTime.textContent = `${minutes}:${String(seconds % 60).padStart(2, "0")}`;
  }

  async function startRecording() {
    if (state.recording) return;
    if (state.call) throw new Error("End the voice call before recording a clip");
    if (state.camera) throw new Error("Stop device video before recording an audio clip");
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error("Microphone capture requires this HTTPS page in a supported browser");
    }
    setComposerStatus("Requesting microphone…");
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
      video: false,
    });
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    const context = new AudioContext();
    await context.resume();
    const source = context.createMediaStreamSource(stream);
    const analyser = context.createAnalyser();
    analyser.fftSize = 1024;
    const processor = context.createScriptProcessor(4096, 1, 1);
    const sink = context.createGain();
    sink.gain.value = 0;
    const chunks = [];
    processor.onaudioprocess = event => chunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
    source.connect(analyser);
    source.connect(processor);
    processor.connect(sink);
    sink.connect(context.destination);
    state.recording = {
      stream, context, source, analyser, processor, sink, chunks,
      started: Date.now(), animationFrame: null, stopping: null, discard: false,
    };
    elements.waveform.hidden = false;
    elements.micButton.classList.add("recording");
    elements.cameraButton.disabled = true;
    elements.micButton.setAttribute("aria-label", "Release to attach recording");
    setComposerStatus("Recording · release to attach");
    drawWaveform();
    updateRecordClock();
    state.recordClock = setInterval(updateRecordClock, 250);
    state.recordTimer = setTimeout(() => {
      state.holdingMic = false;
      stopRecording().catch(showError);
    }, MAX_RECORD_MS);
    if (!state.holdingMic) {
      await stopRecording({ discard: state.discardMicRecording });
    }
  }

  async function stopRecording({ discard = false } = {}) {
    const recording = state.recording;
    if (!recording) return;
    if (discard) recording.discard = true;
    if (recording.stopping) return recording.stopping;
    recording.stopping = (async () => {
      clearTimeout(state.recordTimer);
      clearInterval(state.recordClock);
      cancelAnimationFrame(recording.animationFrame);
      recording.processor.disconnect();
      recording.source.disconnect();
      recording.sink.disconnect();
      recording.stream.getTracks().forEach(track => track.stop());
      const samples = downsample(mergeSamples(recording.chunks), recording.context.sampleRate);
      state.recording = null;
      await recording.context.close();
      elements.waveform.hidden = true;
      elements.micButton.classList.remove("recording");
      elements.cameraButton.disabled = false;
      elements.micButton.setAttribute("aria-label", "Hold to record microphone");
      if (recording.discard || !samples.length) {
        transientComposerStatus("Press and hold to record voice clip");
        return false;
      }
      const blob = pcmWav(samples);
      const file = new File([blob], `microphone-${Date.now()}.wav`, { type: "audio/wav" });
      await addFile(file, "audio");
      setComposerStatus(`Audio attached · ${(samples.length / 16000).toFixed(1)} seconds`);
      return true;
    })();
    return recording.stopping;
  }

  function rms(samples) {
    let sum = 0;
    for (const sample of samples) sum += sample * sample;
    return Math.sqrt(sum / Math.max(1, samples.length));
  }

  function audioEnvelope(samples, sourceRate) {
    return fileDataUrl(pcmWav(downsample(mergeSamples(samples), sourceRate)))
      .then(dataUrl => ({
        mime_type: "audio/wav",
        encoding: "base64",
        data: dataUrl.slice(dataUrl.indexOf(",") + 1),
      }));
  }

  function syncVoiceUi() {
    const enabled = elements.voiceCloneEnabled.checked;
    const selectedPreset = state.voice.presets.find(preset => preset.id === state.voice.selectedPreset);
    elements.voiceCloneToggle.classList.toggle("active", enabled);
    elements.voiceCloneToggle.setAttribute("aria-pressed", String(enabled));
    elements.voiceCloneToggle.querySelector("small").textContent = enabled ? "Enabled" : "Disabled";
    elements.voiceModeRow.classList.toggle("preset-open", state.voice.presetMenuOpen);
    elements.voicePresetToggle.setAttribute("aria-expanded", String(state.voice.presetMenuOpen));
    elements.voicePresetCurrent.textContent = selectedPreset
      ? `${selectedPreset.label}${selectedPreset.default ? " · Default" : " · Secondary"}`
      : "No preset";
    elements.voicePresetOptions.hidden = !state.voice.presetMenuOpen;
    elements.voicePresetOptions.querySelectorAll("[data-voice-preset]").forEach(button => {
      button.classList.toggle("selected", button.dataset.voicePreset === state.voice.selectedPreset);
    });
    elements.voiceReferenceControls.classList.toggle("disabled", !enabled);
    elements.voiceTemperatureValue.value = Number(elements.voiceTemperature.value).toFixed(2);
    elements.voiceTopPValue.value = Number(elements.voiceTopP.value).toFixed(2);
    if (state.voice.reference) {
      elements.voiceReferenceStatus.textContent = `${state.voice.reference.name} · ${humanBytes(state.voice.reference.bytes)}`;
    } else if (state.voice.serverReference && selectedPreset) {
      elements.voiceReferenceStatus.textContent = `Using ${selectedPreset.label} preset`;
    } else if (state.voice.serverReference) {
      elements.voiceReferenceStatus.textContent = "Using server profile reference";
    } else {
      elements.voiceReferenceStatus.textContent = "No reference selected";
    }
  }

  function renderVoicePresets() {
    elements.voicePresetOptions.replaceChildren();
    for (const preset of state.voice.presets) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "voice-preset-option";
      button.dataset.voicePreset = preset.id;
      const label = document.createElement("strong");
      label.textContent = preset.label;
      const detail = document.createElement("small");
      detail.textContent = preset.default ? "Default" : "Secondary";
      button.append(label, detail);
      elements.voicePresetOptions.append(button);
    }
  }

  function setVoicePresetMenu(open) {
    state.voice.presetMenuOpen = Boolean(open);
    syncVoiceUi();
  }

  function applyVoiceDefaults(profile = state.voice.defaults) {
    elements.voiceLanguage.value = profile.language || "en";
    elements.voiceTemperature.value = String(profile.temperature ?? 0.7);
    elements.voiceTopK.value = String(profile.topK ?? 40);
    elements.voiceTopP.value = String(profile.topP ?? 0.9);
    elements.voiceSeed.value = String(profile.seed ?? 42);
    elements.voiceMaxFrames.value = String(profile.maxFrames ?? 512);
    state.voice.selectedPreset = String(
      (state.voice.presets.find(preset => preset.default) || state.voice.presets[0] || {}).id || "",
    );
    elements.voiceCloneEnabled.checked = Boolean(state.voice.serverReference);
    syncVoiceUi();
  }

  function clearVoiceReference() {
    state.voice.reference = null;
    if (state.voice.referenceUrl) URL.revokeObjectURL(state.voice.referenceUrl);
    state.voice.referenceUrl = null;
    elements.voiceReferenceAudio.removeAttribute("src");
    elements.voiceReferenceAudio.hidden = true;
    elements.voiceReferenceInput.value = "";
    elements.voiceCloneEnabled.checked = Boolean(state.voice.serverReference);
    syncVoiceUi();
  }

  async function setVoiceReference(file) {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".wav") && file.type !== "audio/wav") {
      throw new Error("Voice reference must be an uncompressed WAV file");
    }
    if (file.size > MAX_VOICE_REFERENCE_BYTES) {
      throw new Error(`Voice reference exceeds ${humanBytes(MAX_VOICE_REFERENCE_BYTES)}`);
    }
    const dataUrl = await fileDataUrl(file);
    const comma = dataUrl.indexOf(",");
    if (comma < 0) throw new Error("Voice reference could not be encoded");
    state.voice.reference = {
      name: file.name || "voice-reference.wav",
      bytes: file.size,
      envelope: {
        mime_type: "audio/wav",
        encoding: "base64",
        data: dataUrl.slice(comma + 1),
      },
    };
    if (state.voice.referenceUrl) URL.revokeObjectURL(state.voice.referenceUrl);
    state.voice.referenceUrl = URL.createObjectURL(file);
    elements.voiceReferenceAudio.src = state.voice.referenceUrl;
    elements.voiceReferenceAudio.hidden = false;
    elements.voiceCloneEnabled.checked = true;
    syncVoiceUi();
  }

  async function startVoiceReferenceRecording() {
    if (state.voice.recording) return;
    if (state.recording || state.call || state.camera) {
      throw new Error("Stop other microphone or camera capture before recording a voice reference");
    }
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error("Voice reference recording requires HTTPS and microphone access");
    }
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
      video: false,
    });
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    const context = new AudioContext();
    await context.resume();
    const source = context.createMediaStreamSource(stream);
    const processor = context.createScriptProcessor(4096, 1, 1);
    const sink = context.createGain();
    sink.gain.value = 0;
    const chunks = [];
    processor.onaudioprocess = event => chunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
    source.connect(processor);
    processor.connect(sink);
    sink.connect(context.destination);
    state.voice.recording = { stream, context, source, processor, sink, chunks, started: Date.now() };
    elements.voiceRecord.classList.add("recording");
    elements.voiceRecord.textContent = "Stop recording";
    elements.voiceReferenceStatus.textContent = "Recording reference… speak naturally";
    state.voice.recordTimer = window.setTimeout(() => {
      stopVoiceReferenceRecording().catch(showError);
    }, MAX_VOICE_REFERENCE_MS);
  }

  async function stopVoiceReferenceRecording() {
    const recording = state.voice.recording;
    if (!recording) return;
    state.voice.recording = null;
    clearTimeout(state.voice.recordTimer);
    recording.processor.disconnect();
    recording.source.disconnect();
    recording.sink.disconnect();
    recording.stream.getTracks().forEach(track => track.stop());
    const samples = downsample(mergeSamples(recording.chunks), recording.context.sampleRate);
    await recording.context.close();
    elements.voiceRecord.classList.remove("recording");
    elements.voiceRecord.textContent = "Record reference";
    const seconds = samples.length / 16000;
    if (seconds < 0.5) throw new Error("Record at least 0.5 seconds for voice cloning");
    const blob = pcmWav(samples);
    await setVoiceReference(new File([blob], `voice-reference-${Date.now()}.wav`, { type: "audio/wav" }));
  }

  function voicePayload() {
    const integer = (element, fallback) => {
      const value = Number(element.value);
      return Number.isInteger(value) ? value : fallback;
    };
    const payload = {
      clone_enabled: elements.voiceCloneEnabled.checked,
      language: elements.voiceLanguage.value,
      temperature: Number(elements.voiceTemperature.value),
      top_k: integer(elements.voiceTopK, 40),
      top_p: Number(elements.voiceTopP.value),
      seed: integer(elements.voiceSeed, 42),
      max_frames: integer(elements.voiceMaxFrames, 512),
    };
    if (payload.clone_enabled && state.voice.reference) {
      payload.speaker_audio = state.voice.reference.envelope;
    } else if (payload.clone_enabled && state.voice.selectedPreset) {
      payload.preset = state.voice.selectedPreset;
    }
    return payload;
  }

  function cameraMimeType() {
    const choices = [
      "video/webm;codecs=vp8,opus",
      "video/webm",
      "video/mp4",
    ];
    return choices.find(value => MediaRecorder.isTypeSupported(value)) || "";
  }

  async function startCameraCapture() {
    if (state.camera) return;
    if (state.recording) throw new Error("Release the microphone before recording video");
    if (state.call) throw new Error("End the voice call before recording video");
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia || !window.MediaRecorder) {
      throw new Error("Device video requires this HTTPS page in a supported browser");
    }
    setComposerStatus("Requesting camera and microphone…");
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: { ideal: "environment" },
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
    const mime = cameraMimeType();
    let recorder;
    try {
      recorder = new MediaRecorder(stream, {
        ...(mime ? { mimeType: mime } : {}),
        videoBitsPerSecond: 2_500_000,
        audioBitsPerSecond: 64_000,
      });
    } catch (_error) {
      try {
        recorder = new MediaRecorder(stream);
      } catch (error) {
        stream.getTracks().forEach(track => track.stop());
        throw error;
      }
    }
    const chunks = [];
    recorder.addEventListener("dataavailable", event => {
      if (event.data && event.data.size) chunks.push(event.data);
    });
    const stopped = new Promise((resolve, reject) => {
      recorder.addEventListener("stop", resolve, { once: true });
      recorder.addEventListener("error", event => {
        reject(event.error || new Error("Device video recording failed"));
      }, { once: true });
    });
    const camera = {
      stream,
      recorder,
      chunks,
      stopped,
      started: Date.now(),
      timer: null,
      stopping: null,
    };
    state.camera = camera;
    elements.cameraVideo.srcObject = stream;
    await elements.cameraVideo.play().catch(() => {});
    elements.cameraPreview.hidden = false;
    elements.cameraButton.setAttribute("aria-pressed", "true");
    elements.cameraButton.setAttribute("aria-label", "Stop and attach device video");
    elements.cameraButton.title = "Stop and attach video";
    elements.micButton.disabled = true;
    elements.callButton.disabled = false;
    try {
      recorder.start(250);
    } catch (error) {
      stream.getTracks().forEach(track => track.stop());
      elements.cameraVideo.srcObject = null;
      state.camera = null;
      elements.cameraPreview.hidden = true;
      elements.cameraButton.setAttribute("aria-pressed", "false");
      elements.micButton.disabled = false;
      elements.callButton.disabled = false;
      throw error;
    }
    camera.timer = window.setTimeout(() => {
      if (!state.call) stopCameraCapture().catch(showError);
    }, MAX_VIDEO_RECORD_MS);
    setComposerStatus("Camera live · record, send, or start a visual call");
  }

  async function stopCameraCapture() {
    const camera = state.camera;
    if (!camera) return;
    if (camera.stopping) return camera.stopping;
    camera.stopping = (async () => {
      elements.cameraButton.disabled = true;
      clearTimeout(camera.timer);
      if (camera.recorder.state !== "inactive") camera.recorder.stop();
      try {
        await camera.stopped;
      } finally {
        camera.stream.getTracks().forEach(track => track.stop());
        elements.cameraVideo.srcObject = null;
        state.camera = null;
        elements.cameraPreview.hidden = true;
        elements.cameraButton.setAttribute("aria-pressed", "false");
        elements.cameraButton.setAttribute("aria-label", "Start device video recording");
        elements.cameraButton.title = "Record device video";
        elements.cameraButton.disabled = false;
        elements.micButton.disabled = false;
        elements.callButton.disabled = false;
      }
      const container = String(camera.recorder.mimeType || "video/webm").split(";", 1)[0];
      const mime = container === "video/mp4" ? "video/mp4" : "video/webm";
      const blob = new Blob(camera.chunks, { type: mime });
      if (!blob.size) throw new Error("The device video contained no recorded data");
      const extension = mime === "video/mp4" ? "mp4" : "webm";
      const file = new File([blob], `device-video-${Date.now()}.${extension}`, { type: mime });
      await addFile(file, "video", "camera");
      setComposerStatus(`Video attached · ${((Date.now() - camera.started) / 1000).toFixed(1)} seconds`);
    })();
    return camera.stopping;
  }

  async function cameraFrameEnvelope() {
    if (!state.camera || !elements.cameraVideo.videoWidth) return null;
    const canvas = document.createElement("canvas");
    const scale = Math.min(1, 1280 / elements.cameraVideo.videoWidth);
    canvas.width = Math.max(1, Math.round(elements.cameraVideo.videoWidth * scale));
    canvas.height = Math.max(1, Math.round(elements.cameraVideo.videoHeight * scale));
    canvas.getContext("2d").drawImage(elements.cameraVideo, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise(resolve => canvas.toBlob(resolve, "image/jpeg", 0.84));
    if (!blob) return null;
    const dataUrl = await fileDataUrl(blob);
    return {
      mime_type: "image/jpeg",
      encoding: "base64",
      data: dataUrl.slice(dataUrl.indexOf(",") + 1),
    };
  }

  function callListeningStatus(call) {
    const label = state.camera ? "Video call live" : "Call live";
    return call.inflight
      ? `${label} · ${call.inflight} processing · keep speaking`
      : `${label} · listening`;
  }

  function supersedeCallAudio(call, sequence) {
    const result = callPlayback.supersedeBefore(call, sequence);
    for (const turn of result.turns) {
      if (turn.assistant) turn.assistant.node.classList.add("interrupted");
    }
    if (result.playbackInterrupted) stopCurrentPlayback();
    return result.turns.length;
  }

  function flushCallHistory(call) {
    while (call.completedHistory.has(call.nextHistorySequence)) {
      const item = call.completedHistory.get(call.nextHistorySequence);
      call.completedHistory.delete(call.nextHistorySequence);
      call.nextHistorySequence += 1;
      if (!item) continue;
      state.history.push({ role: "user", content: item.transcript });
      if (item.reply) state.history.push({ role: "assistant", content: item.reply });
    }
    scheduleBrowserSessionSave();
  }

  function completeCallTurn(call, turn) {
    call.controllers.delete(turn.controller);
    call.turns.delete(turn);
    call.inflight = Math.max(0, call.inflight - 1);
    if (call.playbackTurn === turn) call.playbackTurn = null;
    if (!turn.historyQueued) {
      call.completedHistory.set(turn.sequence, null);
      flushCallHistory(call);
    }
    if (state.call !== call) return;
    if (!call.vadActive && !call.playbackTurn) {
      setComposerStatus(callListeningStatus(call));
    }
  }

  async function submitCallUtterance(call, chunks, activeDurationMs) {
    const sampleCount = chunks.reduce((total, chunk) => total + chunk.length, 0);
    const capturedDurationMs = sampleCount / call.context.sampleRate * 1000;
    const confirmedDurationMs = Number.isFinite(activeDurationMs)
      ? activeDurationMs
      : capturedDurationMs;
    if (confirmedDurationMs < callVad.DEFAULTS.minActiveMs || state.call !== call) {
      return;
    }

    const turn = {
      sequence: call.nextSequence,
      controller: new AbortController(),
      discardReply: false,
      historyQueued: false,
      assistant: null,
    };
    supersedeCallAudio(call, turn.sequence);
    call.nextSequence += 1;
    call.inflight += 1;
    call.turns.add(turn);
    call.controllers.add(turn.controller);
    setComposerStatus(callListeningStatus(call));

    let envelope;
    let frame;
    try {
      envelope = await audioEnvelope(chunks, call.context.sampleRate);
      frame = await cameraFrameEnvelope();
    } catch (error) {
      completeCallTurn(call, turn);
      showError(error);
      return;
    }
    if (state.call !== call) {
      completeCallTurn(call, turn);
      return;
    }
    const message = {
      role: "user",
      content: frame
        ? (
          "The attached audio is the user's latest spoken turn and the attached "
          + "image is the current camera frame. Continue the conversation by "
          + "answering the user's spoken intent, using the frame only when relevant."
        )
        : (
          "The attached audio is the user's latest spoken turn. Continue the live "
          + "conversation by answering that turn directly."
        ),
      audios: [envelope],
    };
    if (frame) message.images = [frame];
    const callMessages = [
      { role: "system", content: LIVE_CALL_SYSTEM_PROMPT },
      ...state.history.slice(-12),
      message,
    ];
    const user = addMessage({ role: "user", content: frame ? "Video call message" : "Voice message" });
    const assistant = addMessage({ role: "assistant", content: "", streaming: true });
    turn.assistant = assistant;
    assistant.node.hidden = true;
    let streamedContent = "";
    let streamedThinking = "";
    const showThinking = reasoningEnabled();
    let pcmController = null;
    let streamedAudio = false;
    let inputTranscript = "";
    let inputAudioObservation = "";
    const callFallback = frame ? "Video call message" : "Voice message";
    const applyCallAudioEvidence = (transcriptValue, audioObservationValue) => {
      const transcript = String(transcriptValue || "").trim();
      const audioObservation = String(audioObservationValue || "").trim();
      if (transcript) inputTranscript = transcript;
      if (audioObservation) inputAudioObservation = audioObservation;
      updateMessage(user, {
        content: inputTranscript || inputAudioObservation || callFallback,
        audioObservation: inputTranscript ? inputAudioObservation : "",
        soundOnly: !inputTranscript && Boolean(inputAudioObservation),
        streaming: false,
      });
    };
    try {
      const data = await streamChat(
        {
          model: MODEL,
          messages: callMessages,
          omni: { schema: SCHEMA, task: "chat", include_audio_from_video: true },
          response_modalities: ["text", "audio"],
          speech_mode: "always",
          portal_voice: voicePayload(),
          think: showThinking,
          portal_auto_tools: false,
        },
        {
          signal: turn.controller.signal,
          onEvent: event => {
            if (event.type === "observation") {
              applyCallAudioEvidence(event.transcript, event.audio_observation);
            } else if (event.type === "delta") {
              const contentDelta = String((event.message || {}).content || "");
              const thinkingDelta = showThinking
                ? String((event.message || {}).thinking || "")
                : "";
              streamedContent += contentDelta;
              if (showThinking) {
                streamedThinking += thinkingDelta;
              }
              if (contentDelta || thinkingDelta) revealMessage(assistant);
              updateMessage(assistant, {
                content: streamedContent,
                thinking: streamedThinking,
                streaming: true,
              });
            } else if (event.type === "stage" && event.stage === "tts") {
              setComposerStatus("Call · preparing voice…");
            } else if (event.type === "audio_start") {
              if (!callPlayback.canStart(call, turn)) turn.discardReply = true;
              if (!turn.discardReply) {
                if (call.playbackTurn && call.playbackTurn !== turn) {
                  call.playbackTurn.discardReply = true;
                  if (call.playbackTurn.assistant) {
                    call.playbackTurn.assistant.node.classList.add("interrupted");
                  }
                  stopCurrentPlayback();
                }
                call.playbackTurn = turn;
                pcmController = beginPcmPlayback();
                streamedAudio = Boolean(pcmController);
                if (pcmController) assistant.playback = pcmController.promise;
                setComposerStatus("Call · voice ready…");
              }
            } else if (event.type === "audio_delta" && !turn.discardReply) {
              setComposerStatus("Call · streaming voice… keep speaking to interrupt");
              queuePcmPlayback(pcmController, String((event.audio || {}).data || ""));
            } else if (event.type === "audio_end") {
              endPcmPlayback(pcmController);
            }
          },
        },
      );
      const reply = data.message || {};
      if (!callPlayback.canStart(call, turn)) turn.discardReply = true;
      if (!(reply.audio && reply.audio.data)) throw new Error("Voice call reply contained no audio");
      applyCallAudioEvidence(
        (data.adapter || {}).input_transcript,
        (data.adapter || {}).audio_observation,
      );
      const historyContent = audioEvidenceHistory(
        inputTranscript,
        inputAudioObservation,
        callFallback,
      );
      revealMessage(assistant);
      updateMessage(assistant, {
        content: reply.content || "Spoken response",
        thinking: showThinking ? (reply.thinking || streamedThinking) : "",
        generationMetrics: generationMetricsFromResponse(data),
        audio: turn.discardReply ? null : reply.audio,
        streaming: false,
        autoplayAudio: !streamedAudio,
      });
      turn.historyQueued = true;
      call.completedHistory.set(turn.sequence, {
        frame: Boolean(frame),
        transcript: historyContent,
        reply: String(reply.content || ""),
      });
      flushCallHistory(call);
      if (turn.discardReply) {
        assistant.node.classList.add("interrupted");
      } else {
        call.playbackTurn = turn;
        setComposerStatus("Call · speaking… say something to interrupt");
        await assistant.playback;
      }
    } catch (error) {
      if (assistant.node.hidden) removeMessage(assistant);
      if (error.name !== "AbortError") showError(error);
    } finally {
      assistant.node.classList.remove("streaming");
      completeCallTurn(call, turn);
    }
  }

  async function startCall() {
    if (state.call) return;
    if (!state.token) throw new Error("Access token missing from this link");
    if (state.recording) throw new Error("Release the microphone before starting a call");
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error("Voice calls require this HTTPS page in a supported browser");
    }
    await unlockPlayback();
    setComposerStatus("Requesting microphone…");
    const camera = state.camera;
    const stream = camera ? camera.stream : await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
        video: false,
      });
    if (camera) {
      clearTimeout(camera.timer);
      if (camera.recorder.state === "recording") {
        camera.recorder.requestData();
        camera.recorder.pause();
      }
    }
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    const context = new AudioContext();
    await context.resume();
    const source = context.createMediaStreamSource(stream);
    const analyser = context.createAnalyser();
    analyser.fftSize = 1024;
    const processor = context.createScriptProcessor(4096, 1, 1);
    const sink = context.createGain();
    sink.gain.value = 0;
    const call = {
      stream,
      context,
      source,
      analyser,
      processor,
      sink,
      ownsStream: !camera,
      vad: callVad.createState(performance.now()),
      bargeVad: callVad.createState(performance.now(), BARGE_VAD_OPTIONS),
      vadActive: false,
      captureVad: null,
      inflight: 0,
      nextSequence: 0,
      nextHistorySequence: 0,
      completedHistory: new Map(),
      turns: new Set(),
      controllers: new Set(),
      playbackTurn: null,
      animationFrame: null,
    };
    processor.onaudioprocess = event => {
      if (state.call !== call) return;
      const now = performance.now();
      const samples = new Float32Array(event.inputBuffer.getChannelData(0));
      const detector = call.captureVad || (call.playbackTurn ? call.bargeVad : call.vad);
      const detection = callVad.processFrame(detector, {
        level: rms(samples),
        samples,
        now,
        frameMs: samples.length / call.context.sampleRate * 1000,
      });
      if (["candidate", "start", "active"].includes(detection.event)) {
        call.captureVad = detector;
      }
      if (detection.event === "start") {
        setVadActive(call, true);
        if (supersedeCallAudio(call, call.nextSequence)) {
          setComposerStatus("Call · interruption heard…");
        } else {
          setComposerStatus(call.inflight
            ? `Call · listening · ${call.inflight} processing`
            : "Call · listening to you…");
        }
      }
      if (detection.event === "rejected") {
        call.captureVad = null;
        setVadActive(call, false);
        return;
      }
      if (detection.event !== "utterance") return;
      call.captureVad = null;
      setVadActive(call, false);
      void submitCallUtterance(
        call,
        detection.utterance.chunks,
        detection.utterance.activeDurationMs,
      );
    };
    source.connect(analyser);
    source.connect(processor);
    processor.connect(sink);
    sink.connect(context.destination);
    state.call = call;
    setVadActive(call, false);
    elements.callButton.setAttribute("aria-pressed", "true");
    elements.callButton.setAttribute("aria-label", "End voice call");
    elements.callButton.title = "End voice call";
    elements.micButton.disabled = true;
    elements.cameraButton.disabled = true;
    elements.speak.setAttribute("aria-pressed", "true");
    elements.speak.setAttribute("aria-label", "Disable spoken replies");
    elements.speak.title = "Spoken replies on";
    elements.waveform.classList.add("calling");
    elements.waveform.hidden = false;
    elements.recordingTime.textContent = "LIVE";
    setComposerStatus(camera ? "Video call live · listening" : "Call live · listening");
    drawWaveform();
  }

  async function stopCall() {
    const call = state.call;
    if (!call) return;
    state.call = null;
    for (const controller of call.controllers) controller.abort();
    call.controllers.clear();
    cancelAnimationFrame(call.animationFrame);
    call.processor.disconnect();
    call.source.disconnect();
    call.sink.disconnect();
    if (call.ownsStream) call.stream.getTracks().forEach(track => track.stop());
    await call.context.close();
    stopCurrentPlayback();
    if (state.camera && state.camera.recorder.state === "paused") {
      state.camera.recorder.resume();
      state.camera.timer = window.setTimeout(() => {
        if (!state.call) stopCameraCapture().catch(showError);
      }, MAX_VIDEO_RECORD_MS);
    }
    elements.callButton.setAttribute("aria-pressed", "false");
    elements.callButton.setAttribute("aria-label", "Start voice call");
    elements.callButton.title = "Start voice call";
    elements.micButton.disabled = false;
    elements.cameraButton.disabled = false;
    elements.waveform.classList.remove("calling");
    elements.waveform.classList.remove("vad-active");
    elements.waveform.hidden = true;
    setComposerStatus("Voice call ended");
  }

  function requestPayload() {
    const typed = elements.prompt.value.trim();
    const audios = state.attachments.filter(item => item.kind === "audio");
    const images = state.attachments.filter(item => item.kind === "image");
    const videos = state.attachments.filter(item => item.kind === "video");
    const documents = state.attachments.filter(item => item.kind === "document");
    const hasMedia = state.attachments.length > 0;
    const audioOnly = audios.length > 0 && !images.length && !videos.length && !documents.length;
    if (!typed && !state.attachments.length) throw new Error("Enter a message or attach media");

    let task = "chat";
    let content = typed;
    if (!typed && audioOnly) {
      content = (
        "Listen to this audio, use both its speech and non-speech sounds as "
        + "evidence, then reply naturally and concisely."
      );
    } else if (!typed && documents.length) {
      content = (audios.length || images.length || videos.length)
        ? "Use the attached document and media together, then explain the relevant evidence."
        : "Summarize the attached document and identify its key details.";
    } else if (!typed && (images.length || videos.length)) {
      content = (
        "Respond naturally about the currently attached media. Describe the details "
        + "that matter to the ongoing conversation and invite or answer the most "
        + "relevant next point."
      );
    }

    const message = { role: "user", content };
    if (audios.length) message.audios = audios.map(item => ({ mime_type: "audio/wav", encoding: "base64", data: item.data }));
    if (images.length) message.images = images.map(item => ({ mime_type: item.mime, encoding: "base64", data: item.data }));
    if (videos.length) message.videos = videos.map(item => ({
      mime_type: item.mime,
      encoding: "base64",
      data: item.data,
      sampling: { fps: 1, max_frames: 24, include_audio: true },
    }));
    if (documents.length) message.documents = documents.map(item => ({
      name: item.name,
      mime_type: item.mime,
      encoding: "base64",
      data: item.data,
    }));

    const wantsSpeech = elements.speak.getAttribute("aria-pressed") === "true";
    const wantsThinking = reasoningEnabled();
    const messages = hasMedia
      ? [
        { role: "system", content: MEDIA_CONVERSATION_SYSTEM_PROMPT },
        ...state.history.slice(-12),
        message,
      ]
      : [...state.history.slice(-12), message];
    return {
      task,
      hasMedia,
      audioOnly,
      replaceUserWithTranscript: !typed && audioOnly,
      wantsSpeech,
      wantsThinking,
      display: typed || (audioOnly ? "Audio clip" : (documents.length ? "Attached document" : "Attached media")),
      message,
      payload: {
        model: MODEL,
        messages,
        omni: { schema: SCHEMA, task, include_audio_from_video: true },
        response_modalities: wantsSpeech ? ["text", "audio"] : ["text"],
        speech_mode: wantsSpeech ? "always" : "never",
        portal_voice: voicePayload(),
        think: wantsThinking,
        stream: false,
        portal_auto_tools: false,
      },
    };
  }

  function showError(error) {
    const message = error instanceof Error ? error.message : String(error);
    addMessage({ role: "assistant", content: message, error: true });
    setComposerStatus(message, true);
  }

  async function send() {
    if (!state.token) return showError(new Error("Access token missing from this link"));
    if (state.recording) await stopRecording();
    if (state.camera) await stopCameraCapture();
    await unlockPlayback();
    let built;
    try {
      built = requestPayload();
    } catch (error) {
      return showError(error);
    }

    const requestSequence = ++state.requestSequence;
    if (state.call) supersedeCallAudio(state.call, state.call.nextSequence);
    stopCurrentPlayback();

    const sentMedia = [...state.attachments];
    const user = addMessage({
      role: "user",
      content: built.display,
      media: sentMedia,
    });
    state.attachments = [];
    renderAttachments();
    elements.prompt.value = "";
    resizePrompt();
    elements.send.disabled = true;
    setComposerStatus(
      built.audioOnly
        ? "Transcribing audio…"
        : (built.wantsThinking ? "Reasoning…" : "Replying…"),
    );
    const assistant = addMessage({ role: "assistant", content: "", streaming: true });
    assistant.node.hidden = true;
    let streamedContent = "";
    let streamedThinking = "";
    let pcmController = null;
    let streamedAudio = false;
    let inputTranscript = "";
    let inputAudioObservation = "";
    const applyInputAudioEvidence = (transcriptValue, audioObservationValue) => {
      const transcript = String(transcriptValue || "").trim();
      const audioObservation = String(audioObservationValue || "").trim();
      if (transcript) inputTranscript = transcript;
      if (audioObservation) inputAudioObservation = audioObservation;
      if (built.replaceUserWithTranscript) {
        updateMessage(user, {
          content: inputTranscript || inputAudioObservation || built.display,
          audioObservation: inputTranscript ? inputAudioObservation : "",
          soundOnly: !inputTranscript && Boolean(inputAudioObservation),
          streaming: false,
        });
      } else if (inputAudioObservation) {
        updateMessage(user, {
          audioObservation: inputAudioObservation,
          soundOnly: false,
          streaming: false,
        });
      }
    };
    const requestController = new AbortController();
    if (state.requestController) state.requestController.abort();
    state.requestController = requestController;
    try {
      const data = await streamChat(built.payload, {
        signal: requestController.signal,
        onEvent: event => {
          if (requestSequence !== state.requestSequence) return;
          if (event.type === "observation") {
            applyInputAudioEvidence(event.transcript, event.audio_observation);
          } else if (event.type === "delta") {
            const contentDelta = String((event.message || {}).content || "");
            const thinkingDelta = built.wantsThinking
              ? String((event.message || {}).thinking || "")
              : "";
            streamedContent += contentDelta;
            if (built.wantsThinking) {
              streamedThinking += thinkingDelta;
            }
            if (contentDelta || thinkingDelta) revealMessage(assistant);
            updateMessage(assistant, {
              content: streamedContent,
              thinking: streamedThinking,
              streaming: true,
            });
          } else if (event.type === "stage") {
            const labels = {
              comprehension: "Understanding media…",
              language: built.wantsThinking ? "Reasoning…" : "Replying…",
              tts: "Preparing spoken reply…",
            };
            setComposerStatus(labels[event.stage] || "Working…");
          } else if (event.type === "audio_start") {
            pcmController = beginPcmPlayback();
            streamedAudio = Boolean(pcmController);
            if (pcmController) assistant.playback = pcmController.promise;
            setComposerStatus("Voice ready…");
          } else if (event.type === "audio_delta") {
            setComposerStatus("Streaming spoken reply…");
            queuePcmPlayback(pcmController, String((event.audio || {}).data || ""));
          } else if (event.type === "audio_end") {
            endPcmPlayback(pcmController);
          }
        },
      });
      if (requestSequence !== state.requestSequence) {
        removeMessage(assistant);
        return;
      }
      const reply = data.message || {};
      applyInputAudioEvidence(
        (data.adapter || {}).input_transcript,
        (data.adapter || {}).audio_observation,
      );
      if (built.wantsSpeech && !(reply.audio && reply.audio.data)) {
        throw new Error("Spoken replies are enabled, but TTS returned no audio");
      }
      revealMessage(assistant);
      updateMessage(assistant, {
        content: reply.content || (reply.audio ? "Spoken response" : "No response returned."),
        thinking: built.wantsThinking ? (reply.thinking || streamedThinking) : "",
        generationMetrics: generationMetricsFromResponse(data),
        audio: reply.audio,
        streaming: false,
        autoplayAudio: !streamedAudio,
      });
      if (built.task === "chat" || built.hasMedia) {
        state.history.push({
          role: "user",
          content: built.audioOnly
            ? audioEvidenceHistory(
              inputTranscript,
              inputAudioObservation,
              built.message.content,
            )
            : built.message.content,
        });
        if (reply.content) state.history.push({ role: "assistant", content: reply.content });
      }
      scheduleBrowserSessionSave();
      setComposerStatus(reply.audio ? "Text and spoken reply ready" : "Text reply ready");
    } catch (error) {
      removeMessage(assistant);
      if (error.name !== "AbortError") showError(error);
    } finally {
      if (state.requestController === requestController) state.requestController = null;
      assistant.node.classList.remove("streaming");
      elements.send.disabled = false;
      refreshStatus();
    }
  }

  function resizePrompt() {
    elements.prompt.style.height = "auto";
    elements.prompt.style.height = `${Math.min(elements.prompt.scrollHeight, 160)}px`;
  }

  function beginMicHold(event) {
    if (event.type === "pointerdown" && event.button !== 0) return;
    event.preventDefault();
    state.holdingMic = true;
    state.micHoldStartedAt = Date.now();
    state.discardMicRecording = false;
    if (event.pointerId !== undefined) elements.micButton.setPointerCapture(event.pointerId);
    startRecording().catch(error => {
      state.holdingMic = false;
      showError(error);
    });
  }

  function endMicHold(event) {
    event.preventDefault();
    const heldMs = Date.now() - state.micHoldStartedAt;
    const briefTap = heldMs < MIN_MIC_HOLD_MS;
    state.holdingMic = false;
    state.discardMicRecording = briefTap;
    if (briefTap) transientComposerStatus("Press and hold to record voice clip");
    if (state.recording) stopRecording({ discard: briefTap }).catch(showError);
  }

  async function closeVoiceDialog() {
    if (state.voice.recording) await stopVoiceReferenceRecording();
    setVoicePresetMenu(false);
    elements.voiceDialog.close();
    setComposerStatus(elements.voiceCloneEnabled.checked ? "Voice clone configured" : "Voice settings saved");
  }

  elements.voiceButton.addEventListener("click", () => {
    syncVoiceUi();
    elements.voiceDialog.showModal();
  });
  elements.voiceClose.addEventListener("click", () => closeVoiceDialog().catch(showError));
  elements.voiceDone.addEventListener("click", () => closeVoiceDialog().catch(showError));
  elements.voiceDialog.addEventListener("cancel", event => {
    if (!state.voice.recording) return;
    event.preventDefault();
    stopVoiceReferenceRecording().then(() => elements.voiceDialog.close()).catch(showError);
  });
  elements.voiceCloneToggle.addEventListener("click", () => {
    elements.voiceCloneEnabled.checked = !elements.voiceCloneEnabled.checked;
    syncVoiceUi();
  });
  elements.voicePresetToggle.addEventListener("click", () => {
    setVoicePresetMenu(!state.voice.presetMenuOpen);
  });
  elements.voicePresetOptions.addEventListener("click", event => {
    const button = event.target.closest("[data-voice-preset]");
    if (!button) return;
    state.voice.selectedPreset = button.dataset.voicePreset;
    clearVoiceReference();
    elements.voiceCloneEnabled.checked = true;
    setVoicePresetMenu(false);
  });
  elements.voiceTemperature.addEventListener("input", syncVoiceUi);
  elements.voiceTopP.addEventListener("input", syncVoiceUi);
  elements.voiceReferenceInput.addEventListener("change", event => {
    setVoiceReference(event.target.files[0]).catch(showError);
  });
  elements.voiceReferenceClear.addEventListener("click", clearVoiceReference);
  elements.voiceRecord.addEventListener("click", () => {
    const action = state.voice.recording
      ? stopVoiceReferenceRecording()
      : startVoiceReferenceRecording();
    action.catch(showError);
  });
  elements.voiceReset.addEventListener("click", () => {
    clearVoiceReference();
    applyVoiceDefaults();
  });

  elements.micButton.addEventListener("pointerdown", beginMicHold);
  elements.micButton.addEventListener("pointerup", endMicHold);
  elements.micButton.addEventListener("pointercancel", endMicHold);
  elements.micButton.addEventListener("keydown", event => {
    if (!event.repeat && (event.key === " " || event.key === "Enter")) beginMicHold(event);
  });
  elements.micButton.addEventListener("keyup", event => {
    if (event.key === " " || event.key === "Enter") endMicHold(event);
  });
  elements.mediaInput.addEventListener("change", event => {
    Promise.all([...event.target.files].map(file => addFile(file)))
      .then(() => setComposerStatus("Attachment ready"))
      .catch(showError);
    event.target.value = "";
  });
  elements.cameraButton.addEventListener("click", () => {
    const action = state.camera ? stopCameraCapture() : startCameraCapture();
    action.catch(showError);
  });
  elements.speak.addEventListener("click", () => {
    if (state.call) return;
    const enabled = elements.speak.getAttribute("aria-pressed") !== "true";
    elements.speak.setAttribute("aria-pressed", String(enabled));
    elements.speak.setAttribute("aria-label", `${enabled ? "Disable" : "Enable"} spoken replies`);
    elements.speak.title = `Spoken replies ${enabled ? "on" : "off"}`;
    setComposerStatus(enabled ? "Spoken replies on" : "Text replies only");
    if (enabled) unlockPlayback();
  });
  elements.think.addEventListener("click", () => {
    const enabled = elements.think.getAttribute("aria-pressed") !== "true";
    elements.think.setAttribute("aria-pressed", String(enabled));
    elements.think.setAttribute("aria-label", `${enabled ? "Disable" : "Enable"} reasoning`);
    elements.think.title = `Reasoning ${enabled ? "on" : "off"}`;
    setComposerStatus(enabled ? "Reasoning on" : "Reasoning off");
  });
  elements.callButton.addEventListener("click", () => {
    const action = state.call ? stopCall() : startCall();
    action.catch(showError);
  });
  elements.send.addEventListener("click", () => send().catch(showError));
  elements.prompt.addEventListener("input", () => {
    resizePrompt();
    scheduleBrowserSessionSave(700);
  });
  elements.conversation.addEventListener("pointerdown", beginConversationScrollGesture, { passive: true });
  elements.conversation.addEventListener("pointerup", endConversationScrollGesture, { passive: true });
  elements.conversation.addEventListener("pointercancel", endConversationScrollGesture, { passive: true });
  elements.conversation.addEventListener("wheel", beginConversationScrollGesture, { passive: true });
  elements.conversation.addEventListener("wheel", endConversationScrollGesture, { passive: true });
  elements.conversation.addEventListener("scroll", handleConversationScroll, { passive: true });
  elements.scrollLatest.addEventListener("click", resumeConversationAutoFollow);

  elements.prompt.addEventListener("keydown", event => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      send().catch(showError);
    }
  });
  elements.clear.addEventListener("click", () => {
    state.requestSequence += 1;
    if (state.requestController) state.requestController.abort();
    if (state.call) stopCall().catch(showError);
    stopCurrentPlayback();
    clearSessionDiagnostics();
    state.cacheSuppress = true;
    state.history = [];
    for (const item of state.attachments) {
      if (item.previewUrl) URL.revokeObjectURL(item.previewUrl);
    }
    state.attachments = [];
    renderAttachments();
    for (const message of [...state.messages]) removeMessage(message);
    state.messages = [];
    elements.prompt.value = "";
    resizePrompt();
    state.cacheDeleted = true;
    clearBrowserSessionCache();
    state.cacheSuppress = false;
    state.autoFollowConversation = true;
    scrollConversationToBottom({ smooth: false, force: true });
    setComposerStatus("Conversation cleared");
  });

  for (const eventName of ["gesturestart", "gesturechange", "gestureend"]) {
    document.addEventListener(eventName, event => event.preventDefault(), { passive: false });
  }

  window.addEventListener("beforeunload", () => {
    if (layoutResizeObserver) layoutResizeObserver.disconnect();
    if (state.camera) state.camera.stream.getTracks().forEach(track => track.stop());
    if (state.call) state.call.stream.getTracks().forEach(track => track.stop());
    if (state.recording) state.recording.stream.getTracks().forEach(track => track.stop());
    if (state.voice.recording) state.voice.recording.stream.getTracks().forEach(track => track.stop());
    if (state.voice.referenceUrl) URL.revokeObjectURL(state.voice.referenceUrl);
    for (const item of state.attachments) {
      if (item.previewUrl) URL.revokeObjectURL(item.previewUrl);
    }
  });
  window.addEventListener("pagehide", () => {
    persistBrowserSessionOnLeave();
    reportDiagnostic("page_leave");
  });

  async function initializePortal() {
    state.token = accessToken();
    applyVoiceDefaults();
    await restoreBrowserSession();
    if (!state.token) showError(new Error("This link is missing its access fragment"));
    refreshStatus();
    refreshActivity();
    setInterval(refreshActivity, 2_000);
    setInterval(refreshStatus, 15_000);
    setInterval(() => {
      if (state.cacheReady && !state.cacheDeleted && window.OmniSessionCache) {
        enqueueCacheOperation(() => window.OmniSessionCache.touch(state.cacheScope));
      }
    }, 30_000);
  }

  initializePortal().catch(showError);
})();
