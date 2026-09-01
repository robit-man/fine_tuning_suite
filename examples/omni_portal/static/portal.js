(() => {
  "use strict";

  const MODEL = document.body.dataset.model;
  const MAX_UPLOAD_MIB = Number(document.body.dataset.maxUploadMib || 96);
  const SCHEMA = "robit.ollama.omni-adapter.v1";
  const MAX_RECORD_MS = 60_000;
  const MAX_VOICE_REFERENCE_MS = 10_000;
  const MAX_VOICE_REFERENCE_BYTES = 10 * 1024 * 1024;
  const MAX_VIDEO_RECORD_MS = 30_000;
  const CALL_SILENCE_MS = 700;
  const CALL_MIN_SPEECH_MS = 250;
  const CALL_VOICE_THRESHOLD = 0.014;
  const CALL_BARGE_THRESHOLD = 0.035;
  const CALL_BARGE_MIN_MS = 300;
  const LIMITS = {
    audio: 30 * 1024 * 1024,
    image: 18 * 1024 * 1024,
    video: Math.min(68, Math.max(8, MAX_UPLOAD_MIB - 24)) * 1024 * 1024,
  };

  const elements = {
    headerStatus: document.getElementById("header-status"),
    statusText: document.getElementById("status-text"),
    conversation: document.getElementById("conversation"),
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
    recording: null,
    holdingMic: false,
    recordTimer: null,
    recordClock: null,
    playbackContext: null,
    playbackSource: null,
    playbackElement: null,
    playbackEpoch: 0,
    streamController: null,
    call: null,
    camera: null,
    voice: {
      initialized: false,
      serverReference: false,
      reference: null,
      referenceUrl: null,
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

  function setComposerStatus(text, error = false) {
    elements.composerStatus.textContent = text;
    elements.composerStatus.classList.toggle("error", error);
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
        && !/^(#{1,3})\s+|^```|^[-*]\s+|^\d+\.\s+|^>\s+/.test(lines[index])) {
        paragraphLines.push(lines[index]);
        index += 1;
      }
      const paragraph = document.createElement("p");
      appendInlineMarkdown(paragraph, paragraphLines.join("\n"));
      parent.appendChild(paragraph);
    }
  }

  function updateMessage(record, {
    content,
    thinking,
    audio,
    streaming = false,
    autoplayAudio = true,
  }) {
    const contentNode = record.node.querySelector(".message-content");
    if (content !== undefined) {
      if (record.role === "assistant" && !record.error) renderMarkdown(contentNode, content || "");
      else contentNode.textContent = content || "";
    }
    const thinkingBox = record.node.querySelector(".thinking-output");
    const thinkingNode = record.node.querySelector(".thinking-content");
    if (thinking !== undefined) {
      thinkingBox.hidden = !thinking;
      if (thinking) renderMarkdown(thinkingNode, thinking);
      else thinkingNode.replaceChildren();
    }
    record.node.classList.toggle("streaming", streaming);
    if (audio && audio.data && !record.node.querySelector(".audio-output audio")) {
      const playback = attachAudioPlayer(record.node, audio, autoplayAudio);
      if (autoplayAudio) record.playback = playback;
    }
    elements.conversation.scrollTop = elements.conversation.scrollHeight;
    return record;
  }

  function loopingVideo(item, { ownsUrl = false } = {}) {
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
    if (!videos.length) return;
    const gallery = document.createElement("div");
    gallery.className = "message-media";
    for (const item of videos) {
      const preview = document.createElement("div");
      preview.className = "message-video-preview";
      preview.appendChild(loopingVideo(item, { ownsUrl: true }));
      const badge = document.createElement("span");
      badge.textContent = "VIDEO · LOOP";
      preview.appendChild(badge);
      gallery.appendChild(preview);
    }
    node.appendChild(gallery);
  }

  function addMessage({ role, content, thinking, audio, media = [], error = false, streaming = false }) {
    const node = elements.template.content.firstElementChild.cloneNode(true);
    node.classList.add(role === "user" ? "user" : "assistant");
    if (error) node.classList.add("error");
    elements.conversation.appendChild(node);
    const record = { node, role, error, playback: Promise.resolve() };
    updateMessage(record, { content, thinking, audio, streaming });
    appendMessageMedia(node, media);
    elements.conversation.scrollTop = elements.conversation.scrollHeight;
    return record;
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

  async function playWithUnlockedContext(envelope, note, epoch) {
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
    note.textContent = "Playing spoken reply…";
    return new Promise(resolve => {
      source.addEventListener("ended", () => {
        if (state.playbackSource === source) state.playbackSource = null;
        note.textContent = "Spoken reply · replay with the player";
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

  function playHtmlAudio(audio, note, epoch) {
    if (state.playbackEpoch !== epoch) return Promise.resolve(false);
    state.playbackElement = audio;
    return audio.play().then(() => new Promise(resolve => {
      const done = () => {
        if (state.playbackElement === audio) state.playbackElement = null;
        note.textContent = "Spoken reply · replay with the player";
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
    const note = document.createElement("p");
    note.className = "audio-note";
    note.textContent = autoplay ? "Spoken reply ready" : "Streamed reply · replay with the player";
    box.append(audio, note);
    if (!autoplay) return Promise.resolve(false);
    const epoch = state.playbackEpoch;
    return playWithUnlockedContext(envelope, note, epoch)
      .then(playing => {
        if (state.playbackEpoch !== epoch) return false;
        if (!playing) return playHtmlAudio(audio, note, epoch);
        return playing;
      })
      .catch(() => {
        note.textContent = "Spoken reply ready · tap play";
        return playHtmlAudio(audio, note, epoch).catch(() => false);
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
      nextTime: context.currentTime + 0.06,
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
    source.buffer = buffer;
    source.connect(controller.context.destination);
    controller.sources.add(source);
    source.addEventListener("ended", () => {
      controller.sources.delete(source);
      maybeFinishPcmPlayback(controller);
    }, { once: true });
    const startAt = Math.max(controller.nextTime, controller.context.currentTime + 0.03);
    source.start(startAt);
    controller.nextTime = startAt + buffer.duration;
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
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: authHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify({ ...payload, stream: true }),
      signal,
    });
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
      if (!state.voice.initialized && data.voice_profile) {
        const profile = data.voice_profile;
        state.voice.serverReference = Boolean(profile.speaker_reference);
        state.voice.defaults = {
          language: profile.language || "en",
          temperature: Number(profile.temperature ?? 0.7),
          topK: Number(profile.top_k ?? 40),
          topP: Number(profile.top_p ?? 0.9),
          seed: Number(profile.seed ?? 42),
          maxFrames: Number(profile.max_frames ?? 512),
        };
        applyVoiceDefaults();
        state.voice.initialized = true;
      }
    } catch (_error) {
      elements.headerStatus.className = "connection offline";
      elements.statusText.textContent = "Offline";
    }
  }

  function humanBytes(bytes) {
    if (bytes < 1024 * 1024) return `${Math.max(1, Math.ceil(bytes / 1024))} KiB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
  }

  function inferKind(file) {
    const mime = (file.type || "").toLowerCase();
    if (mime === "audio/wav" || file.name.toLowerCase().endsWith(".wav")) return "audio";
    if (mime.startsWith("image/")) return "image";
    if (mime === "video/mp4" || mime === "video/webm") return "video";
    throw new Error("Choose a WAV, JPEG, PNG, WebP, MP4, or WebM file");
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
        preview.textContent = item.kind === "audio" ? "WAV" : "VID";
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
      mime: kind === "audio" ? "audio/wav" : file.type,
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
    context.strokeStyle = state.call ? "#86efac" : "#fb7185";
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
    state.recording = { stream, context, source, analyser, processor, sink, chunks, started: Date.now(), animationFrame: null };
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
    if (!state.holdingMic) await stopRecording();
  }

  async function stopRecording() {
    const recording = state.recording;
    if (!recording) return;
    clearTimeout(state.recordTimer);
    clearInterval(state.recordClock);
    cancelAnimationFrame(recording.animationFrame);
    recording.processor.disconnect();
    recording.source.disconnect();
    recording.sink.disconnect();
    recording.stream.getTracks().forEach(track => track.stop());
    const samples = downsample(mergeSamples(recording.chunks), recording.context.sampleRate);
    await recording.context.close();
    state.recording = null;
    elements.waveform.hidden = true;
    elements.micButton.classList.remove("recording");
    elements.cameraButton.disabled = false;
    elements.micButton.setAttribute("aria-label", "Hold to record microphone");
    if (!samples.length) throw new Error("The microphone clip contained no samples");
    const blob = pcmWav(samples);
    const file = new File([blob], `microphone-${Date.now()}.wav`, { type: "audio/wav" });
    await addFile(file, "audio");
    setComposerStatus(`Audio attached · ${(samples.length / 16000).toFixed(1)} seconds`);
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
    elements.voiceReferenceControls.classList.toggle("disabled", !enabled);
    elements.voiceTemperatureValue.value = Number(elements.voiceTemperature.value).toFixed(2);
    elements.voiceTopPValue.value = Number(elements.voiceTopP.value).toFixed(2);
    if (state.voice.reference) {
      elements.voiceReferenceStatus.textContent = `${state.voice.reference.name} · ${humanBytes(state.voice.reference.bytes)}`;
    } else if (state.voice.serverReference) {
      elements.voiceReferenceStatus.textContent = "Using server profile reference";
    } else {
      elements.voiceReferenceStatus.textContent = "No reference selected";
    }
  }

  function applyVoiceDefaults(profile = state.voice.defaults) {
    elements.voiceLanguage.value = profile.language || "en";
    elements.voiceTemperature.value = String(profile.temperature ?? 0.7);
    elements.voiceTopK.value = String(profile.topK ?? 40);
    elements.voiceTopP.value = String(profile.topP ?? 0.9);
    elements.voiceSeed.value = String(profile.seed ?? 42);
    elements.voiceMaxFrames.value = String(profile.maxFrames ?? 512);
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

  function startPendingCallTurn(call) {
    if (state.call !== call || !call.pendingUtterance || !call.replyComplete) return;
    const pending = call.pendingUtterance;
    call.pendingUtterance = null;
    call.replyComplete = false;
    call.discardReply = false;
    call.busy = true;
    void submitCallUtterance(call, pending);
  }

  function completeCallTurn(call) {
    if (state.call !== call) return;
    call.abortController = null;
    call.replyComplete = true;
    if (call.pendingUtterance) {
      startPendingCallTurn(call);
    } else if (call.bargeActive) {
      setComposerStatus("Call · listening to interruption…");
    } else {
      call.replyComplete = false;
      call.discardReply = false;
      call.busy = false;
      setComposerStatus(state.camera ? "Video call live · listening" : "Call live · listening");
    }
  }

  async function submitCallUtterance(call, chunks) {
    const sampleCount = chunks.reduce((total, chunk) => total + chunk.length, 0);
    const durationMs = sampleCount / call.context.sampleRate * 1000;
    if (durationMs < CALL_MIN_SPEECH_MS || state.call !== call) {
      call.replyComplete = true;
      completeCallTurn(call);
      return;
    }

    const envelope = await audioEnvelope(chunks, call.context.sampleRate);
    if (state.call !== call) return;
    const frame = await cameraFrameEnvelope();
    const message = {
      role: "user",
      content: frame
        ? "Use the current camera view and this speech, then reply naturally and concisely."
        : "Listen to this speech and reply naturally and concisely.",
      audios: [envelope],
    };
    if (frame) message.images = [frame];
    const user = addMessage({ role: "user", content: frame ? "Video call message" : "Voice message" });
    const assistant = addMessage({ role: "assistant", content: "", streaming: true });
    assistant.node.hidden = true;
    let streamedContent = "";
    let streamedThinking = "";
    const showThinking = reasoningEnabled();
    let pcmController = null;
    let streamedAudio = false;
    setComposerStatus(frame ? "Video call · understanding…" : "Call · understanding speech…");
    call.abortController = new AbortController();
    try {
      const data = await streamChat(
        {
          model: MODEL,
          messages: [...state.history.slice(-12), message],
          omni: { schema: SCHEMA, task: "chat", include_audio_from_video: true },
          response_modalities: ["text", "audio"],
          speech_mode: "always",
          portal_voice: voicePayload(),
          think: showThinking,
          options: { num_predict: 512 },
          portal_auto_tools: false,
        },
        {
          signal: call.abortController.signal,
          onEvent: event => {
            if (event.type === "observation" && event.transcript) {
              user.node.querySelector(".message-content").textContent = String(event.transcript);
            } else if (event.type === "delta") {
              const contentDelta = String((event.message || {}).content || "");
              const thinkingDelta = showThinking
                ? String((event.message || {}).thinking || "")
                : "";
              streamedContent += contentDelta;
              if (showThinking) {
                streamedThinking += thinkingDelta;
              }
              if (contentDelta || thinkingDelta) assistant.node.hidden = false;
              updateMessage(assistant, {
                content: streamedContent,
                thinking: streamedThinking,
                streaming: true,
              });
            } else if (event.type === "stage" && event.stage === "tts") {
              setComposerStatus("Call · preparing voice…");
            } else if (event.type === "audio_start") {
              pcmController = beginPcmPlayback();
              streamedAudio = Boolean(pcmController);
              if (pcmController) assistant.playback = pcmController.promise;
              setComposerStatus("Call · streaming voice… say something to interrupt");
            } else if (event.type === "audio_delta" && !call.discardReply) {
              queuePcmPlayback(pcmController, String((event.audio || {}).data || ""));
            } else if (event.type === "audio_end") {
              endPcmPlayback(pcmController);
            }
          },
        },
      );
      const reply = data.message || {};
      if (!(reply.audio && reply.audio.data)) throw new Error("Voice call reply contained no audio");
      const transcript = String(
        (data.adapter || {}).input_transcript
        || (frame ? "Video call message" : "Voice message"),
      ).trim();
      user.node.querySelector(".message-content").textContent = transcript;
      assistant.node.hidden = false;
      updateMessage(assistant, {
        content: reply.content || "Spoken response",
        thinking: showThinking ? (reply.thinking || streamedThinking) : "",
        audio: call.discardReply ? null : reply.audio,
        streaming: false,
        autoplayAudio: !streamedAudio,
      });
      state.history.push({ role: "user", content: transcript });
      if (reply.content) state.history.push({ role: "assistant", content: reply.content });
      if (call.discardReply) {
        assistant.node.classList.add("interrupted");
      } else {
        setComposerStatus("Call · speaking… say something to interrupt");
        await assistant.playback;
      }
    } catch (error) {
      if (assistant.node.hidden) assistant.node.remove();
      if (error.name !== "AbortError") showError(error);
    } finally {
      assistant.node.classList.remove("streaming");
      completeCallTurn(call);
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
      preRoll: [],
      chunks: [],
      speaking: false,
      lastVoiceAt: 0,
      busy: false,
      bargeActive: false,
      bargeCandidate: [],
      bargeStartedAt: 0,
      pendingUtterance: null,
      replyComplete: false,
      discardReply: false,
      abortController: null,
      animationFrame: null,
    };
    processor.onaudioprocess = event => {
      if (state.call !== call) return;
      const now = performance.now();
      const samples = new Float32Array(event.inputBuffer.getChannelData(0));
      const hasVoice = rms(samples) >= CALL_VOICE_THRESHOLD;
      if (call.busy) {
        const isBargeVoice = rms(samples) >= CALL_BARGE_THRESHOLD;
        if (!call.bargeActive) {
          if (!isBargeVoice) {
            call.bargeCandidate = [];
            call.bargeStartedAt = 0;
            return;
          }
          if (!call.bargeStartedAt) call.bargeStartedAt = now;
          call.bargeCandidate.push(samples);
          if (now - call.bargeStartedAt < CALL_BARGE_MIN_MS) return;
          call.bargeActive = true;
          call.chunks = call.bargeCandidate.splice(0);
          call.lastVoiceAt = now;
          call.discardReply = true;
          stopCurrentPlayback();
          setComposerStatus("Call · interruption heard…");
          return;
        }
        call.chunks.push(samples);
        if (isBargeVoice) call.lastVoiceAt = now;
        if (now - call.lastVoiceAt < CALL_SILENCE_MS) return;
        call.pendingUtterance = call.chunks.splice(0);
        call.bargeActive = false;
        call.bargeCandidate = [];
        call.bargeStartedAt = 0;
        setComposerStatus("Call · interruption queued…");
        startPendingCallTurn(call);
        return;
      }
      if (!call.speaking) {
        call.preRoll.push(samples);
        if (call.preRoll.length > 4) call.preRoll.shift();
        if (hasVoice) {
          call.speaking = true;
          call.chunks = call.preRoll.splice(0);
          call.lastVoiceAt = now;
          setComposerStatus("Call · listening to you…");
        }
        return;
      }
      call.chunks.push(samples);
      if (hasVoice) call.lastVoiceAt = now;
      if (now - call.lastVoiceAt < CALL_SILENCE_MS) return;
      const utterance = call.chunks.splice(0);
      call.preRoll = [];
      call.speaking = false;
      call.busy = true;
      void submitCallUtterance(call, utterance);
    };
    source.connect(analyser);
    source.connect(processor);
    processor.connect(sink);
    sink.connect(context.destination);
    state.call = call;
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
    if (call.abortController) call.abortController.abort();
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
    elements.waveform.hidden = true;
    setComposerStatus("Voice call ended");
  }

  function requestPayload() {
    const typed = elements.prompt.value.trim();
    const audios = state.attachments.filter(item => item.kind === "audio");
    const images = state.attachments.filter(item => item.kind === "image");
    const videos = state.attachments.filter(item => item.kind === "video");
    if (!typed && !state.attachments.length) throw new Error("Enter a message or attach media");

    let task = "chat";
    let content = typed;
    if (!typed && audios.length && !images.length && !videos.length) {
      task = "transcribe";
      content = "Transcribe this audio faithfully.";
    } else if (!typed && (images.length || videos.length)) {
      task = "describe";
      content = "Describe this media accurately.";
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

    const wantsSpeech = elements.speak.getAttribute("aria-pressed") === "true";
    const wantsThinking = reasoningEnabled();
    const messages = task === "chat" ? [...state.history.slice(-12), message] : [message];
    return {
      task,
      wantsSpeech,
      wantsThinking,
      display: typed || (task === "transcribe" ? "Audio clip" : "Attached media"),
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
        options: { num_predict: 2048 },
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

    const sentMedia = [...state.attachments];
    const mediaSummary = sentMedia.map(item => item.kind).join(" · ");
    addMessage({
      role: "user",
      content: mediaSummary ? `${built.display}\n${mediaSummary}` : built.display,
      media: sentMedia,
    });
    state.attachments = [];
    renderAttachments();
    elements.prompt.value = "";
    resizePrompt();
    elements.send.disabled = true;
    setComposerStatus(
      built.task === "transcribe"
        ? "Transcribing audio…"
        : (built.wantsThinking ? "Reasoning…" : "Replying…"),
    );
    const assistant = addMessage({ role: "assistant", content: "", streaming: true });
    assistant.node.hidden = true;
    let streamedContent = "";
    let streamedThinking = "";
    let pcmController = null;
    let streamedAudio = false;
    try {
      const data = await streamChat(built.payload, {
        onEvent: event => {
          if (event.type === "delta") {
            const contentDelta = String((event.message || {}).content || "");
            const thinkingDelta = built.wantsThinking
              ? String((event.message || {}).thinking || "")
              : "";
            streamedContent += contentDelta;
            if (built.wantsThinking) {
              streamedThinking += thinkingDelta;
            }
            if (contentDelta || thinkingDelta) assistant.node.hidden = false;
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
            setComposerStatus("Streaming spoken reply…");
          } else if (event.type === "audio_delta") {
            queuePcmPlayback(pcmController, String((event.audio || {}).data || ""));
          } else if (event.type === "audio_end") {
            endPcmPlayback(pcmController);
          }
        },
      });
      const reply = data.message || {};
      if (built.wantsSpeech && !(reply.audio && reply.audio.data)) {
        throw new Error("Spoken replies are enabled, but TTS returned no audio");
      }
      assistant.node.hidden = false;
      updateMessage(assistant, {
        content: reply.content || (reply.audio ? "Spoken response" : "No response returned."),
        thinking: built.wantsThinking ? (reply.thinking || streamedThinking) : "",
        audio: reply.audio,
        streaming: false,
        autoplayAudio: !streamedAudio,
      });
      if (built.task === "chat") {
        state.history.push({ role: "user", content: built.message.content });
        if (reply.content) state.history.push({ role: "assistant", content: reply.content });
      }
      setComposerStatus(reply.audio ? "Text and spoken reply ready" : "Text reply ready");
    } catch (error) {
      assistant.node.remove();
      showError(error);
    } finally {
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
    if (event.pointerId !== undefined) elements.micButton.setPointerCapture(event.pointerId);
    startRecording().catch(error => {
      state.holdingMic = false;
      showError(error);
    });
  }

  function endMicHold(event) {
    event.preventDefault();
    state.holdingMic = false;
    if (state.recording) stopRecording().catch(showError);
  }

  async function closeVoiceDialog() {
    if (state.voice.recording) await stopVoiceReferenceRecording();
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
  elements.voiceCloneEnabled.addEventListener("change", syncVoiceUi);
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
    addFile(event.target.files[0]).then(() => setComposerStatus("Media attached")).catch(showError);
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
  elements.prompt.addEventListener("input", resizePrompt);
  elements.prompt.addEventListener("keydown", event => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      send().catch(showError);
    }
  });
  elements.clear.addEventListener("click", () => {
    state.history = [];
    for (const item of state.attachments) {
      if (item.previewUrl) URL.revokeObjectURL(item.previewUrl);
    }
    state.attachments = [];
    renderAttachments();
    for (const message of [...elements.conversation.querySelectorAll(".message")].slice(1)) {
      for (const media of message.querySelectorAll("[data-object-url]")) {
        URL.revokeObjectURL(media.dataset.objectUrl);
      }
      message.remove();
    }
    setComposerStatus("Conversation cleared");
  });

  for (const eventName of ["gesturestart", "gesturechange", "gestureend"]) {
    document.addEventListener(eventName, event => event.preventDefault(), { passive: false });
  }

  window.addEventListener("beforeunload", () => {
    if (state.camera) state.camera.stream.getTracks().forEach(track => track.stop());
    if (state.call) state.call.stream.getTracks().forEach(track => track.stop());
    if (state.recording) state.recording.stream.getTracks().forEach(track => track.stop());
    if (state.voice.recording) state.voice.recording.stream.getTracks().forEach(track => track.stop());
    if (state.voice.referenceUrl) URL.revokeObjectURL(state.voice.referenceUrl);
    for (const item of state.attachments) {
      if (item.previewUrl) URL.revokeObjectURL(item.previewUrl);
    }
  });

  state.token = accessToken();
  applyVoiceDefaults();
  if (!state.token) showError(new Error("This link is missing its access fragment"));
  refreshStatus();
  setInterval(refreshStatus, 15_000);
})();
