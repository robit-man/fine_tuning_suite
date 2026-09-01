(() => {
  "use strict";

  const MODEL = document.body.dataset.model;
  const MAX_UPLOAD_MIB = Number(document.body.dataset.maxUploadMib || 96);
  const SCHEMA = "robit.ollama.omni-adapter.v1";
  const MAX_RECORD_MS = 60_000;
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
    speak: document.getElementById("speak-toggle"),
    send: document.getElementById("send-button"),
    composerStatus: document.getElementById("composer-status"),
    clear: document.getElementById("clear-button"),
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

  function addMessage({ role, content, audio, error = false }) {
    const node = elements.template.content.firstElementChild.cloneNode(true);
    node.classList.add(role === "user" ? "user" : "assistant");
    if (error) node.classList.add("error");
    node.querySelector(".message-content").textContent = content || "";
    if (audio && audio.data) attachAudioPlayer(node, audio);
    elements.conversation.appendChild(node);
    elements.conversation.scrollTop = elements.conversation.scrollHeight;
    return node;
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

  function attachAudioPlayer(node, envelope) {
    const box = node.querySelector(".audio-output");
    box.hidden = false;
    const audio = document.createElement("audio");
    audio.controls = true;
    audio.playsInline = true;
    audio.preload = "metadata";
    const url = URL.createObjectURL(base64ToBlob(envelope.data, envelope.mime_type));
    audio.src = url;
    audio.addEventListener("ended", () => URL.revokeObjectURL(url), { once: true });
    const note = document.createElement("p");
    note.className = "audio-note";
    note.textContent = "Spoken reply · tap play if autoplay is blocked";
    box.append(audio, note);
    audio.play().catch(() => { note.textContent = "Spoken reply ready · tap play"; });
  }

  async function unlockPlayback() {
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    if (!AudioContext) return;
    if (!state.playbackContext) state.playbackContext = new AudioContext();
    if (state.playbackContext.state === "suspended") await state.playbackContext.resume().catch(() => {});
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

  async function addFile(file, forcedKind = null) {
    if (!file) return;
    const kind = forcedKind || inferKind(file);
    if (file.size > LIMITS[kind]) {
      throw new Error(`${kind} is ${humanBytes(file.size)}; limit is ${humanBytes(LIMITS[kind])}`);
    }
    const dataUrl = await fileDataUrl(file);
    const comma = dataUrl.indexOf(",");
    if (comma < 0) throw new Error("File could not be encoded");
    state.attachments.push({
      kind,
      name: file.name || `microphone-${Date.now()}.wav`,
      mime: kind === "audio" ? "audio/wav" : file.type,
      data: dataUrl.slice(comma + 1),
      bytes: file.size,
    });
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
    const recording = state.recording;
    if (!recording) return;
    const canvas = elements.waveformCanvas;
    const rect = canvas.getBoundingClientRect();
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.floor(rect.width * ratio));
    canvas.height = Math.max(1, Math.floor(rect.height * ratio));
    const context = canvas.getContext("2d");
    context.scale(ratio, ratio);
    const samples = new Uint8Array(recording.analyser.fftSize);
    recording.analyser.getByteTimeDomainData(samples);
    context.clearRect(0, 0, rect.width, rect.height);
    context.strokeStyle = "#fb7185";
    context.lineWidth = 1.5;
    context.beginPath();
    for (let index = 0; index < samples.length; index += 1) {
      const x = index * rect.width / (samples.length - 1);
      const y = samples[index] / 255 * rect.height;
      if (index === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    }
    context.stroke();
    recording.animationFrame = requestAnimationFrame(drawWaveform);
  }

  function updateRecordClock() {
    if (!state.recording) return;
    const seconds = Math.floor((Date.now() - state.recording.started) / 1000);
    const minutes = Math.floor(seconds / 60);
    elements.recordingTime.textContent = `${minutes}:${String(seconds % 60).padStart(2, "0")}`;
  }

  async function startRecording() {
    if (state.recording) return;
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
    elements.micButton.setAttribute("aria-label", "Hold to record microphone");
    if (!samples.length) throw new Error("The microphone clip contained no samples");
    const blob = pcmWav(samples);
    const file = new File([blob], `microphone-${Date.now()}.wav`, { type: "audio/wav" });
    await addFile(file, "audio");
    setComposerStatus(`Audio attached · ${(samples.length / 16000).toFixed(1)} seconds`);
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
      sampling: { fps: 1, max_frames: 96, include_audio: true },
    }));

    const wantsSpeech = elements.speak.getAttribute("aria-pressed") === "true";
    const messages = task === "chat" ? [...state.history.slice(-12), message] : [message];
    return {
      task,
      display: typed || (task === "transcribe" ? "Audio clip" : "Attached media"),
      message,
      payload: {
        model: MODEL,
        messages,
        omni: { schema: SCHEMA, task, include_audio_from_video: true },
        response_modalities: wantsSpeech ? ["text", "audio"] : ["text"],
        speech_mode: wantsSpeech ? "always" : "never",
        think: false,
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
    await unlockPlayback();
    let built;
    try {
      built = requestPayload();
    } catch (error) {
      return showError(error);
    }

    const mediaSummary = state.attachments.map(item => item.kind).join(" · ");
    addMessage({ role: "user", content: mediaSummary ? `${built.display}\n${mediaSummary}` : built.display });
    elements.send.disabled = true;
    setComposerStatus(built.task === "transcribe" ? "Transcribing audio…" : "Thinking…");
    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: authHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify(built.payload),
      });
      const data = await response.json().catch(() => ({ error: `HTTP ${response.status}` }));
      if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
      const reply = data.message || {};
      addMessage({
        role: "assistant",
        content: reply.content || (reply.audio ? "Spoken response" : "No response returned."),
        audio: reply.audio,
      });
      if (built.task === "chat") {
        state.history.push({ role: "user", content: built.message.content });
        if (reply.content) state.history.push({ role: "assistant", content: reply.content });
      }
      state.attachments = [];
      renderAttachments();
      elements.prompt.value = "";
      resizePrompt();
      setComposerStatus(reply.audio ? "Text and spoken reply ready" : "Text reply ready");
    } catch (error) {
      showError(error);
    } finally {
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
  elements.speak.addEventListener("click", () => {
    const enabled = elements.speak.getAttribute("aria-pressed") !== "true";
    elements.speak.setAttribute("aria-pressed", String(enabled));
    elements.speak.setAttribute("aria-label", `${enabled ? "Disable" : "Enable"} spoken replies`);
    elements.speak.title = `Spoken replies ${enabled ? "on" : "off"}`;
    setComposerStatus(enabled ? "Spoken replies on" : "Text replies only");
    if (enabled) unlockPlayback();
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
    state.attachments = [];
    renderAttachments();
    for (const message of [...elements.conversation.querySelectorAll(".message")].slice(1)) message.remove();
    setComposerStatus("Conversation cleared");
  });

  state.token = accessToken();
  if (!state.token) showError(new Error("This link is missing its access fragment"));
  refreshStatus();
  setInterval(refreshStatus, 15_000);
})();
