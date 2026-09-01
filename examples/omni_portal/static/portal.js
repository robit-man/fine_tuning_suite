(() => {
  "use strict";

  const MODEL = document.body.dataset.model;
  const MAX_UPLOAD_MIB = Number(document.body.dataset.maxUploadMib || 96);
  const SCHEMA = "robit.ollama.omni-adapter.v1";
  const MAX_RECORD_MS = 60_000;
  const MAX_VIDEO_RECORD_MS = 30_000;
  const CALL_SILENCE_MS = 700;
  const CALL_MIN_SPEECH_MS = 250;
  const CALL_VOICE_THRESHOLD = 0.014;
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
    speak: document.getElementById("speak-toggle"),
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
    call: null,
    camera: null,
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

  function addMessage({ role, content, audio, error = false }) {
    const node = elements.template.content.firstElementChild.cloneNode(true);
    node.classList.add(role === "user" ? "user" : "assistant");
    if (error) node.classList.add("error");
    const contentNode = node.querySelector(".message-content");
    if (role === "assistant" && !error) renderMarkdown(contentNode, content || "");
    else contentNode.textContent = content || "";
    const playback = audio && audio.data ? attachAudioPlayer(node, audio) : Promise.resolve();
    elements.conversation.appendChild(node);
    elements.conversation.scrollTop = elements.conversation.scrollHeight;
    return { node, playback };
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

  async function playWithUnlockedContext(envelope, note) {
    const context = state.playbackContext;
    if (!context) return false;
    if (context.state === "suspended") await context.resume();
    if (context.state !== "running") return false;
    const encoded = base64ToBlob(envelope.data, envelope.mime_type);
    const decoded = await context.decodeAudioData(await encoded.arrayBuffer());
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

  function playHtmlAudio(audio, note) {
    return audio.play().then(() => new Promise(resolve => {
      const done = () => {
        note.textContent = "Spoken reply · replay with the player";
        resolve(true);
      };
      audio.addEventListener("ended", done, { once: true });
      audio.addEventListener("error", done, { once: true });
    }));
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
    audio.dataset.objectUrl = url;
    const note = document.createElement("p");
    note.className = "audio-note";
    note.textContent = "Spoken reply ready";
    box.append(audio, note);
    return playWithUnlockedContext(envelope, note)
      .then(playing => {
        if (!playing) return playHtmlAudio(audio, note);
        return playing;
      })
      .catch(() => {
        note.textContent = "Spoken reply ready · tap play";
        return playHtmlAudio(audio, note).catch(() => false);
      });
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
    elements.callButton.disabled = true;
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
      stopCameraCapture().catch(showError);
    }, MAX_VIDEO_RECORD_MS);
    setComposerStatus("Recording device video · tap camera or send to finish");
  }

  async function stopCameraCapture() {
    const camera = state.camera;
    if (!camera) return;
    if (camera.stopping) return camera.stopping;
    camera.stopping = (async () => {
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
        elements.micButton.disabled = false;
        elements.callButton.disabled = false;
      }
      const container = String(camera.recorder.mimeType || "video/webm").split(";", 1)[0];
      const mime = container === "video/mp4" ? "video/mp4" : "video/webm";
      const blob = new Blob(camera.chunks, { type: mime });
      if (!blob.size) throw new Error("The device video contained no recorded data");
      const extension = mime === "video/mp4" ? "mp4" : "webm";
      const file = new File([blob], `device-video-${Date.now()}.${extension}`, { type: mime });
      await addFile(file, "video");
      setComposerStatus(`Video attached · ${((Date.now() - camera.started) / 1000).toFixed(1)} seconds`);
    })();
    return camera.stopping;
  }

  async function submitCallUtterance(call, chunks) {
    const sampleCount = chunks.reduce((total, chunk) => total + chunk.length, 0);
    const durationMs = sampleCount / call.context.sampleRate * 1000;
    if (durationMs < CALL_MIN_SPEECH_MS || state.call !== call) {
      call.busy = false;
      return;
    }

    const envelope = await audioEnvelope(chunks, call.context.sampleRate);
    if (state.call !== call) return;
    const message = {
      role: "user",
      content: "Listen to this speech and reply naturally and concisely.",
      audios: [envelope],
    };
    const user = addMessage({ role: "user", content: "Voice message" });
    setComposerStatus("Call · understanding speech…");
    call.abortController = new AbortController();
    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: authHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify({
          model: MODEL,
          messages: [...state.history.slice(-12), message],
          omni: { schema: SCHEMA, task: "chat", include_audio_from_video: true },
          response_modalities: ["text", "audio"],
          speech_mode: "always",
          think: false,
          stream: false,
          options: { num_predict: 512 },
          portal_auto_tools: false,
        }),
        signal: call.abortController.signal,
      });
      const data = await response.json().catch(() => ({ error: `HTTP ${response.status}` }));
      if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
      const reply = data.message || {};
      if (!(reply.audio && reply.audio.data)) throw new Error("Voice call reply contained no audio");
      const transcript = String((data.adapter || {}).observation || "Voice message").trim();
      user.node.querySelector(".message-content").textContent = transcript;
      const assistant = addMessage({
        role: "assistant",
        content: reply.content || "Spoken response",
        audio: reply.audio,
      });
      state.history.push({ role: "user", content: transcript });
      if (reply.content) state.history.push({ role: "assistant", content: reply.content });
      setComposerStatus("Call · speaking…");
      await assistant.playback;
    } catch (error) {
      if (error.name !== "AbortError") showError(error);
    } finally {
      if (state.call === call) {
        call.abortController = null;
        window.setTimeout(() => {
          if (state.call === call) {
            call.busy = false;
            setComposerStatus("Call live · listening");
          }
        }, 250);
      }
    }
  }

  async function startCall() {
    if (state.call) return;
    if (!state.token) throw new Error("Access token missing from this link");
    if (state.recording) throw new Error("Release the microphone before starting a call");
    if (state.camera) throw new Error("Stop device video before starting a voice call");
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error("Voice calls require this HTTPS page in a supported browser");
    }
    await unlockPlayback();
    setComposerStatus("Requesting microphone…");
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
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
    const call = {
      stream,
      context,
      source,
      analyser,
      processor,
      sink,
      preRoll: [],
      chunks: [],
      speaking: false,
      lastVoiceAt: 0,
      busy: false,
      abortController: null,
      animationFrame: null,
    };
    processor.onaudioprocess = event => {
      if (state.call !== call || call.busy) return;
      const now = performance.now();
      const samples = new Float32Array(event.inputBuffer.getChannelData(0));
      const hasVoice = rms(samples) >= CALL_VOICE_THRESHOLD;
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
    setComposerStatus("Call live · listening");
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
    call.stream.getTracks().forEach(track => track.stop());
    await call.context.close();
    if (state.playbackSource) {
      try {
        state.playbackSource.stop();
      } catch (_error) {
        // Playback ended while the call was stopping.
      }
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
      sampling: { fps: 1, max_frames: 96, include_audio: true },
    }));

    const wantsSpeech = elements.speak.getAttribute("aria-pressed") === "true";
    const messages = task === "chat" ? [...state.history.slice(-12), message] : [message];
    return {
      task,
      wantsSpeech,
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
    if (state.camera) await stopCameraCapture();
    await unlockPlayback();
    let built;
    try {
      built = requestPayload();
    } catch (error) {
      return showError(error);
    }

    const mediaSummary = state.attachments.map(item => item.kind).join(" · ");
    addMessage({ role: "user", content: mediaSummary ? `${built.display}\n${mediaSummary}` : built.display });
    state.attachments = [];
    renderAttachments();
    elements.prompt.value = "";
    resizePrompt();
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
      if (built.wantsSpeech && !(reply.audio && reply.audio.data)) {
        throw new Error("Spoken replies are enabled, but TTS returned no audio");
      }
      addMessage({
        role: "assistant",
        content: reply.content || (reply.audio ? "Spoken response" : "No response returned."),
        audio: reply.audio,
      });
      if (built.task === "chat") {
        state.history.push({ role: "user", content: built.message.content });
        if (reply.content) state.history.push({ role: "assistant", content: reply.content });
      }
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
    state.attachments = [];
    renderAttachments();
    for (const message of [...elements.conversation.querySelectorAll(".message")].slice(1)) {
      for (const audio of message.querySelectorAll("audio[data-object-url]")) {
        URL.revokeObjectURL(audio.dataset.objectUrl);
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
  });

  state.token = accessToken();
  if (!state.token) showError(new Error("This link is missing its access fragment"));
  refreshStatus();
  setInterval(refreshStatus, 15_000);
})();
