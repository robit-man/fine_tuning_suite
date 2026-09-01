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
  const SAFE_TOOLS = [
    {
      type: "function",
      function: {
        name: "get_current_time",
        description: "Return the current date, local time, and UTC offset from the portal host.",
        parameters: { type: "object", properties: {} },
      },
    },
    {
      type: "function",
      function: {
        name: "get_portal_capabilities",
        description: "Return the input, output, and task capabilities exposed by this portal.",
        parameters: { type: "object", properties: {} },
      },
    },
  ];

  const elements = {
    headerStatus: document.getElementById("header-status"),
    statusText: document.getElementById("status-text"),
    stageList: document.getElementById("stage-list"),
    routeReadout: document.getElementById("route-readout"),
    conversation: document.getElementById("conversation"),
    template: document.getElementById("message-template"),
    prompt: document.getElementById("prompt"),
    promptLabel: document.getElementById("prompt-label"),
    captureRow: document.querySelector(".capture-row"),
    attachments: document.getElementById("attachments"),
    micButton: document.getElementById("mic-button"),
    micLabel: document.getElementById("mic-label"),
    audioInput: document.getElementById("audio-input"),
    imageInput: document.getElementById("image-input"),
    videoInput: document.getElementById("video-input"),
    think: document.getElementById("think-toggle"),
    speak: document.getElementById("speak-toggle"),
    tools: document.getElementById("tools-toggle"),
    videoAudio: document.getElementById("video-audio-toggle"),
    send: document.getElementById("send-button"),
    composerStatus: document.getElementById("composer-status"),
    clear: document.getElementById("clear-button"),
    modes: [...document.querySelectorAll(".mode-tab")],
  };

  const state = {
    mode: "chat",
    token: "",
    attachments: [],
    history: [],
    recording: null,
    recordTimer: null,
    playbackContext: null,
  };

  function accessToken() {
    const fragment = new URLSearchParams(location.hash.replace(/^#/, ""));
    const supplied = fragment.get("access");
    if (supplied) sessionStorage.setItem("robit_omni_access", supplied);
    return supplied || sessionStorage.getItem("robit_omni_access") || "";
  }

  function authHeaders(extra = {}) {
    return { Authorization: `Bearer ${state.token}`, ...extra };
  }

  function setComposerStatus(text, error = false) {
    elements.composerStatus.textContent = text;
    elements.composerStatus.classList.toggle("error", error);
  }

  function timestamp() {
    return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }

  function addMessage({ role, content, thinking, trace, tools, audio, error = false }) {
    const node = elements.template.content.firstElementChild.cloneNode(true);
    node.classList.add(role === "user" ? "user" : "assistant");
    if (error) node.classList.add("error");
    node.querySelector(".message-meta span").textContent = error ? "ERROR" : role.toUpperCase();
    node.querySelector(".message-meta time").textContent = timestamp();
    node.querySelector(".message-content").textContent = content || "";

    if (thinking) {
      const details = node.querySelector(".thinking");
      details.hidden = false;
      details.querySelector("pre").textContent = thinking;
    }
    if (trace) {
      const details = node.querySelector(".adapter-trace");
      details.hidden = false;
      details.querySelector("pre").textContent = JSON.stringify(trace, null, 2);
    }
    if (tools && tools.length) {
      const box = node.querySelector(".tool-calls");
      box.hidden = false;
      for (const tool of tools) {
        const chip = document.createElement("div");
        chip.className = "tool-chip";
        chip.textContent = `${tool.name || "tool"} // ${tool.result || JSON.stringify(tool)}`;
        box.appendChild(chip);
      }
    }
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
      for (let i = 0; i < slice.length; i += 1) bytes[i] = slice.charCodeAt(i);
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
    note.textContent = `${envelope.sample_rate_hz || 24000} Hz · mono PCM16 · tap play if autoplay is blocked`;
    box.append(audio, note);
    audio.play().catch(() => { note.textContent = "Audio ready · tap play (mobile autoplay policy)"; });
  }

  async function unlockPlayback() {
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    if (!AudioContext) return;
    if (!state.playbackContext) state.playbackContext = new AudioContext();
    if (state.playbackContext.state === "suspended") {
      await state.playbackContext.resume().catch(() => {});
    }
  }

  async function refreshStatus() {
    if (!state.token) {
      elements.headerStatus.className = "header-status offline";
      elements.statusText.textContent = "Access token missing";
      return;
    }
    try {
      const response = await fetch("/api/status", { headers: authHeaders() });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      elements.headerStatus.className = `header-status ${data.ok ? "online" : "offline"}`;
      elements.statusText.textContent = data.ok ? "All systems nominal" : "Stage unavailable";
      for (const item of elements.stageList.querySelectorAll("[data-stage]")) {
        const result = data.stages[item.dataset.stage];
        item.className = result && result.ok ? "online" : "offline";
      }
    } catch (_error) {
      elements.headerStatus.className = "header-status offline";
      elements.statusText.textContent = "Portal unavailable";
    }
  }

  function humanBytes(bytes) {
    if (bytes < 1024 * 1024) return `${Math.ceil(bytes / 1024)} KiB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
  }

  function renderAttachments() {
    elements.attachments.replaceChildren();
    elements.attachments.hidden = state.attachments.length === 0;
    for (const [index, item] of state.attachments.entries()) {
      const node = document.createElement("div");
      node.className = "attachment";
      const kind = document.createElement("b");
      kind.textContent = item.kind;
      const name = document.createElement("span");
      name.textContent = `${item.name} · ${humanBytes(item.bytes)}`;
      const remove = document.createElement("button");
      remove.type = "button";
      remove.textContent = "×";
      remove.setAttribute("aria-label", `Remove ${item.name}`);
      remove.addEventListener("click", () => {
        state.attachments.splice(index, 1);
        renderAttachments();
      });
      node.append(kind, name, remove);
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

  async function addFile(kind, file, overrideMime) {
    if (!file) return;
    if (file.size > LIMITS[kind]) {
      throw new Error(`${kind} is ${humanBytes(file.size)}; portal limit is ${humanBytes(LIMITS[kind])}`);
    }
    const dataUrl = await fileDataUrl(file);
    const comma = dataUrl.indexOf(",");
    if (comma < 0) throw new Error("File could not be base64 encoded");
    state.attachments.push({
      kind,
      name: file.name || `recording-${Date.now()}.wav`,
      mime: overrideMime || file.type,
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
    for (let i = 0; i < output.length; i += 1) {
      const start = Math.floor(i * ratio);
      const end = Math.max(start + 1, Math.floor((i + 1) * ratio));
      let total = 0;
      for (let j = start; j < end && j < samples.length; j += 1) total += samples[j];
      output[i] = total / (end - start);
    }
    return output;
  }

  function pcmWav(samples, sampleRate = 16000) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);
    const write = (offset, text) => {
      for (let i = 0; i < text.length; i += 1) view.setUint8(offset + i, text.charCodeAt(i));
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

  async function startRecording() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error("Microphone capture is unavailable; use the HTTPS tunnel on a supported browser");
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
    state.recording = { stream, context, source, processor, sink, chunks, started: Date.now() };
    elements.micButton.classList.add("recording");
    elements.micButton.setAttribute("aria-pressed", "true");
    elements.micLabel.textContent = "Stop recording";
    setComposerStatus("Recording · 60 second maximum");
    state.recordTimer = setTimeout(() => stopRecording().catch(showError), MAX_RECORD_MS);
  }

  async function stopRecording() {
    const recording = state.recording;
    if (!recording) return;
    clearTimeout(state.recordTimer);
    recording.processor.disconnect();
    recording.source.disconnect();
    recording.sink.disconnect();
    recording.stream.getTracks().forEach(track => track.stop());
    const samples = downsample(mergeSamples(recording.chunks), recording.context.sampleRate);
    await recording.context.close();
    state.recording = null;
    elements.micButton.classList.remove("recording");
    elements.micButton.setAttribute("aria-pressed", "false");
    elements.micLabel.textContent = "Record mic";
    if (!samples.length) throw new Error("Microphone recording contained no samples");
    const blob = pcmWav(samples);
    const file = new File([blob], `microphone-${Date.now()}.wav`, { type: "audio/wav" });
    await addFile("audio", file, "audio/wav");
    setComposerStatus(`Microphone ready · ${(samples.length / 16000).toFixed(1)} seconds`);
  }

  function attachmentSummary() {
    if (!state.attachments.length) return "";
    return `\n\n[${state.attachments.map(item => `${item.kind}: ${item.name}`).join(" · ")}]`;
  }

  function requestPayload() {
    const content = elements.prompt.value.trim();
    const message = { role: "user", content };
    const audios = state.attachments.filter(item => item.kind === "audio").map(item => ({
      mime_type: "audio/wav", encoding: "base64", data: item.data,
    }));
    const images = state.attachments.filter(item => item.kind === "image").map(item => ({
      mime_type: item.mime, encoding: "base64", data: item.data,
    }));
    const videos = state.attachments.filter(item => item.kind === "video").map(item => ({
      mime_type: item.mime, encoding: "base64", data: item.data,
      sampling: { fps: 1, max_frames: 96, include_audio: elements.videoAudio.checked },
    }));
    if (audios.length) message.audios = audios;
    if (images.length) message.images = images;
    if (videos.length) message.videos = videos;

    if (state.mode === "transcribe" && !audios.length) throw new Error("Transcribe mode requires a microphone recording or WAV file");
    if (state.mode === "describe" && !state.attachments.length) throw new Error("Describe mode requires audio, an image, or a video");
    if (state.mode === "synthesize" && !content) throw new Error("Speak mode requires text");
    if (state.mode === "synthesize" && state.attachments.length) throw new Error("Speak mode does not accept input media");
    if (state.mode === "chat" && !content && !state.attachments.length) throw new Error("Enter a message or attach media");

    const messages = state.mode === "chat" ? [...state.history.slice(-12), message] : [message];
    const wantsSpeech = state.mode === "synthesize" || (state.mode === "chat" && elements.speak.checked);
    const payload = {
      model: MODEL,
      messages,
      omni: {
        schema: SCHEMA,
        task: state.mode,
        include_audio_from_video: elements.videoAudio.checked,
      },
      response_modalities: wantsSpeech ? ["text", "audio"] : ["text"],
      speech_mode: wantsSpeech ? "always" : "never",
      think: state.mode === "chat" && elements.think.checked,
      stream: false,
      options: { num_predict: 2048 },
      portal_auto_tools: state.mode === "chat" && elements.tools.checked,
    };
    if (payload.portal_auto_tools) payload.tools = SAFE_TOOLS;
    return { payload, message, wantsSpeech };
  }

  function showError(error) {
    const message = error instanceof Error ? error.message : String(error);
    addMessage({ role: "assistant", content: message, error: true });
    setComposerStatus(message, true);
  }

  async function send() {
    if (!state.token) return showError(new Error("Access token missing from this portal link"));
    if (state.recording) await stopRecording();
    await unlockPlayback();
    let built;
    try {
      built = requestPayload();
    } catch (error) {
      return showError(error);
    }

    addMessage({
      role: "user",
      content: (built.message.content || "Respond to the attached media.") + attachmentSummary(),
    });
    elements.send.disabled = true;
    elements.routeReadout.textContent = `ROUTE // ${state.mode.toUpperCase()} / RUNNING`;
    setComposerStatus("Inference running · keep this tab open");
    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: authHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify(built.payload),
      });
      const data = await response.json().catch(() => ({ error: `HTTP ${response.status}` }));
      if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
      const message = data.message || {};
      const executed = data.portal && data.portal.safe_tools_executed || [];
      addMessage({
        role: "assistant",
        content: message.content || (message.tool_calls ? "Tool call returned without final text." : "No text returned."),
        thinking: message.thinking,
        trace: data.adapter,
        tools: executed.length ? executed : message.tool_calls,
        audio: message.audio,
      });
      const route = data.adapter && data.adapter.route || [];
      elements.routeReadout.textContent = `ROUTE // ${route.length ? route.join(" → ").toUpperCase() : state.mode.toUpperCase()}`;
      if (state.mode === "chat") {
        state.history.push({ role: "user", content: built.message.content || "Respond to the supplied media." });
        if (message.content) state.history.push({ role: "assistant", content: message.content });
      }
      state.attachments = [];
      renderAttachments();
      elements.prompt.value = "";
      setComposerStatus(message.audio ? "Response ready · audio attached" : "Response ready");
    } catch (error) {
      showError(error);
      elements.routeReadout.textContent = `ROUTE // ${state.mode.toUpperCase()} / FAILED`;
    } finally {
      elements.send.disabled = false;
      refreshStatus();
    }
  }

  function setMode(mode) {
    state.mode = mode;
    for (const button of elements.modes) button.classList.toggle("active", button.dataset.mode === mode);
    const direct = mode !== "chat";
    elements.think.disabled = direct;
    elements.tools.disabled = direct;
    elements.speak.disabled = direct || mode === "synthesize";
    elements.captureRow.classList.toggle("disabled", mode === "synthesize");
    const labels = {
      chat: ["Message", "Ask anything, or attach media for comprehension…"],
      transcribe: ["Transcription instruction", "Optional guidance for the transcription…"],
      describe: ["Description instruction", "What should the model inspect in this media?"],
      synthesize: ["Speech text", "Enter exactly what the model should speak…"],
    };
    elements.promptLabel.textContent = labels[mode][0];
    elements.prompt.placeholder = labels[mode][1];
    elements.routeReadout.textContent = `ROUTE // ${mode.toUpperCase()} / IDLE`;
    setComposerStatus(`${mode} mode selected`);
  }

  elements.micButton.addEventListener("click", () => {
    (state.recording ? stopRecording() : startRecording()).catch(showError);
  });
  elements.audioInput.addEventListener("change", event => {
    addFile("audio", event.target.files[0], "audio/wav").catch(showError);
    event.target.value = "";
  });
  elements.imageInput.addEventListener("change", event => {
    addFile("image", event.target.files[0]).catch(showError);
    event.target.value = "";
  });
  elements.videoInput.addEventListener("change", event => {
    addFile("video", event.target.files[0]).catch(showError);
    event.target.value = "";
  });
  elements.send.addEventListener("click", () => send().catch(showError));
  elements.prompt.addEventListener("keydown", event => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) send().catch(showError);
  });
  elements.clear.addEventListener("click", () => {
    state.history = [];
    state.attachments = [];
    renderAttachments();
    for (const message of [...elements.conversation.querySelectorAll(".message")].slice(1)) message.remove();
    setComposerStatus("Session cleared");
  });
  for (const button of elements.modes) button.addEventListener("click", () => setMode(button.dataset.mode));

  state.token = accessToken();
  if (!state.token) showError(new Error("This link is missing its access fragment"));
  setMode("chat");
  refreshStatus();
  setInterval(refreshStatus, 15_000);
})();
