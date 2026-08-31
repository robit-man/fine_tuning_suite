#!/usr/bin/env node
// Minimal audio-chat client using only Node.js built-ins.

import { readFile, writeFile } from "node:fs/promises";

const [audioPath, outputPath = "response.wav"] = process.argv.slice(2);
if (!audioPath) {
  console.error("usage: node javascript_client.mjs input-16khz-mono.wav [response.wav]");
  process.exit(2);
}

const endpoint = process.env.OMNI_ADAPTER_URL ?? "http://127.0.0.1:11435/api/chat";
const model = process.env.OMNI_MODEL ?? "robit/qwen3.8-omni:latest";
const audio = await readFile(audioPath);
const response = await fetch(endpoint, {
  method: "POST",
  headers: { "content-type": "application/json" },
  body: JSON.stringify({
    model,
    messages: [{
      role: "user",
      content: "Answer the question in the recording.",
      audios: [{
        mime_type: "audio/wav",
        encoding: "base64",
        data: audio.toString("base64"),
      }],
    }],
    omni: { schema: "robit.ollama.omni-adapter.v1", task: "chat" },
    response_modalities: ["text", "audio"],
    speech_mode: "always",
    think: true,
    stream: false,
  }),
});

if (!response.ok) {
  throw new Error(`adapter returned ${response.status}: ${await response.text()}`);
}
const result = await response.json();
console.log(result.message?.content ?? "");
if (result.message?.audio?.data) {
  await writeFile(outputPath, Buffer.from(result.message.audio.data, "base64"));
  console.error(`wrote ${outputPath}`);
}
