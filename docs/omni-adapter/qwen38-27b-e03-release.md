# Qwen3.8-27B-E03-Obliterated-Omni Q4_K_M Release Record

Status: release complete. Repository, Hugging Face, Ollama registry round-trip,
live capability, and post-release cleanup gates passed.

## Target tags

- Ollama: `robit/qwen3.8-27b-e03-obliterated-omni:q4km`
- Ollama alias: `robit/qwen3.8-27b-e03-obliterated-omni:latest`
- Hugging Face: `cudabenchmarktest/Qwen3.8-27B-E03-Obliterated-Omni-GGUF`

## Schemas and toolchain

| Item | Value |
|---|---|
| Wire schema | `robit.ollama.omni-adapter.v1` |
| Artifact schema | `robit.ollama-monolithic-omni.v3` |
| Container format | `robit-namespaced-multigraph-gguf-v1` |
| Ollama sidecar media type | `application/vnd.robit.ollama.omni.bundle.v1+gguf` |
| llama.cpp commit | `458681e1d5d4a29a1463c4732e03226cf384b997` |
| Sidecar filename | `qwen3.8-27b-e03-obliterated-omni-q4km.gguf` |
| Sidecar size | `38,843,038,144` bytes |
| Sidecar SHA-256 | `3270f146bae9499b2e40ad230cceeccfc9caa018740c75cfc1856c1abda6ff78` |

## Component provenance

| View | Source | Bytes | Tensors | SHA-256 |
|---|---|---:|---:|---|
| Base model | `manitcor/Qwen3.8-27B-Obliterated-E03@6104397d699fed901e2d4521c3b0fefc9f837d90` | 16,547,400,480 | 851 | `91804f5668d8e5deef47cced0e4ecff9d30da6e965418307d92f3a2822be4248` |
| Base projector | `robit/qwen3.8-27b-obliterated-e03:27b` standard projector | 931,146,016 | 334 | `ac3714bfdddeca31351f2752bf1a63f266f4df87c0b68c895e44945ca704448e` |
| Comprehension model | `ggml-org/Qwen3-Omni-30B-A3B-Instruct-GGUF@6e35a28f4a19b18730f8949b0c579c6429649ab8`, Q4_K_M | 18,557,053,952 | 579 | `d9e2876556e7873e02c0359f832432ee2d67ab7dd0cee3efe0f77fd7a1f4dd85` |
| Comprehension projector | same revision, Q8_0 mmproj | 1,325,020,128 | 860 | `1104376db833f1e89c84834144ac3863340c2cd1ddaeddb39cb0247fb5c20c8d` |
| TTS model | `ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF@ca27d74bc954b73dadab5b71ca265d87fc861a7c`, Q4_K_M | 1,035,965,280 | 311 | `8d18c94acb2addd042f97da63c98be144eafa76d0d9495177eab65130cf85129` |
| TTS projector | same revision, Q8_0 mmproj | 446,422,912 | 378 | `6fd65188839bcd6ecc91b277ad471e22a0edfada4699a0fe82f1165c18cfcce2` |

Total sidecar tensor count: 3,313.

## Ollama standard layer

The local tag retains the source standard layers rather than loading the
heterogeneous sidecar as its model graph:

- architecture: `qwen35`;
- parameters: 26.9B;
- quantization: Q4_K_M;
- context length: 262,144;
- standard capabilities: completion, vision, tools, thinking;
- projector architecture: `clip`, 460.73M parameters;
- Ollama requirement: 0.32.12.

## Local validation

| Gate | Result | Assertion |
|---|---|---|
| Sidecar inspection | PASS | six views, 3,313 tensors, no schema errors |
| Materialized digests | PASS | all five extracted non-base files exactly match pinned source SHA-256 values |
| Stock text | PASS | exact `OMNI BASE OK` response |
| Thinking | PASS | non-empty parsed thinking field |
| Structured tool | PASS | `get_weather` with `location=Seattle` |
| Native image vision | PASS | visible fixture read as `BLUE 42` |
| Qwen3-Omni audio | PASS | exact “The verification phrase is copper lighthouse seven.” |
| Qwen3-Omni image | PASS | blue triangle and number 42 |
| Qwen3-Omni video + audio | PASS | red→blue plus exact spoken phrase |
| Qwen3-TTS direct | PASS | 24 kHz mono PCM16 WAV, 3.28 seconds |
| Adapter TTS | PASS | tagged base64 24 kHz mono PCM16 WAV |
| Adapter TTS repeat | PASS | second valid 24 kHz mono PCM16 WAV, 3.60 seconds |
| Temporary CUDA workers | PASS | stopped; broker reports no active leases |

The live media probes used views freshly materialized from the installed
sidecar blob, not unrelated source paths.

## Known limitations

- Stock Ollama owns text/image/tools/thinking only. The adapter is required for
  audio, video, and TTS.
- Adapter v1 is turn-based and does not support streaming.
- Video audio is demuxed and submitted separately; sample-accurate alignment is
  not guaranteed.
- The reference TTS HTTP worker is serial but keeps one matching voice profile
  resident; a voice-profile change intentionally replaces that worker.
- This is semantic routing, not a trained hidden-state fusion of Qwen3.8 and
  Qwen3-Omni.

## Remote verification

| Gate | Result | Evidence |
|---|---|---|
| GitHub `main` | PASS | Implementation and release documentation pushed through commit `243008ea8740b3ca1bf26e1ed706b7261f40c49e` before this final evidence update |
| Hugging Face artifact | PASS | Repository commit `a0d82e6e076b549289264a6fe6a2625ffe2966ad`; remote file size `38,843,038,144`; LFS SHA-256 `3270f146bae9499b2e40ad230cceeccfc9caa018740c75cfc1856c1abda6ff78` |
| Ollama `q4km` | PASS | Pushed to `robit/qwen3.8-27b-e03-obliterated-omni:q4km` |
| Ollama `latest` | PASS | Pushed to `robit/qwen3.8-27b-e03-obliterated-omni:latest` |
| Ollama registry round-trip | PASS | Removed the disposable local `latest` manifest, pulled it from the registry, resolved the custom sidecar layer, and re-inspected all six views and 3,313 tensors |
| Pulled sidecar descriptor | PASS | Media type `application/vnd.robit.ollama.omni.bundle.v1+gguf`; digest `sha256:3270f146bae9499b2e40ad230cceeccfc9caa018740c75cfc1856c1abda6ff78`; size `38,843,038,144` |
| Pulled stock runtime | PASS | `ollama show` retained completion, vision, tools, and thinking; fresh public-broker inference returned exact `REMOTE OLLAMA OK` |
| Cleanup | PASS | Run directory reduced from 57 GB to 1.4 MB; `/srv` available space increased from 62 GB to 83 GB |

Published locations:

- [Ollama `q4km`](https://ollama.com/robit/qwen3.8-27b-e03-obliterated-omni:q4km)
- [Ollama `latest`](https://ollama.com/robit/qwen3.8-27b-e03-obliterated-omni)
- [Hugging Face sidecar](https://huggingface.co/cudabenchmarktest/Qwen3.8-27B-E03-Obliterated-Omni-GGUF)
- [Fine Tuning Suite](https://github.com/robit-man/fine_tuning_suite)

Cleanup removed the five materialized media-runtime views totaling
`22,295,608,288` bytes, the completed Hugging Face staging tree, and the
release-directory hardlink. The installed Ollama blob and both published local
tags were preserved. This build did not create or download Safetensors, so
there were no Safetensors to remove.
