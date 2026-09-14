# Ornith 1.5 9B Omni Q4_K_M dual-release record

Status: local artifacts and capability gates complete; remote publication in
progress.

## Release matrix

| Variant | Language/vision base | Ollama tags | Hugging Face sidecar |
|---|---|---|---|
| Stock | `ornith-ai/Ornith-1.5-9B@489cb97981b8654bcfcf30ce1f94ed1b62e07b53` | `robit/ornith-1.5-omni:q4km`, `:latest` | `cudabenchmarktest/Ornith-1.5-9B-Omni-GGUF` |
| Obliterated | `OBLITERATUS/Ornith-1.5-9B-OBLITERATED@82df4702cc5e716cf7573a8942fee28ae8f65088` | `robit/ornith-1.5-obliterated-omni:q4km`, `:latest` | `cudabenchmarktest/Ornith-1.5-9B-Obliterated-Omni-GGUF` |

The generic `ornith-1.5-omni` name is reserved for the stock base. The
OBLITERATUS base is never published under that name.

## Artifact identity

| Item | Stock | Obliterated |
|---|---:|---:|
| Filename | `ornith-1.5-9b-omni-q4km.gguf` | `ornith-1.5-9b-obliterated-omni-q4km.gguf` |
| Size | 27,915,305,504 bytes | 28,066,286,368 bytes |
| SHA-256 | `1302c601bbfa81d0efa80167beecaac14b6ecbbada377ff814ce1da4d22e0f6d` | `22a38a92994dd9b21cee273c7fbbe5b21e57498bc6f56e1d6646211fc641b55e` |
| Tensors | 2,889 | 2,904 |
| Base tensors | 427 | 442 |
| Base-projector tensors | 334 | 334 |

Both use artifact schema `robit.ollama-monolithic-omni.v3`, container format
`robit-namespaced-multigraph-gguf-v1`, wire schema
`robit.ollama.omni-adapter.v1`, and custom Ollama media type
`application/vnd.robit.ollama.omni.bundle.v1+gguf`.

## Shared media views

| View | Bytes | Tensors | SHA-256 |
|---|---:|---:|---|
| Qwen3-Omni comprehension model | 18,557,053,952 | 579 | `d9e2876556e7873e02c0359f832432ee2d67ab7dd0cee3efe0f77fd7a1f4dd85` |
| Qwen3-Omni projector | 1,325,020,128 | 860 | `1104376db833f1e89c84834144ac3863340c2cd1ddaeddb39cb0247fb5c20c8d` |
| Qwen3-TTS model | 1,035,965,280 | 311 | `8d18c94acb2addd042f97da63c98be144eafa76d0d9495177eab65130cf85129` |
| Qwen3-TTS codec/projector | 446,422,912 | 378 | `6fd65188839bcd6ecc91b277ad471e22a0edfada4699a0fe82f1165c18cfcce2` |

These views are pinned to Qwen3-Omni revision
`6e35a28f4a19b18730f8949b0c579c6429649ab8` and Qwen3-TTS revision
`ca27d74bc954b73dadab5b71ca265d87fc861a7c`. The runtime is pinned to
llama.cpp `458681e1d5d4a29a1463c4732e03226cf384b997` plus the tracked
persistent/streaming TTS patches.

## Ollama standard layers

| Property | Stock | Obliterated |
|---|---|---|
| Architecture | `qwen35` | `qwen35` |
| Parameters | 9.0B | 9.2B |
| Quantization | Q4_K_M | Q4_K_M |
| Context | 262,144 | 262,144 |
| Capabilities | completion, vision, tools, thinking | completion, vision, tools, thinking |
| Projector | clip, 456.01M | clip, 456.01M |
| Model output default | `num_predict=-1` (unlimited) | `num_predict=-1` (unlimited) |

The source Ollama tags carried a 16,384-token generation cap. Both Omni tags
explicitly override it with Ollama's unlimited value. A caller may still apply
a bounded `num_predict` per request.

## Local validation

| Gate | Result | Evidence |
|---|---|---|
| Sidecar inspection | PASS | Six views; no schema errors; exact tensor counts above |
| Materialized digest round-trip | PASS | Every extracted base, projector, comprehension, and TTS file matched its pinned SHA-256 |
| Stock/obliterated identity | PASS | Distinct base and projector digests; generic tag is stock-backed |
| Text with thinking disabled | PASS | Exact stock and obliterated sentinel replies; no thinking field |
| Parsed thinking | PASS | Non-empty separate reasoning field; no `<think>` tag bleed |
| Structured tools | PASS | Both emitted `get_weather(location=Seattle)` |
| Native image vision | PASS | Both read the held-out blue/42 fixture |
| Audio transcription | PASS | Exact “The verification phrase is copper lighthouse seven.” |
| Environmental audio | PASS | Empty speech tag plus non-speech electronic-tone observation |
| Video and video audio | PASS | Red→blue order and exact spoken phrase |
| Adapter media→language | PASS | Both selected the correct stock or obliterated language backend |
| Full audio chat | PASS | comprehension → correct Ornith language base → TTS for both tags |
| Direct and repeated TTS | PASS | Valid 24 kHz mono PCM16 WAVs, 4.16 s and 3.28 s |
| Raw streamed PCM continuity | PASS | 58 two-frame windows; median normalized boundary jump 0.142; no shared-prefix regression |
| Repository unit tests | PASS | 117 tests passed; compileall passed |

Media inference loaded views freshly reconstructed from the installed stock
Ollama sidecar. Because both artifacts embed byte-identical media views, those
component results apply to both; separate full adapter probes selected each
variant's own language backend.

The temporary worker was launched only after `docker gpu discover`, under an
exact UUID-scoped broker lease, with full GPU offload and no CPU inference
fallback. The worker processes were stopped after validation.

## Execution boundary

Stock Ollama executes text, native image vision, tools, and optional thinking
from each tag's standard layers. It ignores the custom sidecar layer. The
adapter materializes the Qwen3-Omni and Qwen3-TTS views and exposes audio/video
input and audio output while keeping the public request Ollama-shaped.

This is semantic routing between incompatible graphs, not a trained hidden-state
fusion. Video support is understanding only; no video-generation head is
included.

## Publication and cleanup gates

Before release is marked complete:

1. push this record, model cards, manifests, notices, and runtime profiles;
2. upload each GGUF and documentation to its matching Hugging Face repository;
3. verify remote size and LFS SHA-256;
4. push `q4km` and `latest` for both Ollama repositories;
5. remove and re-pull each disposable local `latest` tag, resolve its custom
   layer, and run a post-pull native inference;
6. stop CUDA workers, verify broker release, then remove only materialized
   views, staging data, and redundant release-directory copies.

No Safetensors were downloaded or produced for these releases.

See the two machine-readable manifests, two Hugging Face model cards, and
`ornith15-third-party-notices.md` beside this record.
