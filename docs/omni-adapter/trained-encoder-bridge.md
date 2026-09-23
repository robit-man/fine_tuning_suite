# Trained Omni Encoder Bridge Research Plan

## Goal

Use the E03-obliterated Qwen3.8 27B or standard Ornith 1.5 9B model as the only
language trunk while retaining Qwen3-Omni audio and, where useful, video
perception. The obliterated Ornith variant is explicitly outside this release.
This removes the duplicated Qwen3-Omni Thinker from constrained deployments
without claiming that incompatible tensors can be copied into one another.

The intended graph is:

```text
audio / optional video
        |
        v
frozen Qwen3-Omni encoder
        |
        v
trained final audio projector (1280 -> target language width)
        |
        v
frozen Qwen3.8 or Ornith language trunk -> tools / answer
```

Qwen3.8 and Ornith keep their existing native vision projector when it is the
better path. Qwen3-Omni vision remains an optional bridge input for temporal
video experiments. Qwen3-TTS remains an independently text-conditioned graph.

## Executable stage-one scaffold

The release implementation is in
`training_suite/training/omni_encoder_bridge.py`. It provides:

- an exact replacement for llama.cpp's Qwen3A `mm.a.mlp.2` final audio
  projector: 1,280-to-5,120 for Qwen3.8 or 1,280-to-4,096 for Ornith;
- exact Qwen3-Omni audio output-length and flat-sequence rebatching logic;
- a frozen-encoder/frozen-Qwen3.8 `inputs_embeds` training graph that supervises
  assistant response tokens only;
- a direct loader for the existing Q8_0 comprehension-projector GGUF, so the
  audio tower can be reused without loading or copying the Omni Thinker; and
- safetensors checkpoints containing only the bridge, never either frozen
  base model.

Generate the concrete Qwen3.8 experiment contract with:

```bash
python -m training_suite omni-bridge-plan \
  --text-source manitcor/Qwen3.8-27B-Obliterated-E03 \
  --omni-source Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --out training_suite/outputs/omni/qwen38-audio-bridge.json
```

Use a dedicated environment built from
`training_suite/requirements-omni-bridge.txt`; the legacy SFT environment's
Transformers 4.x pin predates both required model implementations.

The local Q8_0 projector was checked against the installed Transformers model:
all 525 expected audio-tower parameters mapped with no missing or extra
tensors. The frozen encoder contributes the feature sequence entering its
existing final projector, after `proj1` and GELU. The deployable Qwen3.8 bridge
has 6,558,720 trainable parameters (13,117,440 BF16 bytes); the Ornith bridge
has 5,246,976 parameters (10,493,952 BF16 bytes).

## Why direct replacement is invalid

The current components have incompatible execution contracts:

| Property | Qwen3-Omni Thinker | Qwen3.8 27B | Ornith 1.5 |
|---|---:|---:|---:|
| Hidden width | 2,048 | 5,120 | 4,096 |
| Layers | 48 | 64 | 32 |
| Vocabulary | 152,064 | 248,320 | 248,320 |
| Text graph | 128-expert MoE, 8 active | dense | Qwen3.5 conditional graph |

Padding, reshaping, or renaming the Qwen3.8/Ornith tensors would not preserve
their computation. The viable interpretation of "put the text weights in
Omni" is a new hybrid graph: keep the target text graph intact and train the
media interface to its embedding width.

## Local artifact evidence

The pinned Qwen3-Omni Q4_K_M comprehension model is 18,557,053,952 bytes. Its
Q8_0 projector is 1,325,020,128 bytes. Tensor accounting in that projector is:

| Reusable projector portion | Tensor bytes | MiB |
|---|---:|---:|
| Audio encoder/projector | 710,796,352 | 677.87 |
| Vision encoder/projector | 614,174,528 | 585.72 |

The deployable affine audio bridge adds only 13.1 MB of BF16 weights for
Qwen3.8 or 10.5 MB for Ornith. It replaces the donor's final audio projection;
the full 18.56 GB Omni Thinker is not retained.

Approximate weight-only targets using the currently pinned components are:

| Profile | Current separate language + comprehension + TTS | Hybrid target |
|---|---:|---:|
| Standard Ornith 1.5 9B | about 27.9 GB | 8,751,744,576 bytes (8.15 GiB) |
| Qwen3.8 27B E03 Obliterated | about 38.8 GB | 19,682,103,136 bytes (18.33 GiB) |

These are artifact bytes, not peak resident memory. KV, allocator scratch,
decoder state, the OS, and page cache still require live measurement on Tegra.

## Training sequence

1. Freeze the target language graph, its native vision path, and the Omni
   audio encoder. Train only the 1,280-to-target-width final audio projector
   with next-token loss through the target model.
2. Begin with audio because it eliminates the largest functional dependency
   while Ornith/Qwen3.8 retain their validated native image path. Train clean
   speech, noisy speech, non-speech sound, mixed sound/speech, and no-speech
   rejection examples.
3. Add temporal video only after audio passes. Compare the target's native
   multi-image path with an Omni-vision bridge before retaining both encoders.
4. If a linear/gated MLP bridge is insufficient, add a bounded resampler or
   Q-Former. Do not unfreeze the language trunk merely to hide a weak bridge.
5. Only after the frozen-trunk result is measured may a small language LoRA be
   tested. Include text/tool replay and reject any text-only regression.
6. Prototype first in Transformers with `inputs_embeds`. Add a new llama.cpp
   `libmtmd` projector/runtime graph only after the learned interface works.

Teacher supervision may come from the current Qwen3-Omni semantic output and
the selected target model's preferred response. This is behavior distillation,
not tensor merging. Tool/action capability stays in the target language model;
the bridge is trained to supply perceptual evidence, never tool instructions.

## Release gates

- Byte-identical target language weights for the frozen-trunk candidate.
- Text, thinking, structured tools, background-task checkpoints, and safety
  probes match the untouched target.
- Clean/noisy speech transcription and speech-vs-environment separation match
  or exceed the current semantic-router baseline.
- Quiet/click/steady-noise inputs do not invent speech.
- Image OCR/object tests and red -> blue -> red temporal tests pass without
  stale-media reuse.
- Audio-only requests never set current visual provenance.
- Jetson residency is proven through `qwen_omni_adapters.accelerator`, with
  measured peak unified memory and no CPU fallback.

The current semantic-text router remains the rollback path until every gate
passes.

The release builder requires separate machine-readable audio and native-vision
reports. `training_suite.evals.vision_bridge` sends freshly generated solid
red, blue, then red images to the candidate server, disables prompt caching on
every request, and fails on an incorrect color, malformed visual evidence tag,
or a stale prior-frame answer.

The output path has two independent live gates. `training_suite.evals.tts_alignment`
synthesizes and re-transcribes an A -> B -> A sequence through one resident TTS
worker. It validates 24 kHz mono PCM16 output, rejects one-turn state lag, and
requires exact tagged comprehension output. `training_suite.evals.tts_vocabulary`
then extracts the complete normalized vocabulary from the pinned bridge corpus,
shuffles it deterministically, synthesizes every unique word in bounded
ten-word carrier phrases, and re-transcribes every WAV. Publication requires at
least 4,000 unique words,
70% full-vocabulary recall, 60% recall for words occurring no more than twice,
mean WER no greater than 0.45, 99% exact tagged output, and a valid WAV contract
for every batch. These thresholds are fixed before a run; a smaller calibration
run cannot substitute for the complete report.

The gated release builder also consumes the model capability and live structured
tool-smoke reports. It refuses to render a manifest or model card unless vision,
tools, a separate thinking channel, A -> B -> A state reset, and the complete
large-vocabulary run all pass. The release manifest records compact metrics and
SHA-256 digests of those source reports so published claims remain tied to the
tested evidence.

The vocabulary report is paired with a strict audit that must bind its exact
SHA-256 digest and report zero empty transcripts, malformed evidence tags, or
invalid WAVs across every batch. Audio, A -> B -> A, and vocabulary evidence
must also share the exact deployed comprehension-policy digest. If behavioral
bracketing selects a checkpoint other than the minimum validation-loss step,
the optional checkpoint-selection report must bind the published projector
digest, identify the selected candidate, and prove byte identity with the
projector that passed the live gates.

For a trained bridge, the training collator and deployed llama.cpp request must
use the same target-native `enable_thinking=false` prefill. Comprehension is
deterministic and uses a 1.1 repetition penalty to bound rare greedy transcript
loops. This profile-specific setting must not be copied to the stock
Qwen3-Omni path, whose explicit no-thinking template branch is known to fail on
multimodal prompts.

Evidence element bodies are plain text. Nested spans, timestamp or coordinate
markup, Markdown, and code fences fail the contract even when the surrounding
tags are present. A failing large-vocabulary gate is repaired with the
crash-resumable TTS recovery corpus, using the independent full LibriSpeech
transcript vocabulary and a training-only carrier distinct from the evaluator;
release thresholds are never tuned downward around a candidate.
The large-vocabulary evaluator may share an atomic WAV cache across candidates,
but only after matching the corpus and speaker digests, full shuffled batch
sequence, carrier, seed, batch size, and TTS endpoint. ASR responses are never
cached.

## Quantization fallback

The current comprehension model is already Q4_K_M, not full precision. Local
`llama-quantize --dry-run` measurements are:

| Quant | Estimated model MiB | Saving vs current MiB |
|---|---:|---:|
| Q4_K_M current | 17,691.69 | 0 |
| Q4_K_S | 16,642.00 | 1,049.69 |
| Q3_K_M | 14,024.93 | 3,666.76 |
| IQ3_M | 12,881.68 | 4,810.01 |
| Q3_K_S | 12,671.30 | 5,020.39 |
| Q2_K | 10,731.64 | 6,960.05 |

No lossy quant can guarantee identical capability. A release candidate must be
quantized from the pinned 61,096,856,576-byte BF16 GGUF, never requantized from
the current Q4 artifact. Start with Q3_K_M and a representative importance
matrix. Keep the projector at Q8_0 initially: reducing it to Q4_K_M saves only
about 352 MiB and places multimodal alignment at greater risk.

Quantize each component before packing. Never quantize the combined sidecar.

## Speculative decoding

Speculation changes decode speed, not weight memory, and must remain optional
until a target-device benchmark shows a gain.

1. Benchmark weight-free `ngram-simple` first. It uses only the current token
   history and avoids another resident model. Compare open-ended answers,
   structured tool JSON, background-task checkpoints, and summarization.
2. The pinned llama.cpp also supports EAGLE-3. Qwen3-Omni's text trunk matches
   the broad Qwen3-30B-A3B dimensions, but its tokenizer inserts audio/TTS
   tokens beginning at ID 151,669. An off-the-shelf Qwen3-A3B EAGLE draft
   therefore fails the exact shared-token check without a controlled target
   tokenizer export/remap. Treat it as a research candidate, not a drop-in.
3. For the hybrid graph, use a draft trained for the actual Qwen3.8 or Ornith
   target. A same-tokenizer small model is acceptable only if measured draft
   acceptance offsets its extra memory and bandwidth on Jetson.
4. Record time to first token, generated tokens/second, accepted draft tokens,
   peak unified memory, and exact greedy-output equivalence. Disable speculation
   automatically when acceptance or throughput falls below baseline.

### Provisional host benchmark

On 2026-09-22, the pinned Q4_K_M comprehension model and Q8_0 projector were
benchmarked through the pinned llama.cpp server on broker-scoped GPU 1. Each
number below is the median of three warm runs with `temperature=0`,
`cache_prompt=false`, one server slot, and no concurrent broker inference.
The speculative server differed only by `--spec-type ngram-simple`.

| Case | Baseline tok/s | N-gram tok/s | Decode latency change | Accepted drafts |
|---|---:|---:|---:|---:|
| Open-ended policy answer | 170.16 | 167.00 | +1.9% | 0 / 0 |
| Repetitive structured JSON | 170.06 | 229.40 | -25.9% | 201 / 384 |
| Exact policy reproduction | 169.83 | 384.68 | -55.9% | 85 / 96 |

The greedy output SHA-256 matched baseline in all three cases. The open-ended
case demonstrates that weight-free speculation is not universally faster; the
structured case was capped at the same 384-token limit in both modes. Treat
these as a promising runtime result, not a release gate: repeat the benchmark
on the target Jetson with real transcripts, tool calls, and background-task
traces before enabling it by default.

Do not use a server-global cross-session lookup cache in the multi-user portal.
