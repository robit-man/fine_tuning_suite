# Prepared speech weights in official Ollama

Assessed against **official Ollama 0.40.11**, 2026-10-26.

This records the earlier stock-binary assessment. The operator subsequently
authorized a hybrid runtime extension. The working successor is
[Phonon Echo](../phonon-hybrid.md), tested through Ollama's own speech and
transcription endpoints with an Ollama-managed runner.

The deployment requirement is **an unmodified official Ollama binary plus the
weights we prepare**. `POST /v1/audio/speech` must execute the selected tag's
speech weights inside Ollama. Preparation may use Python. A fork, custom runner,
Python inference worker, Omni adapter, or external TTS service is outside this
delivery contract.

## Result

**Original Qwen Omni speech weights cannot currently be delivered under this
contract by the suite.** No supported direct conversion was found. Ollama's
shipped speech graph is Phonon; its operators differ from Omni's Talker,
residual codec predictor, and code2wav graph. Renaming tensors, changing an
architecture string, attaching our sidecar, or folding a text projection does
not supply the missing execution graph.

Offline distillation into a Phonon-compatible student is a separate candidate.
It would create different speech weights and would not preserve the original
Omni graph or its other modalities. This candidate also has unresolved encoder,
training, export, and quality requirements; it is not a working conversion.

The host upgrade from 0.34.4 to 0.40.11 and a native Qwen 27B chat probe passed.
Those checks establish the upgrade, not native Omni speech. No native Omni
audio, distilled student, or offline reference-voice clone has been produced.

Subsequent work on a new Phonon-based model has produced native audio from an
experimentally fitted voice tensor. The candidate fails word-fidelity checks,
and the same Phonon tag is rejected by native ASR. See the
[cloning experiment](../../experiments/native-omni-speech/phonon-cloning.md)
for implementation, metrics, and remaining single-graph runtime constraints.

## How the endpoint executes tensors

```text
server/speech.go: SpeechHandler
  -> llm.Speaker.Speech(ctx, {Input, Voice}, emitPCM)
  -> mlxrunner/client_speech.go: Ollama-managed runner IPC
  -> mlxrunner/speech.go: speechHandler
  -> model.SpeechModel.Speak(ctx, input, emitFloatSamples)
  -> loaded speech graph
  -> float samples -> clamp/round -> little-endian PCM16
  -> streaming HTTP response, with WAV header when requested
```

`Runner.loadModel` reads the manifest and tensor blobs, constructs the registered
architecture, and calls `LoadWeights`. The speech method then runs linear,
attention, flow, and convolution operators against those weights. The endpoint
is transport around this graph, not an interpreter for arbitrary speech tensors.

In this release, `GradiumPhononTTSForConditionalGeneration` is the implemented
native speech architecture. The importer infers its `speech` capability. The
MLX constructor selects `architectures[0]`; listing both Omni and Phonon does
not dispatch between two architectures by endpoint.

Phonon's concrete tensor path is:

1. `flow_lm.condition_provider.conditioners.transcript_in_segment.embed.weight`
   supplies text embeddings.
2. Prepared `voices.*` speaker latents pass through
   `flow_lm.condition_provider.conditioners.speaker_wavs.output_proj.weight`.
   Voice prefill runs them through the transformer and caches attention keys
   and values.
3. `flow_lm.input_linear.*`, `flow_lm.transformer.*`, `flow_lm.out_norm.*`,
   `flow_lm.flow_net.*`, and `flow_lm.out_eos.*` generate continuous latents
   and an end-of-speech signal. The flow head consumes sampled noise.
4. `flow_lm.emb_mean` and `flow_lm.emb_std` restore latent scale.
   `mimi.*` projects, upsamples, runs a causal transformer, and applies the
   convolutional waveform decoder, retaining state across chunks.

The runtime checks width 512, the expected conditioning layout, GELU `erf`,
time norm `rms`, 24 kHz output, 12.5 latent frames/second, and one LSD decode
step. It accepts unquantized F32/F16/BF16 tensors. Capability metadata cannot
bypass these execution constraints.

## Why Qwen speech weights do not map directly

| Execution property | Stock Phonon path | Qwen3-Omni 30B-A3B speech path |
|---|---|---|
| Text conditioning | Phonon tokenizer and embeddings, width 512 | Thinker token embeddings, width 2048, projected to Talker width 1024 |
| Backbone | Flow transformer | 20-layer MoE Talker, 128 experts, 6 active per token |
| Audio prediction | Continuous 32-dimensional latent | First codec stream plus residual predictor for 16 codec groups |
| Waveform decoder | Phonon's Mimi decoder | Omni's code2wav decoder with its own quantizers and transformer |
| Voice conditioning | Prepared speaker-codec latents | Learned speaker-ID embeddings for Chelsie, Ethan, Aiden |

Qwen's `_get_talker_assistant_parts` uses Thinker **token embeddings**, not the
full contextual Thinker transformer, for supplied assistant text. It constructs
a projected text/codec/speaker prefix and trailing text conditioning. This
suggests a preparation optimization: precompute
`text_projection(thinker_embedding[token])` as a speech embedding table.

The [CPU conditioning experiment](../../experiments/native-omni-speech/README.md)
tested that fold against Qwen's actual prefix builder with synthetic tensors.
Nine cases passed, maximum absolute difference 1.1920928955078125e-7, without
initializing CUDA. This proves only that local transformation on those test
tensors. It does not make the remaining MoE, codec predictor, or code2wav
operators available in stock Ollama. Pretrained-weight and audio parity are
unverified.

Our existing Qwen3-TTS GGML implementation similarly runs a text-conditioned
Talker, residual predictor (`a.gen.code.*`), waveform decoder (`a.gen.wav.*`),
and speaker encoder. That graph lives in the adapter's patched llama.cpp
worker. Official Ollama does not call it through `llm.Speaker`. Integrating it
would require a changed runner and is outside the selected contract.

## Distillation candidate and concrete blockers

If a different student model is acceptable, the candidate preparation path is:

```text
Omni teacher -> generated waveform + checked transcript
  -> encode waveform into compatible student codec latents
  -> train/fine-tune a Phonon-compatible student
  -> export native flow_lm.* weights, matching Mimi decoder and tokenizer
  -> ollama create with checkpoint and prepared voice tensors
  -> /v1/audio/speech using only official Ollama and prepared weights
```

Omni codec IDs cannot be Phonon latent labels: their representations and
decoders differ. The student must keep a compatible runtime graph. Saving a
Transformers checkpoint alone is not an Ollama export: native tensor names,
config, latent statistics, decoder weights, and tokenizer must be preserved
and checked against the training implementation.

Transformers main at `6c5f8f4186c15f4bfbd67941a1736b2b347e0155` implements
Phonon training with continuous target `labels` and LSD/EOS losses. This supplies
a training objective, not the missing waveform encoder.

Metadata and safetensors headers were inspected without downloading payloads:

| Published checkpoint | Pinned revision | Relevant finding |
|---|---|---|
| `gradium/phonon-31m-en` | `92741c5b07d085ff840e26fea8ce43110b87852f` | 185 tensors; speech graph and decoder, no waveform encoder |
| `gradium/phonon-mimi` | `33519be1a8e61c6fa00c6e154e2d38e76a367919` | 48 tensors; `encoder_config: null`, decoder only |

The Phonon documentation explicitly says its separate 512-dimensional speaker
codec is unreleased. The processor rejects waveform encoding with
`encoder_config=None`. The published checkpoints therefore do not provide an
established offline path from arbitrary teacher waveforms to training labels
or from recordings to new voices. Obtaining compatible encoder weights, or
developing and validating an alternative, remains work. Student quality and
native export parity are also untested.

## Reference speech under the fixed API

The stock runner request contains only `Input` and `Voice`. It has no native
per-request recording or reference-transcript field. Unknown JSON fields may
be ignored: posting `reference_audio` is not proof of voice conditioning.

Phonon's importer supports **prepared named voices**:

```text
checkpoint/default-voice.safetensors
checkpoint/voices/NAME.safetensors
```

Each file must contain exactly one F32 tensor named `speaker_wavs`, shape
`[1, 512, T]` with positive T. Import adds `voices.default` and `voices.NAME`.
Inference can then select `"voice": "NAME"` with the existing API.

These must be compatible speaker-codec latents. Reshaping a Qwen speaker vector
or using its audio-understanding encoder is not a valid conversion. The
publisher's `clone-voice.py` uploads recordings to Gradium's cloud API and
downloads latents; it does not supply an offline encoder. No recording has
been uploaded by this work. Preparing voice files also does not create an API
for uploading recordings per request.

## API contract and suite acceptance

| Field or behavior | Official Ollama 0.40.11 |
|---|---|
| Endpoint | `/v1/audio/speech` |
| Required | `model`, `input` |
| Input bound | 4096 Unicode characters |
| Voice | Prepared name, or `default` |
| Output | Binary WAV or PCM; mono, 24 kHz, signed 16-bit little-endian |
| Streaming WAV | Lengths use `0xffffffff`; count received samples |
| Speed / stream format | `1` / `audio` |
| Instructions | Nonempty values rejected |
| Reference recording / transcript | No defined fields or cloning contract |
| Discovery | `/api/show` capabilities and voices |

```bash
.venv-bridge/bin/python -m training_suite ollama-speech-smoke YOUR_TAG \
  --ollama-url http://127.0.0.1:11434 \
  --text "Read these exact words." \
  --audio-out training_suite/outputs/native-speech.wav \
  --out training_suite/outputs/native-speech-report.json
```

This structural check requires `speech`, calls the endpoint, validates actual
binary samples, and rejects empty output, incomplete PCM16, digital silence,
and JSON envelopes. It does not certify transcription, speaker similarity, or
authenticity of a server impersonating Ollama. Use the actual official daemon
or its broker and independently verify the binary. Before a CUDA workload,
follow `docker gpu discover` and the exact-scope lease protocol.

`omni-plan` reports `native_speech_delivery.ready=false` with the official
binary constraint and unsupported-graph blocker, even when language interfaces
match. Its existing packing mode describes the sidecar experiment only.
Intake cannot classify missing native Qwen speech as a Modelfile-only repair.
The `omni-audio-smoke` adapter cascade cannot satisfy this delivery gate.
Release evidence must eventually include native WAV/PCM generation, independent
transcription, intended voice similarity, cancellation, and resource release
from the actual exported weights.

## Primary sources

- [Ollama speech API](https://docs.ollama.com/capabilities/speech)
- [0.40.11 handler](https://github.com/ollama/ollama/blob/v0.40.11/server/speech.go) and [runner request](https://github.com/ollama/ollama/blob/v0.40.11/llm/speech.go)
- [Phonon graph](https://github.com/ollama/ollama/blob/v0.40.11/mlxrunner/model/phonon/model.go), [voice importer](https://github.com/ollama/ollama/blob/v0.40.11/create/phonon.go), and [PCM conversion](https://github.com/ollama/ollama/blob/v0.40.11/mlxrunner/speech.go)
- [Architecture selection](https://github.com/ollama/ollama/blob/v0.40.11/mlxrunner/model/model.go)
- [Qwen3-Omni graph](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py) and [checkpoint config](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct/blob/main/config.json)
- [Phonon architecture and unreleased speaker codec](https://huggingface.co/docs/transformers/main/en/model_doc/gradium_phonon_tts)
- [Phonon training objective](https://github.com/huggingface/transformers/blob/6c5f8f4186c15f4bfbd67941a1736b2b347e0155/src/transformers/models/gradium_phonon_tts/modeling_gradium_phonon_tts.py) and [processor encoder checks](https://github.com/huggingface/transformers/blob/6c5f8f4186c15f4bfbd67941a1736b2b347e0155/src/transformers/models/pocket_tts/processing_pocket_tts.py)
- [Published codec config](https://huggingface.co/gradium/phonon-mimi/blob/33519be1a8e61c6fa00c6e154e2d38e76a367919/config.json)
- [Publisher's cloud cloning script](https://github.com/gradium-ai/xn-ptts/blob/main/scripts/clone-voice.py)
