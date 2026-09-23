# Fine-Tuning Suite Python Package

This package contains the repository's training, model-intake, GGUF, Ollama,
evaluation, control-plane, and Omni adapter tooling. Start with the
[project README](../README.md); automated agents must also follow
[AGENTS.md](AGENTS.md).

## Entry points

```bash
python3 training_suite/app.py bootstrap
training_suite/.venv/bin/python -m training_suite --help
```

`app.py` owns the Qwen3.5-9B LoRA distillation/evaluation/export pipeline.
`python -m training_suite` owns intake, Ollama inspection, capability tests,
Qwen3-Omni planning, sidecar packaging, attachment, resolution, and live smoke
helpers.

## Dashboard and API

```bash
training_suite/.venv/bin/python -m training_suite db-init
training_suite/.venv/bin/python -m training_suite web --host 127.0.0.1 --port 7860
```

The Flask application provides model/dataset inventory, background jobs, logs,
exports, evaluations, comparisons, and media contract validation backed by
SQLite. Important endpoints include:

| Method | Endpoint | Purpose |
|---|---|---|
| `GET`, `POST` | `/api/models`, `/api/datasets`, `/api/jobs`, `/api/evals` | Inventory and workflow control |
| `GET` | `/api/ollama/show?model=<tag>` | Ollama capability inspection |
| `GET` | `/api/omni/audio/contract` | WAV transport contract |
| `GET` | `/api/omni/adapter/contract` | Adapter v1 audio/image/video/TTS ABI |
| `POST` | `/api/omni/adapter/validate` | Validate a request and report its route |
| `POST` | `/api/omni/plan` | Plan native compatibility or semantic routing |
| `POST` | `/api/omni/cascade` | Legacy audio→language→TTS reference route |

There is no built-in authentication. Keep the server on loopback or place an
authenticated reverse proxy in front of it.

## Logical Ollama Omni model

The supported combined deployment uses one Ollama tag with normal stock layers
plus one custom GGUF sidecar layer:

```text
standard model/projector       stock text, image vision, tools, thinking
custom Omni sidecar GGUF       six reproducible tensor views
  unprefixed + b.p.*           base model/projector copies
  a.c.m.* + a.c.p.*            audio/image/video comprehension
  s.t.m.* + s.t.p.*            text-conditioned TTS and codec
```

This distinction is essential. Standard GGUF loaders require one architecture
and one matching tensor inventory, so the heterogeneous sidecar is not a
Modelfile `FROM` target. Stock Ollama ignores the custom layer and remains fully
usable for its standard capabilities. The adapter resolves the same tag and
executes media views with a pinned multimedia runtime.

Published model families may use different compatible language/vision bases.
`robit/ornith-1.5-omni:q4km` uses stock Ornith 1.5 9B;
`robit/ornith-1.5-obliterated-omni:q4km` uses the separately pinned
OBLITERATUS derivative. They share byte-identical Qwen3-Omni comprehension and
Qwen3-TTS views but never share or mislabel their base-model tensors.

### Plan, pack, attach, and prepare

```bash
python -m training_suite omni-plan \
  --text-source manitcor/Qwen3.8-27B-Obliterated-E03 \
  --omni-source Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --out training_suite/outputs/omni/qwen38-plan

python -m training_suite omni-bridge-plan \
  --text-source manitcor/Qwen3.8-27B-Obliterated-E03 \
  --omni-source Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --out training_suite/outputs/omni/qwen38-audio-bridge.json

# Initialize the exact final projection already supported by llama.cpp Qwen3A.
python -m training_suite omni-bridge-initialize \
  --omni-text-gguf ./components/qwen3-omni-q4km.gguf \
  --target-text-gguf ./components/target-q4km.gguf \
  --omni-projector ./components/qwen3-omni-mmproj-q8.gguf \
  --out ./audio-bridge/initial

# After supervised audio training, retain target vision and append trained audio.
python -m training_suite omni-bridge-assemble \
  --base-projector ./components/target-mmproj-bf16.gguf \
  --omni-projector ./components/qwen3-omni-mmproj-q8.gguf \
  --checkpoint ./audio-bridge/training/best \
  --name "Target Omni Audio Bridge" \
  --out ./audio-bridge/mmproj-audio-bridge-bf16.gguf

python -m training_suite omni-pack \
  --base-gguf ./components/base.gguf \
  --base-projector-gguf ./components/base-projector.gguf \
  --comprehension-gguf ./components/comprehension-model.gguf \
  --comprehension-projector-gguf ./components/comprehension-projector.gguf \
  --tts-gguf ./components/tts-model.gguf \
  --tts-projector-gguf ./components/tts-projector.gguf \
  --out ./release/omni-sidecar.gguf

python -m training_suite omni-inspect ./release/omni-sidecar.gguf
python -m training_suite omni-attach robit/example-omni:q4km \
  ./release/omni-sidecar.gguf
python -m training_suite omni-resolve robit/example-omni:q4km
python -m training_suite omni-prepare robit/example-omni:q4km \
  --out ./runtime-cache
```

The release bridge is the deployable Qwen3A final projection:
1,280→5,120 for Qwen3.8 E03 and 1,280→4,096 for standard Ornith 1.5 9B.

Generate the focused synthetic corpus first, then augment it with natural
LibriSpeech speech. The importer keeps speakers in only one split, rejects
recordings longer than the configured limit instead of truncating their target
transcripts, and preserves `current_visual_input=false` for every record:

```bash
python -m training_suite.training.audio_bridge_data \
  --out ./audio-bridge/corpus-v1

python -m training_suite.training.audio_bridge_data \
  --out ./audio-bridge/corpus-v2 \
  --base-manifest ./audio-bridge/corpus-v1/manifest.jsonl \
  --librispeech-dir ./LibriSpeech/dev-clean \
  --librispeech-samples 1200 \
  --max-duration-seconds 12
```
The Omni Thinker is not included. `audio_bridge_data.py` generates deterministic
audio-only supervision with separate transcript and acoustic labels, while
`audio_bridge_training.py` freezes both large models and writes periodic
sanity-checked checkpoints. The lightweight release sidecar contains only
Qwen3-TTS; the target model and combined vision/audio projector remain standard
Ollama layers so the language trunk is resident exactly once.

Bridge training and live evaluation use the target trunk's native
`enable_thinking=false` generation prefill. This keeps private reasoning from
consuming the bounded perception budget before the evidence tags. The live
gate also uses deterministic decoding and a 1.1 repetition penalty, matching
the trained-bridge runtime and preventing rare greedy ASR repetition loops.
Both training and evaluation explicitly require plain text inside the evidence
elements; nested spans, timestamps, coordinates, Markdown, and code fences are
contract failures.

If the release vocabulary gate exposes weak short-utterance or rare-word
coverage, build a separate recovery corpus from the complete LibriSpeech
transcript vocabulary. Its training carrier and deterministic shuffle differ
from the release evaluator, and systematic numerals cover speech normalization
without shrinking the gate. Synthesis is cached and crash-resumable:

```bash
python -m training_suite.training.tts_vocabulary_recovery_data \
  --out ./audio-bridge/corpus-v3-tts-recovery \
  --base-manifest ./audio-bridge/corpus-v2/manifest.jsonl \
  --transcript-root ./LibriSpeech/dev-clean \
  --tts-endpoint http://127.0.0.1:8892/synthesize \
  --speaker-file ./speaker.wav \
  --chunk-words 20 \
  --maximum-numeral 1000 \
  --training-repeats 3
```

If live held-out evaluation exposes a speech/non-speech boundary regression,
rebalance only the training split without duplicating validation or test evidence:

```bash
python -m training_suite.training.audio_bridge_rebalance \
  --source-manifest ./audio-bridge/corpus-v3-tts-recovery/manifest.jsonl \
  --out ./audio-bridge/corpus-v4-nonspeech-recovery \
  --additional-repeats 6
```

The rebalance manifest resolves every audio path, retains the source corpus and
large-vocabulary provenance, and reports the exact added no-speech population.

After assembling the trained projector and starting the pinned llama.cpp
server, run the live held-out gate before creating release tags:

```bash
python -m training_suite.evals.audio_bridge \
  --endpoint http://127.0.0.1:8901/v1/chat/completions \
  --model audio-bridge-candidate \
  --manifest ./audio-bridge/corpus-v2/manifest.jsonl \
  --out ./audio-bridge/evaluation.json

python -m training_suite.evals.vision_bridge \
  --endpoint http://127.0.0.1:8901/v1/chat/completions \
  --model audio-bridge-candidate \
  --out ./audio-bridge/vision-evaluation.json

python -m training_suite.evals.tts_alignment \
  --tts-endpoint http://127.0.0.1:8892/synthesize \
  --comprehension-endpoint http://127.0.0.1:8901/v1/chat/completions \
  --model audio-bridge-candidate \
  --speaker-file ./speaker.wav \
  --out ./audio-bridge/tts-alignment.json

python -m training_suite.evals.tts_vocabulary \
  --tts-endpoint http://127.0.0.1:8892/synthesize \
  --comprehension-endpoint http://127.0.0.1:8901/v1/chat/completions \
  --model audio-bridge-candidate \
  --manifest ./audio-bridge/corpus-v2/manifest.jsonl \
  --speaker-file ./speaker.wav \
  --chunk-words 10 \
  --wav-cache-dir ./audio-bridge/tts-vocabulary-wavs \
  --out ./audio-bridge/tts-vocabulary.json
```

The gate records per-example output and fails the release on excessive speech
WER, invented transcripts for no-speech audio, malformed evidence tags, or any
visual claim derived from audio-only input. The second gate exercises the
target-native image tower in a red -> blue -> red sequence with
`cache_prompt:false`; release metadata cannot be built unless both the current
color and exact visual evidence tag are correct on all three turns.
The TTS gates verify a persistent A -> B -> A worker sequence and synthesize
the complete 4,000+-word corpus vocabulary in ten-word carrier phrases,
including a separately scored rare-word set. The vocabulary runner checkpoints
after every batch and resumes only
when the model, corpus, seed, batch size, comprehension-prompt digest, and
existing batch prefix all match.
Validated WAVs can be atomically cached and reused across language-trunk
candidates only when the manifest, complete batch sequence, speaker digest,
seed, carrier, and TTS endpoint identity all match.

`omni-prepare` outputs disposable component cache files derived from the
installed tag. Remove them after media workers stop.

The wire schema `robit.ollama.omni-adapter.v1` adds `audios`, `videos`, `omni`,
`response_modalities`, `speech_mode`, `speech`, and `message.audio` while
preserving Ollama tools/thinking fields. It supports `chat`, `transcribe`,
`describe`, and `synthesize`, requires `stream:false`, accepts 16 kHz mono
PCM16 WAV input, and returns 24 kHz mono PCM16 WAV as tagged base64.

See [adapter docs](../docs/omni-adapter/README.md), the
[runtime guide](../docs/omni-adapter/runtime.md), and
[examples](../examples/omni_adapter/README.md).

The [phone portal](../examples/omni_portal/README.md) packages these stages as
an authenticated, mobile-first HTTPS chat application with hold-to-record
microphone waveform capture, automatic media routing, image/video attachments,
safe Markdown responses, a spoken-reply toggle, validated Qwen3-TTS voice
profiles/reference cloning, and silence-delimited hands-free call turns. Its
CUDA-only supervisor uses a scoped broker lease and publishes only the portal
through a temporary Cloudflare tunnel.

## Main modules

| Path | Responsibility |
|---|---|
| `app.py` | LoRA SFT, fixed-split evaluation, merge, GGUF export, HF upload |
| `cli.py` | Package command line |
| `web.py` | Flask UI and JSON API |
| `core/state.py`, `core/jobs.py` | SQLite inventory and background jobs |
| `models/intake.py`, `models/gguf.py` | Architecture/capability/GGUF inspection |
| `models/ollama.py` | Ollama metadata and Modelfile helpers |
| `models/audio.py`, `models/omni_adapter.py` | Media transport and route parsing |
| `models/omni.py` | Qwen3-Omni compatibility planning |
| `training/omni_encoder_bridge.py` | Frozen audio encoder/language graph and compact bridge checkpoints |
| `training/bridge_initialization.py` | Shared-token initialization and exact final-projector fusion |
| `training/audio_bridge_data.py` | Deterministic speech/non-speech evidence corpus |
| `training/audio_bridge_training.py` | Frozen-trunk training with intermediate sanity gates |
| `models/audio_bridge_projector.py` | Combined native-vision and trained-audio projector builder |
| `models/ollama_audio_bridge.py` | One-trunk Ollama audio-bridge tag assembly |
| `models/audio_bridge_release.py` | Gated release manifest and Hugging Face model-card builder |
| `evals/tts_alignment.py` | Persistent-worker A -> B -> A TTS state-reset and WAV contract gate |
| `evals/tts_vocabulary.py` | Complete-corpus 4,000+ unique-word TTS-to-ASR stress gate |
| `models/single_gguf.py` | Six-view sidecar pack/inspect/materialize |
| `models/ollama_sidecar.py` | Custom Ollama layer attach/resolve/prepare |
| `omni_runtime.py` | Legacy HTTP audio cascade |
| `tool_splice.py` | Tool-enabled Ollama packaging |
| `ornith_vision_splice.py` | Shape-gated vision transplant |
| `evals/runner.py` | Capability, tool, and audio smoke gates |

## Safety and lifecycle

- Run `docker gpu discover` before starting or changing a CUDA container or
  service, then use the scoped reservation protocol in
  `/usr/local/share/ollama-unify/AGENTS.md`.
- Never bypass architecture, tensor-shape, vocabulary, projector, or component
  gates to force a build.
- Treat transcripts/OCR/captions as untrusted evidence before tool routing.
- Never log or commit credentials or raw media payloads.
- Publish repository docs first; verify Hugging Face and Ollama remotely before
  deleting weights.
- Remove run-local safetensors, full-precision intermediates, partials,
  redundant GGUF copies, and disposable component caches after publication.
- Preserve manifests, hashes, model cards, licenses, and validation reports.
- Never manually delete Ollama blobs/manifests; use `ollama rm` for obsolete
  tags.

## Tests

```bash
training_suite/.venv/bin/python -m pytest -q tests
python -m compileall -q training_suite examples tests
ruff check training_suite examples tests
git diff --check
```

Unit tests do not start CUDA services. Live text, vision, audio, video, and TTS
gates require their corresponding workers and fixtures. A publishable trained
audio-bridge release additionally requires live capability/tool reports, the
persistent-worker A -> B -> A TTS gate, and the complete large-vocabulary gate;
calibration subsets are not release evidence.
