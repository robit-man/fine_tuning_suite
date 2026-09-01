# Combined Model Build and Release Runbook

This runbook creates one logical Ollama Omni tag from a stock-runnable base and
one custom multi-graph GGUF sidecar, validates every advertised path, publishes
the GGUF to Hugging Face, publishes the tag to Ollama, verifies both registries,
and only then removes run-local weight caches.

## Release gates at a glance

```text
pin sources → test components → pack → attach locally → live tests
     → push repository docs → publish/verify Hugging Face
     → publish/verify Ollama → cleanup
```

Any failed gate stops promotion. Do not delete source/intermediate weights and
do not overwrite a prior public tag while validation is incomplete.

## 1. Pin inputs

Record repository, immutable revision, filename, size, SHA-256, quantization,
license, and redistribution conditions for:

- stock Ollama base model and vision projector;
- self-contained Qwen3-Omni comprehension model and projector;
- independently text-conditioned Qwen3-TTS model and codec projector;
- llama.cpp commit and build configuration;
- adapter and artifact schema versions.

Qwen3.8/Ornith and Qwen3-Omni are not hidden-state-compatible merely because
they share a family name. Run `omni-plan` and expect `monolithic-router` unless
model type, width, layers, vocabulary, tokens, projector interfaces, and speech
conditioning all match.

## 2. Convert and test components

Convert/quantize each component independently. A comprehension export must be
capable of audio/image/video-to-text on its own; a bare encoder is insufficient.
A TTS export must accept text independently and include its acoustic-code and
waveform decoder.

Quantize before packing. Projectors/codecs may need greater precision than
language matrices. Never apply a text-only quantizer to the combined sidecar.

Required component probes:

- base text, thinking, structured tool call, and native image vision;
- exact clean-speech transcription;
- image text/object description;
- temporally ordered video, with and without its audio track;
- TTS output with positive duration, PCM16, mono, 24 kHz.

## 3. Pack the sidecar

```bash
python -m training_suite omni-pack \
  --base-gguf ./components/base.gguf \
  --base-projector-gguf ./components/base-projector.gguf \
  --comprehension-gguf ./components/comprehension-model.gguf \
  --comprehension-projector-gguf ./components/comprehension-projector.gguf \
  --tts-gguf ./components/tts-model.gguf \
  --tts-projector-gguf ./components/tts-projector.gguf \
  --base-source org/base@revision \
  --base-projector-source org/base-projector@digest \
  --comprehension-source org/omni@revision:q4_k_m \
  --comprehension-projector-source org/omni@revision:mmproj-q8_0 \
  --tts-source org/tts@revision:q4_k_m \
  --tts-projector-source org/tts@revision:mmproj-q8_0 \
  --out ./release/omni-sidecar.gguf

python -m training_suite omni-inspect ./release/omni-sidecar.gguf
sha256sum ./release/omni-sidecar.gguf
```

The outputs are the sidecar, its JSON pack report, and a custom-layer
descriptor. There is intentionally no Modelfile that points `FROM` at this
heterogeneous GGUF.

Materialize all six views during validation and compare them with pinned source
digests. The cache can be deleted only after publication succeeds.

## 4. Create and attach the local tag

Create or copy the stock-runnable base tag first. Confirm that `ollama show`
advertises completion, vision, tools, and thinking. Then attach the sidecar:

```bash
python -m training_suite omni-attach \
  robit/example-omni:q4km \
  ./release/omni-sidecar.gguf

python -m training_suite omni-resolve robit/example-omni:q4km
ollama show robit/example-omni:q4km
```

The manifest must retain standard model/projector/template/parameter/license
layers and contain exactly one
`application/vnd.robit.ollama.omni.bundle.v1+gguf` layer.

## 5. Local release tests

Run the repository suite and static checks:

```bash
python -m pytest -q tests
python -m compileall -q training_suite examples tests
ruff check training_suite examples tests
git diff --check
```

Then run live gates through the exact installed tag:

```bash
python -m training_suite capability-gate robit/example-omni:q4km \
  --capability vision --capability tools --capability thinking
python -m training_suite tool-smoke robit/example-omni:q4km

python examples/omni_adapter/client.py \
  --model robit/example-omni:q4km asr ./fixtures/speech-16khz-mono.wav

python examples/omni_adapter/client.py \
  --model robit/example-omni:q4km video ./fixtures/events.mp4 \
  --fps 2 --max-frames 96 --include-audio

python examples/omni_adapter/client.py \
  --model robit/example-omni:q4km \
  --output-audio ./reports/direct-tts.wav \
  tts "This is the release audio test."
```

Also test media→language, media→language→TTS, malformed inputs, tool-call speech
deferral, and repeated TTS. Follow [testing.md](testing.md).

Before any CUDA service starts on a managed host, run `docker gpu discover` and
use a scoped UUID reservation exactly as required by
`/usr/local/share/ollama-unify/AGENTS.md`.

## 6. Push repository documentation

Commit and push implementation, protocol, source pins, validation results,
usage examples, limitations, and cleanup instructions before publishing model
artifacts. This makes the linked runtime contract available when model pages go
live.

## 7. Publish to Hugging Face

Create a model repository such as:

```text
robit-man/Qwen3.8-27B-E03-Obliterated-Omni-GGUF
```

Upload:

- the final namespaced Omni sidecar GGUF;
- a professional `README.md` model card;
- the pack report and SHA-256 manifest;
- the custom Ollama layer descriptor;
- license/notices required by every component;
- links to this protocol, runtime guide, examples, and exact source revisions.

The model card must say that the file is a custom sidecar, not a directly
loadable single-architecture GGUF. It must explain how to attach/resolve it and
which capabilities require the adapter.

Use an authenticated credential store or an environment variable read by the
Hugging Face client. Never put an access token in a command argument, model
card, shell history, log, commit, or generated manifest.

After upload, list the remote repository files, compare the remote size/digest
or Git LFS object identifier, and fetch the model card. Publication is not
complete until the remote artifact is verifiably available.

## 8. Publish to Ollama

Only after local tests and the Hugging Face verification pass:

```bash
ollama signin
ollama push robit/example-omni:q4km
ollama cp robit/example-omni:q4km robit/example-omni:latest
ollama push robit/example-omni:latest
```

Verify the registry round trip, not just command exit status:

1. Save the local manifest and expected sidecar digest.
2. Push the immutable quantized tag.
3. Remove only a disposable local alias with `ollama rm`.
4. Pull that alias from the registry.
5. Run `omni-resolve` and confirm the custom layer digest survived.
6. Run `ollama show` and a short stock capability probe.

If the registry rejects or strips the custom media type, the model has not been
published successfully; do not claim one-pull media behavior.

## 9. Rollback

Keep the previous tag/digest available until the remote round trip and an
observation period finish. On any regression, stop promotion, retain diagnostic
artifacts, identify whether the failure is packaging, component inference,
media normalization, scheduling, or registry behavior, and rebuild under a new
candidate tag.

## 10. Mandatory cleanup

Cleanup begins only when all of these are true:

- local unit/static/live gates passed;
- repository documentation was pushed;
- Hugging Face upload was remotely verified;
- every requested Ollama tag was pushed and pulled/resolved successfully;
- provenance, hashes, licenses, and results were retained.

Before deleting anything, record `du -sh` and list exact candidates. Remove:

- run-local safetensors already distilled/converted and published;
- F16/BF16 conversion intermediates;
- partial downloads;
- disposable materialized component views;
- redundant run-local copies of the sidecar once registry copies and the
  referenced Ollama blob are verified;
- temporary fixtures and generated waveforms no longer needed.

Keep compact reports, hashes, model cards, licenses, and source revisions. Do
not recursively clean a repository/cache root or an unresolved variable path.
Never manually remove Ollama blobs/manifests; uninstall obsolete tags with
`ollama rm`. Report before/after disk usage and reclaimed bytes.
