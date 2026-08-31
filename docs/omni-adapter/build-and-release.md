# Combined Model Build and Release Runbook

This runbook turns independently executable component GGUFs into one combined
Qwen3.8/Ornith Omni artifact, imports it with the custom Ollama build, verifies
every capability, and publishes an Ollama tag.

The production component converters and custom Ollama executor are still
required engineering work. Commands that operate today are marked accordingly;
do not substitute placeholder GGUFs in a release.

## Release inputs

Pin and record all of the following before downloading weights:

| Role | Required property | Example source |
|---|---|---|
| Base language | Working Ollama-compatible GGUF with expected text/tools/thinking | `manitcor/Qwen3.8-27B-Obliterated-E03` or an Ornith 1.5 derivative |
| Comprehension | Self-contained audio/image/video-to-text graph | Qwen3-Omni Instruct/Thinking-derived component |
| Speech | Independently text-conditioned TTS graph including required codec | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` or another compatible Qwen3-TTS release |
| Runtime | Custom Ollama commit supporting both adapter schemas and all component architectures | pinned fork commit |
| Toolchain | Converter and quantizer commits | pinned llama.cpp/fork revisions |

For each source capture repository, immutable revision, license, expected file
digests, and redistribution conditions. Model availability does not imply that
all combinations may be redistributed under one tag.

## Phase 1: architecture plan

Generate the compatibility report before model downloads or conversion:

```bash
training_suite/.venv/bin/python -m training_suite omni-plan \
  --text-source manitcor/Qwen3.8-27B-Obliterated-E03 \
  --omni-source Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --target-tag robit/qwen3.8-27b-omni:latest \
  --out training_suite/outputs/omni/qwen38-27b
```

For the current Qwen3.8/Ornith targets, expect `mode=monolithic-router`. Treat a
native-fusion result as a separate review: architecture name alone is not
enough; hidden size, layers, vocabulary, special tokens, projector outputs, and
speech conditioning must all match.

Required outputs:

- `omni_bundle.json` with architecture signatures and mismatches;
- `audio_contract.json`;
- source/revision/license manifest;
- disk and GPU capacity estimate;
- target Ollama tag and quantization policy.

## Phase 2: component conversion

Create a unique run directory, for example:

```text
training_suite/outputs/omni/qwen38-27b/run-2026-08-31/
  sources/
  components/
  release/
  reports/
```

Do not place multi-gigabyte artifacts in the Git repository index.

### Base language GGUF

Use the existing model's verified GGUF when possible. If conversion is needed,
use the converter supporting its exact architecture and then run normal
text/tools/thinking probes in Ollama.

The base is release-ready only if:

- its GGUF loads independently;
- tokenizer and chat template are correct;
- renderer/parser produce structured tool calls;
- thinking is parsed separately;
- expected vision behavior still passes if vision is included.

### Comprehension GGUF

The converter must export a complete callable comprehension path, including all
audio/vision preprocessing model tensors, projector tensors, multimodal Thinker
layers needed to produce text, and tokenizer/special-token metadata.

If the converter exports only the audio/vision encoder, stop. That encoder
cannot feed Qwen3.8 or Ornith directly without a learned bridge.

Validate independently with:

- 16 kHz speech transcription;
- non-speech audio captioning;
- image description/OCR fixture;
- silent video temporal description;
- video with aligned speech/sound;
- malformed and oversized media rejection.

### TTS GGUF

The converter must export a text-conditioned model and every decoder/codec
weight required to produce PCM. Validate independently with:

- exact-text synthesis;
- 24 kHz mono PCM output;
- short, empty, multilingual, punctuation-heavy, and long inputs;
- deterministic or documented voice selection;
- bounded output duration.

An original Omni Talker coupled to the donor Thinker is not a valid component
for this route unless a matching conditioning bridge is trained and versioned.

### Quantization

Quantize each component before packing. Record quantization type and calibration
policy per graph. Speech and projector/codec tensors may require higher
precision than language matrices; do not apply one global quantization choice
without quality evaluation.

Keep source F16/BF16 artifacts until the corresponding quantized component has
passed independent probes and the final published tag is verified.

## Phase 3: pack one GGUF

```bash
training_suite/.venv/bin/python -m training_suite omni-pack \
  --base-gguf ./components/qwen38.q4_k_m.gguf \
  --base-source manitcor/Qwen3.8-27B-Obliterated-E03@<revision> \
  --comprehension-gguf ./components/qwen3-omni-comprehension.gguf \
  --comprehension-source Qwen/Qwen3-Omni-30B-A3B-Instruct@<revision> \
  --tts-gguf ./components/qwen3-tts.gguf \
  --tts-source Qwen/Qwen3-TTS-12Hz-0.6B-Base@<revision> \
  --out ./release/model.gguf \
  --renderer qwen3.8 \
  --parser qwen3.5 \
  --requires <custom-ollama-version>
```

The command writes:

- `release/model.gguf` — the single release weight file;
- `release/model.gguf.report.json` — component sizes, SHA-256 values, tensor
  counts, metadata-copy results, and post-write inspection;
- `release/Modelfile` — one `FROM ./model.gguf` reference.

Then run:

```bash
training_suite/.venv/bin/python -m training_suite omni-inspect \
  ./release/model.gguf | tee ./reports/omni-inspect.json

sha256sum ./release/model.gguf > ./reports/model.gguf.sha256
```

Do not re-quantize the packed file.

## Phase 4: build and identify the custom runtime

The Ollama fork release must publish:

- upstream base tag/commit;
- fork commit;
- supported bundle and wire schemas;
- patched API/OpenAPI types;
- component architecture support matrix;
- build command and binary digest;
- regression-test results against ordinary Ollama models.

Build/test the runtime in a separate source checkout. Do not vendor a mutable
Ollama binary into this repository.

Before starting any CUDA-backed container or service on the managed host:

```bash
docker gpu discover
```

Use the scoped GPU reservation protocol documented in
`/usr/local/share/ollama-unify/AGENTS.md`; the externally supervised process must
use exactly its reserved GPU UUIDs.

## Phase 5: create the local Ollama tag

Run with the custom binary/daemon, not stock Ollama:

```bash
cd ./release
ollama create qwen3.8-omni-candidate:q4km -f Modelfile
ollama show qwen3.8-omni-candidate:q4km --verbose
```

Creation passes only when:

- the model manifest contains one GGUF model layer;
- the custom runtime recognizes both schemas;
- the base, comprehension, and TTS filtered views match their reports;
- a text-only request does not load media graphs unnecessarily;
- unsupported runtime versions fail before inference with a useful error.

## Phase 6: release gates

Run at least these probes against the local tag through the adapter endpoint:

```bash
# Text/tools/thinking regression
training_suite/.venv/bin/python -m training_suite capability-gate \
  qwen3.8-omni-candidate:q4km \
  --capability tools --capability thinking
training_suite/.venv/bin/python -m training_suite tool-smoke \
  qwen3.8-omni-candidate:q4km

# ASR
training_suite/.venv/bin/python examples/omni_adapter/client.py \
  --model qwen3.8-omni-candidate:q4km \
  asr ./fixtures/speech-16khz-mono.wav

# Video comprehension
training_suite/.venv/bin/python examples/omni_adapter/client.py \
  --model qwen3.8-omni-candidate:q4km \
  video ./fixtures/temporal-events.mp4 --fps 2 --max-frames 96

# Direct TTS
training_suite/.venv/bin/python examples/omni_adapter/client.py \
  --model qwen3.8-omni-candidate:q4km \
  --output-audio ./reports/direct-tts.wav \
  tts "This is the release audio test."

# Combined audio → reasoning → TTS
training_suite/.venv/bin/python examples/omni_adapter/client.py \
  --model qwen3.8-omni-candidate:q4km \
  --output-audio ./reports/combined.wav \
  chat --audio ./fixtures/question.wav --speak \
  --prompt "Answer the recorded question."
```

Also run malformed-base64, wrong sample-rate, media-size, decoder-timeout,
cancellation, concurrent-load, component-eviction, and ordinary text-only
regression tests. Full criteria are in [testing.md](testing.md).

## Phase 7: publish

Only after all gates pass:

```bash
ollama signin
ollama cp qwen3.8-omni-candidate:q4km robit/qwen3.8-omni:q4km
ollama push robit/qwen3.8-omni:q4km
```

Pull or inspect the remote manifest after the push. Confirm the remote tag points
to the expected single model-layer digest and record the registry URL.

The Ollama model page should link to:

- this adapter documentation index;
- the runnable examples;
- the exact wire schema/version;
- required custom Ollama runtime build;
- accepted audio/video formats and limits;
- tools/thinking behavior;
- limitations and licenses;
- release test report and component provenance.

Do not phrase the tag as compatible with unmodified Ollama if the stock runtime
can only execute the base text view.

## Rollback

Keep the previous public tag/digest available until remote validation and an
observation period complete. If any capability regresses:

1. Stop promoting the candidate.
2. Restore the previous tag or direct users to its immutable digest.
3. Preserve failing requests after redaction, runtime logs, bundle report, and
   component digests.
4. Determine whether the fault is protocol, media normalization, component
   conversion, quantization, filtered loading, or scheduling.
5. Rebuild under a new candidate tag; never overwrite evidence required for
   diagnosis.

## End-of-session cleanup

Cleanup is mandatory only after the final tag is created, all gates pass, the
push completes, the remote digest is verified, and reproducibility metadata is
saved.

Before deletion, list exact run-local candidates and measure the run directory:

```bash
du -sh training_suite/outputs/omni/qwen38-27b/run-2026-08-31
find training_suite/outputs/omni/qwen38-27b/run-2026-08-31 \
  -type f \( -name '*.safetensors' -o -name '*.partial' -o -name '*.f16.gguf' \) \
  -print
```

After review, remove only those explicit run-local files. Retain manifests,
hashes, Modelfiles, licenses, pack reports, test reports, and runtime revision
records. Use `ollama rm <obsolete-local-tag>` for removable local tags. Never
delete Ollama blob or manifest files directly, and never recursively clean the
repository root, a shared cache, a broad `outputs/` directory, or an unresolved
environment-variable path.

Record disk usage before and after cleanup in the release report.
