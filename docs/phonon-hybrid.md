# Phonon Echo: official Ollama TTS and optional cloning/ASR

The current companion is **`robit/phonon-echo:clone-asr-v2`**, with baked
`female`, `robot`, and `male` profiles plus `default`. This suite includes the
profile baker, checkpoint loader, and updated Ollama runner patch. The
[standalone public repository](https://github.com/robit-man/phonon-echo-ollama)
provides the same runtime as a separate installation. Its
[model README](https://github.com/robit-man/phonon-echo-ollama/blob/main/OLLAMA_README.md)
is intended for the published Ollama model page. `:clone-asr-v1` remains
compatible and has only `default` baked in.

**`robit/phonon-echo:hybrid-v1` now works in unmodified official Ollama 0.40.11
for TTS.** `:tts-v1` and `:latest` name the same native speech model. The original
extension-only composite is preserved separately as **`:clone-asr-v1`**.

The earlier extension-only publication did not meet the operator's requirement.
The corrected TTS tag uses the actual supported Phonon flow/Mimi graph, its
canonical tensors, and public default speaker latents. The official binary
executes those weights directly; it needs no custom runner or Python inference
worker. This TTS variant is prepared from the published Phonon checkpoint, not
a jointly trained new model. Its generation temperature is set to zero.

## TTS with the official binary

Use official Ollama 0.40.11 with its official MLX runtime. On Linux the latter is
also distributed as `ollama-linux-amd64-mlx.tar.zst` in the
[official release](https://github.com/ollama/ollama/releases/tag/v0.40.11).
The official Linux speech graph currently uses CUDA, so this host runs it under
a scoped GPU lease on port **11470**. The broker's CPU metadata daemon on 11436
must not be used for speech inference. On a normal single-daemon installation,
use its usual port 11434 instead.

```bash
OLLAMA_HOST=127.0.0.1:11470 /usr/local/bin/ollama pull robit/phonon-echo:hybrid-v1
curl http://127.0.0.1:11470/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"robit/phonon-echo:hybrid-v1","input":"The small robot speaks clearly.","voice":"default"}' \
  --output speech.wav
```

`response_format: "pcm"` produces raw 24 kHz mono PCM16 samples. The default
response is streaming WAV. The official endpoint has no per-request reference
recording field; the native tag advertises only `speech`, not ASR or cloning.

Reproduce the prepared native tag without installing the extension:

```bash
python -m training_suite phonon-native-tts prepare --out training_suite/outputs/phonon-official-tts
OLLAMA_HOST=127.0.0.1:11470 /usr/local/bin/ollama create robit/phonon-echo:hybrid-v1 \
  -f training_suite/outputs/phonon-official-tts/Modelfile
```

Preparation verifies five pinned source SHA-256 digests and retains the Apache
license. [deploy/phonon-official-tts.service](../deploy/phonon-official-tts.service)
shows this host's official-binary service and GPU lease. Adapt the host paths
and GPU UUID for another machine; discover leases before starting CUDA work.

The official executable and its actual MLX runner generated WAV and PCM from
the corrected tag. An independent Whisper evaluation recovered three held-out
sentences exactly after case/punctuation normalization. The executable's SHA-256
remained `b0a4becfaf872e3ad5d2e9051244e0dfe55409f03d6e2cd9bd9c34400e201b4c`.
These tests establish native TTS, not native reference cloning or ASR.

## Optional cloning and ASR composite

The separate `robit/phonon-echo:clone-asr-v2` combines Gradium Phonon text-to-speech, its Mimi waveform decoder,
OpenVoice V2 reference-conditioned voice conversion, and Whisper English ASR.
It is a composite of pretrained components, with 111.6 million unique parameters;
it has not been jointly trained. The export is one safetensors checkpoint.
Ollama stores the component weights as 853 named tensors, plus three baked
speaker buffers in v2 and its tokenizer, processor, configuration, and licenses.

**It requires the Phonon hybrid extension to Ollama 0.40.11.** The official
unmodified binary does not implement this architecture. The extension retains
Ollama's model resolution, scheduler, speech handler, PCM/WAV transport,
streaming, and transcription route. Ollama starts and owns the hybrid runner;
the runner loads the requested tag's blobs and verifies their SHA-256 digests.
It does not call the old Omni adapter or an independently configured service.
Its manifest declares the `phonon_hybrid` runner; downloading and executing it
does not require MLX or a GPU.

The first runtime executes all four neural components on CPU in one managed
process. Its parameter weights come from the installed model tag, not the
original source checkpoint folders. The helper Python environment contains
runtime code and dependencies; it is not an alternate source of model weights.

## Install the optional composite runtime

Use Python 3.12 or newer and a Go toolchain supporting Ollama's `go 1.26.10`
directive. From a checkout of this repository:

```bash
uv venv --python /usr/bin/python3 --system-site-packages .venv-phonon
uv pip install --python .venv-phonon/bin/python -r training_suite/requirements-phonon-hybrid.txt
uv pip install --python .venv-phonon/bin/python --no-deps \
  'git+https://github.com/myshell-ai/OpenVoice.git@74a1d147b17a8c3092dd5430504bd83ef6c7eb23'
bash training_suite/build_phonon_ollama.sh
```

OpenVoice's Torch architecture is used directly. Its optional TTS/demo packages
are unnecessary; Phonon supplies the speech. The build script applies the
checked-in patch to the pinned official `v0.40.11` commit and creates a separate
`0.40.11+phonon-echo.2` binary. Source directories are keyed by the patch digest,
so an updated patch gets its own checkout rather than conflicting with an older build.
It does not overwrite an existing official Ollama installation.

Run the extended Ollama daemon, substituting absolute paths for your checkout:

```bash
export OLLAMA_PHONON_HYBRID_PYTHON="$PWD/.venv-phonon/bin/python"
export OLLAMA_PHONON_HYBRID_ROOT="$PWD"
export OLLAMA_HOST=127.0.0.1:11471
export OLLAMA_NOPRUNE=1
CUDA_VISIBLE_DEVICES=-1 training_suite/outputs/phonon-hybrid/ollama-runtime/bin/ollama serve
```

From another terminal, use that daemon to pull the model:

```bash
OLLAMA_HOST=127.0.0.1:11471 \
  training_suite/outputs/phonon-hybrid/ollama-runtime/bin/ollama \
  pull robit/phonon-echo:clone-asr-v2
```

The host service example is [deploy/phonon-hybrid.service](../deploy/phonon-hybrid.service).
Adapt its paths and account before installing it on another host.

## Generate and clone speech

Choose `default`, `female`, `robot`, or `male` with the existing `voice` field:

```bash
curl http://127.0.0.1:11471/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"robit/phonon-echo:clone-asr-v2","input":"Hello from Phonon Echo.","voice":"female"}' \
  --output speech.wav
```

For cloning, the extension accepts a bounded `reference_audio` object. The
waveform is consumed by the trained reference encoder and voice-conversion
graph; it is never treated as text or silently ignored:

```python
import base64
from pathlib import Path
import httpx

reference = base64.b64encode(Path("reference.wav").read_bytes()).decode()
with httpx.Client(timeout=180) as client:
    with client.stream("POST", "http://127.0.0.1:11471/v1/audio/speech", json={
        "model": "robit/phonon-echo:clone-asr-v2",
        "input": "The small robot speaks clearly.",
        "voice": "default",
        "reference_audio": {"format": "wav", "data": reference},
        "response_format": "wav",
    }) as response:
        response.raise_for_status()
        with Path("cloned.wav").open("wb") as output:
            for chunk in response.iter_bytes():
                output.write(chunk)
```

Per-request reference recordings are processed locally and are not persisted
in the checkpoint. The three pre-cloned profiles are included as speaker
embeddings; their source recordings are absent. No transcript of the reference
is required. Short, clean recordings work best; the speech handler
has a 1 MiB request limit, so use a short mono WAV. The reference decoder also
rejects silence, invalid samples, and recordings outside its duration bounds.

Output is 24 kHz mono PCM16. `response_format=pcm` produces raw little-endian
samples. WAV responses use unknown-length headers and stream one sentence at a
time. Count received samples rather than trusting the header's maximum length.
Input is limited to 4,096 characters; speed is 1, instructions are empty, and
stream format is audio, matching the base speech API's current controls.

## Transcribe audio with the same tag

```bash
curl http://127.0.0.1:11471/v1/audio/transcriptions \
  -F model=robit/phonon-echo:clone-asr-v2 \
  -F file=@recording.wav
```

ASR supports English recordings up to 30 seconds. This model's text output is
transcription, not general chat, tools, or reasoning. Speech generation is also
English. The exported model is not Qwen Omni and does not claim its modalities.

## Preparation and validation

Bake additional named voices from an installed hybrid tag on CPU:

```bash
CUDA_VISIBLE_DEVICES=-1 .venv-phonon/bin/python -m training_suite phonon-bake-voices \
  --models-directory /srv/ollama/models \
  --source-model robit/phonon-echo:clone-asr-v1 \
  --voice female=/path/to/female-reference.wav \
  --voice robot=/path/to/robot-reference.wav \
  --voice male=/path/to/male-reference.wav \
  --out training_suite/outputs/my-voices/model
```

`--models-directory` defaults to `OLLAMA_MODELS`, then `~/.ollama/models`.
The loader verifies SHA-256 digests of the installed manifest and tensor blobs;
no original component checkpoints are required. The baker trims outer silence,
normalizes reference amplitude, and stores `[1,256,1]` OpenVoice embeddings as
`voices.NAME` buffers. Export includes the full graph, licenses, a model card,
and a file-integrity manifest. The receipt beside the export records source
hashes and profile statistics without recording filenames or copying audio.
Use a new output directory. Existing baked names cannot be overwritten; start
from v1 for a fresh set, or extend v2 with new names. `default` is reserved.

Create a local tag from the export with the extended daemon:

```bash
printf 'FROM %s\n' "$PWD/training_suite/outputs/my-voices/model" \
  > training_suite/outputs/my-voices/Modelfile
OLLAMA_HOST=127.0.0.1:11471 \
  training_suite/outputs/phonon-hybrid/ollama-runtime/bin/ollama create my-voices \
  -f training_suite/outputs/my-voices/Modelfile
```

Named profiles survive checkpoint reloads and appear in Ollama's speech voice
inventory. The development server's `/v1/audio/voices` also lists them and
rejects attempts to overwrite them through enrollment. Before publishing a new
tag, run the suite's `ollama-speech-smoke` and `ollama-transcription-smoke`
commands against that daemon, plus reference-speaker fidelity checks.

`python -m training_suite phonon-hybrid download --out SOURCES` downloads the
pinned source checkpoints. `phonon-hybrid build` combines them, and the optional
`phonon-hybrid serve` command runs the same graph directly for development.
Production Ollama delivery uses the managed runner described above.

The native API fixture generated WAV and PCM and transcribed the results through
the same model tag. Four default/A/B/A requests reproduced the held-out sentence
"The small robot speaks clearly." with zero word errors. Independent WavLM
speaker verification measured 0.872 cosine similarity to reference A for the
first A output, compared with 0.771 for the default voice and 0.696 for output B.
The two A outputs had 0.994 cosine similarity, checking speaker-state isolation.
These are bounded fixture results, not a general speaker benchmark.

The earlier soft-prompt fitting experiment failed intelligibility and remains
archived in [phonon-cloning.md](../experiments/native-omni-speech/phonon-cloning.md).
It is not the cloning path used by this release.

## Sources and licenses

| Component | Pinned source | Published license |
|---|---|---|
| Phonon | `gradium/phonon-31m-en@92741c5b07d085ff840e26fea8ce43110b87852f` | Apache 2.0 |
| Mimi decoder | `gradium/phonon-mimi@33519be1a8e61c6fa00c6e154e2d38e76a367919` | Apache 2.0 |
| Voice converter | `myshell-ai/OpenVoiceV2@f36e7edfe1684461a8343844af60babc2efbb727` | MIT |
| ASR | `openai/whisper-tiny.en@87c7102498dcde7456f24cfd30239ca606ed9063` | Apache 2.0 model card; upstream code MIT |

License texts are included in the export and Ollama license layers.
Runtime architecture source: [OpenVoice](https://github.com/myshell-ai/OpenVoice),
[Phonon](https://huggingface.co/gradium/phonon-31m-en), and
[Ollama speech](https://docs.ollama.com/capabilities/speech).
