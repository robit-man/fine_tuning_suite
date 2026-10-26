# Phonon Echo: speech cloning and ASR in one Ollama model

Model: **`robit/phonon-echo:hybrid-v1`**.

Phonon Echo combines Gradium Phonon text-to-speech, its Mimi waveform decoder,
OpenVoice V2 reference-conditioned voice conversion, and Whisper English ASR.
It is a composite of pretrained components, with 111.6 million unique parameters;
it has not been jointly trained. The export is one safetensors checkpoint.
Ollama stores that checkpoint as 853 named tensor layers plus its tokenizer,
processor, configuration, and license layers.

**It requires the Phonon hybrid extension to Ollama 0.40.11.** The official
unmodified binary does not implement this architecture. The extension retains
Ollama's model resolution, scheduler, speech handler, PCM/WAV transport,
streaming, and transcription route. Ollama starts and owns the hybrid runner;
the runner loads the requested tag's blobs and verifies their SHA-256 digests.
It does not call the old Omni adapter or an independently configured service.

The first runtime executes all four neural components on CPU in one managed
process. Its parameter weights come from the installed model tag, not the
original source checkpoint folders. The helper Python environment contains
runtime code and dependencies; it is not an alternate source of model weights.

## Install the runtime

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
checked-in patch to the official `v0.40.11` source and creates a separate binary.
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
  pull robit/phonon-echo:hybrid-v1
```

The host service example is [deploy/phonon-hybrid.service](../deploy/phonon-hybrid.service).
Adapt its paths and account before installing it on another host.

## Generate and clone speech

The existing speech request fields work unchanged for the model's default voice:

```bash
curl http://127.0.0.1:11471/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"robit/phonon-echo:hybrid-v1","input":"Hello from Phonon Echo.","voice":"default"}' \
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
        "model": "robit/phonon-echo:hybrid-v1",
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

Reference recordings are processed locally. They and their derived enrollment
embeddings are not included in the public checkpoint. No transcript of the
reference is required. Short, clean recordings work best; the speech handler
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
  -F model=robit/phonon-echo:hybrid-v1 \
  -F file=@recording.wav
```

ASR supports English recordings up to 30 seconds. This model's text output is
transcription, not general chat, tools, or reasoning. Speech generation is also
English. The exported model is not Qwen Omni and does not claim its modalities.

## Preparation and validation

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
