# Phonon Echo

Use these three published tags:

| Tag | Runtime | Available voices |
| --- | --- | --- |
| `robit/phonon-echo:tts` | Unmodified official Ollama | `default` |
| `robit/phonon-echo:hybrid` | Unmodified official Ollama; native TTS alias | `default` |
| `robit/phonon-echo:clone` | Phonon Echo extension | `default`, `female`, `robot`, `male` |

The [public adapter repository](https://github.com/robit-man/phonon-echo-ollama)
contains the current setup, tests, and preparation tools. Its
[Ollama model README](https://github.com/robit-man/phonon-echo-ollama/blob/main/OLLAMA_README.md)
is ready to publish and contains the complete API examples.

## Official TTS

The native tags execute Gradium Phonon's supported flow/Mimi graph in unmodified
Ollama 0.40.11 with the official MLX runtime. No adapter is needed for these tags.
On this host the official speech service listens on port 11470 under a scoped
GPU lease. Use the usual 11434 port on a standard single-daemon installation.

```bash
OLLAMA_HOST=127.0.0.1:11470 /usr/local/bin/ollama pull robit/phonon-echo:tts
curl http://127.0.0.1:11470/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"robit/phonon-echo:tts","input":"The small robot speaks clearly.","voice":"default"}' \
  --output speech.wav
```

The official Linux speech runtime uses CUDA and is separately available as
`ollama-linux-amd64-mlx.tar.zst` in the official release assets. The device-fenced
CPU metadata daemon on port 11436 must not execute CUDA speech.

## Baked profiles and ASR

Install the standalone extension from its public repository. On this host
`phonon-hybrid.service` runs that CPU-only runtime on 11471. Its `clone` tag
includes the three prepared profile embeddings in its tensor blobs. The source
recordings are not distributed and are not needed when generating speech.

```bash
curl http://127.0.0.1:11471/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"robit/phonon-echo:clone","input":"The new voice profile is ready.","voice":"female"}' \
  --output female.wav
```

Choose `robot`, `male`, or `default` in the same field. The extension also accepts
inline `reference_audio` for other voices and supports English transcription:

```bash
curl http://127.0.0.1:11471/v1/audio/transcriptions \
  -F model=robit/phonon-echo:clone -F file=@recording.wav
```

The `clone` graph combines pretrained Phonon, Mimi, OpenVoice, and Whisper. It
requires the extension and is not executable by the unmodified official binary.
It has not been jointly trained. Audio output is 24 kHz mono PCM16, with WAV or
raw PCM transport. Speech input is limited to 4,096 characters and the speech
request body to 1 MiB; English ASR recordings are limited to 30 seconds.

## Validation and preparation

All three cloned profiles passed held-out sentence transcription and WAV/PCM
checks. Independent speaker verification preferred each intended profile output
over the other profile outputs for the matching reference. The repeated female
outputs had 0.993 cosine similarity. These are bounded fixture results rather
than a general voice-cloning benchmark.

The standalone `phonon_echo.bake_voices` module prepares additional named
embeddings from local recordings. It verifies source Ollama blob digests and
exports a self-contained checkpoint with `voices.NAME` tensors. Follow the
repository's preparation instructions and preserve manifests, license texts,
and validation reports when publishing new weights.
