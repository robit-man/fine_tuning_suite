# Phonon cloning and ASR with official Ollama

This is the historical prompt-inversion experiment. The working successor,
[Phonon Echo](../../docs/phonon-hybrid.md), uses pretrained reference voice
conversion and ASR inside an Ollama-managed hybrid runner. It passed native
speech, transcription, and speaker-state checks; the failed results below
belong to the earlier approach.

The hard fixture is the current official executable and `/v1/audio/speech`.
On 2026-10-26, GitHub's latest release and the installed daemon both report
**0.40.11**. The request snapshot is
[`tests/fixtures/ollama-speech-0.40.11.json`](../../tests/fixtures/ollama-speech-0.40.11.json).
This is a compatibility test, not an installation/version gate.

## Implemented offline preparation

[`phonon_voice.py`](../../training_suite/training/phonon_voice.py) freezes the
released Phonon model and Mimi decoder, optimizes continuous decoder latents
against a recording using spectral reconstruction, then teacher-forces the
transcript/latents and fits only the speaker prompt. Export writes exactly one
F32 `speaker_wavs` tensor `[1,512,T]`. Put it in `voices/NAME.safetensors`, use
official `ollama create`, and select `"voice":"NAME"`. Python is preparation
tooling only; audio is never uploaded. The fitting function supports multiple
recordings sharing one prompt.

```bash
uv venv --python /usr/bin/python3 --system-site-packages .venv-phonon
uv pip install --python .venv-phonon/bin/python -r training_suite/requirements-phonon.txt
CUDA_VISIBLE_DEVICES='' .venv-phonon/bin/python -m training_suite.training.phonon_voice \
  --checkpoint training_suite/outputs/phonon-hybrid/checkpoints/phonon-31m-en \
  --codec-checkpoint training_suite/outputs/phonon-hybrid/checkpoints/phonon-mimi \
  --reference reference.wav --text "Exact recorded words." \
  --out training_suite/outputs/phonon-hybrid/candidate
```

Use the pinned checkpoint revisions in the
[compatibility assessment](../../docs/omni-adapter/native-speech.md). The CPU
tool requires existing local checkpoints. No compatible speaker encoder is
assumed; this experiment tests an alternative based on prompt inversion.

## Native API results

The pilot used an existing 1.92-second Qwen3-TTS-generated fixture, "Copper
lighthouse seven." It was not a new human voice enrollment. Sixty inversion
steps reduced spectral loss **2.9795 -> 1.2400**; eighty fitting steps reduced
the fixed-noise objective **0.9513 -> 0.2128**. These losses do not prove cloning.

The `[1,512,125]` prompt imported as `reference-fit` in
`phonon-clone-poc:latest`. Official Ollama listed both voices and produced:

| Native request | Result |
|---|---|
| Fitted voice, held-out "Blue river eight.", WAV | HTTP 200; 51,840 frames |
| Default voice, same text, WAV | HTTP 200; 42,240 frames |
| Fitted voice, reference text, WAV | HTTP 200; 63,360 frames |
| Fitted voice, held-out text, PCM | HTTP 200; 48,000 frames |
| Same tag, `/v1/audio/transcriptions` | HTTP 400: does not support chat |

All generated audio passed the 24 kHz mono PCM16 structural gate. The official
Linux installation initially lacked its separate MLX payload. Installed
`ollama-linux-amd64-mlx.tar.zst`, SHA-256
`0122fdb0f929db3b2fc43c0b82bd1a562152684814f37f404415b2a118e27a31`.
The executable remained unchanged, SHA-256
`b0a4becfaf872e3ad5d2e9051244e0dfe55409f03d6e2cd9bd9c34400e201b4c`.

A device-fenced CPU fixture failed because this Linux MLX allocator needs CUDA.
The first GPU request exceeded 90 seconds during loading. A subsequent bounded
load succeeded; warm requests completed in milliseconds. The official fixture
server ran on 127.0.0.1:11470 with a 4,096 MiB broker lease on exact UUID
`GPU-3538758b-6ba7-fa1b-a80c-e315a606c3f0`. Readiness proved runner residency at
642 MiB. The temporary server/runners stopped and the lease was released; other
owners were preserved. The main broker and metadata service were not restarted.
This is not a deployed public broker speech lane.

## Quality result: failed candidate

WavLM speaker verification and Whisper transcription ran offline for evaluation;
they are not deployed workers or ASR added to Phonon.

| Output | Speaker cosine to reference | Whisper transcript |
|---|---:|---|
| Another fixture from the reference voice | 0.8633 | Not scored here |
| Native default, stochastic | 0.7740 | Blue River 8 |
| Native fitted, stochastic | 0.6287 | Boog river eat! |
| Native default, deterministic | 0.7544 | Blue River 8 |
| Native fitted, deterministic | 0.7989 | Bloor and wait! |

Deterministic inference used a separately imported configuration with
temperature zero, through the same API. It improved speaker similarity but
failed word fidelity. The WER tool counts digit/word spellings separately; raw
transcripts expose that limitation. **Do not publish this candidate as a
working clone.** A later CPU pilot fit one prompt across three utterances,
reducing its objective from 1.0929 to 0.3606. Its held-out preview was transcribed
as "Play River 8!", still failing exact-word fidelity. The inverted first
reference was transcribed as "corporate lighthouse seven", exposing an error
in the reconstruction targets themselves. These CPU results remain separate
from native generation evidence.

Further training must preserve linguistic content while learning identity.
Candidates include constrained speaker prompts, several reference recordings
with held-out sentences, and a learned offline encoder supervised by compatible
native voice prompts. Lower reconstruction loss alone is insufficient.

## ASR and a single hybrid tag

Official 0.40.11 has native `/v1/audio/transcriptions`, implemented through chat
and an audio-capable model such as Gemma4. The suite now sends actual multipart
audio and checks returned words:

```bash
.venv-bridge/bin/python -m training_suite ollama-transcription-smoke MODEL \
  --audio reference.wav --expected-text "Exact recorded words." --out asr-report.json
```

Phonon's implementation has a nil text tokenizer, context length zero, and no
text-decoding forward implementation. ASR tensors cannot add that method. Its
native transcription request was tested and rejected. No registered graph was
found that executes both Phonon speech and ASR from one tag. A release with two
native tags is different from one hybrid graph and must not be substituted
silently. No complete hybrid model or ASR-capable Phonon artifact was produced.

## Evidence and retention

`training_suite/outputs/phonon-hybrid/` retains checkpoints, candidate prompts,
WAV/PCM, fitting/quality reports, native readiness, discovery snapshots, server
logs, and official runtime digest receipts. Weights remain because quality and
publication gates have not passed. The verified 1.23 GB installation archive
was removed after installation; its receipt remains. No model was pushed.

Primary sources: [speech docs](https://docs.ollama.com/capabilities/speech),
[official runtime asset](https://github.com/ollama/ollama/releases/tag/v0.40.11),
[Phonon graph and text placeholders](https://github.com/ollama/ollama/blob/v0.40.11/mlxrunner/model/phonon/model.go),
[native transcription middleware](https://github.com/ollama/ollama/blob/v0.40.11/middleware/openai.go),
[Phonon architecture](https://huggingface.co/docs/transformers/main/en/model_doc/gradium_phonon_tts).
