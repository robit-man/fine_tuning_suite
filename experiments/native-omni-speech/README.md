# Native Omni speech preparation experiment

The deployment goal is **an unmodified official Ollama binary plus prepared
Omni weights**, with no fork, custom runner, Omni adapter, external TTS service,
or separately installed Python inference worker.
Python in this directory is preparation/verification tooling only.

The [Phonon cloning experiment](phonon-cloning.md) now fits and exports native
voice tensors, proves generation through the official speech API, and records
the failed voice-quality and single-tag ASR checks. Native audio generation
does not establish a usable clone or completed hybrid model.

`probe_conditioning.py` checks one export transformation against the installed
Qwen reference function, using synthetic CPU tensors:

```text
projected_text_table[token] = text_projection(original_text_table[token])
```

The prepared table replaces the original text embedding lookup followed by
the positionwise projection. The reference `_get_talker_assistant_parts`
function receives either the original embeddings/projection or the folded
embeddings/identity projection. It must produce matching speech prefixes,
token IDs, and trailing text conditioning across multiple lengths and voices.

```bash
.venv-bridge/bin/python experiments/native-omni-speech/probe_conditioning.py \
  --out training_suite/outputs/native-omni-speech/conditioning-parity.json
```

Observed with PyTorch2.12.0 and Transformers5.9.0: nine cases passed; maximum
absolute tensor difference was1.1920928955078125e-7. No Thinker transformer
ran and CUDA was not initialized. This does **not** establish pretrained speech
quality, full decoding parity, or compatibility with stock Ollama.

The remaining execution graph is Omni's MoE Talker, residual codec predictor,
and code2wav decoder. Those operators are not implemented by the official
binary's speech graph. A custom runner would violate the selected delivery
constraint.

The official0.40.11 speech model registry currently implements Phonon's flow/Mimi
graph. Omni's discrete codec/MoE graph cannot be substituted by tensor renaming.
If Ollama must remain entirely unmodified, distillation into a supported speech
architecture is a separate approach that produces student weights; it is not
a lossless repackaging of Omni's original speech tensors.
That candidate remains unproven: the released Phonon codec is decoder-only and
the speaker encoder is unreleased. See the
[compatibility assessment](../../docs/omni-adapter/native-speech.md) for training
and export requirements and source evidence. This experiment is not an
Omni-to-Phonon converter and has produced no native speech artifact.
