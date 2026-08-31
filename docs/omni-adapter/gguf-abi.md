# Monolithic GGUF ABI

This document defines how the Omni adapter stores three independently
executable graphs inside one GGUF v3 file. The artifact schema is
`robit.ollama-monolithic-audio.v1`; it is distinct from the API schema
`robit.ollama.omni-adapter.v1`.

## Goals

- Produce exactly one `.gguf` file and one Ollama `FROM` model layer.
- Leave the existing Qwen3.8/Ornith base model loadable under its original
  tensor names and metadata.
- Preserve complete comprehension and TTS graphs without pretending their
  hidden states are shape-compatible with the base language graph.
- Retain component provenance and hashes for release verification.
- Permit lazy loading and independent eviction of the embedded components.

## Tensor namespaces

| View | Stored tensor name | Name exposed to component loader |
|---|---|---|
| Base language | `blk.0.attn_q.weight` | `blk.0.attn_q.weight` |
| Comprehension | `a.c.blk.0.attn_q.weight` | `blk.0.attn_q.weight` |
| TTS | `s.t.blk.0.attn_q.weight` | `blk.0.attn_q.weight` |

The prefixes are storage namespaces. They are stripped only in the filtered
view passed to the relevant component loader. Loader code MUST NOT rewrite the
file in place.

Reserved prefixes:

- `a.c.` — self-contained audio/image/video comprehension graph;
- `s.t.` — independently text-conditioned TTS graph.

The packer rejects a base GGUF that already contains either prefix.

## Metadata namespaces

The base GGUF metadata remains at its original keys. Component metadata is
copied verbatim beneath a component prefix:

| Stored key | Meaning |
|---|---|
| `general.architecture` | Architecture of the unprefixed base graph |
| `robit.audio_bundle.schema` | Artifact schema identifier |
| `robit.audio_bundle.manifest` | Compact JSON manifest and wire contract |
| `robit.audio_bundle.tensor_prefixes` | JSON list of reserved tensor prefixes |
| `robit.audio_bundle.component.comprehension.kv.<original-key>` | Comprehension GGUF key |
| `robit.audio_bundle.component.tts.kv.<original-key>` | TTS GGUF key |

For example, the comprehension component's original `general.architecture` is
stored as:

```text
robit.audio_bundle.component.comprehension.kv.general.architecture
```

The filtered comprehension metadata view removes the namespace and exposes the
key again as `general.architecture`.

## Manifest

`robit.audio_bundle.manifest` is JSON serialized without insignificant
whitespace. Its important fields are:

```json
{
  "schema": "robit.ollama-monolithic-audio.v1",
  "physical_artifacts": 1,
  "container": "GGUF v3",
  "ollama_import": "FROM ./model.gguf",
  "base": {
    "source": "manitcor/Qwen3.8-27B-Obliterated-E03",
    "architecture": "qwen3_5_text",
    "tensor_namespace": "unmodified",
    "role": "reasoning-tools-thinking"
  },
  "components": [{
    "name": "comprehension",
    "tensor_prefix": "a.c.",
    "input_modalities": ["audio", "video", "text"],
    "output_modalities": ["text"]
  }, {
    "name": "tts",
    "tensor_prefix": "s.t.",
    "input_modalities": ["text"],
    "output_modalities": ["audio"]
  }],
  "contract": {
    "schema": "robit.ollama.omni-adapter.v1"
  }
}
```

The pack report records each component's source label, filename, byte size,
SHA-256 digest, architecture, tensor count, and storage prefix. A release
manifest SHOULD additionally record exact upstream revisions, conversion tool
commits, quantization commands, licenses, and the custom Ollama runtime commit.

## Filtered view algorithm

The custom loader opens the same physical file for every view:

```text
base view:
  include tensors not beginning with a.c. or s.t.
  expose unprefixed metadata

comprehension view:
  include tensors beginning with a.c.
  strip a.c. from each visible tensor name
  include metadata beginning with
    robit.audio_bundle.component.comprehension.kv.
  strip that metadata prefix

tts view:
  include tensors beginning with s.t.
  strip s.t. from each visible tensor name
  include metadata beginning with
    robit.audio_bundle.component.tts.kv.
  strip that metadata prefix
```

Pseudocode:

```text
open_monolithic(path):
  gguf = parse_header_and_index(path)
  require gguf["robit.audio_bundle.schema"] == supported_schema

  base = filtered_view(gguf, reject_prefixes=["a.c.", "s.t."])
  comprehension = filtered_view(
      gguf,
      tensor_prefix="a.c.",
      metadata_prefix="robit.audio_bundle.component.comprehension.kv.")
  tts = filtered_view(
      gguf,
      tensor_prefix="s.t.",
      metadata_prefix="robit.audio_bundle.component.tts.kv.")

  return base, comprehension, tts
```

Every component loader MUST see exactly the tensor names, shapes, dtypes, and
metadata it would see if its component GGUF were opened independently. Missing,
duplicate, overlong, or unexpectedly visible tensor names are fatal.

## Component requirements

### Base language graph

The unprefixed graph is the Ollama-facing Qwen3.8 or Ornith model. It owns:

- tokenization and chat rendering;
- language reasoning and final response text;
- parsed thinking;
- structured tool calls;
- native vision, if the base artifact already has a compatible vision tower.

### Comprehension graph

For the first implementation, `a.c.*` MUST be self-contained and return text or
another stable semantic representation. A bare Qwen3-Omni audio/vision encoder
is not sufficient when the target language trunk has a different hidden width,
vocabulary, or special-token map.

A future trained bridge may replace the text boundary only if its architecture,
weights, tokenizer alignment, and runtime graph are versioned in a new bundle
schema.

### TTS graph

`s.t.*` MUST accept ordinary text independently. Do not pack only an Omni Talker
whose conditioning input is tied to another Thinker's hidden states. Qwen3-TTS
or another text-conditioned model plus its codec/decoder is the intended first
component.

If the speech model requires a separate codec graph, that codec MUST be included
inside the `s.t.*` component GGUF and described by its component metadata. One
logical TTS component may contain several internal subgraphs.

## Conversion and quantization order

Each component is converted and validated independently:

```text
safetensors → component F16/BF16 GGUF → component quantized GGUF
                                               ↓
base GGUF + comprehension GGUF + TTS GGUF → omni-pack → one model.gguf
```

Never run an ordinary text-model quantizer over the packed file. Such a tool may
skip unknown tensor namespaces or apply the base architecture's quantization
rules to incompatible components. Quantize first, then pack byte-preserving
tensor payloads.

## Packer and inspector

```bash
training_suite/.venv/bin/python -m training_suite omni-pack \
  --base-gguf ./components/qwen38.q4_k_m.gguf \
  --comprehension-gguf ./components/qwen3-omni-comprehension.q4_k_m.gguf \
  --tts-gguf ./components/qwen3-tts.q4_k_m.gguf \
  --base-source manitcor/Qwen3.8-27B-Obliterated-E03 \
  --comprehension-source Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --tts-source Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --out ./release/model.gguf \
  --renderer qwen3.8 \
  --parser qwen3.5

training_suite/.venv/bin/python -m training_suite omni-inspect ./release/model.gguf
```

The pack is written to a sibling `.partial` file and atomically renamed only
after serialization. The inspector requires nonzero tensor counts in all three
views and a supported manifest.

## Ollama import behavior

The generated Modelfile uses one reference:

```text
FROM ./model.gguf
RENDERER qwen3.8
PARSER qwen3.5
```

Official Ollama supports importing a GGUF through `FROM /path/to/file.gguf`, but
stock model execution does not interpret this repository's component namespaces
or wire fields. The custom build must keep the embedded tensors in the model
blob and select the base view for ordinary text loading.

## Verification invariants

Before runtime testing:

1. `omni-inspect` reports the expected schema and nonzero counts for every view.
2. The output SHA-256 is recorded after the final write.
3. The base view's tokenizer metadata and tensor inventory match the source base
   GGUF.
4. The stripped comprehension view matches its source component inventory.
5. The stripped TTS view matches its source component inventory.
6. Each component can allocate and execute a minimal probe in isolation.
7. The base text/tools/thinking regression suite remains green.

The one-file property is satisfied by a single GGUF and a single Ollama model
blob. Memory mapping regions or creating multiple execution contexts over that
file does not violate the property.
