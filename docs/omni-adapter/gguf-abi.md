# GGUF Sidecar and Ollama Layer ABI

This document defines artifact schema `robit.ollama-monolithic-omni.v3` and
its delivery as a custom layer in one Ollama model manifest. The public HTTP
schema is independently versioned as `robit.ollama.omni-adapter.v1`.

## Invariants

- The sidecar is one valid GGUF v3 file with a contiguous tensor index.
- It contains six reproducible views: base model, base projector,
  comprehension model/projector, and TTS model/projector.
- Each view materializes to the same bytes as its independently tested source
  GGUF when the source was produced by this packer.
- The stock Ollama model and projector remain ordinary standard layers.
- The sidecar is a custom layer; it is never a Modelfile `FROM` target.
- Quantization occurs per component before packing. Never quantize the bundle.

The logical model therefore has several OCI layers, while the custom media
payload is one physical GGUF. “One tag” and “one sidecar GGUF” are accurate;
“one directly runnable GGUF containing every architecture” is not.

## Tensor views

| View | Stored prefix | Example visible name after extraction |
|---|---|---|
| Base model | none | `blk.0.attn_q.weight` |
| Base projector | `b.p.` | `v.blk.0.attn_q.weight` |
| Comprehension model | `a.c.m.` | `blk.0.attn_q.weight` |
| Comprehension projector | `a.c.p.` | `a.blk.0.attn_q.weight` |
| TTS model | `s.t.m.` | `blk.0.attn_q.weight` |
| TTS projector/codec | `s.t.p.` | model-specific acoustic tensor |

Prefixes are stripped only while creating a filtered/materialized component
view. The bundle is read-only at runtime. The packer rejects a base file that
already uses a reserved prefix and rejects names longer than the GGML limit
after namespacing.

## Metadata

Base metadata remains unprefixed. Each embedded component's metadata is copied
under:

```text
robit.audio_bundle.component.<view>.kv.<original-key>
```

Bundle-level keys are:

| Key | Purpose |
|---|---|
| `robit.audio_bundle.schema` | Exact artifact schema identifier |
| `robit.audio_bundle.manifest` | Compact JSON contract, sources, and component map |

The component materializer removes the metadata namespace and restores the
original key. It must reproduce architecture, tokenizer, tensor names, shapes,
types, order, and packed tensor bytes exactly.

## Manifest shape

The embedded manifest includes:

```json
{
  "schema": "robit.ollama-monolithic-omni.v3",
  "physical_bundle_artifacts": 1,
  "container": "GGUF v3",
  "container_format": "robit-namespaced-multigraph-gguf-v1",
  "ollama_delivery": "attach as application/vnd.robit.ollama.omni.bundle.v1+gguf layer",
  "runtime": {
    "stock_ollama_direct_bundle_import": false,
    "stock_ollama_logical_tag": true,
    "custom_media_handler_required": true
  }
}
```

`component_files` records source label, filename, byte size, SHA-256,
architecture, tensor count, and storage prefix for every view. Release records
add immutable upstream revisions, licenses, toolchain commit, and live tests.

## Packing

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
```

The writer uses a sibling `.partial` file and atomically renames only after the
GGUF is complete. `omni-inspect` validates the schema, manifest declarations,
and non-empty tensor views. It does not substitute for inference tests.

## Attaching to Ollama

First create or copy a normal Ollama tag whose standard layers already pass
text, vision, tools, and thinking. Then attach the validated custom layer:

```bash
python -m training_suite omni-attach \
  robit/example-omni:q4km \
  ./release/omni-sidecar.gguf

python -m training_suite omni-resolve robit/example-omni:q4km
```

`omni-attach` computes the content digest, hard-links or copies the blob into
the local Ollama store, replaces any older Robit Omni sidecar entry, and
atomically updates the manifest. It preserves every standard layer.

The resulting descriptor is:

```json
{
  "mediaType": "application/vnd.robit.ollama.omni.bundle.v1+gguf",
  "digest": "sha256:<bundle-digest>",
  "size": 38843038144,
  "annotations": {
    "io.robit.omni.schema": "robit.ollama-monolithic-omni.v3"
  }
}
```

Stop concurrent model create/copy operations while attaching. This is a local
manifest operation because upstream Ollama's Modelfile grammar has no custom
layer directive.

## Runtime materialization

Runtimes without a filtered mmap implementation create disposable views:

```bash
python -m training_suite omni-prepare robit/example-omni:q4km \
  --out ./runtime-cache
```

Equivalent single-view extraction is available through `omni-unpack`. The
runtime cache is not a release artifact and must be removed after workers stop.
The tag retains the source sidecar and can recreate it at any time.

## Why direct stock loading is rejected

Stock llama.cpp chooses one graph from `general.architecture` and then checks
the GGUF tensor inventory. The v3 sidecar has heterogeneous inventories by
design, so a direct Ollama import/load cannot satisfy that graph's expected
tensor count. The following alternatives were tested and rejected:

- an all-namespaced tensor model layer: imports but fails required-tensor
  accounting at load;
- bytes appended after the base GGUF: removed when Ollama normalizes/imports
  the model;
- a pre-tensor opaque gap: rejected because tensor offsets must be contiguous;
- large byte arrays in GGUF metadata: eagerly loaded into host memory and not a
  practical 20+ GB payload mechanism.

Changing this outcome requires upstream loader/runtime work, not a metadata
flag. The custom manifest layer is therefore part of the ABI.

## Verification checklist

1. Record the sidecar SHA-256 and byte size.
2. Confirm exact expected tensor counts for all six views.
3. Materialize every view and compare it with its pinned source digest.
4. Confirm `ollama show` still advertises completion, vision, tools, and
   thinking from the standard layers.
5. Run live audio, image, video, and TTS component probes.
6. Push the tag and pull it back; `omni-resolve` must find the same layer digest.
7. Remove only disposable materialized views after all workers stop.
