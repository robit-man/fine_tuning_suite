# ADR-002: Logical Ollama Omni Tag with Namespaced GGUF Sidecar

- Status: Accepted and locally validated
- Date: 2026-08-31
- Decision owners: Fine-Tuning Suite maintainers
- Schemas: `robit.ollama-monolithic-omni.v3`,
  `robit.ollama.omni-adapter.v1`

## Context

The project needs one model name and one pullable Ollama release that retains
Qwen3.8/Ornith text, thinking, tools, and image vision while adding audio/video
understanding and TTS. The added models do not share a compatible execution
graph or hidden-state interface with the base language model.

Experiments established these constraints:

1. A GGUF with all heterogeneous tensors in one inventory imports but stock
   llama.cpp rejects it because the selected architecture expects only its own
   tensor set.
2. Appended bytes are discarded when Ollama normalizes a GGUF import.
3. A pre-tensor data gap violates required contiguous tensor offsets.
4. A tens-of-gigabytes byte array in GGUF metadata is eagerly allocated by the
   reader and is operationally unacceptable.
5. Qwen3-Omni hidden states and Talker conditioning cannot be safely spliced
   into Qwen3.8/Ornith by renaming, padding, or reshaping tensors.

## Decision

Ship one logical Ollama model manifest containing:

- standard Ollama model/projector/template/parameter/license layers that stock
  Ollama executes normally;
- one custom layer with media type
  `application/vnd.robit.ollama.omni.bundle.v1+gguf`;
- one valid GGUF v3 in that custom layer, containing six namespaced source
  views for reproducibility and adapter execution.

The tensor namespaces are:

| View | Namespace |
|---|---|
| Base language | unprefixed |
| Base projector | `b.p.*` |
| Comprehension model/projector | `a.c.m.*`, `a.c.p.*` |
| TTS model/projector | `s.t.m.*`, `s.t.p.*` |

Stock Ollama ignores the unknown custom layer and therefore preserves native
completion, vision, tools, and thinking. The custom adapter resolves the layer
from the same model tag, validates schema/digest, materializes or maps component
views, and routes audio/video/TTS. The client sees one model name and one
Ollama-shaped endpoint.

## Routing boundary

For Qwen3.8/Ornith combinations, comprehension returns semantic text. The
adapter inserts it as explicitly delimited untrusted evidence before the base
language pass. TTS is independently text-conditioned. No incompatible hidden
state crosses component boundaries.

## Consequences

### Positive

- One tag/pull contains all release weights.
- Stock Ollama remains usable for native capabilities.
- Media graphs can evolve independently without corrupting the base loader.
- Every component retains exact source metadata, tensor bytes, and digest.
- Runtimes can load/evict media contexts independently.
- The sidecar can be mirrored on Hugging Face with complete provenance.

### Negative

- Audio/video/TTS require the adapter and are not native upstream Ollama
  capabilities.
- The logical model uses multiple OCI layers, not one total physical blob.
- Materialization temporarily duplicates approximately 21 GB for the current
  Qwen3-Omni/TTS media views.
- Semantic routing loses dense cross-modal information.
- Adapter v1 is turn-based; streaming needs a new protocol revision.
- The reference TTS wrapper is serial and not a production scheduler.

## Rejected alternatives

### Direct heterogeneous GGUF model layer

Rejected after required-tensor accounting failed in stock Ollama/llama.cpp.

### Trailing payload after a byte-identical base GGUF

Rejected because Ollama import retained only the normalized base model blob.

### Opaque metadata payload

Rejected because the GGUF reader materializes metadata values in host memory.

### Direct Qwen3-Omni → Qwen3.8 hidden-state splice

Rejected because architectures, widths, vocabularies, special tokens, layers,
and Talker conditioning are incompatible. A learned bridge would be a new model
architecture and a new ADR/schema.

### Multiple unrelated model names

Rejected as the public contract because it does not meet one-tag deployment.
Internally, separate execution contexts remain necessary and are hidden behind
the adapter.

## Verification

A release must prove:

- exact six-view tensor counts and source hashes;
- stock text, image vision, thinking, and structured tools;
- live audio, image, video, video-audio, and TTS from sidecar-derived views;
- adapter response conformance;
- Hugging Face remote availability;
- Ollama push/pull preserving the custom layer digest;
- temporary view cleanup only after both registries are verified.

## Follow-up

A future in-process implementation may add filtered mmap views and persistent
libmtmd workers to an Ollama fork. It must preserve the same logical layer and
wire contracts or introduce explicitly versioned successors.
