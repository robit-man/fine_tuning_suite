from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import httpx

from training_suite.core.config import utc_now
from training_suite.models.audio import DEFAULT_AUDIO_CONTRACT
from training_suite.models.ollama import ModelfileSpec, write_modelfile


OMNI_SCHEMA_VERSION = 1
QWEN3_OMNI_INSTRUCT = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
OLLAMA_AUDIO_LIMITATION = (
    "Ollama 0.32.x can store the language GGUF and multimodal projector, but its "
    "public chat response has no native generated-audio field and it cannot execute "
    "the Qwen3-Omni Talker plus code2wav chain."
)


@dataclass(frozen=True)
class ArchitectureSignature:
    model_type: str | None
    hidden_size: int | None
    num_hidden_layers: int | None
    vocab_size: int | None
    architecture: str | None = None
    audio_output_size: int | None = None
    vision_output_size: int | None = None
    has_audio_encoder: bool = False
    has_vision_encoder: bool = False
    has_talker: bool = False
    has_code2wav: bool = False
    has_video_input: bool = False
    talker_thinker_hidden_size: int | None = None
    num_experts: int | None = None
    num_experts_per_token: int | None = None
    talker_hidden_size: int | None = None
    talker_num_hidden_layers: int | None = None
    num_code_groups: int | None = None
    codebook_size: int | None = None
    num_quantizers: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _first(values: Any) -> str | None:
    return str(values[0]) if isinstance(values, list) and values else None


def architecture_signature(config: Mapping[str, Any]) -> ArchitectureSignature:
    """Extract the language and multimodal coupling dimensions from an HF config."""
    model_type = str(config.get("model_type") or "") or None
    thinker = config.get("thinker_config") if isinstance(config.get("thinker_config"), Mapping) else {}
    text = thinker.get("text_config") if isinstance(thinker.get("text_config"), Mapping) else {}
    if not text:
        text = config.get("text_config") if isinstance(config.get("text_config"), Mapping) else config

    audio = thinker.get("audio_config") if isinstance(thinker.get("audio_config"), Mapping) else {}
    vision = thinker.get("vision_config") if isinstance(thinker.get("vision_config"), Mapping) else {}
    if not vision and isinstance(config.get("vision_config"), Mapping):
        vision = config.get("vision_config") or {}
    talker = config.get("talker_config") if isinstance(config.get("talker_config"), Mapping) else {}
    talker_text = talker.get("text_config") if isinstance(talker.get("text_config"), Mapping) else {}
    code2wav = config.get("code2wav_config") if isinstance(config.get("code2wav_config"), Mapping) else {}

    return ArchitectureSignature(
        architecture=_first(config.get("architectures")),
        model_type=str(text.get("model_type") or model_type or "") or None,
        hidden_size=_int_or_none(text.get("hidden_size")),
        num_hidden_layers=_int_or_none(text.get("num_hidden_layers")),
        vocab_size=_int_or_none(text.get("vocab_size")),
        audio_output_size=_int_or_none(audio.get("output_dim")),
        vision_output_size=_int_or_none(vision.get("out_hidden_size")),
        has_audio_encoder=bool(audio),
        has_vision_encoder=bool(vision),
        has_talker=bool(talker),
        has_code2wav=bool(code2wav),
        has_video_input=bool(vision and thinker.get("video_token_id") is not None),
        talker_thinker_hidden_size=_int_or_none(talker.get("thinker_hidden_size")),
        num_experts=_int_or_none(text.get("num_experts")),
        num_experts_per_token=_int_or_none(text.get("num_experts_per_tok")),
        talker_hidden_size=_int_or_none(talker_text.get("hidden_size")),
        talker_num_hidden_layers=_int_or_none(talker_text.get("num_hidden_layers")),
        num_code_groups=_int_or_none(talker.get("num_code_groups")),
        codebook_size=_int_or_none(code2wav.get("codebook_size")),
        num_quantizers=_int_or_none(code2wav.get("num_quantizers")),
    )


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def assess_native_omni_splice(
    text_config: Mapping[str, Any],
    omni_config: Mapping[str, Any],
) -> dict[str, Any]:
    source = architecture_signature(text_config)
    donor = architecture_signature(omni_config)
    comparisons = (
        ("model_type", source.model_type, donor.model_type),
        ("hidden_size", source.hidden_size, donor.hidden_size),
        ("num_hidden_layers", source.num_hidden_layers, donor.num_hidden_layers),
        ("vocab_size", source.vocab_size, donor.vocab_size),
    )
    mismatches = [
        f"{name}: source={left!r}, omni_thinker={right!r}"
        for name, left, right in comparisons
        if left is None or right is None or left != right
    ]
    if donor.audio_output_size != donor.hidden_size:
        mismatches.append(
            "omni audio encoder output does not match its thinker hidden size"
        )
    if donor.vision_output_size not in (None, donor.hidden_size):
        mismatches.append(
            "omni vision encoder output does not match its thinker hidden size"
        )
    if donor.talker_thinker_hidden_size not in (None, donor.hidden_size):
        mismatches.append(
            "omni Talker conditioning size does not match its thinker hidden size"
        )
    if not donor.has_audio_encoder:
        mismatches.append("omni donor has no audio encoder configuration")

    return {
        "compatible": not mismatches,
        "source": source.to_dict(),
        "omni": donor.to_dict(),
        "mismatches": mismatches,
        "rule": (
            "Native tensor substitution requires the language architecture, hidden size, "
            "layer count, vocabulary, and encoder/Talker conditioning dimensions to match."
        ),
    }


def plan_omni_bundle(
    *,
    text_config: Mapping[str, Any],
    omni_config: Mapping[str, Any],
    text_source: str,
    omni_source: str = QWEN3_OMNI_INSTRUCT,
    target_tag: str | None = None,
) -> dict[str, Any]:
    compatibility = assess_native_omni_splice(text_config, omni_config)
    native = bool(compatibility["compatible"])
    mode = "native-omni" if native else "monolithic-router"

    if native:
        components = {
            "thinker": {"source": text_source, "artifact": "model.gguf"},
            "multimodal_projector": {"source": omni_source, "artifact": "mmproj.gguf"},
            "talker": {
                "source": omni_source,
                "artifact": "runtime-native weights",
                "runtime": "vLLM-Omni or Transformers",
            },
            "code2wav": {
                "source": omni_source,
                "artifact": "runtime-native weights",
                "runtime": "vLLM-Omni or Transformers",
            },
        }
        rationale = (
            "The language trunk matches the Omni Thinker signature. Preserve the donor audio/vision "
            "projector, Talker, and code2wav components as a versioned runtime bundle."
        )
    else:
        components = {
            "audio_understanding": {
                "source": omni_source,
                "artifact": "model.gguf + mmproj.gguf",
                "runtime": "llama.cpp libmtmd",
                "output": "text transcript or semantic description",
            },
            "video_understanding": {
                "source": omni_source,
                "artifact": "model.gguf + mmproj.gguf or runtime-native Omni weights",
                "runtime": "llama.cpp libmtmd or a native Qwen3-Omni runtime",
                "input": "decoded frame sequence with temporal metadata",
                "output": "text transcript or semantic description",
                "status": "planned component; no Flask video transport adapter yet",
            },
            "language_model": {
                "source": text_source,
                "artifact": "Ollama-compatible model.gguf",
                "runtime": "Ollama",
                "input": "text",
                "output": "text",
            },
            "speech_synthesis": {
                "source": "Qwen/Qwen3-TTS-12Hz-0.6B-Base or another text-conditioned TTS model",
                "artifact": "runtime-native TTS weights; GGUF only when supported by the selected loader",
                "runtime": "llama.cpp experimental audio generation or dedicated TTS runtime",
                "input": "text",
                "output": "24 kHz mono PCM16 WAV",
            },
        }
        rationale = (
            "Direct hidden-state grafting is unsafe. Package the Qwen3.8/Ornith language model, a "
            "self-contained audio/video comprehension graph, and text-conditioned TTS into one namespaced "
            "GGUF. A custom Ollama runner routes between graphs without pretending their hidden states match."
        )

    return {
        "schema_version": OMNI_SCHEMA_VERSION,
        "created_at": utc_now(),
        "mode": mode,
        "status": "ready-for-native-bundle" if native else "ready-for-monolithic-pack",
        "target_tag": target_tag,
        "text_source": text_source,
        "omni_source": omni_source,
        "requested_capabilities": [
            "completion",
            "vision",
            "audio-input",
            "audio-output",
            "video-input",
            "tools",
            "thinking",
        ],
        "compatibility": compatibility,
        "components": components,
        "rationale": rationale,
        "artifact_policy": {
            "single_gguf": True,
            "single_gguf_layout": "base tensors + a.c.* comprehension + s.t.* text-conditioned TTS",
            "ollama_model_gguf": True,
            "component_inputs_must_be_gguf": True,
            "custom_ollama_handler_required": True,
            "stock_ollama_audio_api": False,
            "full_audio_output_runtime": "custom Ollama runner loading namespaced component views",
            "reason": (
                "A monolithic GGUF can hold all component tensors, while the custom runner defines the "
                "execution graph and tagged audio protocol."
            ),
        },
        "runtime_support": {
            "ollama_0_32": {
                "language_model": True,
                "vision_projector": True,
                "qwen3_omni_audio_input": False,
                "generated_audio_response": False,
            },
            "llama_cpp_libmtmd": {
                "qwen3_omni_audio_input": True,
                "qwen3_omni_vision_input": True,
                "qwen3_omni_talker_output": False,
                "qwen3_tts_output": "experimental",
            },
            "vllm": {
                "qwen3_omni_thinker": True,
                "qwen3_omni_talker_output": False,
            },
            "vllm_omni": {
                "qwen3_omni_audio_input": True,
                "qwen3_omni_talker_output": True,
                "realtime_websocket": True,
            },
        },
        "deployment_profile": {
            "profile_source": "model config; no inferred marketing size",
            "thinker": {
                "model_type": compatibility["omni"]["model_type"],
                "hidden_size": compatibility["omni"]["hidden_size"],
                "num_hidden_layers": compatibility["omni"]["num_hidden_layers"],
                "num_experts": compatibility["omni"]["num_experts"],
                "num_experts_per_token": compatibility["omni"]["num_experts_per_token"],
            },
            "talker": {
                "hidden_size": compatibility["omni"]["talker_hidden_size"],
                "num_hidden_layers": compatibility["omni"]["talker_num_hidden_layers"],
                "thinker_hidden_size": compatibility["omni"]["talker_thinker_hidden_size"],
                "num_code_groups": compatibility["omni"]["num_code_groups"],
            },
            "code2wav": {
                "codebook_size": compatibility["omni"]["codebook_size"],
                "num_quantizers": compatibility["omni"]["num_quantizers"],
                "sample_rate_hz": 24000,
            },
        },
        "streaming_contract": {
            "status": "planned",
            "transport": "websocket",
            "input": "PCM16 mono 16 kHz frames",
            "input_frame_ms": 20,
            "output": "PCM16 mono 24 kHz chunks",
            "output_chunk_ms": 80,
            "note": "Turn-based base64 WAV is implemented first; realtime requires VAD, cancellation, and backpressure.",
        },
        "video_policy": {
            "video_input": bool(compatibility["omni"]["has_video_input"]),
            "video_output": False,
            "scope": "video comprehension only; video generation is out of scope",
            "transport_status": "planned; decode video into frames before donor inference",
        },
        "audio_contract": DEFAULT_AUDIO_CONTRACT.to_dict(),
        "api_response_shape": {
            "message": {"content": "text response"},
            "audio": {
                "type": "audio",
                "mime_type": "audio/wav",
                "encoding": "base64",
                "sample_rate_hz": 24000,
                "channels": 1,
                "data": "<base64 RIFF/WAVE bytes>",
            },
        },
        "native_fusion_blockers": list(compatibility["mismatches"]),
        "stock_ollama_audio_blockers": [OLLAMA_AUDIO_LIMITATION],
        "source_signatures": {
            "text": compatibility["source"],
            "omni": compatibility["omni"],
        },
    }


def load_config_reference(reference: str) -> dict[str, Any]:
    """Load config JSON from a local path, HF repo ID, or huggingface.co URL."""
    local = Path(reference).expanduser()
    if local.is_file():
        return json.loads(local.read_text(encoding="utf-8"))

    repo_id = reference.strip()
    if "huggingface.co/" in repo_id:
        repo_id = repo_id.split("huggingface.co/", 1)[1]
        repo_id = repo_id.split("/blob/", 1)[0].split("/raw/", 1)[0].strip("/")
    if repo_id.startswith("hf.co/"):
        repo_id = repo_id.split("hf.co/", 1)[1]
    parts = repo_id.strip("/").split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError(
            "config reference must be a local config.json path or Hugging Face org/repo"
        )
    response = httpx.get(
        f"https://huggingface.co/{parts[0]}/{parts[1]}/raw/main/config.json",
        timeout=30,
        follow_redirects=True,
    )
    response.raise_for_status()
    return response.json()


def component_file(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    resolved = Path(path).expanduser().resolve()
    return {
        "path": str(resolved),
        "exists": resolved.is_file(),
        "size_bytes": resolved.stat().st_size if resolved.is_file() else None,
    }


def write_omni_bundle(
    out_dir: Path,
    plan: Mapping[str, Any],
    *,
    text_gguf: str | None = None,
    mmproj_gguf: str | None = None,
    talker_gguf: str | None = None,
    code2wav_gguf: str | None = None,
    renderer: str = "qwen3.8",
    parser: str = "qwen3.5",
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_plan = dict(plan)
    artifact_plan["component_files"] = {
        "language_model": component_file(text_gguf),
        "multimodal_projector": component_file(mmproj_gguf),
        "talker": component_file(talker_gguf),
        "code2wav": component_file(code2wav_gguf),
    }
    manifest = out_dir / "omni_bundle.json"
    manifest.write_text(json.dumps(artifact_plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    contract = out_dir / "audio_contract.json"
    contract.write_text(
        json.dumps(DEFAULT_AUDIO_CONTRACT.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    outputs = {"manifest": str(manifest), "audio_contract": str(contract)}

    if text_gguf:
        modelfile = out_dir / "Modelfile"
        write_modelfile(
            modelfile,
            ModelfileSpec(
                from_ref=str(Path(text_gguf).expanduser().resolve()),
                additional_from=(
                    [str(Path(mmproj_gguf).expanduser().resolve())]
                    if mmproj_gguf and plan.get("mode") == "native-omni"
                    else []
                ),
                renderer=renderer,
                parser=parser,
                parameters={"num_ctx": 262144, "temperature": 0.6, "top_p": 0.95, "top_k": 20},
            ),
        )
        outputs["modelfile"] = str(modelfile)
    return outputs
