from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from training_suite.core.config import DEFAULT_TARGET_CAPABILITIES
from training_suite.models.gguf import compatible_for_splice, inspect_gguf
from training_suite.models.ollama import capability_delta, show_model
from training_suite.models.omni import architecture_signature

HF_MODEL_RE = re.compile(r"huggingface\.co/([^/\s]+/[^/\s?#]+)")


@dataclass
class IntakeResult:
    name: str
    source: str
    source_type: str
    raw_source: str | None = None
    donor_model: str | None = None
    local_ollama_model: str | None = None
    architecture: str | None = None
    quantization: str | None = None
    context_length: int | None = None
    tensor_count: int | None = None
    detected_capabilities: list[str] = field(default_factory=list)
    target_capabilities: list[str] = field(default_factory=lambda: list(DEFAULT_TARGET_CAPABILITIES))
    repair_plan: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    status: str = "intake"

    def to_model_row(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source,
            "source_type": self.source_type,
            "raw_source": self.raw_source,
            "donor_model": self.donor_model,
            "local_ollama_model": self.local_ollama_model,
            "architecture": self.architecture,
            "quantization": self.quantization,
            "context_length": self.context_length,
            "tensor_count": self.tensor_count,
            "detected_capabilities": self.detected_capabilities,
            "target_capabilities": self.target_capabilities,
            "repair_plan": self.repair_plan,
            "metadata": self.metadata,
            "status": self.status,
        }


def parse_hf_repo(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip()
    match = HF_MODEL_RE.search(value)
    if match:
        return match.group(1).rstrip("/")
    if re.match(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$", value):
        return value
    if value.startswith("hf.co/"):
        return value.split("hf.co/", 1)[1].strip("/")
    return None


def hf_model_info(repo_id: str, timeout: float = 20) -> dict[str, Any]:
    url = f"https://huggingface.co/api/models/{repo_id}"
    response = httpx.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def hf_raw_json(repo_id: str, filename: str, timeout: float = 20) -> dict[str, Any] | None:
    url = f"https://huggingface.co/{repo_id}/raw/main/{filename}"
    response = httpx.get(url, timeout=timeout)
    if response.status_code >= 400:
        return None
    return response.json()


def _source_type(source: str, gguf_path: str | None, ollama_model: str | None) -> str:
    if ollama_model:
        return "ollama"
    if gguf_path:
        return "gguf"
    if parse_hf_repo(source):
        return "huggingface"
    return "unknown"


def _model_name_from_source(source: str, ollama_model: str | None = None) -> str:
    if ollama_model:
        return ollama_model
    repo = parse_hf_repo(source)
    if repo:
        return repo.split("/", 1)[1]
    if source:
        return Path(source).stem
    return "model"


def _config_signature(config: dict[str, Any] | None) -> dict[str, Any]:
    if not config:
        return {}
    signature = architecture_signature(config)
    return {
        "architecture": signature.model_type or signature.architecture,
        "hidden_size": signature.hidden_size,
        "text_hidden_size": signature.hidden_size,
        "num_hidden_layers": signature.num_hidden_layers,
        "vocab_size": signature.vocab_size,
        "vision_out_hidden_size": signature.vision_output_size,
        "audio_out_hidden_size": signature.audio_output_size,
        "has_audio_encoder": signature.has_audio_encoder,
        "has_talker": signature.has_talker,
        "has_code2wav": signature.has_code2wav,
        "has_video_input": signature.has_video_input,
        "has_vision_config": signature.has_vision_encoder,
        "transformers_version": config.get("transformers_version"),
    }


def build_repair_plan(
    *,
    detected: list[str],
    target: list[str],
    source_architecture: str | None,
    gguf_has_vision: bool,
    raw_signature: dict[str, Any],
    gguf_signature: dict[str, Any],
    raw_source: str | None,
    source: str,
) -> dict[str, Any]:
    delta = capability_delta(detected, target)
    actions: list[str] = []
    blockers: list[str] = []

    if {"tools", "thinking"} & set(delta["missing"]):
        actions.append("generate Modelfile with RENDERER qwen3.5 and PARSER qwen3.5")
    if "vision" in delta["missing"]:
        if gguf_has_vision:
            actions.append("patch or preserve GGUF vision metadata and repackage")
        elif raw_source:
            actions.append("reconstruct multimodal GGUF from matching raw weights")
        else:
            actions.append("provide matching raw multimodal weights before vision splice")
    if "audio-input" in delta["missing"]:
        actions.append(
            "attach a self-contained Qwen3-Omni/Qwen3-ASR model/projector under a.c.m.*/a.c.p.* or train a compatible bridge"
        )
    if "audio-output" in delta["missing"]:
        actions.append(
            "attach independently text-conditioned TTS model/projector views under s.t.m.*/s.t.p.* and return tagged base64 PCM WAV"
        )
    if "video-input" in delta["missing"]:
        actions.append("preserve the Omni vision/video projector and frame-interleave contract")

    if raw_signature and gguf_signature:
        ok, errors = compatible_for_splice(raw_signature, gguf_signature)
        if not ok:
            blockers.extend(errors)

    if "Ornith-1.0-9B" in source and raw_source and "Ornith-1.0-35B" in raw_source:
        blockers.append("Ornith 35B MoE raw weights are not compatible with the 9B dense GGUF")

    mode = "none"
    if blockers:
        mode = "blocked"
    elif not delta["missing"]:
        mode = "verified"
    elif any("Modelfile" in action for action in actions) and "vision" not in delta["missing"]:
        mode = "package-only"
    elif gguf_has_vision:
        mode = "metadata-patch"
    elif {"audio-input", "audio-output"} & set(delta["missing"]):
        mode = "monolithic-router"
    elif raw_source:
        mode = "vision-splice"

    return {
        "mode": mode,
        "delta": delta,
        "actions": actions,
        "blockers": blockers,
        "source_architecture": source_architecture,
    }


def inspect_intake(
    *,
    source: str,
    raw_source: str | None = None,
    gguf_path: str | None = None,
    ollama_model: str | None = None,
    donor_model: str | None = None,
    target_capabilities: list[str] | None = None,
) -> IntakeResult:
    target = target_capabilities or list(DEFAULT_TARGET_CAPABILITIES)
    source = source.strip()
    raw_source = raw_source.strip() if raw_source else None
    gguf_path = gguf_path.strip() if gguf_path else None
    ollama_model = ollama_model.strip() if ollama_model else None
    source_type = _source_type(source, gguf_path, ollama_model)
    metadata: dict[str, Any] = {"input": {"source": source, "raw_source": raw_source}}
    architecture = None
    quantization = None
    context_length = None
    tensor_count = None
    detected: list[str] = []
    gguf_has_vision = False
    gguf_signature: dict[str, Any] = {}

    repo_id = parse_hf_repo(source)
    if repo_id:
        try:
            info = hf_model_info(repo_id)
            metadata["hf"] = {
                "id": info.get("id") or repo_id,
                "pipeline_tag": info.get("pipeline_tag"),
                "library_name": info.get("library_name"),
                "tags": info.get("tags", []),
                "lastModified": info.get("lastModified"),
                "siblings": [s.get("rfilename") for s in info.get("siblings", [])],
            }
            gguf = info.get("gguf") or {}
            architecture = gguf.get("architecture") or (info.get("config") or {}).get("model_type")
            context_length = gguf.get("context_length")
            tensor_count = gguf.get("tensor_count")
            chat_template = gguf.get("chat_template") or (info.get("config") or {}).get("chat_template_jinja")
            detected = ["completion"]
            if chat_template and "tool_call" in chat_template:
                detected.append("tools")
            if chat_template and "<think>" in chat_template:
                detected.append("thinking")
            if "image-text-to-text" in info.get("tags", []) or (info.get("config") or {}).get("vision_config"):
                detected.append("vision")
            source_config = hf_raw_json(repo_id, "config.json")
            if source_config:
                signature = architecture_signature(source_config)
                metadata["config_signature"] = signature.to_dict()
                architecture = signature.model_type or architecture
                if signature.has_vision_encoder:
                    detected.append("vision")
                if signature.has_audio_encoder:
                    detected.append("audio-input")
                if signature.has_talker and signature.has_code2wav:
                    detected.append("audio-output")
                if signature.has_video_input:
                    detected.append("video-input")
            metadata["hf_gguf"] = gguf
            if gguf.get("total"):
                tensor_count = tensor_count or None
        except Exception as exc:  # noqa: BLE001 - optional remote/reader backends vary
            metadata["hf_error"] = f"{type(exc).__name__}: {exc}"

    if raw_source:
        raw_repo = parse_hf_repo(raw_source)
        if raw_repo:
            try:
                raw_config = hf_raw_json(raw_repo, "config.json")
                metadata["raw_config_signature"] = _config_signature(raw_config)
            except Exception as exc:  # noqa: BLE001 - optional remote backends vary
                metadata["raw_error"] = f"{type(exc).__name__}: {exc}"

    if gguf_path:
        inspection = inspect_gguf(Path(gguf_path))
        metadata["gguf_inspection"] = inspection.to_dict()
        architecture = inspection.architecture or architecture
        quantization = inspection.quantization or quantization
        context_length = inspection.context_length or context_length
        tensor_count = inspection.tensor_count or tensor_count
        detected = sorted(set(detected) | set(inspection.capabilities))
        gguf_has_vision = "vision" in inspection.capabilities
        gguf_signature = {
            "architecture": inspection.architecture,
            "hidden_size": inspection.metadata.get(f"{inspection.architecture}.embedding_length")
            if inspection.architecture
            else None,
            "text_hidden_size": inspection.metadata.get(f"{inspection.architecture}.embedding_length")
            if inspection.architecture
            else None,
            "num_hidden_layers": inspection.metadata.get(f"{inspection.architecture}.block_count")
            if inspection.architecture
            else None,
        }

    if ollama_model:
        shown = show_model(ollama_model, verbose=True, include_modelfile=True)
        metadata["ollama"] = shown.to_dict()
        if shown.exists:
            architecture = shown.architecture or architecture
            quantization = shown.quantization or quantization
            context_length = shown.context_length or context_length
            detected = sorted(set(detected) | set(shown.capabilities))
            gguf_has_vision = gguf_has_vision or "vision" in shown.capabilities

    if not detected:
        detected = ["completion"] if architecture or source else []

    raw_signature = metadata.get("raw_config_signature", {})
    repair_plan = build_repair_plan(
        detected=detected,
        target=target,
        source_architecture=architecture,
        gguf_has_vision=gguf_has_vision,
        raw_signature=raw_signature,
        gguf_signature=gguf_signature,
        raw_source=raw_source,
        source=source,
    )

    return IntakeResult(
        name=_model_name_from_source(source, ollama_model),
        source=source,
        source_type=source_type,
        raw_source=raw_source,
        donor_model=donor_model,
        local_ollama_model=ollama_model,
        architecture=architecture,
        quantization=quantization,
        context_length=context_length,
        tensor_count=tensor_count,
        detected_capabilities=sorted(set(detected)),
        target_capabilities=target,
        repair_plan=repair_plan,
        metadata=metadata,
        status="blocked" if repair_plan.get("blockers") else "intake",
    )
