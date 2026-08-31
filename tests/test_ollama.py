from __future__ import annotations

from training_suite.models.ollama import ModelfileSpec, generate_modelfile, parse_ollama_show


SHOW_TEXT = """
  Model
    architecture        qwen35
    parameters          9.7B
    context length      262144
    embedding length    4096
    quantization        Q4_K_M

  Capabilities
    completion
    vision
    tools
    thinking

  Parameters
    temperature         0.6
    top_p               0.95
"""


def test_parse_ollama_show_capabilities() -> None:
    shown = parse_ollama_show(SHOW_TEXT, name="qwen3.5:9b")

    assert shown.exists is True
    assert shown.architecture == "qwen35"
    assert shown.context_length == 262144
    assert shown.embedding_length == 4096
    assert shown.quantization == "Q4_K_M"
    assert shown.capabilities == ["completion", "vision", "tools", "thinking"]
    assert shown.runtime_parameters["temperature"] == "0.6"


def test_generate_modelfile_contains_qwen_renderer_and_parser() -> None:
    text = generate_modelfile(
        ModelfileSpec(
            from_ref="./ornith.gguf",
            parameters={"num_ctx": 262144, "temperature": 0.6, "stop": "<|im_end|>"},
        )
    )

    assert "FROM ./ornith.gguf" in text
    assert "RENDERER qwen3.5" in text
    assert "PARSER qwen3.5" in text
    assert "PARAMETER num_ctx 262144" in text
    assert 'PARAMETER stop "<|im_end|>"' in text


def test_generate_modelfile_supports_separate_multimodal_projector() -> None:
    text = generate_modelfile(
        ModelfileSpec(
            from_ref="./model.gguf",
            additional_from=["./mmproj.gguf"],
        )
    )

    assert text.count("FROM ") == 2
    assert "FROM ./model.gguf" in text
    assert "FROM ./mmproj.gguf" in text
