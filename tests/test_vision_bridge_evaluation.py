from __future__ import annotations

import struct

import pytest

from training_suite.evals.vision_bridge import (
    _request_payload,
    parse_visual_observation,
    solid_color_png,
)


def test_vision_request_disables_cache_and_thinking() -> None:
    payload = _request_payload(model="candidate", color="red", max_tokens=128)

    assert payload["cache_prompt"] is False
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["temperature"] == 0
    assert payload["max_tokens"] == 128
    image = payload["messages"][1]["content"][1]["image_url"]["url"]
    assert image.startswith("data:image/png;base64,")


def test_solid_color_fixture_is_a_224_pixel_rgb_png() -> None:
    image = solid_color_png("blue")

    assert image.startswith(b"\x89PNG\r\n\x1a\n")
    width, height, depth, color_type = struct.unpack(">IIBB", image[16:26])
    assert (width, height, depth, color_type) == (224, 224, 8, 2)


def test_visual_parser_requires_one_exact_tag_and_unambiguous_color() -> None:
    assert parse_visual_observation(
        "<visual_observation>dominant color: red</visual_observation>"
    ) == {
        "valid": True,
        "observation": "dominant color: red",
        "predicted_color": "red",
    }
    assert parse_visual_observation("red")["valid"] is False
    assert (
        parse_visual_observation(
            "<visual_observation><dominant_color>red</dominant_color>"
            "</visual_observation>"
        )["valid"]
        is False
    )
    assert (
        parse_visual_observation(
            "<visual_observation>red and blue</visual_observation>"
        )["predicted_color"]
        == ""
    )


def test_solid_color_fixture_rejects_unknown_color() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        solid_color_png("green")
