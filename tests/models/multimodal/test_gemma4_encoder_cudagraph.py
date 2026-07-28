# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import cast

import pytest
import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.models.gemma4_mm import (
    Gemma4ForConditionalGeneration,
    _get_gemma4_vision_use_cudnn_sdpa,
)
from vllm.model_executor.models.interfaces import (
    supports_encoder_cudagraph,
)


def _make_model() -> Gemma4ForConditionalGeneration:
    model = Gemma4ForConditionalGeneration.__new__(Gemma4ForConditionalGeneration)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        vision_config=SimpleNamespace(
            default_output_length=4,
            pooling_kernel_size=3,
            patch_size=2,
        ),
        text_config=SimpleNamespace(hidden_size=16),
    )
    model._vision_use_cudnn_sdpa = False
    return model


def _make_inputs(valid_patches: list[int]) -> dict[str, torch.Tensor]:
    num_items = len(valid_patches)
    num_patches = 18
    patch_pixels = 12
    pixel_values = torch.randn(num_items, num_patches, patch_pixels)
    pixel_position_ids = torch.full((num_items, num_patches, 2), -1)
    for i, count in enumerate(valid_patches):
        pixel_position_ids[i, :count, 0] = torch.arange(count)
        pixel_position_ids[i, :count, 1] = 0
    return {
        "pixel_values": pixel_values,
        "pixel_position_ids": pixel_position_ids,
    }


def test_gemma4_supports_encoder_cudagraph_protocol() -> None:
    assert supports_encoder_cudagraph(Gemma4ForConditionalGeneration)


def test_gemma4_encoder_cudagraph_item_metadata() -> None:
    model = _make_model()
    mm_kwargs = _make_inputs([18, 9])

    config = model.get_encoder_cudagraph_config()
    assert config.modalities == ["image"]
    assert config.input_key_by_modality == {"image": "pixel_values"}
    assert config.buffer_keys == ["pixel_position_ids"]
    assert config.out_hidden_size == 16
    assert model.get_input_modality(mm_kwargs) == "image"
    assert model.get_encoder_cudagraph_budget_range(cast(VllmConfig, None)) == (4, 4)
    assert model.get_encoder_cudagraph_num_items(mm_kwargs) == 2
    assert model.get_encoder_cudagraph_per_item_input_sizes(mm_kwargs) == [18, 9]
    assert model.get_encoder_cudagraph_per_item_output_tokens(mm_kwargs) == [2, 1]


def test_gemma4_encoder_cudagraph_select_pads_to_capture_shape() -> None:
    model = _make_model()
    mm_kwargs = _make_inputs([18, 9])

    selected = model.select_encoder_cudagraph_items(mm_kwargs, [1])

    assert selected["pixel_values"].shape == (1, 36, 12)
    assert selected["pixel_position_ids"].shape == (1, 36, 2)
    torch.testing.assert_close(
        selected["pixel_values"][0, :18],
        mm_kwargs["pixel_values"][1],
    )
    torch.testing.assert_close(
        selected["pixel_position_ids"][0, :18],
        mm_kwargs["pixel_position_ids"][1],
    )
    assert torch.count_nonzero(selected["pixel_values"][0, 18:]) == 0
    assert torch.all(selected["pixel_position_ids"][0, 18:] == -1)


def test_gemma4_encoder_cudagraph_capture_and_replay_buffers() -> None:
    model = _make_model()

    capture = model.prepare_encoder_cudagraph_capture_inputs(
        token_budget=4,
        max_batch_size=1,
        max_frames_per_batch=0,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    assert capture.mm_kwargs["pixel_values"].shape == (1, 36, 12)
    assert capture.mm_kwargs["pixel_values"].dtype == torch.bfloat16
    assert capture.mm_kwargs["pixel_position_ids"].shape == (1, 36, 2)
    assert model.get_encoder_cudagraph_per_item_output_tokens(capture.mm_kwargs) == [4]
    assert capture.buffers == {
        "pixel_position_ids": capture.mm_kwargs["pixel_position_ids"]
    }

    replay = model.prepare_encoder_cudagraph_replay_buffers(
        capture.mm_kwargs,
        max_batch_size=1,
        max_frames_per_batch=0,
    )
    assert replay.buffers == {
        "pixel_position_ids": capture.mm_kwargs["pixel_position_ids"]
    }


def test_gemma4_vision_cudnn_sdpa_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_name = "VLLM_GEMMA4_VISION_USE_CUDNN_SDPA"
    monkeypatch.delenv(env_name, raising=False)
    assert not _get_gemma4_vision_use_cudnn_sdpa()

    monkeypatch.setenv(env_name, "1")
    assert _get_gemma4_vision_use_cudnn_sdpa()

    monkeypatch.setenv(env_name, "true")
    assert _get_gemma4_vision_use_cudnn_sdpa()

    monkeypatch.setenv(env_name, "0")
    assert not _get_gemma4_vision_use_cudnn_sdpa()
