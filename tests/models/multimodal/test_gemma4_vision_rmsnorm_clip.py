# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.models.gemma4_mm import (
    _convert_gemma4_vision_rmsnorm,
    _gemma4_vision_gelu_mul_clip,
    _Gemma4LinearWithoutOutputClip,
    _Gemma4RMSNormClip,
    _Gemma4SharedInputClip,
)


class _ClippedLinear(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        input_min: float,
        input_max: float,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.register_buffer("input_min", torch.tensor(input_min))
        self.register_buffer("input_max", torch.tensor(input_max))
        self.register_buffer("output_min", torch.tensor(-3.0))
        self.register_buffer("output_max", torch.tensor(3.0))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = torch.clamp(
            hidden_states,
            self.input_min,
            self.input_max,
        )
        hidden_states = self.linear(hidden_states)
        return torch.clamp(hidden_states, self.output_min, self.output_max)


def _rmsnorm_reference(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    hidden_states_float = hidden_states.float()
    mean_squared = hidden_states_float.pow(2).mean(-1, keepdim=True) + eps
    hidden_states_float = hidden_states_float * torch.pow(mean_squared, -0.5)
    hidden_states_float = hidden_states_float * weight.float()
    return hidden_states_float.type_as(hidden_states)


def test_gemma4_shared_input_clip_validation() -> None:
    projections = [_ClippedLinear(8, -2.0, 2.0) for _ in range(3)]
    shared_clip = _Gemma4SharedInputClip(projections)
    assert shared_clip.validate()

    projections[1].input_max.fill_(1.5)
    assert not shared_clip.validate()


@pytest.mark.parametrize(
    ("clip_input", "expected"),
    [
        (False, [-8.0, 8.0]),
        (True, [-4.0, 4.0]),
    ],
)
def test_gemma4_linear_without_output_clip(
    clip_input: bool,
    expected: list[float],
) -> None:
    projection = _ClippedLinear(2, -2.0, 2.0)
    with torch.no_grad():
        projection.linear.weight.copy_(2 * torch.eye(2))
    wrapped_projection = _Gemma4LinearWithoutOutputClip(
        projection, clip_input=clip_input
    )

    torch.testing.assert_close(
        wrapped_projection(torch.tensor([[-4.0, 4.0]])),
        torch.tensor([expected]),
    )


def test_gemma4_rmsnorm_clip_rejects_non_cuda_input() -> None:
    projections = [_ClippedLinear(8, -2.0, 2.0) for _ in range(3)]
    shared_clip = _Gemma4SharedInputClip(projections)
    rms_norm = SimpleNamespace(weight=nn.Parameter(torch.randn(8)), eps=1e-6)
    fused_norm = _Gemma4RMSNormClip(rms_norm, shared_clip)

    with pytest.raises(RuntimeError, match="contiguous CUDA FP16 or BF16"):
        fused_norm(torch.randn(4, 8))


@pytest.mark.parametrize("with_scale", [False, True])
def test_convert_gemma4_vision_rmsnorm_preserves_parameters(
    with_scale: bool,
) -> None:
    rms_norm = SimpleNamespace(eps=1e-5, with_scale=with_scale)
    if with_scale:
        rms_norm.weight = nn.Parameter(torch.randn(8))

    fused_norm = _convert_gemma4_vision_rmsnorm(rms_norm)

    assert fused_norm.eps == rms_norm.eps
    assert fused_norm.has_weight == with_scale
    if with_scale:
        assert fused_norm.weight is rms_norm.weight


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@pytest.mark.parametrize("hidden_size", [768, 1152])
def test_gemma4_vision_triton_rmsnorm_clip_cuda_graph(hidden_size: int) -> None:
    torch.manual_seed(0)
    hidden_states = torch.randn(
        2520,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    weight = torch.randn(hidden_size, device="cuda", dtype=torch.bfloat16)
    clip_min = torch.tensor(-2.5, device="cuda", dtype=torch.bfloat16)
    clip_max = torch.tensor(2.5, device="cuda", dtype=torch.bfloat16)
    projection = SimpleNamespace(input_min=clip_min, input_max=clip_max)
    shared_clip = _Gemma4SharedInputClip([projection, projection, projection])
    rms_norm = SimpleNamespace(weight=nn.Parameter(weight), eps=1e-6)
    fused_norm = _Gemma4RMSNormClip(rms_norm, shared_clip)
    assert shared_clip.validate()
    expected = torch.clamp(
        _rmsnorm_reference(hidden_states, weight, 1e-6),
        clip_min,
        clip_max,
    )

    actual = fused_norm(hidden_states)
    torch.testing.assert_close(actual, expected, atol=0.015625, rtol=0.01)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = fused_norm(hidden_states)
    graph.replay()
    torch.testing.assert_close(replayed, expected, atol=0.015625, rtol=0.01)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@pytest.mark.parametrize("with_scale", [False, True])
def test_gemma4_vision_fused_rmsnorm_cuda_graph(with_scale: bool) -> None:
    torch.manual_seed(0)
    hidden_size = 72
    hidden_states = torch.randn(
        2520,
        16,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    rms_norm = SimpleNamespace(eps=1e-6, with_scale=with_scale)
    if with_scale:
        rms_norm.weight = nn.Parameter(
            torch.randn(hidden_size, device="cuda", dtype=torch.bfloat16)
        )
        weight = rms_norm.weight
    else:
        weight = torch.ones(hidden_size, device="cuda", dtype=torch.bfloat16)
    fused_norm = _convert_gemma4_vision_rmsnorm(rms_norm).cuda()
    expected = _rmsnorm_reference(hidden_states, weight, rms_norm.eps)

    actual = fused_norm(hidden_states)
    torch.testing.assert_close(actual, expected, atol=0.015625, rtol=0.01)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = fused_norm(hidden_states)
    graph.replay()
    torch.testing.assert_close(replayed, expected, atol=0.015625, rtol=0.01)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@pytest.mark.parametrize("with_scale", [False, True])
def test_gemma4_vision_fused_input_clip_rmsnorm_cuda_graph(
    with_scale: bool,
) -> None:
    torch.manual_seed(0)
    hidden_size = 72
    hidden_states = 8 * torch.randn(
        2520,
        16,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    projection = _Gemma4LinearWithoutOutputClip(
        _ClippedLinear(hidden_size, -2.0, 2.0).cuda(),
        clip_input=False,
    )
    rms_norm = SimpleNamespace(eps=1e-6, with_scale=with_scale)
    if with_scale:
        rms_norm.weight = nn.Parameter(
            torch.randn(hidden_size, device="cuda", dtype=torch.bfloat16)
        )
        weight = rms_norm.weight
    else:
        weight = torch.ones(hidden_size, device="cuda", dtype=torch.bfloat16)
    fused_norm = _convert_gemma4_vision_rmsnorm(rms_norm, projection).cuda()
    expected = _rmsnorm_reference(
        torch.clamp(
            hidden_states,
            projection.output_min,
            projection.output_max,
        ),
        weight,
        rms_norm.eps,
    )

    actual = fused_norm(hidden_states)
    torch.testing.assert_close(actual, expected, atol=0.015625, rtol=0.01)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = fused_norm(hidden_states)
    graph.replay()
    torch.testing.assert_close(replayed, expected, atol=0.015625, rtol=0.01)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_gemma4_vision_fused_clip_rmsnorm_residual_cuda_graph() -> None:
    torch.manual_seed(0)
    hidden_size = 1152
    hidden_states = 8 * torch.randn(
        2520,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    residual = torch.randn_like(hidden_states)
    projection = _Gemma4LinearWithoutOutputClip(
        _ClippedLinear(hidden_size, -2.0, 2.0).cuda(),
        clip_input=False,
    )
    rms_norm = SimpleNamespace(
        eps=1e-6,
        with_scale=True,
        weight=nn.Parameter(
            torch.randn(hidden_size, device="cuda", dtype=torch.bfloat16)
        ),
    )
    fused_norm = _convert_gemma4_vision_rmsnorm(rms_norm, projection).cuda()
    expected = residual + _rmsnorm_reference(
        torch.clamp(
            hidden_states,
            projection.output_min,
            projection.output_max,
        ),
        rms_norm.weight,
        rms_norm.eps,
    )

    actual = fused_norm(hidden_states, residual)
    torch.testing.assert_close(actual, expected, atol=0.015625, rtol=0.01)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = fused_norm(hidden_states, residual)
    graph.replay()
    torch.testing.assert_close(replayed, expected, atol=0.015625, rtol=0.01)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_gemma4_vision_fused_gelu_mul_clip_cuda_graph() -> None:
    torch.manual_seed(0)
    gate = 8 * torch.randn(
        2520,
        4304,
        device="cuda",
        dtype=torch.bfloat16,
    )
    up = 8 * torch.randn_like(gate)
    gate_min = torch.tensor(-2.5, device="cuda", dtype=torch.bfloat16)
    gate_max = torch.tensor(2.25, device="cuda", dtype=torch.bfloat16)
    up_min = torch.tensor(-3.0, device="cuda", dtype=torch.bfloat16)
    up_max = torch.tensor(2.75, device="cuda", dtype=torch.bfloat16)
    output_min = torch.tensor(-4.0, device="cuda", dtype=torch.bfloat16)
    output_max = torch.tensor(3.5, device="cuda", dtype=torch.bfloat16)
    expected = torch.clamp(
        torch.nn.functional.gelu(
            torch.clamp(gate, gate_min, gate_max), approximate="tanh"
        )
        * torch.clamp(up, up_min, up_max),
        output_min,
        output_max,
    )

    actual = _gemma4_vision_gelu_mul_clip(
        gate,
        up,
        gate_min,
        gate_max,
        up_min,
        up_max,
        output_min,
        output_max,
    )
    torch.testing.assert_close(actual, expected, atol=0.015625, rtol=0.01)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = _gemma4_vision_gelu_mul_clip(
            gate,
            up,
            gate_min,
            gate_max,
            up_min,
            up_max,
            output_min,
            output_max,
        )
    graph.replay()
    torch.testing.assert_close(replayed, expected, atol=0.015625, rtol=0.01)
