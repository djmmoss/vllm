# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from SGLang:
# https://github.com/sgl-project/sglang/blob/ded068a76e00878881d52d5bfb791e0f60d7311b/sgl-kernel/tests/test_es_fp8_blockwise_moe.py

"""Tests for SM100 CUTLASS MXFP8 grouped MoE kernels."""

import random
from types import SimpleNamespace

import pytest
import torch

from tests.kernels.moe.utils import (
    is_sm100_supported,
    quantize_expert_rows_mxfp8,
)
from tests.kernels.utils import torch_moe_single
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
    apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.experts.cutlass_moe import (
    _format_deepep_mxfp8_scales_for_cutlass,
    _make_batched_mxfp8_problem_data,
    run_cutlass_batched_moe_mxfp8,
)
from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
    Fp8MoeBackend,
    convert_to_fp8_moe_kernel_format,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
)
from vllm.utils.math_utils import round_up
from vllm.utils.torch_utils import set_random_seed

random.seed(42)
set_random_seed(42)


# Copy from: https://github.com/deepseek-ai/DeepGEMM/blob/main/deep_gemm/utils.py
def calc_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


def compute_ref_output(
    input_tensor: torch.Tensor,
    weight_list: list[torch.Tensor],
    expert_offsets: list[int],
    expert_offset: int,
    num_experts: int,
) -> torch.Tensor:
    # Build a top-1 routing score so each token maps to its owning expert.
    score = torch.full(
        (expert_offset, num_experts),
        -1e9,
        device=input_tensor.device,
        dtype=torch.float32,
    )
    for g in range(num_experts):
        start = expert_offsets[g]
        end = expert_offsets[g + 1] if g + 1 < num_experts else expert_offset
        score[start:end, g] = 0.0

    return torch_moe_single(
        input_tensor, torch.stack(weight_list, dim=0), score, topk=1
    )


def compute_kernel_output(
    input_tensor: torch.Tensor,
    weight_tensor: torch.Tensor,
    problem_sizes: list[list[int]],
    aux_problem_sizes: list[list[int]],
    expert_offsets: list[int],
    aux_expert_offsets: list[int],
    input_blockscale_offsets: list[int],
    weight_blockscale_offsets: list[int],
    input_blockscale_offset: int,
    n_g: int,
    k_g: int,
    num_experts: int,
    expert_offset: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    device = input_tensor.device
    _problem_sizes = torch.tensor(problem_sizes).to(device=device, dtype=torch.int32)
    _aux_problem_sizes = torch.tensor(aux_problem_sizes).to(
        device=device, dtype=torch.int32
    )
    _expert_offsets = torch.tensor(expert_offsets).to(device=device, dtype=torch.int32)
    _aux_expert_offsets = torch.tensor(aux_expert_offsets).to(
        device=device, dtype=torch.int32
    )
    _input_blockscale_offsets = torch.tensor(input_blockscale_offsets).to(
        device=device, dtype=torch.int32
    )
    _weight_blockscale_offsets = torch.tensor(weight_blockscale_offsets).to(
        device=device, dtype=torch.int32
    )

    input_quant = torch.zeros_like(
        input_tensor, dtype=torch.float8_e4m3fn, device=device
    )
    input_scale_factor = torch.zeros(
        (input_blockscale_offset, k_g // MXFP8_BLOCK_SIZE),
        dtype=torch.uint8,
        device=device,
    )

    weight_quant = torch.zeros_like(
        weight_tensor, dtype=torch.float8_e4m3fn, device=device
    )
    weight_scale_factor = torch.zeros(
        (num_experts, n_g, k_g // MXFP8_BLOCK_SIZE),
        dtype=torch.uint8,
        device=device,
    )

    ops.mxfp8_experts_quant(
        input_tensor,
        _problem_sizes,
        _expert_offsets,
        _input_blockscale_offsets,
        input_quant,
        input_scale_factor,
    )

    ops.mxfp8_experts_quant(
        weight_tensor,
        _aux_problem_sizes,
        _aux_expert_offsets,
        _weight_blockscale_offsets,
        weight_quant,
        weight_scale_factor,
    )
    weight_quant = weight_quant.view(num_experts, n_g, k_g).transpose(1, 2)
    weight_scale_factor = weight_scale_factor.view(
        num_experts, n_g, k_g // MXFP8_BLOCK_SIZE
    ).transpose(1, 2)

    output = torch.empty((expert_offset, n_g), device=device, dtype=out_dtype)
    ops.cutlass_mxfp8_grouped_mm(
        input_quant,
        weight_quant,
        input_scale_factor,
        weight_scale_factor,
        output,
        _problem_sizes,
        _expert_offsets,
        _input_blockscale_offsets,
    )
    return output


def test_make_batched_mxfp8_problem_data():
    expert_num_tokens = torch.tensor([3, 0, 129], dtype=torch.int64)
    problem_sizes, expert_offsets, blockscale_offsets, scale_rows = (
        _make_batched_mxfp8_problem_data(
            expert_num_tokens=expert_num_tokens,
            max_num_tokens=160,
            n=256,
            k=128,
        )
    )

    assert problem_sizes.tolist() == [
        [3, 256, 128],
        [0, 256, 128],
        [129, 256, 128],
    ]
    assert expert_offsets.tolist() == [0, 160, 320]
    assert blockscale_offsets.tolist() == [0, 256, 512]
    assert scale_rows == 768


def test_format_deepep_mxfp8_scales_for_cutlass():
    act_scales = torch.arange(2 * 128 * 16, dtype=torch.uint8).view(2, 128, 16)

    formatted = _format_deepep_mxfp8_scales_for_cutlass(
        act_scales,
        max_num_tokens=64,
        hidden_dim=512,
        scale_rows=256,
    )

    assert formatted.shape == (256, 16)
    assert torch.equal(formatted[:128], act_scales[0])
    assert torch.equal(formatted[128:], act_scales[1])

    act_scales = torch.arange(2 * 256 * 16, dtype=torch.uint8).view(2, 256, 16)
    formatted = _format_deepep_mxfp8_scales_for_cutlass(
        act_scales,
        max_num_tokens=129,
        hidden_dim=512,
        scale_rows=512,
    )
    assert formatted.shape == (512, 16)
    assert torch.equal(formatted[:256], act_scales[0])
    assert torch.equal(formatted[256:], act_scales[1])

    with pytest.raises(AssertionError):
        _format_deepep_mxfp8_scales_for_cutlass(
            act_scales[:, :128],
            max_num_tokens=129,
            hidden_dim=512,
            scale_rows=512,
        )

    with pytest.raises(AssertionError):
        _format_deepep_mxfp8_scales_for_cutlass(
            torch.empty(2, 256, 32, dtype=torch.uint8)[:, :, ::2],
            max_num_tokens=129,
            hidden_dim=512,
            scale_rows=512,
        )


def test_convert_batched_cutlass_mxfp8_kernel_format():
    num_experts = 2
    hidden_size = 128
    intermediate_size = 128
    w13_rows = 2 * intermediate_size
    layer = SimpleNamespace(weight_block_size=[1, MXFP8_BLOCK_SIZE])

    w13 = torch.empty(num_experts, w13_rows, hidden_size, dtype=torch.float8_e4m3fn)
    w2 = torch.empty(
        num_experts, hidden_size, intermediate_size, dtype=torch.float8_e4m3fn
    )
    w13_scale = torch.arange(
        num_experts * w13_rows * (hidden_size // MXFP8_BLOCK_SIZE), dtype=torch.uint8
    ).view(num_experts, w13_rows, hidden_size // MXFP8_BLOCK_SIZE)
    w2_scale = torch.arange(
        num_experts * hidden_size * (intermediate_size // MXFP8_BLOCK_SIZE),
        dtype=torch.uint8,
    ).view(num_experts, -1)

    w13_out, w2_out, w13_scale_out, w2_scale_out = convert_to_fp8_moe_kernel_format(
        fp8_backend=Fp8MoeBackend.BATCHED_VLLM_CUTLASS,
        layer=layer,
        w13=w13,
        w2=w2,
        w13_scale=w13_scale,
        w2_scale=w2_scale,
        w13_input_scale=None,
        w2_input_scale=None,
    )

    assert w13_out.shape == (num_experts, hidden_size, w13_rows)
    assert w2_out.shape == (num_experts, intermediate_size, hidden_size)
    assert w13_out.stride(1) == 1
    assert w2_out.stride(1) == 1
    assert w13_out.data_ptr() == w13.data_ptr()
    assert w2_out.data_ptr() == w2.data_ptr()

    assert torch.equal(w13_scale_out, w13_scale.contiguous().view(num_experts, -1))
    assert torch.equal(w2_scale_out, w2_scale.contiguous().view(num_experts, -1))


@pytest.mark.skipif(
    not is_sm100_supported(),
    reason=(
        "cutlass_mxfp8_grouped_mm and mxfp8_experts_quant "
        "are only supported on CUDA SM100"
    ),
)
@pytest.mark.parametrize("num_experts", [8, 16, 32, 64])
@pytest.mark.parametrize("out_dtype", [torch.half, torch.bfloat16])
def test_cutlass_mxfp8_grouped_mm(num_experts, out_dtype):
    device = "cuda"
    alignment = 128
    n_g = random.randint(1, 64) * alignment
    k_g = random.randint(1, 64) * alignment

    expert_offset = 0
    expert_offsets = []
    aux_expert_offset = 0
    aux_expert_offsets = []
    input_blockscale_offset = 0
    input_blockscale_offsets = []
    weight_blockscale_offset = 0
    weight_blockscale_offsets = []
    problem_sizes = []
    aux_problem_sizes = []
    input_list = []
    weight_list = []

    for g in range(num_experts):
        m_g = random.randint(1, 512)
        expert_offsets.append(expert_offset)
        expert_offset += m_g
        aux_expert_offsets.append(aux_expert_offset)
        aux_expert_offset += n_g
        input_blockscale_offsets.append(input_blockscale_offset)
        input_blockscale_offset += round_up(m_g, 128)
        weight_blockscale_offsets.append(weight_blockscale_offset)
        weight_blockscale_offset += n_g  # n_g already align to 128
        problem_sizes.append([m_g, n_g, k_g])
        aux_problem_sizes.append([n_g, m_g, k_g])

        input_tensor = torch.normal(
            0.0, std=1.0, size=(m_g, k_g), device=device, dtype=out_dtype
        )  # (M, K):(K, 1)
        weight_tensor = torch.normal(
            0.0, std=1.0, size=(n_g, k_g), device=device, dtype=out_dtype
        )  # (N, K):(K, 1)

        input_list.append(input_tensor)
        weight_list.append(weight_tensor)
    input_tensor = torch.concat(input_list, dim=0)
    weight_tensor = torch.concat(weight_list, dim=0)

    ref_output = compute_ref_output(
        input_tensor=input_tensor,
        weight_list=weight_list,
        expert_offsets=expert_offsets,
        expert_offset=expert_offset,
        num_experts=num_experts,
    )
    output = compute_kernel_output(
        input_tensor=input_tensor,
        weight_tensor=weight_tensor,
        problem_sizes=problem_sizes,
        aux_problem_sizes=aux_problem_sizes,
        expert_offsets=expert_offsets,
        aux_expert_offsets=aux_expert_offsets,
        input_blockscale_offsets=input_blockscale_offsets,
        weight_blockscale_offsets=weight_blockscale_offsets,
        input_blockscale_offset=input_blockscale_offset,
        n_g=n_g,
        k_g=k_g,
        num_experts=num_experts,
        expert_offset=expert_offset,
        out_dtype=out_dtype,
    )

    for g in range(num_experts):
        baseline = ref_output[
            expert_offsets[g] : (expert_offsets[g] + problem_sizes[g][0])
        ]
        actual = output[expert_offsets[g] : (expert_offsets[g] + problem_sizes[g][0])]
        diff = calc_diff(actual, baseline)
        assert diff < 0.001
        print(
            f"m_g={baseline.shape[0]} n_g={n_g} k_g={k_g} num_experts={num_experts}, "
            f"out_dtype={out_dtype}, diff={diff:.5f}: OK"
        )


@pytest.mark.skipif(
    not is_sm100_supported(),
    reason=(
        "cutlass_mxfp8_grouped_mm and mxfp8_experts_quant "
        "are only supported on CUDA SM100"
    ),
)
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.half])
def test_cutlass_batched_mxfp8_experts(out_dtype):
    device = "cuda"
    num_experts = 4
    max_tokens = 17
    hidden_size = 128
    intermediate_size = 128
    activation = MoEActivation.SILU
    w13_rows = 2 * intermediate_size

    hidden_states = (
        torch.randn(
            num_experts,
            max_tokens,
            hidden_size,
            device=device,
            dtype=out_dtype,
        )
        / 20
    )
    w13 = (
        torch.randn(
            num_experts,
            w13_rows,
            hidden_size,
            device=device,
            dtype=out_dtype,
        )
        / 20
    )
    w2 = (
        torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            device=device,
            dtype=out_dtype,
        )
        / 20
    )

    w13_q, w13_scale = quantize_expert_rows_mxfp8(
        w13.reshape(num_experts * w13_rows, hidden_size),
        w13_rows,
        num_experts,
    )
    w13_q = w13_q.view(num_experts, w13_rows, hidden_size)
    w13_scale = w13_scale.view(num_experts, w13_rows, hidden_size // MXFP8_BLOCK_SIZE)

    w2_q, w2_scale = quantize_expert_rows_mxfp8(
        w2.reshape(num_experts * hidden_size, intermediate_size),
        hidden_size,
        num_experts,
    )
    w2_q = w2_q.view(num_experts, hidden_size, intermediate_size)
    w2_scale = w2_scale.view(
        num_experts, hidden_size, intermediate_size // MXFP8_BLOCK_SIZE
    )

    layer = SimpleNamespace(weight_block_size=[1, MXFP8_BLOCK_SIZE])
    w13_q, w2_q, w13_scale, w2_scale = convert_to_fp8_moe_kernel_format(
        fp8_backend=Fp8MoeBackend.BATCHED_VLLM_CUTLASS,
        layer=layer,
        w13=w13_q,
        w2=w2_q,
        w13_scale=w13_scale,
        w2_scale=w2_scale,
        w13_input_scale=None,
        w2_input_scale=None,
    )

    output = torch.empty_like(hidden_states)
    workspace13 = torch.empty(
        num_experts, max_tokens, w13_rows, device=device, dtype=out_dtype
    )
    workspace2 = torch.empty(
        num_experts, max_tokens, hidden_size, device=device, dtype=out_dtype
    )
    expert_num_tokens = torch.tensor([3, 17, 0, 8], dtype=torch.int32, device=device)

    run_cutlass_batched_moe_mxfp8(
        output=output,
        hidden_states=hidden_states,
        w1=w13_q,
        w2=w2_q,
        w1_scale=w13_scale,
        w2_scale=w2_scale,
        expert_num_tokens=expert_num_tokens,
        activation=activation,
        workspace13=workspace13,
        workspace2=workspace2,
    )

    ref = torch.empty_like(output)
    act_out = torch.empty(max_tokens, intermediate_size, device=device, dtype=out_dtype)
    for expert, num_tokens in enumerate(expert_num_tokens.tolist()):
        if num_tokens == 0:
            continue
        mm1 = hidden_states[expert, :num_tokens] @ w13[expert].transpose(0, 1)
        apply_moe_activation(activation, act_out[:num_tokens], mm1)
        ref[expert, :num_tokens] = act_out[:num_tokens] @ w2[expert].transpose(0, 1)

    for expert, num_tokens in enumerate(expert_num_tokens.tolist()):
        if num_tokens == 0:
            continue
        diff = calc_diff(output[expert, :num_tokens], ref[expert, :num_tokens])
        assert diff < 0.05


if __name__ == "__main__":
    pytest.main([__file__])
