# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    ExpertTokensMetadata,
)
from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
    Fp8MoeBackend,
    convert_to_fp8_moe_kernel_format,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    mxfp8_e4m3_quantize,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import (
    has_flashinfer_cutedsl_grouped_gemm_nt_masked_zero_output,
)

if not current_platform.has_device_capability(100):
    pytest.skip(
        reason="MXFP8 CuteDSL MoE requires compute capability 10 or above.",
        allow_module_level=True,
    )


def _run_case_in_subprocess(case_name: str):
    code = """
import importlib.util
import sys

path, case_name = sys.argv[1:3]
spec = importlib.util.spec_from_file_location("_cutedsl_mxfp8_moe_case", path)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)
getattr(mod, case_name)()
"""
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    subprocess.run(
        [sys.executable, "-c", code, os.path.abspath(__file__), case_name],
        check=True,
        env=env,
        timeout=300,
    )


def _make_mxfp8_weights(
    num_experts: int,
    hidden_dim: int,
    intermediate_dim: int,
    device: str,
):
    w1_ref = (
        torch.randn(
            num_experts,
            2 * intermediate_dim,
            hidden_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        / 10
    )
    w2_ref = (
        torch.randn(
            num_experts,
            hidden_dim,
            intermediate_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        / 10
    )
    w1 = torch.empty_like(w1_ref, dtype=torch.float8_e4m3fn)
    w2 = torch.empty_like(w2_ref, dtype=torch.float8_e4m3fn)
    w1_scale = torch.empty(
        (num_experts, 2 * intermediate_dim, hidden_dim // MXFP8_BLOCK_SIZE),
        dtype=torch.uint8,
        device=device,
    )
    w2_scale = torch.empty(
        (num_experts, hidden_dim, intermediate_dim // MXFP8_BLOCK_SIZE),
        dtype=torch.uint8,
        device=device,
    )
    for expert in range(num_experts):
        w1[expert], w1_scale[expert] = mxfp8_e4m3_quantize(
            w1_ref[expert], is_sf_swizzled_layout=False
        )
        w2[expert], w2_scale[expert] = mxfp8_e4m3_quantize(
            w2_ref[expert], is_sf_swizzled_layout=False
        )

    layer = SimpleNamespace(weight_block_size=[1, MXFP8_BLOCK_SIZE])
    return convert_to_fp8_moe_kernel_format(
        fp8_backend=Fp8MoeBackend.BATCHED_FLASHINFER_MXFP8,
        layer=layer,
        w13=w1,
        w2=w2,
        w13_scale=w1_scale,
        w2_scale=w2_scale,
        w13_input_scale=None,
        w2_input_scale=None,
    )


def _align128(val: int) -> int:
    return (val + 127) // 128 * 128


def _make_native_mxfp8_input(
    hidden_states: torch.Tensor, expert_num_tokens: torch.Tensor
):
    num_experts, max_tokens, hidden_dim = hidden_states.shape
    scale_stride = _align128(max_tokens)
    problem_sizes = torch.empty(
        (num_experts, 3), dtype=torch.int32, device=hidden_states.device
    )
    problem_sizes[:, 0].copy_(expert_num_tokens.to(torch.int32))
    problem_sizes[:, 1].fill_(hidden_dim)
    problem_sizes[:, 2].fill_(hidden_dim)
    expert_offsets = (
        torch.arange(num_experts, dtype=torch.int32, device=hidden_states.device)
        * max_tokens
    )
    blockscale_offsets = (
        torch.arange(num_experts, dtype=torch.int32, device=hidden_states.device)
        * scale_stride
    )
    hidden_fp8 = torch.zeros_like(hidden_states, dtype=torch.float8_e4m3fn)
    hidden_scale_flat = torch.zeros(
        (num_experts * scale_stride, hidden_dim // MXFP8_BLOCK_SIZE),
        dtype=torch.uint8,
        device=hidden_states.device,
    )
    ops.mxfp8_experts_quant(
        hidden_states.view(-1, hidden_dim),
        problem_sizes,
        expert_offsets,
        blockscale_offsets,
        hidden_fp8.view(-1, hidden_dim),
        hidden_scale_flat,
    )
    return hidden_fp8, hidden_scale_flat.view(
        num_experts, scale_stride, hidden_dim // MXFP8_BLOCK_SIZE
    )


def _make_cutedsl_mxfp8_experts(
    num_experts: int,
    max_tokens: int,
    hidden_dim: int,
    intermediate_dim: int,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    device: str,
):
    from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutedsl_batched_mxfp8_moe import (  # noqa: E501
        FlashInferCuteDSLBatchedExpertsMxfp8,
    )

    parallel_config = FusedMoEParallelConfig(
        tp_size=1,
        pcp_size=1,
        dp_size=1,
        ep_size=1,
        tp_rank=0,
        pcp_rank=0,
        dp_rank=0,
        ep_rank=0,
        sp_size=1,
        use_ep=False,
        all2all_backend="deepep_low_latency",
        enable_eplb=False,
    )
    moe_config = FusedMoEConfig(
        num_experts=num_experts,
        experts_per_token=1,
        hidden_dim=hidden_dim,
        intermediate_size_per_partition=intermediate_dim,
        num_local_experts=num_experts,
        num_logical_experts=num_experts,
        activation=MoEActivation.SILU,
        device=device,
        routing_method=RoutingMethodType.Default,
        moe_parallel_config=parallel_config,
        in_dtype=torch.bfloat16,
        max_num_tokens=max_tokens,
    )
    quant_config = FusedMoEQuantConfig.make(
        "mxfp8",
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        block_shape=[1, MXFP8_BLOCK_SIZE],
        weight_dtype="mxfp8",
    )
    return FlashInferCuteDSLBatchedExpertsMxfp8(
        moe_config, quant_config, max_num_tokens=max_tokens, num_dispatchers=1
    )


def _apply_cutedsl_mxfp8_experts(
    experts,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    expert_num_tokens: torch.Tensor,
    a1q_scale: torch.Tensor | None,
):
    num_experts, max_tokens, hidden_dim = hidden_states.shape
    intermediate_dim = w2.size(2)
    output = torch.empty(
        (num_experts, max_tokens, hidden_dim),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )
    workspace13 = torch.empty(
        (num_experts, max_tokens, max(2 * intermediate_dim, hidden_dim)),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )
    workspace2 = torch.empty(
        (num_experts, max_tokens, max(intermediate_dim, hidden_dim)),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )
    experts.apply(
        output=output,
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=torch.ones((max_tokens, 1), dtype=torch.float32, device="cuda"),
        topk_ids=torch.zeros((max_tokens, 1), dtype=torch.long, device="cuda"),
        activation=MoEActivation.SILU,
        global_num_experts=num_experts,
        expert_map=None,
        a1q_scale=a1q_scale,
        a2_scale=None,
        workspace13=workspace13,
        workspace2=workspace2,
        expert_tokens_meta=ExpertTokensMetadata(
            expert_num_tokens=expert_num_tokens, expert_num_tokens_cpu=None
        ),
        apply_router_weight_on_input=False,
    )
    return output


@pytest.mark.skipif(
    not has_flashinfer_cutedsl_grouped_gemm_nt_masked_zero_output(),
    reason="FlashInfer CuteDSL grouped masked GEMM zeroing is unavailable.",
)
def test_cutedsl_mxfp8_non_native_dispatch_locally_quantizes_bf16_input():
    _run_case_in_subprocess(
        "_case_cutedsl_mxfp8_non_native_dispatch_locally_quantizes_bf16_input"
    )


@torch.inference_mode()
def _case_cutedsl_mxfp8_non_native_dispatch_locally_quantizes_bf16_input():
    from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutedsl_batched_mxfp8_moe import (  # noqa: E501
        FlashInferCuteDSLBatchedExpertsMxfp8,
    )

    torch.manual_seed(0)
    device = "cuda"
    num_experts = 2
    max_tokens = 4
    hidden_dim = 128
    intermediate_dim = 128
    hidden_states = (
        torch.randn(
            num_experts,
            max_tokens,
            hidden_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        / 10
    )
    expert_num_tokens = torch.tensor([4, 3], dtype=torch.int32, device=device)

    w1, w2, w1_scale, w2_scale = _make_mxfp8_weights(
        num_experts, hidden_dim, intermediate_dim, device
    )
    parallel_config = FusedMoEParallelConfig(
        tp_size=1,
        pcp_size=1,
        dp_size=1,
        ep_size=1,
        tp_rank=0,
        pcp_rank=0,
        dp_rank=0,
        ep_rank=0,
        sp_size=1,
        use_ep=False,
        all2all_backend="deepep_low_latency",
        enable_eplb=False,
    )
    moe_config = FusedMoEConfig(
        num_experts=num_experts,
        experts_per_token=1,
        hidden_dim=hidden_dim,
        intermediate_size_per_partition=intermediate_dim,
        num_local_experts=num_experts,
        num_logical_experts=num_experts,
        activation=MoEActivation.SILU,
        device=device,
        routing_method=RoutingMethodType.Default,
        moe_parallel_config=parallel_config,
        in_dtype=torch.bfloat16,
        max_num_tokens=max_tokens,
    )
    quant_config = FusedMoEQuantConfig.make(
        "mxfp8",
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        block_shape=[1, MXFP8_BLOCK_SIZE],
        weight_dtype="mxfp8",
    )
    experts = FlashInferCuteDSLBatchedExpertsMxfp8(
        moe_config, quant_config, max_num_tokens=max_tokens, num_dispatchers=1
    )
    output = torch.empty(
        (num_experts, max_tokens, hidden_dim),
        dtype=torch.bfloat16,
        device=device,
    )
    workspace13 = torch.empty(
        (num_experts, max_tokens, max(2 * intermediate_dim, hidden_dim)),
        dtype=torch.bfloat16,
        device=device,
    )
    workspace2 = torch.empty(
        (num_experts, max_tokens, max(intermediate_dim, hidden_dim)),
        dtype=torch.bfloat16,
        device=device,
    )

    experts.apply(
        output=output,
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=torch.ones((max_tokens, 1), dtype=torch.float32, device=device),
        topk_ids=torch.zeros((max_tokens, 1), dtype=torch.long, device=device),
        activation=MoEActivation.SILU,
        global_num_experts=num_experts,
        expert_map=None,
        a1q_scale=None,
        a2_scale=None,
        workspace13=workspace13,
        workspace2=workspace2,
        expert_tokens_meta=ExpertTokensMetadata(
            expert_num_tokens=expert_num_tokens, expert_num_tokens_cpu=None
        ),
        apply_router_weight_on_input=False,
    )

    assert torch.isfinite(output[:, : expert_num_tokens.max()]).all()


@pytest.mark.skipif(
    not has_flashinfer_cutedsl_grouped_gemm_nt_masked_zero_output(),
    reason="FlashInfer CuteDSL grouped masked GEMM zeroing is unavailable.",
)
def test_cutedsl_mxfp8_native_dispatch_masks_invalid_a1_rows():
    _run_case_in_subprocess(
        "_case_cutedsl_mxfp8_native_dispatch_masks_invalid_a1_rows"
    )


@pytest.mark.skipif(
    not has_flashinfer_cutedsl_grouped_gemm_nt_masked_zero_output(),
    reason="FlashInfer CuteDSL grouped masked GEMM zeroing is unavailable.",
)
def test_cutedsl_mxfp8_native_dispatch_ignores_stale_workspaces():
    _run_case_in_subprocess(
        "_case_cutedsl_mxfp8_native_dispatch_ignores_stale_workspaces"
    )


@pytest.mark.skipif(
    not has_flashinfer_cutedsl_grouped_gemm_nt_masked_zero_output(),
    reason="FlashInfer CuteDSL grouped masked GEMM zeroing is unavailable.",
)
def test_cutedsl_mxfp8_native_dispatch_does_not_full_zero_workspaces():
    _run_case_in_subprocess(
        "_case_cutedsl_mxfp8_native_dispatch_does_not_full_zero_workspaces"
    )


def test_cutedsl_mxfp8_native_dispatch_uses_flashinfer_zeroing():
    _run_case_in_subprocess(
        "_case_cutedsl_mxfp8_native_dispatch_uses_flashinfer_zeroing"
    )


@pytest.mark.skipif(
    not has_flashinfer_cutedsl_grouped_gemm_nt_masked_zero_output(),
    reason="FlashInfer CuteDSL grouped masked GEMM zeroing is unavailable.",
)
def test_cutedsl_mxfp8_native_dispatch_high_hidden_dim_a2_scales():
    _run_case_in_subprocess(
        "_case_cutedsl_mxfp8_native_dispatch_high_hidden_dim_a2_scales"
    )


@torch.inference_mode()
def _case_cutedsl_mxfp8_native_dispatch_masks_invalid_a1_rows():
    torch.manual_seed(1)
    device = "cuda"
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    num_experts = 2
    max_tokens = 5
    hidden_dim = 128
    intermediate_dim = 128
    expert_num_tokens = torch.tensor([5, 2], dtype=torch.int32, device=device)
    hidden_states = (
        torch.randn(
            num_experts,
            max_tokens,
            hidden_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        / 10
    )
    hidden_fp8, hidden_scale = _make_native_mxfp8_input(
        hidden_states, expert_num_tokens
    )
    poisoned_fp8 = hidden_fp8.clone()
    poisoned_scale = hidden_scale.clone()
    poisoned_fp8[1, 2:].view(torch.uint8).fill_(0x7F)
    poisoned_scale[1, 2:].fill_(0xFF)

    w1, w2, w1_scale, w2_scale = _make_mxfp8_weights(
        num_experts, hidden_dim, intermediate_dim, device
    )
    experts = _make_cutedsl_mxfp8_experts(
        num_experts,
        max_tokens,
        hidden_dim,
        intermediate_dim,
        w1_scale,
        w2_scale,
        device,
    )

    clean_output = _apply_cutedsl_mxfp8_experts(
        experts, hidden_fp8, w1, w2, expert_num_tokens, hidden_scale
    )
    poisoned_output = _apply_cutedsl_mxfp8_experts(
        experts, poisoned_fp8, w1, w2, expert_num_tokens, poisoned_scale
    )

    valid = (
        torch.arange(max_tokens, device=device).unsqueeze(0)
        < expert_num_tokens.unsqueeze(1)
    )
    assert torch.isfinite(poisoned_output[valid]).all()
    torch.testing.assert_close(poisoned_output[valid], clean_output[valid])


@torch.inference_mode()
def _case_cutedsl_mxfp8_native_dispatch_high_hidden_dim_a2_scales():
    torch.manual_seed(123)
    device = "cuda"
    num_experts = 16
    max_tokens = 128
    hidden_dim = 2560
    intermediate_dim = 128
    expert_num_tokens = torch.tensor(
        [10, 8, 10, 12, 6, 9, 5, 13, 5, 6, 3, 11, 13, 7, 5, 12],
        dtype=torch.int32,
        device=device,
    )
    hidden_states = (
        torch.randn(
            num_experts,
            max_tokens,
            hidden_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        / 10
    )
    hidden_fp8, hidden_scale = _make_native_mxfp8_input(
        hidden_states, expert_num_tokens
    )
    w1, w2, w1_scale, w2_scale = _make_mxfp8_weights(
        num_experts, hidden_dim, intermediate_dim, device
    )
    experts = _make_cutedsl_mxfp8_experts(
        num_experts,
        max_tokens,
        hidden_dim,
        intermediate_dim,
        w1_scale,
        w2_scale,
        device,
    )

    output = _apply_cutedsl_mxfp8_experts(
        experts, hidden_fp8, w1, w2, expert_num_tokens, hidden_scale
    )
    valid = (
        torch.arange(max_tokens, device=device).unsqueeze(0)
        < expert_num_tokens.unsqueeze(1)
    )

    assert torch.isfinite(output[valid]).all()
    assert output[valid].float().abs().max() < 10


@torch.inference_mode()
def _case_cutedsl_mxfp8_native_dispatch_ignores_stale_workspaces():
    torch.manual_seed(2)
    device = "cuda"
    num_experts = 2
    max_tokens = 7
    hidden_dim = 512
    intermediate_dim = 128
    expert_num_tokens = torch.tensor([7, 3], dtype=torch.int32, device=device)
    hidden_states = (
        torch.randn(
            num_experts,
            max_tokens,
            hidden_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        / 10
    )
    hidden_fp8, hidden_scale = _make_native_mxfp8_input(
        hidden_states, expert_num_tokens
    )
    w1, w2, w1_scale, w2_scale = _make_mxfp8_weights(
        num_experts, hidden_dim, intermediate_dim, device
    )
    experts = _make_cutedsl_mxfp8_experts(
        num_experts,
        max_tokens,
        hidden_dim,
        intermediate_dim,
        w1_scale,
        w2_scale,
        device,
    )

    workspace13_shape = (
        num_experts,
        max_tokens,
        max(2 * intermediate_dim, hidden_dim),
    )
    output_shape = (num_experts, max_tokens, hidden_dim)
    common_numel = max(
        num_experts * max_tokens * workspace13_shape[-1],
        num_experts * max_tokens * output_shape[-1],
    )
    valid = (
        torch.arange(max_tokens, device=device).unsqueeze(0)
        < expert_num_tokens.unsqueeze(1)
    )

    def run(poison: bool) -> torch.Tensor:
        common = torch.empty(common_numel, dtype=torch.bfloat16, device=device)
        workspace13 = common[: num_experts * max_tokens * workspace13_shape[-1]].view(
            workspace13_shape
        )
        output = common[: num_experts * max_tokens * output_shape[-1]].view(
            output_shape
        )
        workspace2 = torch.empty(
            (num_experts, max_tokens, max(intermediate_dim, hidden_dim)),
            dtype=torch.bfloat16,
            device=device,
        )
        if poison:
            common.fill_(float("nan"))
            workspace2.fill_(float("nan"))
        else:
            common.zero_()
            workspace2.zero_()
        experts.apply(
            output=output,
            hidden_states=hidden_fp8.clone(),
            w1=w1,
            w2=w2,
            topk_weights=torch.ones(
                (max_tokens, 1), dtype=torch.float32, device=device
            ),
            topk_ids=torch.zeros((max_tokens, 1), dtype=torch.long, device=device),
            activation=MoEActivation.SILU,
            global_num_experts=num_experts,
            expert_map=None,
            a1q_scale=hidden_scale.clone(),
            a2_scale=None,
            workspace13=workspace13,
            workspace2=workspace2,
            expert_tokens_meta=ExpertTokensMetadata(
                expert_num_tokens=expert_num_tokens, expert_num_tokens_cpu=None
            ),
            apply_router_weight_on_input=False,
        )
        return output.clone()

    clean_output = run(poison=False)
    poisoned_output = run(poison=True)

    torch.testing.assert_close(poisoned_output[valid], clean_output[valid])


@torch.inference_mode()
def _case_cutedsl_mxfp8_native_dispatch_does_not_full_zero_workspaces():
    from vllm.model_executor.layers.fused_moe.experts import (
        flashinfer_cutedsl_batched_mxfp8_moe as cutedsl_mxfp8_moe,
    )

    torch.manual_seed(3)
    device = "cuda"
    num_experts = 2
    max_tokens = 7
    hidden_dim = 512
    intermediate_dim = 128
    expert_num_tokens = torch.tensor([7, 3], dtype=torch.int32, device=device)
    hidden_states = (
        torch.randn(
            num_experts,
            max_tokens,
            hidden_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        / 10
    )
    hidden_fp8, hidden_scale = _make_native_mxfp8_input(
        hidden_states, expert_num_tokens
    )
    w1, w2, w1_scale, w2_scale = _make_mxfp8_weights(
        num_experts, hidden_dim, intermediate_dim, device
    )
    experts = _make_cutedsl_mxfp8_experts(
        num_experts,
        max_tokens,
        hidden_dim,
        intermediate_dim,
        w1_scale,
        w2_scale,
        device,
    )
    workspace13 = torch.empty(
        (num_experts, max_tokens, max(2 * intermediate_dim, hidden_dim)),
        dtype=torch.bfloat16,
        device=device,
    )
    workspace2 = torch.empty(
        (num_experts, max_tokens, max(intermediate_dim, hidden_dim)),
        dtype=torch.bfloat16,
        device=device,
    )
    output = torch.empty(
        (num_experts, max_tokens, hidden_dim),
        dtype=torch.bfloat16,
        device=device,
    )
    workspace13.fill_(float("nan"))
    workspace2.fill_(float("nan"))
    output.fill_(float("nan"))

    def fail_full_zero(*args, **kwargs):
        raise AssertionError("full workspace zero should not run")

    old_zero_two_tensors = getattr(cutedsl_mxfp8_moe, "_zero_two_tensors", None)
    if old_zero_two_tensors is not None:
        cutedsl_mxfp8_moe._zero_two_tensors = fail_full_zero
    try:
        experts.apply(
            output=output,
            hidden_states=hidden_fp8.clone(),
            w1=w1,
            w2=w2,
            topk_weights=torch.ones(
                (max_tokens, 1), dtype=torch.float32, device=device
            ),
            topk_ids=torch.zeros((max_tokens, 1), dtype=torch.long, device=device),
            activation=MoEActivation.SILU,
            global_num_experts=num_experts,
            expert_map=None,
            a1q_scale=hidden_scale.clone(),
            a2_scale=None,
            workspace13=workspace13,
            workspace2=workspace2,
            expert_tokens_meta=ExpertTokensMetadata(
                expert_num_tokens=expert_num_tokens, expert_num_tokens_cpu=None
            ),
            apply_router_weight_on_input=False,
        )
    finally:
        if old_zero_two_tensors is not None:
            cutedsl_mxfp8_moe._zero_two_tensors = old_zero_two_tensors

    valid = (
        torch.arange(max_tokens, device=device).unsqueeze(0)
        < expert_num_tokens.unsqueeze(1)
    )
    assert torch.isfinite(output[valid]).all()


@torch.inference_mode()
def _case_cutedsl_mxfp8_native_dispatch_uses_flashinfer_zeroing():
    from vllm.model_executor.layers.fused_moe.experts import (
        flashinfer_cutedsl_batched_mxfp8_moe as cutedsl_mxfp8_moe,
    )

    torch.manual_seed(4)
    device = "cuda"
    num_experts = 2
    max_tokens = 7
    hidden_dim = 512
    intermediate_dim = 128
    expert_num_tokens = torch.tensor([7, 3], dtype=torch.int32, device=device)
    hidden_fp8 = (
        torch.randn(num_experts, max_tokens, hidden_dim, device=device) / 10
    ).to(torch.float8_e4m3fn)
    scale_stride = _align128(max_tokens)
    hidden_scale = torch.zeros(
        (num_experts, scale_stride, hidden_dim // MXFP8_BLOCK_SIZE),
        dtype=torch.uint8,
        device=device,
    )
    w1 = (
        torch.randn(num_experts, 2 * intermediate_dim, hidden_dim, device=device) / 10
    ).to(torch.float8_e4m3fn)
    w2 = (
        torch.randn(num_experts, hidden_dim, intermediate_dim, device=device) / 10
    ).to(torch.float8_e4m3fn)
    w1_scale = torch.zeros(
        (num_experts, 2 * intermediate_dim, hidden_dim // MXFP8_BLOCK_SIZE),
        dtype=torch.uint8,
        device=device,
    )
    w2_scale = torch.zeros(
        (num_experts, hidden_dim, intermediate_dim // MXFP8_BLOCK_SIZE),
        dtype=torch.uint8,
        device=device,
    )
    experts = _make_cutedsl_mxfp8_experts(
        num_experts,
        max_tokens,
        hidden_dim,
        intermediate_dim,
        w1_scale,
        w2_scale,
        device,
    )
    workspace13 = torch.empty(
        (num_experts, max_tokens, max(2 * intermediate_dim, hidden_dim)),
        dtype=torch.bfloat16,
        device=device,
    )
    workspace2 = torch.empty(
        (num_experts, max_tokens, max(intermediate_dim, hidden_dim)),
        dtype=torch.bfloat16,
        device=device,
    )
    output = torch.empty(
        (num_experts, max_tokens, hidden_dim),
        dtype=torch.bfloat16,
        device=device,
    )

    old_flashinfer_gemm = cutedsl_mxfp8_moe.flashinfer_cutedsl_grouped_gemm_nt_masked
    old_quantize_act_for_cute = cutedsl_mxfp8_moe._quantize_act_for_cute
    zero_masked_output_values = []
    zero_masked_sfa_values = []

    def checked_flashinfer_gemm(*args, **kwargs):
        zero_masked_output_values.append(kwargs.get("zero_masked_output"))
        zero_masked_sfa_values.append(kwargs.get("zero_masked_sfa"))
        out = args[2]
        masked_m = args[3]
        out.zero_()
        for expert in range(masked_m.numel()):
            out[: int(masked_m[expert].item()), :, expert] = 1

    def fake_quantize_act_for_cute(
        act_out,
        expert_num_tokens_arg,
        num_experts_arg,
        max_num_tokens_arg,
        n_out,
    ):
        del expert_num_tokens_arg
        fp8 = torch.zeros(
            (num_experts_arg * max_num_tokens_arg, n_out),
            dtype=torch.float8_e4m3fn,
            device=act_out.device,
        )
        sf = torch.zeros(
            (
                num_experts_arg,
                _align128(max_num_tokens_arg),
                n_out // MXFP8_BLOCK_SIZE,
            ),
            dtype=torch.uint8,
            device=act_out.device,
        )
        return fp8, sf.view(num_experts_arg, -1).view(torch.float8_e8m0fnu)

    cutedsl_mxfp8_moe.flashinfer_cutedsl_grouped_gemm_nt_masked = (
        checked_flashinfer_gemm
    )
    cutedsl_mxfp8_moe._quantize_act_for_cute = fake_quantize_act_for_cute
    try:
        experts.apply(
            output=output,
            hidden_states=hidden_fp8,
            w1=w1,
            w2=w2,
            topk_weights=torch.ones(
                (max_tokens, 1), dtype=torch.float32, device=device
            ),
            topk_ids=torch.zeros((max_tokens, 1), dtype=torch.long, device=device),
            activation=MoEActivation.SILU,
            global_num_experts=num_experts,
            expert_map=None,
            a1q_scale=hidden_scale,
            a2_scale=None,
            workspace13=workspace13,
            workspace2=workspace2,
            expert_tokens_meta=ExpertTokensMetadata(
                expert_num_tokens=expert_num_tokens, expert_num_tokens_cpu=None
            ),
            apply_router_weight_on_input=False,
        )
    finally:
        cutedsl_mxfp8_moe.flashinfer_cutedsl_grouped_gemm_nt_masked = (
            old_flashinfer_gemm
        )
        cutedsl_mxfp8_moe._quantize_act_for_cute = old_quantize_act_for_cute

    assert zero_masked_output_values == [True, True]
    assert zero_masked_sfa_values == [True, True]
    valid = (
        torch.arange(max_tokens, device=device).unsqueeze(0)
        < expert_num_tokens.unsqueeze(1)
    )
    assert torch.isfinite(output[valid]).all()
