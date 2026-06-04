# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer CuteDSL MXFP8 batched MoE experts (BatchedExperts format).

Used when DeepEP low-latency native MXFP8 dispatch is on. The kernel is
FlashInfer's masked block-scaled grouped GEMM:
``flashinfer.gemm.grouped_gemm_nt_masked`` with ``ab_dtype="float8_e4m3fn"``,
``sf_dtype="float8_e8m0fnu"``, ``sf_vec_size=32``.
"""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
    apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceDelegate,
)
from vllm.model_executor.layers.fused_moe.utils import (
    _resize_cache,
    mxfp8_moe_nvtx_range,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    swizzle_mxfp8_scales_batched_for_cute,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp8Dynamic,
    kMxfp8Static,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.flashinfer import (
    flashinfer_cutedsl_grouped_gemm_nt_masked,
    has_flashinfer_cutedsl_grouped_gemm_nt_masked,
)

logger = init_logger(__name__)

_BATCHED_PROBLEM_OFFSETS_CACHE: dict[
    tuple[str, int, int, int], tuple[torch.Tensor, torch.Tensor]
] = {}


@triton.jit
def _zero_invalid_rows_kernel(
    tensor_ptr,
    expert_num_tokens_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    expert = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid_tokens = tl.load(expert_num_tokens_ptr + expert)
    mask = (offsets < M * K) & (offsets >= valid_tokens * K)
    zero = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    tl.store(tensor_ptr + expert * M * K + offsets, zero, mask=mask)


def _zero_invalid_rows(tensor: torch.Tensor, expert_num_tokens: torch.Tensor) -> None:
    assert tensor.is_contiguous()
    assert tensor.dim() == 3
    E, M, K = tensor.shape
    if E == 0 or M == 0 or K == 0:
        return
    block_size = 1024
    _zero_invalid_rows_kernel[(E, triton.cdiv(M * K, block_size))](
        tensor, expert_num_tokens, M, K, BLOCK_SIZE=block_size
    )


@triton.jit
def _zero_invalid_scale_rows_kernel(
    scale_ptr,
    expert_num_tokens_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    expert = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row_offsets = offsets // K
    valid_tokens = tl.load(expert_num_tokens_ptr + expert)
    in_bounds = offsets < M * K
    values = tl.load(
        scale_ptr + expert * M * K + offsets,
        mask=in_bounds,
        other=0,
    )
    mask = in_bounds & ((row_offsets >= valid_tokens) | (values == 0xFF))
    zero = tl.zeros((BLOCK_SIZE,), dtype=tl.uint8)
    tl.store(scale_ptr + expert * M * K + offsets, zero, mask=mask)


def _zero_invalid_scale_rows(
    scale: torch.Tensor, expert_num_tokens: torch.Tensor
) -> None:
    assert scale.is_contiguous()
    assert scale.dtype == torch.uint8
    assert scale.dim() == 3
    E, M, K = scale.shape
    if E == 0 or M == 0 or K == 0:
        return
    block_size = 1024
    _zero_invalid_scale_rows_kernel[(E, triton.cdiv(M * K, block_size))](
        scale, expert_num_tokens, M, K, BLOCK_SIZE=block_size
    )


@triton.jit
def _zero_invalid_swizzled_scale_rows_kernel(
    scale_ptr,
    expert_num_tokens_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    NUM_K_TILES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    expert = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    in_bounds = offsets < M * K

    k4 = offsets % 4
    tmp = offsets // 4
    m4 = tmp % 4
    tmp = tmp // 4
    m32 = tmp % 32
    tmp = tmp // 32
    kt = tmp % NUM_K_TILES
    mt = tmp // NUM_K_TILES
    row_offsets = mt * 128 + m4 * 32 + m32
    scale_cols = kt * 4 + k4

    values = tl.load(
        scale_ptr + expert * M * K + offsets,
        mask=in_bounds,
        other=0,
    )
    valid_tokens = tl.load(expert_num_tokens_ptr + expert)
    mask = in_bounds & (
        (row_offsets >= valid_tokens) | (scale_cols >= K) | (values == 0xFF)
    )
    zero = tl.zeros((BLOCK_SIZE,), dtype=tl.uint8)
    tl.store(scale_ptr + expert * M * K + offsets, zero, mask=mask)


def _zero_invalid_swizzled_scale_rows(
    scale: torch.Tensor, expert_num_tokens: torch.Tensor
) -> None:
    assert scale.is_contiguous()
    assert scale.dtype == torch.uint8
    assert scale.dim() == 3
    E, M, K = scale.shape
    assert M % 128 == 0
    assert K % 4 == 0
    if E == 0 or M == 0 or K == 0:
        return
    block_size = 1024
    _zero_invalid_swizzled_scale_rows_kernel[(E, triton.cdiv(M * K, block_size))](
        scale,
        expert_num_tokens,
        M,
        K,
        NUM_K_TILES=triton.cdiv(K, 4),
        BLOCK_SIZE=block_size,
    )


def _align128(val: int) -> int:
    return (val + 127) // 128 * 128


@triton.jit
def _fill_problem_sizes_kernel(
    problem_sizes_ptr,
    expert_num_tokens_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < NUM_EXPERTS
    m = tl.load(expert_num_tokens_ptr + offsets, mask=mask, other=0).to(tl.int32)
    base = offsets * 3
    tl.store(problem_sizes_ptr + base, m, mask=mask)
    tl.store(problem_sizes_ptr + base + 1, N, mask=mask)
    tl.store(problem_sizes_ptr + base + 2, K, mask=mask)


def _fill_problem_sizes(
    problem_sizes: torch.Tensor,
    expert_num_tokens: torch.Tensor,
    n: int,
    k: int,
) -> None:
    assert problem_sizes.is_contiguous()
    assert problem_sizes.dtype == torch.int32
    num_experts = problem_sizes.size(0)
    if num_experts == 0:
        return
    block_size = 128
    _fill_problem_sizes_kernel[(triton.cdiv(num_experts, block_size),)](
        problem_sizes,
        expert_num_tokens,
        N=n,
        K=k,
        NUM_EXPERTS=num_experts,
        BLOCK_SIZE=block_size,
    )


def _get_batched_problem_offsets(
    num_experts: int,
    max_num_tokens: int,
    scale_stride: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    key = (str(device), num_experts, max_num_tokens, scale_stride)
    offsets = _BATCHED_PROBLEM_OFFSETS_CACHE.get(key)
    if offsets is not None:
        return offsets

    expert_ids = torch.arange(num_experts, dtype=torch.int32, device=device)
    expert_offsets = expert_ids * max_num_tokens
    blockscale_offsets = expert_ids * scale_stride
    offsets = (expert_offsets, blockscale_offsets)
    _BATCHED_PROBLEM_OFFSETS_CACHE[key] = offsets
    return offsets


def _make_batched_problem_data(
    expert_num_tokens: torch.Tensor,
    max_num_tokens: int,
    n: int,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    assert n % 128 == 0
    assert k % 128 == 0
    num_experts = expert_num_tokens.numel()
    device = expert_num_tokens.device
    problem_sizes = torch.empty((num_experts, 3), dtype=torch.int32, device=device)
    scale_stride = _align128(max_num_tokens)
    _fill_problem_sizes(problem_sizes, expert_num_tokens, n, k)
    expert_offsets, blockscale_offsets = _get_batched_problem_offsets(
        num_experts, max_num_tokens, scale_stride, device
    )
    scale_rows = num_experts * scale_stride
    return problem_sizes, expert_offsets, blockscale_offsets, scale_rows


def _quantize_act_for_cute(
    act_out: torch.Tensor,
    expert_num_tokens: torch.Tensor,
    num_experts: int,
    max_num_tokens: int,
    n_out: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize bf16 activations to MXFP8 and return (fp8, sf_swizzled_flat).

    ``act_out`` is laid out as ``(num_experts * max_num_tokens, n_out)``.

    ``mxfp8_experts_quant`` writes SF per ``ceil(m/128)*128`` rows per expert
    but ``copy_if`` predicates row reads on ``row < m``: rows past ``m`` keep
    stale fragment registers, which empirically changes the SF that's
    computed even for valid rows (the warp shuffle reduces over a register
    fragment whose lanes include the unloaded positions). To get SF that
    matches the CUTLASS quantize for valid rows, pass ``m = expert_num_tokens``
    here, then zero out past-mask SF bytes ourselves so the cute masked MMA
    doesn't accumulate stale TMA-pushed scale contributions.
    """
    assert act_out.dim() == 2
    assert act_out.size(0) == num_experts * max_num_tokens
    with mxfp8_moe_nvtx_range("mxfp8_moe:a2_quant_problem_data"):
        problem_sizes, expert_offsets, blockscale_offsets, scale_rows = (
            _make_batched_problem_data(
                expert_num_tokens, max_num_tokens, n=n_out, k=n_out
            )
        )
    with mxfp8_moe_nvtx_range("mxfp8_moe:a2_quant_alloc_zero"):
        fp8 = torch.zeros(
            (num_experts * max_num_tokens, n_out),
            dtype=torch.float8_e4m3fn,
            device=act_out.device,
        )
        sf_rm = torch.zeros(
            (scale_rows, n_out // MXFP8_BLOCK_SIZE),
            dtype=torch.uint8,
            device=act_out.device,
        )
    with mxfp8_moe_nvtx_range("mxfp8_moe:a2_quant_kernel"):
        ops.mxfp8_experts_quant(
            act_out, problem_sizes, expert_offsets, blockscale_offsets, fp8, sf_rm
        )
    # ``mxfp8_experts_quant`` writes scale factors in the Cute/CUTLASS swizzled
    # layout already. Zero past-mask SF bytes in that layout: the grouped GEMM
    # can TMA-read a whole scale tile even when masked_m is smaller.
    scale_stride = scale_rows // num_experts
    with mxfp8_moe_nvtx_range("mxfp8_moe:a2_quant_zero_past_sf"):
        _zero_invalid_swizzled_scale_rows(
            sf_rm.view(num_experts, scale_stride, n_out // MXFP8_BLOCK_SIZE),
            expert_num_tokens,
        )
    return fp8, sf_rm.view(num_experts, -1).view(torch.float8_e8m0fnu)


class FlashInferCuteDSLBatchedExpertsMxfp8(mk.FusedMoEExpertsModular):
    """Batched FlashInfer CuteDSL MXFP8 fused MoE expert implementation."""

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ):
        super().__init__(
            moe_config=moe_config,
            quant_config=quant_config,
            max_num_tokens=max_num_tokens,
            num_dispatchers=num_dispatchers,
        )
        assert quant_config.quant_dtype == "mxfp8"
        assert quant_config.weight_quant_dtype == "mxfp8"
        assert quant_config.block_shape == [1, MXFP8_BLOCK_SIZE]
        self.out_dtype = moe_config.in_dtype

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.BatchedExperts

    @staticmethod
    def _supports_current_device() -> bool:
        return (
            current_platform.is_cuda()
            and current_platform.is_device_capability_family(100)
            and has_flashinfer_cutedsl_grouped_gemm_nt_masked()
        )

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) == (kMxfp8Static, kMxfp8Dynamic)

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation == MoEActivation.SILU

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return moe_parallel_config.use_deepep_ll_kernels

    def supports_expert_map(self) -> bool:
        return False

    def supports_native_mxfp8_act_scales(self) -> bool:
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceDelegate(sync_before_low_latency_combine=True)

    def workspace_dtype(self, act_dtype: torch.dtype) -> torch.dtype:
        return self.out_dtype if self.out_dtype is not None else act_dtype

    def moe_problem_size(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        assert a1.dim() == 3
        assert w1.dim() == 3 and w2.dim() == 3
        E = w1.size(0)
        assert a1.size(0) == E
        M = a1.size(1)
        N = w1.size(2)
        K = a1.size(-1)
        topk = topk_ids.size(1)
        return E, M, N, K, topk

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        num_dp = self.num_dispatchers
        assert num_dp is not None
        experts_per_worker = self.moe_config.num_local_experts
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        workspace13 = (experts_per_worker, M * num_dp, max(N, K))
        workspace2 = (
            experts_per_worker,
            M * num_dp,
            max(activation_out_dim, K),
        )
        output = (experts_per_worker, M, K)
        return (workspace13, workspace2, output)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        assert a2_scale is None
        assert expert_tokens_meta is not None
        assert activation == MoEActivation.SILU, (
            "Only SILU activation is currently supported"
        )
        assert self.w1_scale is not None and self.w2_scale is not None

        # Inputs:
        #   hidden_states: (E, M_max, K) fp8 if a1q_scale else bf16
        #   a1q_scale: (E, scale_stride, K // MXFP8_BLOCK_SIZE) uint8 row-major
        #     (from DeepEP-LL native MXFP8 dispatch)
        #   w1: (E, 2N, K) fp8, w2: (E, K, N) fp8
        #   self.w1_scale/w2_scale: pre-swizzled flat E8M0 (see oracle.fp8
        #     ``convert_to_fp8_moe_kernel_format``).
        assert hidden_states.dim() == 3
        E, M_max, K = hidden_states.shape
        N2 = w1.size(1)
        N = N2 // 2
        scale_stride = _align128(M_max)

        if a1q_scale is not None:
            assert hidden_states.dtype == torch.float8_e4m3fn
            assert a1q_scale.dim() == 3
            # The cute masked GEMM may read FP8 payload rows beyond masked_m
            # while forming tiles. Keep those rows finite without allocating a
            # full torch.where copy of the dispatch buffer.
            with mxfp8_moe_nvtx_range("mxfp8_moe:a1_zero_past_fp8"):
                _zero_invalid_rows(
                    hidden_states, expert_tokens_meta.expert_num_tokens
                )
            a1_fp8 = hidden_states
            # Sanitize beyond-mask scale rows from the DeepEP recv buffer. The
            # cute masked GEMM can see stale TMA-pushed scale data past
            # ``masked_m``; CUTLASS avoids this through per-expert problem
            # sizes that bound both reads and writes.
            with mxfp8_moe_nvtx_range("mxfp8_moe:a1_zero_past_sf"):
                _zero_invalid_scale_rows(
                    a1q_scale, expert_tokens_meta.expert_num_tokens
                )
            with mxfp8_moe_nvtx_range("mxfp8_moe:a1_swizzle_sf"):
                a1_sf_cute = swizzle_mxfp8_scales_batched_for_cute(
                    a1q_scale, E, scale_stride, K
                ).view(torch.float8_e8m0fnu)
        else:
            assert hidden_states.dtype in (torch.float16, torch.bfloat16)
            with mxfp8_moe_nvtx_range("mxfp8_moe:a1_quantize_for_cute"):
                a1_fp8, a1_sf_cute = _quantize_act_for_cute(
                    hidden_states.view(-1, K),
                    expert_tokens_meta.expert_num_tokens,
                    E,
                    M_max,
                    K,
                )
            a1_fp8 = a1_fp8.view(E, M_max, K)

        with mxfp8_moe_nvtx_range("mxfp8_moe:workspace_resize"):
            gemm1_out = _resize_cache(workspace13, (E, M_max, N2))
        with mxfp8_moe_nvtx_range("mxfp8_moe:gemm1"):
            flashinfer_cutedsl_grouped_gemm_nt_masked(
                (a1_fp8.permute(1, 2, 0), a1_sf_cute),
                (w1.permute(1, 2, 0), self.w1_scale.view(torch.float8_e8m0fnu)),
                gemm1_out.permute(1, 2, 0),
                expert_tokens_meta.expert_num_tokens,
                ab_dtype="float8_e4m3fn",
                sf_dtype="float8_e8m0fnu",
                c_dtype="bfloat16",
                sf_vec_size=MXFP8_BLOCK_SIZE,
            )
        # The activation kernel reads the full workspace row range. Keep only
        # invalid GEMM1 rows finite instead of pre-zeroing the whole workspace.
        with mxfp8_moe_nvtx_range("mxfp8_moe:gemm1_zero_past_rows"):
            _zero_invalid_rows(gemm1_out, expert_tokens_meta.expert_num_tokens)

        act_out = _resize_cache(workspace2, (E * M_max, N))
        with mxfp8_moe_nvtx_range("mxfp8_moe:activation"):
            apply_moe_activation(activation, act_out, gemm1_out.view(E * M_max, N2))

        with mxfp8_moe_nvtx_range("mxfp8_moe:a2_quantize_for_cute"):
            a2_fp8, a2_sf_cute = _quantize_act_for_cute(
                act_out,
                expert_tokens_meta.expert_num_tokens,
                E,
                M_max,
                N,
            )
        a2_fp8 = a2_fp8.view(E, M_max, N)

        with mxfp8_moe_nvtx_range("mxfp8_moe:gemm2"):
            flashinfer_cutedsl_grouped_gemm_nt_masked(
                (a2_fp8.permute(1, 2, 0), a2_sf_cute),
                (w2.permute(1, 2, 0), self.w2_scale.view(torch.float8_e8m0fnu)),
                output.permute(1, 2, 0),
                expert_tokens_meta.expert_num_tokens,
                ab_dtype="float8_e4m3fn",
                sf_dtype="float8_e8m0fnu",
                c_dtype="bfloat16",
                sf_vec_size=MXFP8_BLOCK_SIZE,
            )
