# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for MLA fused RoPE+FP8 quantization accuracy.

Reference: Commit f77a47c66 - fused RoPE/FP8 quantization optimization.
Validates that fused path matches unfused reference implementation.
"""

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.platforms import current_platform

if not current_platform.has_device_capability(100):
    pytest.skip(
        reason="FlashInfer MLA requires compute capability 10.0+ (Hopper)",
        allow_module_level=True,
    )
else:
    from flashinfer.rope import mla_rope_quantize_fp8

# DeepSeek MLA configuration
NUM_HEADS = 128
KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
V_HEAD_DIM = 128
MAX_POSITION = 8192
ROTARY_BASE = 10000


def create_mla_tensors(
    batch_size: int,
    dtype: torch.dtype,
    seed: int,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate Q, K inputs matching MLA shapes."""
    torch.manual_seed(seed)

    ql_nope = torch.randn(batch_size, NUM_HEADS, KV_LORA_RANK, dtype=dtype,
                           device=device)
    q_pe = torch.randn(batch_size, NUM_HEADS, QK_ROPE_HEAD_DIM, dtype=dtype,
                        device=device)
    k_nope = torch.randn(batch_size, KV_LORA_RANK, dtype=dtype, device=device)
    k_pe = torch.randn(batch_size, QK_ROPE_HEAD_DIM, dtype=dtype, device=device)
    positions = torch.randint(0, MAX_POSITION, (batch_size,), dtype=torch.long,
                              device=device)

    return ql_nope, q_pe, k_nope, k_pe, positions


def apply_unfused_rope(
    q_pe: torch.Tensor,
    k_pe: torch.Tensor,
    positions: torch.Tensor,
    rotary_emb: RotaryEmbedding,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE using standard RotaryEmbedding (unfused reference)."""
    k_pe_expanded = k_pe.unsqueeze(1).expand(-1, NUM_HEADS, -1)
    q_pe_rope, k_pe_rope = rotary_emb(positions, q_pe, k_pe_expanded)
    # contiguous() is required: the slice creates a non-contiguous view
    # (stride 8192 instead of 64) which ops.convert_fp8 doesn't handle.
    k_pe_rope = k_pe_rope[:, 0, :].contiguous()
    return q_pe_rope, k_pe_rope


def apply_unfused_rope_fp8(
    ql_nope: torch.Tensor,
    q_pe: torch.Tensor,
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    positions: torch.Tensor,
    rotary_emb: RotaryEmbedding,
    q_scale: float,
    k_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply RoPE then quantize to FP8 separately (unfused reference)."""
    q_pe_rope, k_pe_rope = apply_unfused_rope(q_pe, k_pe, positions, rotary_emb)

    q_concat = torch.cat([ql_nope, q_pe_rope], dim=-1)

    q_out = torch.empty_like(q_concat, dtype=torch.float8_e4m3fn)
    k_nope_out = torch.empty_like(k_nope, dtype=torch.float8_e4m3fn)
    k_pe_out = torch.empty_like(k_pe_rope, dtype=torch.float8_e4m3fn)

    ops.convert_fp8(q_out, q_concat, q_scale, kv_dtype="fp8_e4m3")
    ops.convert_fp8(k_nope_out, k_nope, k_scale, kv_dtype="fp8_e4m3")
    ops.convert_fp8(k_pe_out, k_pe_rope, k_scale, kv_dtype="fp8_e4m3")

    return q_out, k_nope_out, k_pe_out


def apply_fused_rope_fp8(
    ql_nope: torch.Tensor,
    q_pe: torch.Tensor,
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    q_scale: float,
    k_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply fused RoPE+FP8 using FlashInfer kernel (matches _fused_rope_quant)."""
    attn_dtype = torch.float8_e4m3fn

    # Separate output tensors (the fix from the bug)
    q_nope_fp8 = ql_nope.new_empty(ql_nope.shape, dtype=attn_dtype)
    q_pe_fp8 = q_pe.new_empty(q_pe.shape, dtype=attn_dtype)
    k_nope_out = k_nope.new_empty(k_nope.shape, dtype=attn_dtype)
    k_pe_out = k_pe.new_empty(k_pe.shape, dtype=attn_dtype)

    cos_sin_cache_f32 = cos_sin_cache.float()

    # FlashInfer uses multiply convention (FP8 = x * scale) while
    # vLLM uses divide convention (FP8 = x / scale), so pass reciprocal.
    mla_rope_quantize_fp8(
        q_rope=q_pe,
        k_rope=k_pe,
        q_nope=ql_nope,
        k_nope=k_nope,
        cos_sin_cache=cos_sin_cache_f32,
        pos_ids=positions,
        is_neox=False,
        quantize_dtype=attn_dtype,
        q_rope_out=q_pe_fp8,
        q_nope_out=q_nope_fp8,
        k_rope_out=k_pe_out,
        k_nope_out=k_nope_out,
        quant_scale_q=1.0 / q_scale,
        quant_scale_kv=1.0 / k_scale,
    )

    q_out = torch.cat([q_nope_fp8, q_pe_fp8], dim=-1)
    return q_out, k_nope_out, k_pe_out


@pytest.mark.parametrize("batch_size", [1, 4, 8, 16, 32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("seed", [0, 42])
def test_fused_rope_fp8_accuracy(
    default_vllm_config, batch_size: int, dtype: torch.dtype, seed: int
):
    """Test fused RoPE+FP8 matches unfused RoPE then FP8.

    Validates the full fused optimization path by calling the FlashInfer
    mla_rope_quantize_fp8 kernel directly and comparing against the
    unfused reference (RotaryEmbedding + convert_fp8).

    Reference: commit f77a47c66
    """
    device = "cuda"

    ql_nope, q_pe, k_nope, k_pe, positions = create_mla_tensors(
        batch_size, dtype, seed, device
    )

    rotary_emb = RotaryEmbedding(
        head_size=QK_ROPE_HEAD_DIM,
        rotary_dim=QK_ROPE_HEAD_DIM,
        max_position_embeddings=MAX_POSITION,
        base=ROTARY_BASE,
        is_neox_style=False,
        dtype=torch.float32,
    ).to(device=device)

    q_scale = 0.1
    k_scale = 0.1

    # Reference: unfused RoPE + FP8
    q_ref, k_nope_ref, k_pe_ref = apply_unfused_rope_fp8(
        ql_nope.clone(), q_pe.clone(), k_nope.clone(), k_pe.clone(),
        positions, rotary_emb, q_scale, k_scale,
    )

    # Test: fused RoPE+FP8
    q_test, k_nope_test, k_pe_test = apply_fused_rope_fp8(
        ql_nope.clone(), q_pe.clone(), k_nope.clone(), k_pe.clone(),
        positions, rotary_emb.cos_sin_cache, q_scale, k_scale,
    )

    # Nope parts should match exactly (no RoPE computation difference).
    # FP8 tensors only support bitwise comparison, so use atol=0, rtol=0.
    torch.testing.assert_close(
        q_test[:, :, :KV_LORA_RANK], q_ref[:, :, :KV_LORA_RANK],
        atol=0, rtol=0,
        msg=f"Fused Q nope mismatch: batch={batch_size}, dtype={dtype}, seed={seed}"
    )
    torch.testing.assert_close(
        k_nope_test, k_nope_ref, atol=0, rtol=0,
        msg=f"Fused k_nope mismatch: batch={batch_size}, dtype={dtype}, seed={seed}"
    )

    # Rope parts may differ by up to 1 FP8 ULP due to different intermediate
    # precision (fused kernel uses float32, unfused uses input dtype for RoPE).
    # Cast to float for tolerance-based comparison since FP8 assert_close
    # only supports exact bitwise comparison.
    torch.testing.assert_close(
        q_test[:, :, KV_LORA_RANK:].float(), q_ref[:, :, KV_LORA_RANK:].float(),
        atol=4.0, rtol=0.05,
        msg=f"Fused Q rope mismatch: batch={batch_size}, dtype={dtype}, seed={seed}"
    )
    torch.testing.assert_close(
        k_pe_test.float(), k_pe_ref.float(),
        atol=4.0, rtol=0.05,
        msg=f"Fused k_pe mismatch: batch={batch_size}, dtype={dtype}, seed={seed}"
    )
