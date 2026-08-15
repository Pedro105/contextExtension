"""max_concurrency: KV-limited vs scheduler-limited, and which one binds."""

import pytest

from ctxcost.costmodel import ArchSpec, max_concurrency

SPEC = ArchSpec(
    hf_model_id="synthetic/test",
    num_hidden_layers=4,
    num_attention_heads=8,
    num_key_value_heads=2,
    head_dim=64,
    hidden_size=512,
    max_position_embeddings=8192,
    rope_theta=10000.0,
    torch_dtype="bfloat16",
)


def test_scheduler_binds_when_memory_is_abundant():
    result = max_concurrency(SPEC, gpu_vram_gib=80, util=0.9, ctx_len=1024, max_num_seqs=8)
    assert result.binding == "scheduler"
    assert result.scheduler_limited == 8
    assert result.kv_limited > result.scheduler_limited
    assert result.effective == 8


def test_kv_binds_when_memory_is_scarce():
    result = max_concurrency(SPEC, gpu_vram_gib=1, util=0.9, ctx_len=8192, max_num_seqs=256)
    assert result.binding == "kv"
    assert result.kv_limited < result.scheduler_limited
    assert result.effective == result.kv_limited


def test_effective_is_always_the_min():
    for vram, max_seqs in [(1, 4), (4, 4), (80, 4), (80, 1000)]:
        result = max_concurrency(SPEC, gpu_vram_gib=vram, util=0.9, ctx_len=4096, max_num_seqs=max_seqs)
        assert result.effective == min(result.kv_limited, result.scheduler_limited)


def test_longer_context_reduces_kv_limited_concurrency():
    short = max_concurrency(SPEC, gpu_vram_gib=40, util=0.9, ctx_len=4096, max_num_seqs=10_000)
    long = max_concurrency(SPEC, gpu_vram_gib=40, util=0.9, ctx_len=32768, max_num_seqs=10_000)
    assert long.kv_limited < short.kv_limited


def test_invalid_util_rejected():
    with pytest.raises(ValueError):
        max_concurrency(SPEC, gpu_vram_gib=40, util=1.5, ctx_len=4096, max_num_seqs=10)
    with pytest.raises(ValueError):
        max_concurrency(SPEC, gpu_vram_gib=40, util=0, ctx_len=4096, max_num_seqs=10)
