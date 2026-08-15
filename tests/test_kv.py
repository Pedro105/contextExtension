"""kv_bytes_per_token across the three attention regimes it must handle."""

from ctxcost.costmodel import ArchSpec, fetch_arch
from ctxcost.costmodel.kv import kv_bytes_per_token

BASE_KWARGS = {
    "hf_model_id": "synthetic/test",
    "num_hidden_layers": 4,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "hidden_size": 512,
    "max_position_embeddings": 8192,
    "rope_theta": 10000.0,
    "torch_dtype": "bfloat16",
}


def _per_token_per_layer(spec: ArchSpec) -> int:
    return 2 * spec.num_key_value_heads * spec.head_dim * 2  # bf16 = 2 bytes


def test_full_attention_is_linear_in_ctx_len(fixtures_dir):
    spec = fetch_arch("HuggingFaceTB/SmolLM2-1.7B", cache_dir=fixtures_dir)
    per_layer = spec.num_hidden_layers * spec.num_key_value_heads * spec.head_dim * 2 * 2
    assert kv_bytes_per_token(spec, 1000) == per_layer * 1000
    assert kv_bytes_per_token(spec, 2000) == per_layer * 2000  # exactly doubles


def test_disabled_sliding_window_behaves_as_full_attention(fixtures_dir):
    """Qwen2.5 declares sliding_window but has it disabled: must not cap the cache."""
    spec = fetch_arch("Qwen/Qwen2.5-1.5B", cache_dir=fixtures_dir)
    per_layer = spec.num_hidden_layers * spec.num_key_value_heads * spec.head_dim * 2 * 2
    ctx_len = 200_000  # well beyond the declared (but inactive) sliding_window
    assert kv_bytes_per_token(spec, ctx_len) == per_layer * ctx_len


def test_uniform_sliding_window_caps_beyond_window():
    spec = ArchSpec(**BASE_KWARGS, sliding_window=100, sliding_window_enabled=True)
    per_token_per_layer = _per_token_per_layer(spec)

    # below the window: still linear
    assert kv_bytes_per_token(spec, 50) == spec.num_hidden_layers * 50 * per_token_per_layer
    # beyond the window: capped at the window size, constant beyond that point
    at_window = kv_bytes_per_token(spec, 100)
    beyond = kv_bytes_per_token(spec, 5000)
    assert at_window == spec.num_hidden_layers * 100 * per_token_per_layer
    assert beyond == at_window


def test_interleaved_swa_mixes_full_and_capped_layers():
    layer_types = ("full_attention", "sliding_attention", "full_attention", "sliding_attention")
    spec = ArchSpec(
        **BASE_KWARGS,
        sliding_window=100,
        sliding_window_enabled=True,
        layer_types=layer_types,
    )
    per_token_per_layer = _per_token_per_layer(spec)
    ctx_len = 1000

    full_layers = sum(1 for lt in layer_types if lt == "full_attention")
    sliding_layers = sum(1 for lt in layer_types if lt == "sliding_attention")
    expected = full_layers * ctx_len * per_token_per_layer + sliding_layers * 100 * per_token_per_layer

    assert kv_bytes_per_token(spec, ctx_len) == expected
    # sanity: strictly between the all-full and all-sliding totals
    all_full = spec.num_hidden_layers * ctx_len * per_token_per_layer
    all_sliding = spec.num_hidden_layers * 100 * per_token_per_layer
    assert all_sliding < expected < all_full
