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


def test_interleaved_swa_stays_linear_not_constant():
    """Gemma3-style interleaved SWA (sliding_window_pattern) must keep growing linearly
    in ctx_len -- unlike uniform SWA, which caps to a constant beyond the window,
    because the periodic global layers keep scaling with ctx_len even though the local
    layers cap out."""
    spec = ArchSpec(
        **{**BASE_KWARGS, "num_hidden_layers": 30},
        sliding_window=1024,
        sliding_window_pattern=6,  # 5 local + 1 global, repeating -> 5 full, 25 sliding
    )
    below_window = kv_bytes_per_token(spec, 512)
    at_window = kv_bytes_per_token(spec, 1024)
    beyond_32k = kv_bytes_per_token(spec, 32768)
    beyond_64k = kv_bytes_per_token(spec, 65536)

    assert below_window < at_window < beyond_32k < beyond_64k  # never flatlines


def test_interleaved_vs_global_only_ratio_grows_toward_pattern_period():
    """At 32k the interleaved (pattern=6, window=1024) model should be ~5x cheaper per
    token than the same architecture with every layer global, and that ratio should
    grow toward 6x (the pattern period: 1 global layer per 6) as ctx_len grows toward
    128k, since the fixed per-layer window cost matters less as ctx_len dominates."""
    interleaved = ArchSpec(
        **{**BASE_KWARGS, "num_hidden_layers": 30},
        sliding_window=1024,
        sliding_window_pattern=6,
    )
    all_global = ArchSpec(**{**BASE_KWARGS, "num_hidden_layers": 30})  # no sliding window at all

    def ratio(ctx_len: int) -> float:
        return kv_bytes_per_token(all_global, ctx_len) / kv_bytes_per_token(interleaved, ctx_len)

    ratio_32k = ratio(32768)
    ratio_128k = ratio(131072)

    assert 4.5 <= ratio_32k <= 5.5
    assert 5.5 <= ratio_128k < 6.0
    assert ratio_32k < ratio_128k  # grows toward 6x as ctx_len grows
    assert ratio_128k < 6.0  # never reaches the layer-count-only asymptote


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
