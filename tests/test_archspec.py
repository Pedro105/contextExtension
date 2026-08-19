"""fetch_arch parsing, exercised against cached fixture configs (no network)."""

import pytest

from ctxcost.costmodel import ArchSpec, fetch_arch


def test_fetch_arch_hits_cache_not_network(fixtures_dir, monkeypatch):
    def _fail(*a, **k):
        raise AssertionError("fetch_arch should not hit the network when cached")

    monkeypatch.setattr("ctxcost.costmodel.archspec.hf_hub_download", _fail)
    fetch_arch("HuggingFaceTB/SmolLM2-1.7B", cache_dir=fixtures_dir)


def test_smollm2_derives_head_dim_and_defaults_kv_heads(fixtures_dir):
    spec = fetch_arch("HuggingFaceTB/SmolLM2-1.7B", cache_dir=fixtures_dir)
    assert spec.num_hidden_layers == 24
    assert spec.num_attention_heads == 32
    # config.json has no num_key_value_heads -> MHA, defaults to num_attention_heads
    assert spec.num_key_value_heads == 32
    # config.json has no head_dim -> derived as hidden_size / num_attention_heads
    assert spec.head_dim == 2048 // 32
    assert spec.max_position_embeddings == 8192
    assert spec.native_max_position_embeddings == 8192
    assert spec.rope_theta == 130000.0
    assert spec.rope_scaling is None
    assert spec.sliding_window is None
    assert spec.sliding_window_enabled is False
    assert spec.torch_dtype == "bfloat16"


def test_qwen_declares_sliding_window_but_disabled(fixtures_dir):
    spec = fetch_arch("Qwen/Qwen2.5-1.5B", cache_dir=fixtures_dir)
    assert spec.num_key_value_heads == 2  # GQA
    assert spec.head_dim == 1536 // 12
    # sliding_window is a real value in the config, but use_sliding_window is False,
    # so it must be recorded as present-but-inactive rather than dropped or treated
    # as active.
    assert spec.sliding_window == 131072
    assert spec.sliding_window_enabled is False
    assert spec.is_layer_sliding(0) is False


def test_ministral_unwraps_text_config_and_yarn_scaling(fixtures_dir):
    spec = fetch_arch("mistralai/Ministral-3-3B-Base-2512", cache_dir=fixtures_dir)
    # fields must come from the nested text_config, not the multimodal wrapper top level
    assert spec.num_hidden_layers == 26
    assert spec.num_attention_heads == 32
    assert spec.num_key_value_heads == 8  # GQA
    assert spec.head_dim == 128
    # dtype lives at the top level ("dtype"), not inside text_config
    assert spec.torch_dtype == "bfloat16"
    # rope_theta/scaling live under the unified "rope_parameters" dict, not top-level
    # rope_theta/rope_scaling keys
    assert spec.rope_theta == 1000000.0
    assert spec.rope_scaling is not None
    assert spec.rope_scaling["rope_type"] == "yarn"
    # max_position_embeddings is the YaRN-extended figure; the native trained length
    # is the smaller original_max_position_embeddings inside rope_scaling
    assert spec.max_position_embeddings == 262144
    assert spec.native_max_position_embeddings == 16384
    # explicitly no sliding-window attention
    assert spec.sliding_window is None
    assert spec.sliding_window_enabled is False
    assert spec.is_layer_sliding(0) is False


def test_gemma3_interleaved_pattern_and_local_theta(fixtures_dir):
    spec = fetch_arch("google/gemma-3-1b-pt", cache_dir=fixtures_dir)
    assert spec.num_hidden_layers == 26
    assert spec.num_key_value_heads == 1  # MQA
    assert spec.head_dim == 256
    # expressed as sliding_window_pattern, not an explicit layer_types list
    assert spec.layer_types is None
    assert spec.sliding_window_pattern == 6
    assert spec.sliding_window == 512
    assert spec.sliding_window_enabled is True
    # global theta vs. local theta must be captured separately
    assert spec.rope_theta == 1000000.0
    assert spec.rope_theta_local == 10000.0
    assert spec.rope_scaling is None
    assert spec.native_max_position_embeddings == 32768

    # pattern=6 -> 5 sliding layers then 1 global layer, repeating: layer indices
    # 0-4 sliding, 5 global, 6-10 sliding, 11 global, ...
    expected_sliding = [i % 6 != 5 for i in range(spec.num_hidden_layers)]
    actual_sliding = [spec.is_layer_sliding(i) for i in range(spec.num_hidden_layers)]
    assert actual_sliding == expected_sliding

    # rope_theta_for_layer must follow the same split
    assert spec.rope_theta_for_layer(0) == 10000.0  # sliding -> local theta
    assert spec.rope_theta_for_layer(5) == 1000000.0  # global -> global theta


def test_yaml_roundtrip(fixtures_dir, tmp_path):
    spec = fetch_arch("mistralai/Ministral-3-3B-Base-2512", cache_dir=fixtures_dir)
    out = tmp_path / "spec.yaml"
    spec.save_yaml(out)
    loaded = ArchSpec.load_yaml(out)
    assert loaded == spec


def test_missing_head_dim_and_hidden_size_raises():
    from ctxcost.costmodel.archspec import parse_config

    raw = {
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "max_position_embeddings": 2048,
    }
    with pytest.raises(ValueError):
        parse_config(raw, "fake/model")
