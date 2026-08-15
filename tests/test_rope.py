"""rope_spectrum: per-dimension frequency/wavelength/periods, and its plotting helper."""

import math
from itertools import pairwise

from ctxcost.costmodel import ArchSpec, plot_rope_spectrum, rope_spectrum

SPEC = ArchSpec(
    hf_model_id="synthetic/test",
    num_hidden_layers=4,
    num_attention_heads=8,
    num_key_value_heads=8,
    head_dim=64,
    hidden_size=512,
    max_position_embeddings=8192,
    rope_theta=10000.0,
    torch_dtype="bfloat16",
)


def test_spectrum_has_one_point_per_dimension_pair():
    points = rope_spectrum(SPEC)
    assert len(points) == SPEC.head_dim // 2


def test_dim_zero_is_highest_frequency_shortest_wavelength():
    points = rope_spectrum(SPEC)
    # theta_i = base^(-2i/d) is strictly decreasing in i, so wavelength (2*pi/theta_i)
    # is strictly increasing -> dim 0 rotates fastest and completes the most periods.
    for a, b in pairwise(points):
        assert a.theta > b.theta
        assert a.wavelength < b.wavelength
        assert a.periods_completed > b.periods_completed


def test_periods_completed_matches_definition():
    points = rope_spectrum(SPEC, trained_ctx_len=8192)
    for p in points:
        assert math.isclose(p.wavelength, 2 * math.pi / p.theta)
        assert math.isclose(p.periods_completed, 8192 / p.wavelength)


def test_defaults_to_native_max_position_embeddings():
    with_default = rope_spectrum(SPEC)
    explicit = rope_spectrum(SPEC, trained_ctx_len=SPEC.native_max_position_embeddings)
    assert with_default == explicit


def test_ministral_yarn_native_ctx_used_by_default(fixtures_dir):
    from ctxcost.costmodel import fetch_arch

    spec = fetch_arch("mistralai/Ministral-3-3B-Base-2512", cache_dir=fixtures_dir)
    default_points = rope_spectrum(spec)
    scaled_points = rope_spectrum(spec, trained_ctx_len=spec.max_position_embeddings)
    # using the true native (16384) vs the YaRN-extended figure (262144) as the
    # baseline must give very different periods_completed for the same dimension
    assert default_points[-1].periods_completed < scaled_points[-1].periods_completed


def test_plot_rope_spectrum_runs_and_writes_file(tmp_path):
    out = tmp_path / "spectrum.png"
    ax = plot_rope_spectrum(SPEC, save_path=out)
    assert ax is not None
    assert out.exists()
    assert out.stat().st_size > 0
