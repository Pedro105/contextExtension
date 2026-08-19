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


def test_gemma3_local_and_global_spectra_differ(fixtures_dir):
    from ctxcost.costmodel import fetch_arch

    spec = fetch_arch("google/gemma-3-1b-pt", cache_dir=fixtures_dir)
    global_points = rope_spectrum(spec, layer_type="global")
    local_points = rope_spectrum(spec, layer_type="local")
    # dim 0 is a fixed point (theta_i = base**0 = 1) for any base, so the two spectra
    # only diverge from dim 1 onward -- a higher base (global, 1e6) decays theta_i
    # faster than a lower base (local, 1e4), so global's theta is smaller past dim 0.
    assert math.isclose(global_points[0].theta, 1.0)
    assert math.isclose(local_points[0].theta, 1.0)
    assert global_points[1].theta < local_points[1].theta
    # different base frequencies -> different wavelengths at the same dimension
    assert global_points[-1].wavelength != local_points[-1].wavelength
    # default (no layer_type) is "global", not some average of the two
    assert rope_spectrum(spec) == global_points


def test_local_layer_type_requires_rope_theta_local():
    import pytest

    with pytest.raises(ValueError):
        rope_spectrum(SPEC, layer_type="local")  # SPEC has no rope_theta_local


def test_invalid_layer_type_rejected():
    import pytest

    with pytest.raises(ValueError):
        rope_spectrum(SPEC, layer_type="bogus")


def test_plot_rope_spectrum_runs_and_writes_file(tmp_path):
    out = tmp_path / "spectrum.png"
    ax = plot_rope_spectrum(SPEC, save_path=out)
    assert ax is not None
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_rope_spectrum_overlays_both_thetas_for_gemma3(fixtures_dir, tmp_path):
    from ctxcost.costmodel import fetch_arch

    spec = fetch_arch("google/gemma-3-1b-pt", cache_dir=fixtures_dir)
    out = tmp_path / "gemma3_spectrum.png"
    ax = plot_rope_spectrum(spec, save_path=out)
    # one line per layer type (global, local) plus the "1 period" reference line
    assert len(ax.get_lines()) == 3
    assert out.exists()
