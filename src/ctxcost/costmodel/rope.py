"""RoPE frequency spectrum: how much of it has completed a full period by training end.

Each dimension pair of a RoPE embedding rotates at its own frequency. Low-index (high-
frequency) dimensions complete many full rotations over a trained context and so have
their phase fully observed during pretraining; high-index (low-frequency) dimensions
may complete less than one rotation, meaning large stretches of their phase space were
never seen in training. That's precisely the gap context-extension methods (position
interpolation, YaRN, NTK-aware scaling, ...) have to paper over: the fewer periods a
dimension completed natively, the more its extrapolated behavior beyond the trained
length is unconstrained by data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .archspec import ArchSpec


@dataclass(frozen=True)
class RopeSpectrumPoint:
    dim: int  # index of the dimension pair, in [0, head_dim / 2)
    theta: float  # angular frequency (radians/position) for this dimension pair
    wavelength: float  # positions needed for one full rotation: 2*pi / theta
    periods_completed: float  # trained_ctx_len / wavelength


def rope_spectrum(
    spec: ArchSpec, trained_ctx_len: int | None = None, layer_type: str = "global"
) -> list[RopeSpectrumPoint]:
    """Per-dimension RoPE theta, wavelength, and periods completed within training.

    `trained_ctx_len` defaults to `spec.native_max_position_embeddings` — the length
    the model actually saw phase information for, before any context-extension scaling.

    Models with interleaved local/global attention (e.g. Gemma3) can use a different
    RoPE base frequency per layer type (`spec.rope_theta_local` vs `spec.rope_theta`).
    Reporting only one spectrum for such a model would silently hide the other, so
    `layer_type` ("global" or "local") selects which base frequency to use; "local" is
    only valid when `spec.rope_theta_local` is set.
    """
    if layer_type == "global":
        base_theta = spec.rope_theta
    elif layer_type == "local":
        if spec.rope_theta_local is None:
            raise ValueError(
                f"{spec.hf_model_id}: no separate local-layer rope_theta; "
                "use layer_type='global' (the only spectrum this model has)"
            )
        base_theta = spec.rope_theta_local
    else:
        raise ValueError(f"layer_type must be 'global' or 'local', got {layer_type!r}")

    ctx_len = trained_ctx_len if trained_ctx_len is not None else spec.native_max_position_embeddings
    n_pairs = spec.head_dim // 2

    points = []
    for i in range(n_pairs):
        theta_i = base_theta ** (-2 * i / spec.head_dim)
        wavelength_i = 2 * math.pi / theta_i
        points.append(
            RopeSpectrumPoint(
                dim=i,
                theta=theta_i,
                wavelength=wavelength_i,
                periods_completed=ctx_len / wavelength_i,
            )
        )
    return points


def plot_rope_spectrum(spec: ArchSpec, trained_ctx_len: int | None = None, save_path=None, ax=None):
    """Plot periods-completed vs. dimension index, marking the under-trained region.

    Models with a separate local-layer RoPE theta (`spec.rope_theta_local`, e.g. Gemma3)
    get both the global- and local-layer spectra overlaid, since plotting only one would
    misrepresent a model that actually has two.

    Returns the matplotlib Axes. Pass `save_path` to also write the figure to disk.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))

    layer_types = ["global", "local"] if spec.rope_theta_local is not None else ["global"]
    for lt in layer_types:
        points = rope_spectrum(spec, trained_ctx_len, layer_type=lt)
        dims = [p.dim for p in points]
        periods = [p.periods_completed for p in points]
        theta = spec.rope_theta if lt == "global" else spec.rope_theta_local
        ax.plot(dims, periods, marker=".", linewidth=1, label=f"{lt} (theta={theta:g})")

    ax.axhline(1.0, color="red", linestyle="--", linewidth=1, label="1 period completed")
    ax.set_yscale("log")
    ax.set_xlabel("RoPE dimension pair index")
    ax.set_ylabel("periods completed (log scale)")
    ctx_len = trained_ctx_len if trained_ctx_len is not None else spec.native_max_position_embeddings
    ax.set_title(f"{spec.hf_model_id}: RoPE spectrum at trained ctx_len={ctx_len}")
    ax.legend()

    if save_path is not None:
        ax.figure.savefig(save_path, bbox_inches="tight")
    return ax
