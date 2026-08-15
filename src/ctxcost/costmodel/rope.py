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


def rope_spectrum(spec: ArchSpec, trained_ctx_len: int | None = None) -> list[RopeSpectrumPoint]:
    """Per-dimension RoPE theta, wavelength, and periods completed within training.

    `trained_ctx_len` defaults to `spec.native_max_position_embeddings` — the length
    the model actually saw phase information for, before any context-extension scaling.
    """
    ctx_len = trained_ctx_len if trained_ctx_len is not None else spec.native_max_position_embeddings
    n_pairs = spec.head_dim // 2

    points = []
    for i in range(n_pairs):
        theta_i = spec.rope_theta ** (-2 * i / spec.head_dim)
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

    Returns the matplotlib Axes. Pass `save_path` to also write the figure to disk.
    """
    import matplotlib.pyplot as plt

    points = rope_spectrum(spec, trained_ctx_len)
    dims = [p.dim for p in points]
    periods = [p.periods_completed for p in points]

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(dims, periods, marker=".", linewidth=1)
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
