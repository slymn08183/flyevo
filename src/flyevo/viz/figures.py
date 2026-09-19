"""Static figures and GIFs (matplotlib). The interactive 3D view lives in ``rerun_log.py``."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.artist import Artist


def brain_activity_gif(
    out_path: Path,
    positions: np.ndarray,
    binned: np.ndarray,
    bin_ms: float,
    *,
    title: str,
    fps: int,
    decay: float,
    rate_vmax_hz: float,
    cmap: str,
    density_bins: int,
    dpi: int,
) -> Path:
    """Frontal (x-y) view of one trial: brain outline as a density map, spikes glowing on top.

    positions: f32[N, 3], NaN rows hidden.  binned: int[T, N] spikes per bin.
    ``decay`` in [0, 1) leaves a fading trail so single 10 ms frames are readable.
    """
    n_bins, n = binned.shape
    visible = np.isfinite(positions).all(axis=1)
    x, y = positions[:, 0], positions[:, 1]
    bin_s = bin_ms * 1e-3

    fig, (ax, ax_rate) = plt.subplots(
        2, 1, figsize=(8, 6), dpi=dpi, height_ratios=[5, 1], facecolor="#08080c"
    )
    density, xe, ye = np.histogram2d(x[visible], y[visible], bins=density_bins)
    ax.imshow(
        np.log1p(density.T),
        extent=(xe[0], xe[-1], ye[-1], ye[0]),  # y grows downwards = dorsal up in FlyWire space
        cmap="bone",
        alpha=0.55,
        interpolation="bilinear",
        aspect="equal",
    )
    ax.set_axis_off()
    ax.set_title(title, color="white", fontsize=11, loc="left")
    dots = ax.scatter([], [], s=[], c=[], cmap=cmap, vmin=0, vmax=rate_vmax_hz, linewidths=0)
    clock = ax.text(0.99, 0.02, "", color="white", ha="right", transform=ax.transAxes)

    population = binned.sum(axis=1) / bin_s / n
    ax_rate.set_facecolor("#08080c")
    ax_rate.plot(np.arange(n_bins) * bin_ms, population, color="#f0a030", lw=1.2)
    ax_rate.set_xlim(0, n_bins * bin_ms)
    ax_rate.tick_params(colors="white", labelsize=7)
    ax_rate.set_ylabel("mean Hz", color="white", fontsize=7)
    ax_rate.set_xlabel("ms", color="white", fontsize=7)
    for side in ax_rate.spines.values():
        side.set_color("#444")
    cursor = ax_rate.axvline(0, color="white", lw=0.8)

    glow = np.zeros(n, dtype=np.float32)  # Hz, exponentially decaying trace

    def draw(t: int) -> tuple[Artist, ...]:
        nonlocal glow
        glow = np.maximum(glow * decay, binned[t] / bin_s)
        idx = np.flatnonzero((glow > 1.0) & visible)
        idx = idx[np.argsort(glow[idx])]  # brightest on top
        dots.set_offsets(np.column_stack([x[idx], y[idx]]))
        dots.set_array(glow[idx])
        dots.set_sizes(4 + 26 * np.clip(glow[idx] / rate_vmax_hz, 0, 1))
        clock.set_text(f"{t * bin_ms:.0f} ms")
        cursor.set_xdata([t * bin_ms, t * bin_ms])
        return dots, clock, cursor

    fig.subplots_adjust(left=0.07, right=0.99, top=0.95, bottom=0.09, hspace=0.12)
    FuncAnimation(fig, draw, frames=n_bins, blit=False).save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out_path
