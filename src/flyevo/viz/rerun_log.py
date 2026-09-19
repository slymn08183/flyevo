"""Rerun logging: neurons as a 3D point cloud, spikes on a time axis, saved as ``.rrd``.

Data-agnostic: takes positions and binned spike counts, knows nothing about where they came from.
Open the result with ``rerun <file.rrd>`` (the viewer ships with ``rerun-sdk``).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

BACKGROUND_RGBA = (120, 130, 150, 110)


def _heat_colors(values: np.ndarray, vmax: float, cmap: str) -> np.ndarray:
    """values f32[K] → u8[K, 4] colours on ``cmap``, saturating at ``vmax``."""
    rgba = matplotlib.colormaps[cmap](np.clip(values / vmax, 0.0, 1.0) * 0.85 + 0.15)
    return (np.asarray(rgba) * 255).astype(np.uint8)


def log_spiking_brain(
    out_path: Path,
    positions: np.ndarray,
    binned: np.ndarray,
    bin_ms: float,
    *,
    app_id: str,
    background_colors: np.ndarray | None = None,
    groups: Mapping[str, np.ndarray] | None = None,
    point_radius: float,
    active_radius: float,
    rate_vmax_hz: float,
    cmap: str,
    rate_smooth_ms: float,
    view_axis: tuple[float, float, float],
    view_up: tuple[float, float, float],
) -> Path:
    """Write a recording of one simulated trial.

    positions: f32[N, 3] (any unit; NaN rows are hidden).  binned: int[T, N] spikes per bin.
    background_colors: u8[N, 4] colour of each resting neuron (default: faint grey).
    groups: name → neuron indices; each gets a mean-rate trace next to the population rate,
    smoothed with a ``rate_smooth_ms`` moving average (single neurons are unreadable per bin).
    view_axis / view_up: initial camera looks along ``view_axis`` at the centre of the cloud.

    Entities: ``brain/all`` (static), ``brain/active`` (per bin, coloured by firing rate),
    ``rate/population`` and ``rate/<group>`` (Hz). Timeline: ``sim_time``.
    """
    n_bins, n = binned.shape
    if positions.shape != (n, 3):
        raise ValueError(f"positions {positions.shape} do not match binned {binned.shape}")
    visible = np.isfinite(positions).all(axis=1)
    bin_s = bin_ms * 1e-3
    centre = np.nanmean(positions[visible], axis=0)
    extent = float(np.ptp(positions[visible], axis=0).max())
    eye = centre - np.asarray(view_axis, dtype=np.float32) * extent * 1.1
    kernel = np.ones(max(1, round(rate_smooth_ms / bin_ms)))
    kernel /= kernel.size

    rec = rr.RecordingStream(app_id)
    rec.save(str(out_path))
    rec.send_blueprint(
        rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(
                    origin="brain",
                    name="brain",
                    background=[8, 8, 12],
                    line_grid=False,
                    eye_controls=rrb.EyeControls3D(
                        position=eye.tolist(), look_target=centre.tolist(), eye_up=list(view_up)
                    ),
                ),
                rrb.TimeSeriesView(origin="rate", name="firing rate (Hz)"),
                column_shares=[3, 1],
            ),
            collapse_panels=True,
        )
    )

    colors = BACKGROUND_RGBA if background_colors is None else background_colors[visible]
    rec.log(
        "brain/all",
        rr.Points3D(positions[visible], radii=point_radius, colors=colors),
        static=True,
    )

    traces = {
        name: np.convolve(binned[:, members].mean(axis=1) / bin_s, kernel, mode="same")
        for name, members in (groups or {}).items()
        if len(members)
    }
    traces["population"] = np.convolve(binned.sum(axis=1) / bin_s / n, kernel, mode="same")

    for t in range(n_bins):
        rec.set_time("sim_time", duration=t * bin_s)
        idx = np.flatnonzero((binned[t] > 0) & visible)
        rate = binned[t, idx] / bin_s
        rec.log(
            "brain/active",
            rr.Points3D(
                positions[idx],
                radii=active_radius * (0.6 + 0.4 * np.clip(rate / rate_vmax_hz, 0, 1)),
                colors=_heat_colors(rate, rate_vmax_hz, cmap),
            ),
        )
        for name, trace in traces.items():
            rec.log(f"rate/{name}", rr.Scalars(float(trace[t])))

    rec.flush()
    return out_path
