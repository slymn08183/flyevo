"""Rerun logging: a brain in 3D with its activity on a time axis, saved as ``.rrd``.

Data-agnostic: takes positions, surfaces, skeletons and binned spike counts, and knows nothing
about where they came from. Open the result with ``rerun <file.rrd>``.

What ends up in the recording (entity paths):
    brain/regions/<name>   static  translucent surface of a brain region, with a label
    brain/guide/*          static  scale bar and direction arrows
    brain/all              static  one faint dot per neuron (its anchor point)
    brain/skeletons        static shape + per-bin colour: the real branching shape of the
                                   neurons that matter, lighting up when they fire
    brain/active           per bin one bright dot per firing neuron
    rate/<group>           per bin firing-rate traces (Hz)
Timeline: ``sim_time``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

import matplotlib
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

BACKGROUND_RGBA = (120, 130, 150, 110)
SILENT_SKELETON_RGBA = (150, 160, 190, 14)


class Region(NamedTuple):
    """Closed surface of one brain region."""

    vertices: np.ndarray  # f32[V, 3]
    faces: np.ndarray  # i32[F, 3]
    rgba: tuple[int, int, int, int]
    label: str | None  # shown once, at the centroid


class Guide(NamedTuple):
    """Orientation help: arrows from ``origin`` and a scale bar."""

    origin: np.ndarray  # f32[3]
    arrows: Mapping[str, np.ndarray]  # label -> vector f32[3]
    scale_bar: float  # length, same unit as positions
    scale_label: str


def _heat_colors(values: np.ndarray, vmax: float, cmap: str) -> np.ndarray:
    """values f32[K] → u8[K, 4] colours on ``cmap``, saturating at ``vmax``."""
    rgba = matplotlib.colormaps[cmap](np.clip(values / vmax, 0.0, 1.0) * 0.85 + 0.15)
    return (np.asarray(rgba) * 255).astype(np.uint8)


def swc_to_paths(node_id: np.ndarray, parent_id: np.ndarray, xyz: np.ndarray) -> list[np.ndarray]:
    """Cut a neuron tree (SWC table: every node points to its parent, root = -1) into polylines.

    Each leaf is walked towards the root until it meets a node that is already drawn, so every
    edge appears exactly once and branch points are shared. Returns a list of f32[K, 3], K >= 2.
    """
    row = {int(n): i for i, n in enumerate(node_id)}
    parent_row = np.array([row.get(int(p), -1) for p in parent_id])
    has_child = np.zeros(len(node_id), dtype=bool)
    has_child[parent_row[parent_row >= 0]] = True
    drawn = np.zeros(len(node_id), dtype=bool)
    paths = []
    for leaf in np.flatnonzero(~has_child):
        path, i = [leaf], leaf
        drawn[leaf] = True
        while parent_row[i] >= 0:
            i = parent_row[i]
            path.append(i)
            if drawn[i]:
                break
            drawn[i] = True
        if len(path) >= 2:
            paths.append(xyz[path].astype(np.float32))
    return paths


def thin_path(path: np.ndarray, keep_every: int) -> np.ndarray:
    """Keep every n-th point of a polyline, always including both ends."""
    if keep_every <= 1 or len(path) <= 2:
        return path
    idx = np.unique(np.r_[np.arange(0, len(path), keep_every), len(path) - 1])
    return path[idx]


def log_spiking_brain(
    out_path: Path,
    positions: np.ndarray,
    binned: np.ndarray,
    bin_ms: float,
    *,
    app_id: str,
    background_colors: np.ndarray | None = None,
    groups: Mapping[str, np.ndarray] | None = None,
    regions: Mapping[str, Region] | None = None,
    skeletons: Mapping[int, Sequence[np.ndarray]] | None = None,
    guide: Guide | None = None,
    point_radius: float,
    active_radius: float,
    skeleton_radius: float,
    skeleton_decay: float,
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
    regions: name → surface.  skeletons: neuron index → polylines (see ``swc_to_paths``);
    drawn faintly all the time and coloured by firing rate while the neuron spikes; the colour
    fades by ``skeleton_decay`` per bin so that a 10 ms spike stays visible for a few frames.
    view_axis / view_up: initial camera looks along ``view_axis`` at the centre of the cloud.
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

    for name, region in (regions or {}).items():
        rec.log(
            f"brain/regions/{name}",
            rr.Mesh3D(
                vertex_positions=region.vertices,
                triangle_indices=region.faces,
                albedo_factor=region.rgba,
            ),
            static=True,
        )
        if region.label:
            rec.log(
                f"brain/regions/{name}/label",
                rr.Points3D(
                    [region.vertices.mean(axis=0)],
                    radii=0.01,
                    colors=[(*region.rgba[:3], 255)],
                    labels=[region.label],
                ),
                static=True,
            )

    if guide is not None:
        labels = list(guide.arrows)
        rec.log(
            "brain/guide/directions",
            rr.Arrows3D(
                origins=[guide.origin] * len(labels),
                vectors=[guide.arrows[k] for k in labels],
                labels=labels,
                colors=(230, 230, 230, 255),
                radii=point_radius,
            ),
            static=True,
        )
        bar = np.array([guide.origin, guide.origin + np.array([guide.scale_bar, 0.0, 0.0])])
        bar[:, 1] += guide.scale_bar * 0.35
        rec.log(
            "brain/guide/scale",
            rr.LineStrips3D(
                [bar], radii=point_radius, colors=(255, 255, 255, 255), labels=[guide.scale_label]
            ),
            static=True,
        )

    colors = BACKGROUND_RGBA if background_colors is None else background_colors[visible]
    rec.log(
        "brain/all",
        rr.Points3D(positions[visible], radii=point_radius, colors=colors),
        static=True,
    )

    strips: list[np.ndarray] = []
    strip_owner: list[int] = []
    for neuron, paths in (skeletons or {}).items():
        strips.extend(paths)
        strip_owner.extend([neuron] * len(paths))
    owner = np.array(strip_owner, dtype=np.int64)
    if strips:
        # Static = shape only. Colours must NOT be static: static data overrides timed data in
        # Rerun, and the per-bin colour updates below would never show.
        rec.log("brain/skeletons", rr.LineStrips3D(strips, radii=skeleton_radius), static=True)
    glow = np.zeros(n, dtype=np.float32)  # Hz, fading trace of each neuron's firing

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
        if strips:
            glow = np.maximum(glow * skeleton_decay, binned[t] / bin_s)
            strip_rate = glow[owner]
            strip_colors = np.where(
                (strip_rate >= 1.0)[:, None],
                _heat_colors(strip_rate, rate_vmax_hz, cmap),
                np.array(SILENT_SKELETON_RGBA, dtype=np.uint8),
            )
            # partial update: geometry stays the static one, only the colours change over time
            rec.log("brain/skeletons", rr.LineStrips3D.from_fields(colors=strip_colors))
        for name, trace in traces.items():
            rec.log(f"rate/{name}", rr.Scalars(float(trace[t])))

    rec.flush()
    return out_path
