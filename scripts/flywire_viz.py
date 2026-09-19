"""Turn one FlyWire experiment into pictures: ``viz.rrd`` (interactive 3D) and ``activity.gif``.

    uv run python scripts/flywire_viz.py runs/pretest/sugar_grn

Neuron positions come from the FlyWire annotation table (Schlegel et al. 2024,
doi:10.1038/s41586-024-07686-5, github.com/flyconnectome/flywire_annotations, CC BY-NC like the
rest of FlyWire → this stays in ``scripts/``). Runs on the CPU; no GPU needed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from omegaconf import OmegaConf

from flyevo.viz.figures import brain_activity_gif
from flyevo.viz.rerun_log import log_spiking_brain

REPO = Path(__file__).resolve().parents[1]
VOXEL_UM = np.array([4.0, 4.0, 40.0]) * 1e-3  # FlyWire voxel size → micrometres
SUPER_CLASS_RGBA = {
    "optic": (70, 110, 170, 110),
    "central": (150, 150, 165, 110),
    "sensory": (90, 170, 110, 130),
    "visual_projection": (110, 140, 200, 110),
    "ascending": (170, 120, 170, 130),
    "descending": (190, 130, 110, 130),
    "motor": (220, 90, 90, 160),
}
DEFAULT_RGBA = (130, 130, 140, 110)


def load_positions(ids: np.ndarray, annotations: Path) -> tuple[np.ndarray, np.ndarray]:
    """Positions (um, f32[N, 3], NaN if unknown) and colours (u8[N, 4]), ordered like ``ids``."""
    ann = pl.read_csv(
        annotations,
        separator="\t",
        columns=["root_id", "pos_x", "pos_y", "pos_z", "super_class"],
        schema_overrides={"root_id": pl.Int64},
    )
    table = pl.DataFrame({"root_id": ids}).join(ann, on="root_id", how="left")
    pos = table.select("pos_x", "pos_y", "pos_z").to_numpy().astype(np.float32) * VOXEL_UM
    colors = np.array(
        [SUPER_CLASS_RGBA.get(c, DEFAULT_RGBA) for c in table["super_class"].to_list()],
        dtype=np.uint8,
    )
    return pos.astype(np.float32), colors


def main(run_dir: str) -> None:
    run = REPO / run_dir
    cfg = OmegaConf.load(REPO / "configs" / "viz" / "brain.yaml")
    meta = json.loads((run / "meta.json").read_text())
    exp = meta["experiment"]
    data = np.load(run / "binned_trial0.npz")
    binned, ids = data["binned"], data["ids"]
    index_of = {int(r): i for i, r in enumerate(ids)}

    positions, colors = load_positions(ids, REPO / "data" / "flywire" / "neuron_annotations.tsv")
    print(f"{int(np.isfinite(positions).all(axis=1).sum())}/{len(ids)} neurons have a position")

    groups = {
        "stimulated": np.array([index_of[i] for i in exp["stimulated_ids"] if i in index_of]),
        **{
            label: np.array([index_of[int(i)]])
            for label, i in exp.get("readout_ids", {}).items()
            if int(i) in index_of
        },
    }
    rrd = log_spiking_brain(
        run / "viz.rrd",
        positions,
        binned,
        exp["bin_ms"],
        app_id=f"flyevo-{exp['name']}",
        background_colors=colors,
        groups=groups,
        point_radius=cfg.rerun.point_radius,
        active_radius=cfg.rerun.active_radius,
        rate_vmax_hz=cfg.rate_vmax_hz,
        cmap=cfg.cmap,
        rate_smooth_ms=cfg.rerun.rate_smooth_ms,
        view_axis=tuple(cfg.rerun.view_axis),
        view_up=tuple(cfg.rerun.view_up),
    )
    gif = brain_activity_gif(
        run / "activity.gif",
        positions,
        binned,
        exp["bin_ms"],
        title=f"{exp['name']}: {len(groups['stimulated'])} neurons driven at {exp['rate_hz']} Hz",
        fps=cfg.gif.fps,
        decay=cfg.gif.decay,
        rate_vmax_hz=cfg.rate_vmax_hz,
        cmap=cfg.cmap,
        density_bins=cfg.gif.density_bins,
        dpi=cfg.gif.dpi,
    )
    print(f"wrote {rrd.relative_to(REPO)} and {gif.relative_to(REPO)}")


if __name__ == "__main__":
    main(sys.argv[1])
