"""Turn one FlyWire experiment into pictures: ``viz.rrd`` (interactive 3D) and ``activity.gif``.

    uv run python scripts/flywire_viz.py runs/pretest/sugar_grn [key=value ...]

Trailing ``key=value`` pairs override configs/viz/brain.yaml, e.g. ``regions.enabled=false``.

What is drawn, and where it comes from (all in FlyWire space, converted to micrometres here):
- one dot per neuron: "anchor" point from the annotation table (Schlegel et al. 2024,
  doi:10.1038/s41586-024-07686-5, github.com/flyconnectome/flywire_annotations)
- brain-region surfaces: ``neuropil_meshes_flywire.zip`` from github.com/flyconnectome/fafbseg-py
- real neuron shapes: ``sk_lod1_783_healed_ds2.parquet`` (5 GB, doi:10.5281/zenodo.10877326),
  only for the neurons that fire in the trial; skipped with a note if the file is missing
Runs on the CPU; no GPU needed.
"""

from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import polars as pl
import trimesh
from omegaconf import DictConfig, OmegaConf

from flyevo.viz.figures import brain_activity_gif
from flyevo.viz.rerun_log import Guide, Region, log_spiking_brain, swc_to_paths, thin_path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "flywire"
VOXEL_UM = np.array([4.0, 4.0, 40.0]) * 1e-3  # annotation table: voxels of 4 x 4 x 40 nm
NM_TO_UM = 1e-3  # meshes and skeletons are in nanometres
SUPER_CLASS_RGBA = {
    "optic": (70, 110, 170, 70),
    "central": (150, 150, 165, 70),
    "sensory": (90, 170, 110, 90),
    "visual_projection": (110, 140, 200, 70),
    "ascending": (170, 120, 170, 90),
    "descending": (190, 130, 110, 90),
    "motor": (220, 90, 90, 140),
}
DEFAULT_RGBA = (130, 130, 140, 70)


def load_annotations(ids: np.ndarray) -> pl.DataFrame:
    ann = pl.read_csv(
        DATA / "neuron_annotations.tsv",
        separator="\t",
        columns=["root_id", "pos_x", "pos_y", "pos_z", "super_class", "side"],
        schema_overrides={"root_id": pl.Int64},
    )
    return pl.DataFrame({"root_id": ids}).join(ann, on="root_id", how="left")


def load_regions(cfg: DictConfig) -> dict[str, Region]:
    """Every region surface, coloured by its group; a group's biggest mesh carries the label."""
    regions: dict[str, Region] = {}
    largest: dict[str, tuple[str, int]] = {}
    with zipfile.ZipFile(DATA / "neuropil_meshes_flywire.zip") as z:
        for member in z.namelist():
            if not member.endswith(".ply") or member.startswith("__MACOSX"):
                continue
            name = Path(member).stem
            mesh = trimesh.load(io.BytesIO(z.read(member)), file_type="ply")
            group = next(
                (g for g, spec in cfg.groups.items() if name.startswith(tuple(spec.prefixes))), None
            )
            rgb = cfg.groups[group].rgb if group else cfg.other_rgb
            regions[name] = Region(
                vertices=(np.asarray(mesh.vertices) * NM_TO_UM).astype(np.float32),
                faces=np.asarray(mesh.faces, dtype=np.int32),
                rgba=(int(rgb[0]), int(rgb[1]), int(rgb[2]), int(cfg.alpha)),
                label=None,
            )
            if group and len(mesh.vertices) > largest.get(group, ("", 0))[1]:
                largest[group] = (name, len(mesh.vertices))
    for group, (name, _) in largest.items():
        regions[name] = regions[name]._replace(label=cfg.groups[group].label)
    return regions


def make_guide(positions: np.ndarray, side: list[str | None], length: float) -> Guide:
    """Arrows from the front-bottom corner. Which way is the fly's left is read from the data."""
    ok = np.isfinite(positions).all(axis=1)
    x = positions[:, 0]
    left = np.array([s == "left" for s in side]) & ok
    right = np.array([s == "right" for s in side]) & ok
    left_sign = 1.0 if x[left].mean() > x[right].mean() else -1.0
    lo, hi = positions[ok].min(axis=0), positions[ok].max(axis=0)
    origin = np.array([lo[0] if left_sign > 0 else hi[0], hi[1], lo[2]], dtype=np.float32)
    return Guide(
        origin=origin,
        arrows={
            "fly's left": np.array([left_sign * length, 0, 0], dtype=np.float32),
            "up (top of the head)": np.array([0, -length, 0], dtype=np.float32),
            "back of the head": np.array([0, 0, length], dtype=np.float32),
        },
        scale_bar=length * left_sign,
        scale_label=f"{length:.0f} um (a human hair is ~70 um thick)",
    )


def load_skeletons(
    root_ids: np.ndarray, keep_every: int, min_twig_um: float
) -> dict[int, list[np.ndarray]]:
    """root id → polylines in micrometres, for the requested neurons only.

    Every polyline starts at a branch tip, so short ones are end twigs; those below
    ``min_twig_um`` are fuzz from turning surface meshes into skeletons and are dropped.
    """
    path = DATA / "sk_lod1_783_healed_ds2.parquet"
    try:
        pl.scan_parquet(path).collect_schema()
    except (OSError, pl.exceptions.ComputeError) as err:  # missing, or still downloading
        print(f"note: cannot read {path.name} ({type(err).__name__}) → dots only, no real shapes")
        return {}
    wanted = pl.Series("root_id", root_ids, dtype=pl.Int64)
    table = (
        pl.scan_parquet(path)
        .filter(pl.col("neuron").is_in(wanted.implode()))
        .select("neuron", "node_id", "parent_id", "x", "y", "z")
        .collect()
    )
    skeletons = {}
    for (root_id,), nodes in table.group_by("neuron"):
        xyz = nodes.select("x", "y", "z").to_numpy() * NM_TO_UM
        paths = swc_to_paths(nodes["node_id"].to_numpy(), nodes["parent_id"].to_numpy(), xyz)
        long_enough = [
            p for p in paths if np.linalg.norm(np.diff(p, axis=0), axis=1).sum() >= min_twig_um
        ]
        skeletons[int(root_id)] = [thin_path(p, keep_every) for p in long_enough]  # type: ignore[arg-type]
    return skeletons


def main(run_dir: str, overrides: list[str]) -> None:
    run = REPO / run_dir
    cfg = OmegaConf.merge(
        OmegaConf.load(REPO / "configs" / "viz" / "brain.yaml"), OmegaConf.from_dotlist(overrides)
    )
    exp = json.loads((run / "meta.json").read_text())["experiment"]
    data = np.load(run / "binned_trial0.npz")
    binned, ids = data["binned"], data["ids"]
    index_of = {int(r): i for i, r in enumerate(ids)}

    ann = load_annotations(ids)
    positions = (ann.select("pos_x", "pos_y", "pos_z").to_numpy() * VOXEL_UM).astype(np.float32)
    colors = np.array(
        [SUPER_CLASS_RGBA.get(c, DEFAULT_RGBA) for c in ann["super_class"].to_list()],
        dtype=np.uint8,
    )
    print(f"{int(np.isfinite(positions).all(axis=1).sum())}/{len(ids)} neurons have a position")

    rate_hz = binned.sum(axis=0) / (binned.shape[0] * exp["bin_ms"] * 1e-3)
    firing = ids[rate_hz >= cfg.rerun.skeleton_min_rate_hz]
    by_root = load_skeletons(firing, cfg.rerun.skeleton_keep_every, cfg.rerun.skeleton_min_twig_um)
    skeletons = {index_of[r]: paths for r, paths in by_root.items()}
    print(f"real shapes drawn for {len(skeletons)} of {len(firing)} firing neurons")

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
        regions=load_regions(cfg.regions) if cfg.regions.enabled else None,
        skeletons=skeletons,
        guide=make_guide(positions, ann["side"].to_list(), cfg.rerun.guide_um),
        point_radius=cfg.rerun.point_radius,
        active_radius=cfg.rerun.active_radius,
        skeleton_radius=cfg.rerun.skeleton_radius,
        skeleton_decay=cfg.rerun.skeleton_decay,
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
    main(sys.argv[1], sys.argv[2:])
