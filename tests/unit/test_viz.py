"""Visualization writes real files from synthetic activity; looks are not asserted."""

from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from flyevo.viz.figures import brain_activity_gif
from flyevo.viz.rerun_log import Guide, Region, log_spiking_brain, swc_to_paths, thin_path

LOOK = {
    "point_radius": 1.0,
    "active_radius": 3.0,
    "skeleton_radius": 0.5,
    "skeleton_decay": 0.5,
    "rate_vmax_hz": 200.0,
    "cmap": "inferno",
    "rate_smooth_ms": 20.0,
    "view_axis": (0.0, 0.0, 1.0),
    "view_up": (0.0, -1.0, 0.0),
}


@pytest.fixture
def activity() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    positions = rng.normal(0, 50, (300, 3)).astype(np.float32)
    positions[0] = np.nan  # neuron without a known position must not break anything
    binned = (rng.random((6, 300)) < 0.05).astype(np.int16)
    return positions, binned


def test_rerun_recording_with_everything(
    activity: tuple[np.ndarray, np.ndarray], tmp_path: Path
) -> None:
    positions, binned = activity
    tetrahedron = Region(
        vertices=np.array([[0, 0, 0], [9, 0, 0], [0, 9, 0], [0, 0, 9]], dtype=np.float32),
        faces=np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32),
        rgba=(255, 0, 0, 40),
        label="a region",
    )
    guide = Guide(
        origin=np.zeros(3, dtype=np.float32),
        arrows={"up": np.array([0, -10, 0], dtype=np.float32)},
        scale_bar=10.0,
        scale_label="10 um",
    )
    skeletons = {5: [np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float32)]}
    plain = log_spiking_brain(
        tmp_path / "plain.rrd", positions, binned, 10.0, app_id="flyevo-test", **LOOK
    )
    full = log_spiking_brain(
        tmp_path / "full.rrd",
        positions,
        binned,
        10.0,
        app_id="flyevo-test",
        groups={"first_ten": np.arange(10), "empty": np.array([], dtype=int)},
        regions={"R": tetrahedron},
        skeletons=skeletons,
        guide=guide,
        **LOOK,
    )
    assert plain.stat().st_size > 1000
    assert full.stat().st_size > plain.stat().st_size


def test_rerun_rejects_mismatched_shapes(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="do not match"):
        log_spiking_brain(
            tmp_path / "x.rrd",
            np.zeros((5, 3), dtype=np.float32),
            np.zeros((2, 7), dtype=np.int16),
            10.0,
            app_id="flyevo-test",
            **LOOK,
        )


def test_swc_to_paths_draws_every_edge_once() -> None:
    #        0 (root)
    #        |
    #        1
    #       / \
    #      2   4
    #      |
    #      3
    node_id = np.array([10, 11, 12, 13, 14])
    parent_id = np.array([-1, 10, 11, 12, 11])
    xyz = np.arange(15, dtype=np.float32).reshape(5, 3)
    paths = swc_to_paths(node_id, parent_id, xyz)
    edges = {tuple(sorted((tuple(a), tuple(b)))) for p in paths for a, b in pairwise(p)}
    assert len(edges) == 4  # a tree with 5 nodes has 4 edges
    assert sum(len(p) - 1 for p in paths) == 4  # ... and none is drawn twice
    assert all(len(p) >= 2 for p in paths)


def test_thin_path_keeps_both_ends() -> None:
    path = np.arange(30, dtype=np.float32).reshape(10, 3)
    thinned = thin_path(path, 4)
    assert len(thinned) < len(path)
    assert (thinned[0] == path[0]).all() and (thinned[-1] == path[-1]).all()
    assert thin_path(path, 1) is path


def test_gif_has_one_frame_per_bin(activity: tuple[np.ndarray, np.ndarray], tmp_path: Path) -> None:
    positions, binned = activity
    out = brain_activity_gif(
        tmp_path / "a.gif",
        positions,
        binned,
        10.0,
        title="test",
        fps=10,
        decay=0.5,
        rate_vmax_hz=200.0,
        cmap="inferno",
        density_bins=20,
        dpi=40,
    )
    assert Image.open(out).n_frames == binned.shape[0]
