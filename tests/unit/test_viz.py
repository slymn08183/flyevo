"""Visualization writes real files from synthetic activity; looks are not asserted."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from flyevo.viz.figures import brain_activity_gif
from flyevo.viz.rerun_log import log_spiking_brain


@pytest.fixture
def activity() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    positions = rng.normal(0, 50, (300, 3)).astype(np.float32)
    positions[0] = np.nan  # neuron without a known position must not break anything
    binned = (rng.random((6, 300)) < 0.05).astype(np.int16)
    return positions, binned


def test_rerun_recording_is_written(
    activity: tuple[np.ndarray, np.ndarray], tmp_path: Path
) -> None:
    positions, binned = activity
    out = log_spiking_brain(
        tmp_path / "viz.rrd",
        positions,
        binned,
        10.0,
        app_id="flyevo-test",
        groups={"first_ten": np.arange(10), "empty": np.array([], dtype=int)},
        point_radius=1.0,
        active_radius=3.0,
        rate_vmax_hz=200.0,
        cmap="inferno",
        rate_smooth_ms=20.0,
        view_axis=(0.0, 0.0, 1.0),
        view_up=(0.0, -1.0, 0.0),
    )
    assert out.stat().st_size > 1000


def test_rerun_rejects_mismatched_shapes(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="do not match"):
        log_spiking_brain(
            tmp_path / "x.rrd",
            np.zeros((5, 3), dtype=np.float32),
            np.zeros((2, 7), dtype=np.int16),
            10.0,
            app_id="flyevo-test",
            point_radius=1.0,
            active_radius=3.0,
            rate_vmax_hz=200.0,
            cmap="inferno",
            rate_smooth_ms=20.0,
            view_axis=(0.0, 0.0, 1.0),
            view_up=(0.0, -1.0, 0.0),
        )


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
