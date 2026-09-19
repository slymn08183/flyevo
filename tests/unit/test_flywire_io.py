"""The FlyWire loader lives in scripts/ (CC BY-NC data); tested here on a synthetic file pair."""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from flywire_io import load_flywire, sha256


@pytest.fixture
def files(tmp_path: Path) -> tuple[Path, Path]:
    ids = [720575940000000010, 720575940000000020, 720575940000000030]
    comp = tmp_path / "completeness.csv"
    pl.DataFrame({"": ids, "Completed": [True] * 3}).write_csv(comp)
    con = tmp_path / "connectivity.parquet"
    pl.DataFrame(
        {
            "Presynaptic_ID": [ids[2], ids[0]],
            "Postsynaptic_ID": [ids[0], ids[1]],
            "Presynaptic_Index": [2, 0],
            "Postsynaptic_Index": [0, 1],
            "Connectivity": [4, 7],
            "Excitatory": [-1, 1],
            "Excitatory x Connectivity": [-4, 7],
        }
    ).write_parquet(con)
    return con, comp


def test_index_is_row_of_completeness_and_weights_keep_sign(files: tuple[Path, Path]) -> None:
    brain = load_flywire(*files)
    assert brain.net.n_neurons == 3
    assert brain.index_of[720575940000000020] == 1
    assert np.asarray(brain.net.pre).tolist() == [0, 2]  # CSR order
    assert np.asarray(brain.net.weight).tolist() == [7.0, -4.0]


def test_missing_ids_are_skipped_not_fatal(
    files: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    brain = load_flywire(*files)
    assert brain.indices([720575940000000030, 1], "test") == [2]
    assert "1/2 test ids not in this FlyWire version" in capsys.readouterr().out


def test_sha256_is_stable(files: tuple[Path, Path]) -> None:
    assert sha256(files[1]) == sha256(files[1])
    assert len(sha256(files[0])) == 64
