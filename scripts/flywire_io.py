"""FlyWire connectome loader for the whole-brain LIF model.

FlyWire data is CC BY-NC 4.0 → this module lives in ``scripts/`` and must never be imported from
``src/flyevo/``. Files are the ones shipped with the reference model
(github.com/philshiu/Drosophila_brain_model): ``Connectivity_783.parquet`` and
``Completeness_783.csv``. Cite Dorkenwald et al. 2024 (doi:10.1038/s41586-024-07558-y) and
Shiu et al. 2024 (doi:10.1038/s41586-024-07763-9) for anything derived from them.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from flyevo.models.lif import Network, make_network


@dataclass(frozen=True)
class FlyWireBrain:
    net: Network
    ids: np.ndarray  # i64[N] FlyWire root id of neuron index i
    index_of: dict[int, int]  # root id → neuron index

    def indices(self, root_ids: list[int], what: str) -> list[int]:
        """Map root ids to indices; ids missing from this version are reported, not fatal."""
        missing = [r for r in root_ids if r not in self.index_of]
        if missing:
            print(f"warn: {len(missing)}/{len(root_ids)} {what} ids not in this FlyWire version")
        return [self.index_of[r] for r in root_ids if r in self.index_of]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_flywire(connectivity: Path, completeness: Path) -> FlyWireBrain:
    """Neuron index = row of the completeness file, as in the reference model."""
    ids = pl.read_csv(completeness).to_series(0).to_numpy()
    con = pl.read_parquet(
        connectivity,
        columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"],
    )
    net = make_network(
        con["Presynaptic_Index"].to_numpy(),
        con["Postsynaptic_Index"].to_numpy(),
        con["Excitatory x Connectivity"].to_numpy().astype(np.float32),
        n_neurons=len(ids),
    )
    return FlyWireBrain(net=net, ids=ids, index_of={int(r): i for i, r in enumerate(ids)})
