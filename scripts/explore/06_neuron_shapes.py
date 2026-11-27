"""What does a real neuron look like in the data? (CPU, ~10 s)

    uv run python scripts/explore/06_neuron_shapes.py

The 5 GB parquet holds the "skeleton" of every neuron: the branching tree of the cell reduced to
points and parent links (the SWC format used across neuroscience). Columns are printed below.
A neuron = one tree: every node has x, y, z (nanometres) and points to its parent; root = -1.
"""

from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

DATA = Path(__file__).resolve().parents[2] / "data" / "flywire"
FILE = DATA / "sk_lod1_783_healed_ds2.parquet"
MN9 = 720575940660219265  # the feeding motor neuron from the sugar experiment

meta = pq.ParquetFile(FILE)
print(f"file: {FILE.stat().st_size / 1e9:.2f} GB, {meta.metadata.num_rows:,} skeleton nodes")
print("columns:", {f.name: str(f.type) for f in meta.schema_arrow})

id_column = "neuron" if "neuron" in meta.schema_arrow.names else meta.schema_arrow.names[-1]
one = pl.scan_parquet(FILE).filter(pl.col(id_column) == MN9).collect()
print(f"\nMN9 ({MN9}): {one.height} nodes")
print(one.head(5))

xyz = one.select("x", "y", "z").to_numpy() / 1000.0  # nm -> um
parent = dict(zip(one["node_id"].to_list(), one["parent_id"].to_list(), strict=True))
position = dict(zip(one["node_id"].to_list(), xyz, strict=True))
cable = sum(np.linalg.norm(position[n] - position[p]) for n, p in parent.items() if p in position)
children = set(parent.values())
print(f"spans {np.round(np.ptp(xyz, axis=0)).astype(int)} um in x, y, z")
print(
    f"total cable length: {cable:.0f} um  |  branch tips: {sum(n not in children for n in parent)}"
)
