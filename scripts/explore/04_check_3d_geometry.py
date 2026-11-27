"""Is the 3D picture geometrically right? Print the size of the point cloud in micrometres. (CPU)

    docker compose -f docker/compose.yaml run --rm --entrypoint "uv run --no-sync python" \
        flyevo scripts/explore/04_check_3d_geometry.py

The annotation table gives two positions per neuron, both in voxels of 4 x 4 x 40 nm:
  pos_*   an "anchor" point, typically on the neuron's backbone  <- what viz.rrd draws
  soma_*  the cell body, missing for ~15 % of neurons
A neuron is a branching tree that can span the whole brain; we draw ONE dot for it.
"""

from pathlib import Path

import numpy as np
import polars as pl

DATA = Path(__file__).resolve().parents[2] / "data" / "flywire"
VOXEL_UM = np.array([4.0, 4.0, 40.0]) * 1e-3

ann = pl.read_csv(
    DATA / "neuron_annotations.tsv",
    separator="\t",
    columns=["pos_x", "pos_y", "pos_z", "soma_x", "soma_y", "soma_z", "super_class"],
)
anchor = ann.select("pos_x", "pos_y", "pos_z").to_numpy() * VOXEL_UM
soma = ann.select("soma_x", "soma_y", "soma_z").to_numpy().astype(float) * VOXEL_UM
has_soma = np.isfinite(soma).all(axis=1)

for name, p in (("anchor (drawn)", anchor), ("soma", soma[has_soma])):
    size = p.max(axis=0) - p.min(axis=0)
    print(f"{name:15s} width {size[0]:4.0f} um  height {size[1]:4.0f} um  depth {size[2]:4.0f} um")
print(f"neurons with a known soma: {has_soma.mean():.0%}")
gap = np.linalg.norm(anchor[has_soma] - soma[has_soma], axis=1)
print(f"distance anchor <-> soma of the same neuron: median {np.median(gap):.0f} um, "
      f"90th percentile {np.percentile(gap, 90):.0f} um")  # fmt: skip
print("without the 10x z-scaling the depth would read", f"{(anchor[:, 2] / 10).max():.0f} um")

print("\nwhere the dots of each region sit, left-right (x, um):")
for r in ann.group_by("super_class").len().sort("len", descending=True).head(4).to_dicts():
    x = anchor[(ann["super_class"] == r["super_class"]).to_numpy(), 0]
    print(f"  {r['super_class']:18s} {np.percentile(x, 5):4.0f} .. {np.percentile(x, 95):4.0f}")
