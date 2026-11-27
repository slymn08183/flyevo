"""What are the brain-region surfaces, and do they line up with the neuron positions? (CPU)

    uv run python scripts/explore/05_brain_regions.py

The zip holds one closed surface (.ply mesh = vertices + triangles) per brain region
("neuropil"), made for a template brain and transformed into FlyWire space by the fafbseg
authors. If they are in the same coordinate system as the neurons, most Kenyon-cell anchors
must fall inside / next to the mushroom-body surfaces - that is the check at the bottom.
"""

import io
import zipfile
from pathlib import Path

import numpy as np
import polars as pl
import trimesh

DATA = Path(__file__).resolve().parents[2] / "data" / "flywire"
VOXEL_NM = np.array([4.0, 4.0, 40.0])

meshes = {}
with zipfile.ZipFile(DATA / "neuropil_meshes_flywire.zip") as z:
    for name in z.namelist():
        if name.endswith(".ply") and not name.startswith("__MACOSX"):
            meshes[Path(name).stem] = trimesh.load(io.BytesIO(z.read(name)), file_type="ply")

print(f"{len(meshes)} regions, e.g. {sorted(meshes)[:8]} ...")
allv = np.vstack([m.vertices for m in meshes.values()])
print("extent of all surfaces :", np.round(allv.max(0) - allv.min(0)).astype(int), "(raw units)")

ann = pl.read_csv(
    DATA / "neuron_annotations.tsv",
    separator="\t",
    columns=["pos_x", "pos_y", "pos_z", "cell_class"],
)
anchors_nm = ann.select("pos_x", "pos_y", "pos_z").to_numpy() * VOXEL_NM
print(
    "extent of neuron anchors:", np.round(anchors_nm.max(0) - anchors_nm.min(0)).astype(int), "nm"
)

is_kc = (ann["cell_class"] == "Kenyon_Cell").fill_null(False).to_numpy()  # unlabelled -> False
kc = anchors_nm[is_kc]
mb = np.vstack([m.vertices for n, m in meshes.items() if n.startswith("MB_")])
lo, hi = mb.min(0), mb.max(0)
inside = ((kc >= lo) & (kc <= hi)).all(axis=1).mean()
print(f"Kenyon-cell anchors inside the mushroom-body bounding box: {inside:.0%}")
print("mushroom body parts:", sorted(n for n in meshes if n.startswith("MB_")))
