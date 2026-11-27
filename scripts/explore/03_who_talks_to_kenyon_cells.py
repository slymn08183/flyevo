"""Step 3 - why does the memory centre saturate? Count who feeds the Kenyon cells. (CPU, seconds)

    docker compose -f docker/compose.yaml run --rm --entrypoint "uv run --no-sync python" \
        flyevo scripts/explore/03_who_talks_to_kenyon_cells.py

Also compares the Kenyon-cell sets saved by step 2, if you ran it for more than one odour.
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "flywire"

ann = pl.read_csv(
    DATA / "neuron_annotations.tsv",
    separator="\t",
    columns=["root_id", "cell_class", "cell_type"],
    schema_overrides={"root_id": pl.Int64},
)
con = pl.read_parquet(
    DATA / "Connectivity_783.parquet",
    columns=["Presynaptic_ID", "Postsynaptic_ID", "Excitatory x Connectivity"],
)
kc = ann.filter(pl.col("cell_class") == "Kenyon_Cell")["root_id"]

source = (
    pl.when(pl.col("cell_type") == "APL")
    .then(pl.lit("APL (the brake)"))
    .otherwise(pl.col("cell_class"))
    .alias("source")
)
into_kc = (
    con.filter(pl.col("Postsynaptic_ID").is_in(kc.implode()))
    .join(ann.rename({"root_id": "Presynaptic_ID"}), on="Presynaptic_ID", how="left")
    .with_columns(source)
    .group_by("source")
    .agg(pl.col("Excitatory x Connectivity").sum().alias("synapses"))
    .sort("synapses", descending=True)
)
print("input to an average Kenyon cell (signed synapses; + pushes it to fire, - holds it back):")
for r in into_kc.to_dicts():
    per_kc = r["synapses"] / kc.len()
    if abs(per_kc) >= 0.5:
        print(f"  {r['source'] or 'unlabelled':18s} {per_kc:+7.1f}")

saved = sorted((REPO / "runs").glob("explore_kc_*.npy"))
if len(saved) >= 2:
    print("\noverlap of the Kenyon-cell sets from step 2 (1.00 = exactly the same cells):")
    sets = {f.stem.removeprefix("explore_kc_"): np.load(f) for f in saved}
    for a, b in combinations(sets, 2):
        jaccard = (sets[a] & sets[b]).sum() / max(1, (sets[a] | sets[b]).sum())
        print(f"  {a} vs {b}: {jaccard:.3f}")
