"""Step 1 - what is actually in the files we downloaded? (CPU, a few seconds)

    docker compose -f docker/compose.yaml run --rm --entrypoint "uv run --no-sync python" \
        flyevo scripts/explore/01_whats_in_the_data.py
"""

from pathlib import Path

import polars as pl

DATA = Path(__file__).resolve().parents[2] / "data" / "flywire"
GLOMERULI = ["DL2d", "DP1l", "DC3", "VM5v", "DM2", "DA1", "VA2"]  # the 7 "odours" of M1

print("== files ==")
for f in sorted(DATA.glob("*.*")):
    print(f"  {f.name:32s} {f.stat().st_size / 1e6:7.1f} MB")

# One row per neuron. Row number = the neuron's index in the simulation.
neurons = pl.read_csv(DATA / "Completeness_783.csv")
print(f"\n== neurons: {neurons.height} ==")

# One row per CONNECTION (neuron A -> neuron B), not per synapse.
con = pl.read_parquet(DATA / "Connectivity_783.parquet")
print(f"== connections: {con.height} ==")
row = con.row(2, named=True)
print(
    f"  example: neuron #{row['Presynaptic_Index']} -> neuron #{row['Postsynaptic_Index']}: "
    f"{row['Connectivity']} synapses, sign {row['Excitatory']:+d} (+1 excites, -1 inhibits) "
    f"-> weight {row['Excitatory x Connectivity']:+d}"
)
print(f"  {(con['Excitatory'] < 0).mean():.0%} of connections are inhibitory")

# Who is who: a human-made label for every neuron (Schlegel et al. 2024).
ann = pl.read_csv(
    DATA / "neuron_annotations.tsv",
    separator="\t",
    schema_overrides={"root_id": pl.Int64},
    columns=["root_id", "super_class", "cell_class", "cell_type", "side"],
).filter(pl.col("root_id").is_in(neurons.to_series(0).implode()))

print("\n== the brain by region (super_class) ==")
for r in ann.group_by("super_class").len().sort("len", descending=True).to_dicts():
    print(f"  {r['super_class'] or '?':20s} {r['len']:6d}")

print("\n== the smell pathway, in order ==")
for label, expr in [
    ("ORN  nose sensors", pl.col("cell_class") == "olfactory"),
    ("PN   cables to the higher brain", pl.col("cell_class") == "ALPN"),
    ("KC   Kenyon cells (memory centre)", pl.col("cell_class") == "Kenyon_Cell"),
    ("APL  the brake neuron (one per side)", pl.col("cell_type") == "APL"),
    ("MBON memory-centre outputs", pl.col("cell_class") == "MBON"),
]:
    print(f"  {label:38s} {ann.filter(expr).height:5d}")

print("\n== the 7 odour channels (glomeruli) used in M1 ==")
for g in GLOMERULI:
    orn = ann.filter(pl.col("cell_type") == f"ORN_{g}").height
    pn = ann.filter((pl.col("cell_class") == "ALPN") & pl.col("cell_type").str.starts_with(f"{g}_"))
    print(f"  {g:5s} {orn:4d} sensors -> {pn.height:2d} cables  {sorted(set(pn['cell_type']))}")
