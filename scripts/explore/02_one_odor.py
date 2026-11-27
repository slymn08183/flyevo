"""Step 2 - give the normal brain ONE odour and watch the memory centre. (GPU, ~1 minute)

    docker compose -f docker/compose.yaml run --rm --entrypoint "uv run --no-sync python -u" \
        flyevo scripts/explore/02_one_odor.py DL2d 100

An "odour" here = all nose sensors (ORNs) of one glomerulus firing at <rate> Hz for 1 second.
A healthy fly answers with ~5-10 % of its Kenyon cells, a different subset for each odour.
"""

import sys
from pathlib import Path

import jax
import numpy as np
import polars as pl
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flyevo.models.lif import LIFParams, make_stimulus, simulate
from flywire_io import load_flywire

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "flywire"
glomerulus, rate_hz = sys.argv[1], float(sys.argv[2])

lif_yaml = OmegaConf.to_container(OmegaConf.load(REPO / "configs" / "model" / "lif.yaml"))
params = LIFParams.from_config(lif_yaml)  # type: ignore[arg-type]
brain = load_flywire(DATA / "Connectivity_783.parquet", DATA / "Completeness_783.csv")
ann = pl.read_csv(
    DATA / "neuron_annotations.tsv",
    separator="\t",
    columns=["root_id", "cell_class", "cell_type"],
    schema_overrides={"root_id": pl.Int64},
)


def indices(expr: pl.Expr) -> np.ndarray:
    ids = ann.filter(expr)["root_id"].to_list()
    return np.array([brain.index_of[r] for r in ids if r in brain.index_of])


is_pn = (pl.col("cell_class") == "ALPN") & pl.col("cell_type").str.starts_with(f"{glomerulus}_")
orn = indices(pl.col("cell_type") == f"ORN_{glomerulus}")
pn = indices(is_pn)
kc = indices(pl.col("cell_class") == "Kenyon_Cell")
apl = indices(pl.col("cell_type") == "APL")
mbon = indices(pl.col("cell_class") == "MBON")

print(f"device: {jax.devices()[0]}")
print(f"odour : {glomerulus} = {len(orn)} nose sensors driven at {rate_hz:.0f} Hz for 1 s")
stimulus = make_stimulus(brain.net.n_neurons, dict.fromkeys(orn.tolist(), rate_hz))
out = simulate(jax.random.key(0), brain.net, stimulus, params, t_run_ms=1000, bin_ms=10)
hz = np.asarray(out.spike_counts)  # 1 s run -> spike count = rate in Hz

responding = hz[kc] >= 1
print(f"  cables (PN) of this channel  : {np.sort(hz[pn])[::-1].tolist()} Hz")
print(f"  Kenyon cells that respond    : {responding.sum()} of {len(kc)} = {responding.mean():.0%}")
print("                                 (a real fly: roughly 5-10 %)")
print(f"  brake neuron (APL)           : {hz[apl].tolist()} Hz  (ceiling of this model: ~430 Hz)")
print(f"  outputs (MBON) active        : {(hz[mbon] > 0).sum()} of {len(mbon)}")
print(f"  fastest output neuron        : {hz[mbon].max()} Hz")
print(f"  neurons active in whole brain: {(hz > 0).sum()} of {len(hz)}")

saved = REPO / "runs" / f"explore_kc_{glomerulus}.npy"
saved.parent.mkdir(exist_ok=True)
np.save(saved, responding)
print(f"  saved which Kenyon cells responded -> {saved.relative_to(REPO)} (step 3 compares them)")
