"""Run one whole-brain activation/silencing experiment on the FlyWire connectome (GPU).

    docker compose -f docker/compose.yaml run --rm --entrypoint "uv run --no-sync python" \
        flyevo scripts/flywire_experiment.py configs/pretest/sugar_grn.yaml [key=value ...]

Trailing ``key=value`` pairs override the YAML (e.g. ``rate_hz=100 name=sugar_grn_100hz``).

Same protocol as the reference model's ``run_exp``: Poisson-activate a set of neurons, optionally
silence another, n trials from rest, report mean firing rate per neuron. Everything about the
experiment comes from the YAML; model constants come from ``configs/model/lif.yaml``.

Writes to ``runs/pretest/<name>/``: ``rates.parquet`` (active neurons only), ``binned_trial0.npz``
(spikes per time bin of the first trial, for visualization) and ``meta.json``.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import jax
import numpy as np
import polars as pl
from omegaconf import OmegaConf

from flyevo.models.lif import LIFParams, make_stimulus, simulate
from flywire_io import load_flywire, sha256

REPO = Path(__file__).resolve().parents[1]


def _git_sha() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True)
        return sha + ("-dirty" if dirty.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main(experiment_yaml: str, overrides: list[str]) -> None:
    exp = OmegaConf.merge(OmegaConf.load(experiment_yaml), OmegaConf.from_dotlist(overrides))
    params = LIFParams.from_config(OmegaConf.to_container(OmegaConf.load(REPO / exp.model)))  # type: ignore[arg-type]
    con_path, comp_path = REPO / exp.data.connectivity, REPO / exp.data.completeness

    print(f"devices {jax.devices()}")
    brain = load_flywire(con_path, comp_path)
    n = brain.net.n_neurons
    print(f"brain: {n} neurons, {brain.net.pre.shape[0]} connections")

    stimulated = brain.indices(list(exp.stimulated_ids), "stimulated")
    silenced = brain.indices(list(exp.get("silenced_ids", [])), "silenced")
    stim = make_stimulus(n, dict.fromkeys(stimulated, float(exp.rate_hz)), silenced)

    counts = np.zeros((exp.n_trials, n), dtype=np.int32)
    binned_trial0 = None
    t0 = time.time()
    for trial, key in enumerate(jax.random.split(jax.random.key(exp.seed), exp.n_trials)):
        out = simulate(key, brain.net, stim, params, t_run_ms=exp.t_run_ms, bin_ms=exp.bin_ms)
        counts[trial] = np.asarray(out.spike_counts)
        if trial == 0:
            binned_trial0 = np.asarray(out.binned)
        elapsed = time.time() - t0
        print(f"trial {trial + 1}/{exp.n_trials}: {counts[trial].sum()} spikes, {elapsed:.0f}s")
    wall = time.time() - t0

    rates = counts / (exp.t_run_ms * 1e-3)  # Hz, [n_trials, N]
    mean, std = rates.mean(axis=0), rates.std(axis=0)
    active = np.flatnonzero(mean > 0)
    table = pl.DataFrame(
        {"flywire_id": brain.ids[active], "rate_hz": mean[active], "rate_std_hz": std[active]}
    ).sort("rate_hz", descending=True)

    out_dir = REPO / "runs" / "pretest" / exp.name
    out_dir.mkdir(parents=True, exist_ok=True)
    table.write_parquet(out_dir / "rates.parquet")
    np.savez_compressed(out_dir / "binned_trial0.npz", binned=binned_trial0, ids=brain.ids)
    meta = {
        "experiment": OmegaConf.to_container(exp),
        "lif_params": params.__dict__,
        "git_sha": _git_sha(),
        "data_sha256": {p.name: sha256(p) for p in (con_path, comp_path)},
        "wall_seconds": round(wall, 1),
        "active_neurons": len(active),
        "spikes_per_trial_mean": float(counts.sum(axis=1).mean()),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(
        f"\n{len(active)} active neurons, {counts.sum()} spikes, {exp.n_trials} trials, {wall:.0f}s"
    )
    for label, root_id in dict(exp.get("readout_ids", {})).items():
        idx = brain.index_of.get(int(root_id))
        if idx is not None:
            print(f"readout {label}: {mean[idx]:.1f} ± {std[idx]:.1f} Hz")
    print(f"results: {out_dir.relative_to(REPO)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
