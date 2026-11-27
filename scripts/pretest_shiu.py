"""M1 go/no-go: does the whole-brain LIF model react to the changes evolution really made?

    docker compose -f docker/compose.yaml run --rm --entrypoint "uv run --no-sync python -u" \
        flyevo scripts/pretest_shiu.py configs/pretest/m1_sechellia.yaml

Two brains: the FlyWire melanogaster brain as is ("baseline") and a copy edited by hand with the
measured D. sechellia PN→KC differences ("sechellia"). Both get the same odours with the same
random keys (paired trials). We compare the mushroom-body response. The decision rule lives in
the YAML and was written before the first run.

Writes ``runs/pretest/<name>/results.json`` (+ ``per_trial.npz``, ``meta.json``).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import jax
import numpy as np
import polars as pl
import yaml
from omegaconf import OmegaConf

from flyevo.models.lif import LIFParams, make_network, make_stimulus, simulate
from flywire_io import sha256

REPO = Path(__file__).resolve().parents[1]
Edges = tuple[np.ndarray, np.ndarray, np.ndarray]  # pre i64[E], post i64[E], weight f32[E]


def remove_outputs(edges: Edges, neurons: np.ndarray) -> Edges:
    """Delete every outgoing connection of ``neurons`` (a neuron that does not exist)."""
    pre, post, w = edges
    keep = ~np.isin(pre, neurons)
    return pre[keep], post[keep], w[keep]


def add_partners(
    edges: Edges, neuron: int, pool: np.ndarray, factor: float, rng: np.random.Generator
) -> Edges:
    """Give ``neuron`` new targets in ``pool`` until it has ``factor`` times as many as before.

    New targets are drawn at random from the pool members it does not reach yet; weights are
    resampled from its existing connections into the pool. No existing partners → nothing added.
    """
    pre, post, w = edges
    mine = (pre == neuron) & np.isin(post, pool)
    n_new = round(int(mine.sum()) * (factor - 1.0))
    candidates = np.setdiff1d(pool, post[mine])
    n_new = max(0, min(n_new, len(candidates)))
    if n_new == 0:
        return edges
    targets = rng.choice(candidates, n_new, replace=False)
    weights = rng.choice(w[mine], n_new, replace=True)
    return (
        np.concatenate([pre, np.full(n_new, neuron, dtype=pre.dtype)]),
        np.concatenate([post, targets.astype(post.dtype)]),
        np.concatenate([w, weights]),
    )


def paired_effect_size(base: np.ndarray, pert: np.ndarray) -> float:
    """Cohen's d for paired trials: mean difference / sd of the differences."""
    diff = np.asarray(pert, dtype=float) - np.asarray(base, dtype=float)
    sd = diff.std(ddof=1)
    if sd == 0:
        return 0.0 if diff.mean() == 0 else float(np.sign(diff.mean()) * np.inf)
    return float(diff.mean() / sd)


def population_shift(base: np.ndarray, pert: np.ndarray) -> float:
    """Distance between mean rate vectors [trials, M], in units of trial-to-trial scatter."""
    scatter = np.mean([np.linalg.norm(x - x.mean(axis=0), axis=1).mean() for x in (base, pert)])
    shift = float(np.linalg.norm(pert.mean(axis=0) - base.mean(axis=0)))
    return shift / scatter if scatter > 0 else (0.0 if shift == 0 else float("inf"))


def main(experiment_yaml: str, overrides: list[str]) -> None:
    exp = OmegaConf.merge(OmegaConf.load(experiment_yaml), OmegaConf.from_dotlist(overrides))
    params = LIFParams.from_config(OmegaConf.to_container(OmegaConf.load(REPO / exp.model)))  # type: ignore[arg-type]
    comparative = yaml.safe_load((REPO / exp.comparative).read_text(encoding="utf-8"))
    rng = np.random.default_rng(exp.seed)

    ids = pl.read_csv(REPO / exp.data.completeness).to_series(0).to_numpy()
    index_of = {int(r): i for i, r in enumerate(ids)}
    n = len(ids)
    con = pl.read_parquet(
        REPO / exp.data.connectivity,
        columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"],
    )
    baseline: Edges = (
        con["Presynaptic_Index"].to_numpy(),
        con["Postsynaptic_Index"].to_numpy(),
        con["Excitatory x Connectivity"].to_numpy().astype(np.float32),
    )
    ann = pl.read_csv(
        REPO / exp.data.annotations,
        separator="\t",
        columns=["root_id", "cell_class", "cell_type", "side"],
        schema_overrides={"root_id": pl.Int64},
    ).filter(pl.col("root_id").is_in(ids))

    def indices(expr: pl.Expr) -> np.ndarray:
        return np.array(
            [index_of[r] for r in ann.filter(expr)["root_id"].to_list()], dtype=np.int64
        )

    kc, mbon = (
        indices(pl.col("cell_class") == "Kenyon_Cell"),
        indices(pl.col("cell_class") == "MBON"),
    )
    kc_side = {
        s: indices((pl.col("cell_class") == "Kenyon_Cell") & (pl.col("side") == s))
        for s in ("left", "right")
    }

    # ---- build the sechellia-fied brain ----
    edited, log = baseline, []
    for name, g in comparative["glomeruli"].items():
        is_pn = (pl.col("cell_class") == "ALPN") & pl.col("cell_type").str.starts_with(f"{name}_")
        if g["mechanism"] == "bouton_gain":
            pct = g["kc_connection_pct"]
            factor = pct["sechellia"] / pct["melanogaster"]
            for side, pool in kc_side.items():
                for pn in indices(is_pn & (pl.col("side") == side)):
                    before = len(edited[0])
                    edited = add_partners(edited, int(pn), pool, factor, rng)
                    log.append(
                        f"{name} PN {ids[pn]} ({side}): +{len(edited[0]) - before} KC partners (x{factor:.2f})"
                    )
        elif g["mechanism"] == "pn_loss":
            for side in kc_side:
                pns = indices(
                    is_pn & (pl.col("side") == side) & pl.col("cell_type").str.ends_with("_adPN")
                )
                extra = len(pns) - int(exp.perturbation.pn_target_per_side)
                if extra > 0:
                    gone = rng.choice(pns, extra, replace=False)
                    edited = remove_outputs(edited, gone)
                    log.append(f"{name} ({side}): removed {extra} of {len(pns)} adPNs")
    print("\n".join(log))

    odors = [(o, "changed") for o in exp.odors.changed] + [
        (o, "control") for o in exp.odors.control
    ]
    keys = jax.random.split(jax.random.key(exp.seed), exp.n_trials)
    early_bins = max(1, round(exp.early_ms / exp.bin_ms))
    data: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    t0 = time.time()
    print(f"devices {jax.devices()}")
    for cond, edges in (("baseline", baseline), ("sechellia", edited)):
        net = make_network(*edges, n_neurons=n)
        data[cond] = {}
        for odor, _ in odors:
            orn = indices(pl.col("cell_type") == f"ORN_{odor}")
            stim = make_stimulus(n, dict.fromkeys(orn.tolist(), float(exp.rate_hz)))
            kc_rate, mbon_rate, kc_early = [], [], []
            for key in keys:
                out = simulate(key, net, stim, params, t_run_ms=exp.t_run_ms, bin_ms=exp.bin_ms)
                counts = np.asarray(out.spike_counts)
                kc_rate.append(counts[kc] / (exp.t_run_ms * 1e-3))
                mbon_rate.append(counts[mbon] / (exp.t_run_ms * 1e-3))
                kc_early.append(np.asarray(out.binned[:early_bins][:, kc]).sum(axis=0) > 0)
            data[cond][odor] = {
                "kc_rate": np.array(kc_rate),
                "mbon_rate": np.array(mbon_rate),
                "kc_early": np.array(kc_early),
            }
            responding = (data[cond][odor]["kc_rate"] >= exp.perturbation.active_hz).sum(axis=1)
            print(
                f"{cond:9s} {odor:5s}: {len(orn)} ORNs, KCs responding {responding.mean():.0f}/{len(kc)}, {time.time() - t0:.0f}s"
            )
        del net

    # ---- analysis ----
    rows = []
    for odor, kind in odors:
        b, s = data["baseline"][odor], data["sechellia"][odor]
        n_b = (b["kc_rate"] >= exp.perturbation.active_hz).sum(axis=1)
        n_s = (s["kc_rate"] >= exp.perturbation.active_hz).sum(axis=1)
        direction = comparative["glomeruli"].get(odor, {}).get("direction")
        d_kc = paired_effect_size(n_b, n_s)
        d_mbon = population_shift(b["mbon_rate"], s["mbon_rate"])
        if kind == "changed":
            right_way = d_kc > 0 if direction == "up" else d_kc < 0
            passed = bool(
                right_way
                and abs(d_kc) >= exp.criteria.effect_size_min
                and d_mbon >= exp.criteria.effect_size_min
            )
        else:
            passed = bool(abs(d_kc) < exp.criteria.effect_size_min)
        rows.append({
            "odor": odor, "kind": kind, "expected": direction,
            "kc_responding_baseline": float(n_b.mean()), "kc_responding_sechellia": float(n_s.mean()),
            "kc_effect_size_d": d_kc, "mbon_shift_in_noise_units": d_mbon,
            "kc_fraction_baseline": float(n_b.mean() / len(kc)),
            # exploratory, not part of the decision rule:
            "early_kc_baseline": float(b["kc_early"].sum(axis=1).mean()),
            "early_kc_sechellia": float(s["kc_early"].sum(axis=1).mean()),
            "early_kc_effect_size_d": paired_effect_size(b["kc_early"].sum(axis=1), s["kc_early"].sum(axis=1)),
            "passed": passed,
        })  # fmt: skip

    # how odour-specific is the baseline KC response? (1 = identical sets, 0 = disjoint)
    sets = {
        o: data["baseline"][o]["kc_rate"].mean(axis=0) >= exp.perturbation.active_hz
        for o, _ in odors
    }
    names = [o for o, _ in odors]
    overlap = {
        f"{a}|{b}": float((sets[a] & sets[b]).sum() / max(1, (sets[a] | sets[b]).sum()))
        for i, a in enumerate(names) for b in names[i + 1 :]
    }  # fmt: skip

    changed_pass = sum(r["passed"] for r in rows if r["kind"] == "changed")
    controls_ok = all(r["passed"] for r in rows if r["kind"] == "control")
    verdict = "GO" if changed_pass >= exp.criteria.min_changed_pass and controls_ok else "NO-GO"

    out_dir = REPO / "runs" / "pretest" / exp.name
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "verdict": verdict, "changed_passed": changed_pass, "controls_ok": controls_ok,
        "rows": rows, "baseline_kc_set_overlap_jaccard": overlap, "edits": log,
    }  # fmt: skip
    (out_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.savez_compressed(
        out_dir / "per_trial.npz",
        **{f"{c}/{o}/{k}": v for c, d in data.items() for o, m in d.items() for k, v in m.items()},
    )
    meta = {
        "experiment": OmegaConf.to_container(exp), "lif_params": params.__dict__,
        "data_sha256": {k: sha256(REPO / exp.data[k]) for k in ("connectivity", "completeness", "annotations")},
        "wall_seconds": round(time.time() - t0, 1),
    }  # fmt: skip
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(
        f"\n{'odor':6s}{'kind':9s}{'exp':5s}{'KC base':>9s}{'KC sech':>9s}{'d(KC)':>8s}{'MBON shift':>11s}  pass"
    )
    for r in rows:
        print(
            f"{r['odor']:6s}{r['kind']:9s}{r['expected'] or '-':5s}{r['kc_responding_baseline']:9.0f}{r['kc_responding_sechellia']:9.0f}{r['kc_effect_size_d']:8.2f}{r['mbon_shift_in_noise_units']:11.2f}  {r['passed']}"
        )
    print(
        f"\nverdict: {verdict} ({changed_pass}/4 changed odours passed, controls ok: {controls_ok})"
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
