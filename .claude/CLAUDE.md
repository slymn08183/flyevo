# CLAUDE.md — flyevo

In-silico evolution of the Drosophila olfactory circuit (AL → MB → MBON) under ecological fitness, tested against real comparative-connectomics data (D. sechellia vs D. melanogaster). Full plan: `docs/PLAN.md`. Read it before non-trivial work.

> This folder (`.claude/`) holds AI-assistant working rules only. Project documentation lives in `README.md` and `docs/`. Keep the two separate.

## Stack (do not swap without discussion)

Python 3.13 (`requires-python = ">=3.13,<3.15"`) · uv · JAX · evosax (≥0.3) · QDax (≥0.5) · pymoo · neuprint-python · navis · hydra-core · typer · rerun-sdk · mlflow · pytest · hypothesis · ruff · mypy · Docker (single GPU image).

Python version rule: the pin is "newest CPython for which every core dependency ships binary wheels" (matrix in `docs/PLAN.md` §2.1). Do not bump or lower it without re-checking that matrix; never accept a source build of a C/Cython dependency inside Docker to reach a newer interpreter. 3.14 runs in CI as allowed-to-fail; when it is green for QDax and omegaconf, bump the default. Aim was removed on purpose (no wheels past 3.12, stale since May 2025) — do not reintroduce it.

## Commands

```bash
make setup            # uv sync --all-extras + pre-commit install + minimal data fetch
make test             # pytest (unit + property + golden)
make lint             # ruff + mypy
make docker           # build the GPU image (Docker Desktop + NVIDIA driver)
uv run flyevo doctor  # env health: JAX backend, GPU, data checksums, versions
uv run flyevo data fetch
uv run flyevo evolve experiment=smoke                    # 2-min CPU run
uv run flyevo evolve experiment=sech_main env=sechellia evo=cmaes seed=0
uv run flyevo analyze runs/<run>/
uv run flyevo viz runs/<run>/ --serve                    # Rerun web viewer
docker compose -f docker/compose.yaml up -d mlflow   # UI at :5000
docker compose -f docker/compose.yaml run --rm flyevo evolve experiment=smoke   # GPU run
```

## Repo conventions

* Source of truth is `src/flyevo/` + `tests/`. Notebooks are exploration only; nothing counts as a result until it lives in `src/` with a test.
* Every experiment runs through the CLI + Hydra config. No hard-coded parameters in code. New knobs → new config field with a default.
* Every run writes to `runs/<date>/<time>_<name>/`: resolved config, git SHA, data checksums, `genotypes.npz`, `metrics.parquet`, `viz.rrd`. Warn if git tree is dirty.
* JAX style: pure functions, explicit `key` threading, `vmap` over population, no Python-side loops over individuals. Shapes documented in docstrings.
* Seeds are explicit everywhere (`seed` config → `jax.random.key`). Generator (genotype → phenotype) must be deterministic for a given key; there is a test for it.
* No plotting inside `models/`, `develop/`, `env/`, `fitness/`, `evo/`. Visualization lives only in `viz/`.
* `src/flyevo/motifs/` stays pure and dependency-light (numpy/JAX only) so it can be split into a separate package later.
* Keep evosax sign convention in one place (`evo/evosax_runner.py`): evosax minimizes; we maximize fitness → negate there, nowhere else.
* Root stays clean: only README, LICENSE, pyproject, Makefile and dotfiles. Docs → `docs/`, assistant rules → `.claude/`.

## Data & licensing (hard rules)

* Core package uses hemibrain v1.2.1 and MaleCNS v1.0 (both CC BY 4.0) via neuprint. Dataset ids: `hemibrain:v1.2.1`, `male-cns:v1.0`.
* FlyWire data is CC BY-NC. It is used only under `scripts/`. Never import it from `src/flyevo/`; anything rendered from it (images, recordings) is CC BY-NC too and must say so.
* Brian2 (CeCILL) and brian2cuda (GPL) are `[pretest]`-only. Core deps stay permissive (Apache/MIT/BSD).
* Raw data and `runs/` are gitignored. Only small derived tables (< 1 MB) with a source DOI in the file are committed (`src/flyevo/data/comparative/`).
* Code license: Apache-2.0.

## Gates

* M1 pre-test is a go/no-go (`scripts/pretest_shiu.py`). Result recorded in `docs/M1_RESULT.md`: **NO-GO** for the whole-brain model on mushroom-body questions → L1 path. Do not use the whole-brain LIF to draw conclusions about KC/MBON coding unless its mushroom body is fixed and re-validated first.
* Before M3, write `docs/PREREG.md` (hypotheses, metrics, null model, seeds, stopping rule). It is frozen after that; changes go in a dated "Deviations" section.
* M6 forward prediction (`docs/PLAN.md` §1.7) runs only if the M3 calibration passes the PREREG criteria. Predictions go to `docs/PREDICTIONS.md`, are dated, tagged and frozen before the target species' data exists; never edit them afterwards.
* Fitness is always ecological (niche + cost). Never use similarity to a real species' data as fitness — that data is the held-out exam.
* Golden test (`tests/golden/`) must stay green; tolerance changes require a note in the PR.

## Testing

* Unit: generator determinism, KC sparsity bounds, divisive normalization scale invariance, plasticity sign.
* Property (hypothesis): any in-bounds genotype → valid phenotype, finite fitness.
* Golden: `experiment=smoke` on CPU < 2 min → reference metrics ± tolerance.
* CI is currently disabled (`.github/workflows/ci.yml` is commented out; no pipeline yet). Tests run locally with `make test`.

## Working style

* **Explain, don't just do.** No experiments or analyses out of sight: no ad-hoc `python -c` one-liners whose numbers then show up in a report. Anything worth quoting comes from a readable script in the repo (`scripts/explore/NN_*.py` for look-around steps).
* Show every command when running it and say what each part does. One step at a time; real experiments are run in the open and interpreted together.
* Plain language: define each biology term and each tool in one sentence the first time it matters; say why a step is done and what the result means.
* On Windows the `.venv` depends on a uv-managed Python 3.13 (`winget install --id=astral-sh.uv -e`, then `uv sync --extra dev`). Docker runs do not depend on it.

## Windows / GPU

* Real runs happen on the GPU through the Docker image (`make docker`). Docker Desktop's own WSL2 backend passes the GPU through; **no separate WSL distro is needed**.
* The Windows `.venv` is CPU-only and is for tests/lint/quick checks. JAX-CUDA is Linux-only; never attempt it on native Windows.
* There is no CPU Docker image (removed on purpose). The image uses `python:3.13-slim`, not `nvidia/cuda`: the `jax[cuda12]` wheels bring their own CUDA libs.
* The GPU is a 4 GB GTX 1650 shared with the desktop: `XLA_PYTHON_CLIENT_PREALLOCATE=false` is mandatory (set in `cli.py` **and** as `ENV` in the Dockerfile, because `scripts/` and `python -c` bypass the CLI). Without it JAX dies with `CUDA_ERROR_OUT_OF_MEMORY` on the first large transfer — that, not leaked memory, was the cause of the 2026-09-18 crashes. Mind memory in batch sizes.
* Disk is not a constraint; download what the work needs. Do not ask for installs or downloads that are not really needed.

## Do / Don't

* Do: small PRs, one milestone step each; tests with each feature; pin versions in `pyproject.toml`; update `docs/PLAN.md` when a decision changes.
* Don't: add topology evolution (architecture is fixed from connectome); evolve MBON weights (they are learned within a lifetime); add new dependencies without a one-line justification in the PR; write results into notebooks.

## First tasks (in order)

1. ~~Repo skeleton per `docs/PLAN.md` §3, `.python-version` = `3.13`, `pyproject.toml` with `requires-python = ">=3.13,<3.15"` and extras `cuda`, `pretest`, `dev`; Apache-2.0.~~ ✅
2. ~~`Makefile`, `docker/Dockerfile` (GPU), `docker/compose.yaml`, `.devcontainer/`.~~ ✅ (`uv.lock` generated and verified on 3.13)
3. ~~`models/lif.py`: JAX port of the Shiu 2024 whole-brain LIF (event-driven, exact dense fallback), unit-tested.~~ ✅ Measured on the GTX 1650 with a synthetic whole-brain-sized net (138k neurons, 15M edges, ~200 active): ~23 s per biological second (dense version: ~150 s). Budgets in `configs/model/lif.yaml` are the tuning knob.
4. `scripts/`: FlyWire v783 loader (`flywire_io.py`) + experiment runner (`flywire_experiment.py`, YAML in `configs/pretest/`). ✅ Sugar-GRN experiment runs on the GPU (30 trials ≈ 10 min): 423 active neurons, 467,570 spikes, MN9 96 ± 6 Hz — matches the reference notebook's coarse description (~400 active, >400k spikes; v630 data). Per-neuron check against the reference result files (v630 data, ours v783): at 100 Hz — stimulated GRNs 99.0 vs 99.3 Hz, active neurons 404 vs 406, MN9 67.0 vs 70.5 Hz, r = 0.95 over neurons active in both; ours has ~13% more spikes overall, consistent with (but not proven to be) the connectome version difference. The shipped `sugarR.parquet` was evidently generated at 200 Hz, not the notebook's 150 Hz (its GRNs fire at ~197 Hz). An exact same-data comparison would need the v630 connectivity file (86.6 MB). FlyWire stays out of `src/flyevo/`.
5. ~~`viz/rerun_log.py` (3D point cloud + spikes on `sim_time`, `.rrd`) and `viz/figures.py` (GIF); look in `configs/viz/brain.yaml`; FlyWire positions via `scripts/flywire_viz.py`.~~ ✅ Verified in the Rerun web viewer. Scene now also has region surfaces, guide arrows and real neuron skeletons; README image in `docs/assets/`.
6. ~~`scripts/pretest_shiu.py` (M1 go/no-go).~~ ✅ **Result: NO-GO** (`docs/M1_RESULT.md`): in the whole-brain LIF the mushroom body is saturated and odour-blind (~75% of Kenyon cells fire for any odour, same cells, MBON output identical across odours) because KC→KC input (+73 synapses/KC) is treated as plain excitation and the single spiking APL cannot brake it. Per `docs/PLAN.md` §8: evolution runs in the L1 rate model; the "holds in whole-brain context" claim is dropped for now.
7. `flyevo data fetch` (hemibrain olfactory subgraph, cached, checksummed) → M2.
