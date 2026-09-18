# CLAUDE.md — flyevo

In-silico evolution of the Drosophila olfactory circuit (AL → MB → MBON) under ecological fitness, tested against real comparative-connectomics data (D. sechellia vs D. melanogaster). Full plan: `docs/PLAN.md`. Read it before non-trivial work.

> This folder (`.claude/`) holds AI-assistant working rules only. Project documentation lives in `README.md` and `docs/`. Keep the two separate.

## Stack (do not swap without discussion)

Python 3.13 (`requires-python = ">=3.13,<3.15"`) · uv · JAX · evosax (≥0.3) · QDax (≥0.5) · pymoo · neuprint-python · navis · hydra-core · typer · rerun-sdk · mlflow · pytest · hypothesis · ruff · mypy · Docker (targets `cpu`, `cuda`).

Python version rule: the pin is "newest CPython for which every core dependency ships binary wheels" (matrix in `docs/PLAN.md` §2.1). Do not bump or lower it without re-checking that matrix; never accept a source build of a C/Cython dependency inside Docker to reach a newer interpreter. 3.14 runs in CI as allowed-to-fail; when it is green for QDax and omegaconf, bump the default. Aim was removed on purpose (no wheels past 3.12, stale since May 2025) — do not reintroduce it.

## Commands

```bash
make setup            # uv sync --all-extras + pre-commit install + minimal data fetch
make test             # pytest (unit + property + golden)
make lint             # ruff + mypy
make docker-cpu       # build cpu image
make docker-cuda      # build cuda image (WSL2 + NVIDIA toolkit required)
uv run flyevo doctor  # env health: JAX backend, GPU, data checksums, versions
uv run flyevo data fetch
uv run flyevo evolve experiment=smoke                    # 2-min CPU run
uv run flyevo evolve experiment=sech_main env=sechellia evo=cmaes seed=0
uv run flyevo analyze runs/<run>/
uv run flyevo viz runs/<run>/ --serve                    # Rerun web viewer
docker compose -f docker/compose.yaml --profile cuda up -d mlflow   # UI at :5000
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
* FlyWire data is CC BY-NC. It is used only in `scripts/pretest_shiu.py` behind the `[pretest]` extra. Never import it from `src/flyevo/`.
* Brian2 (CeCILL) and brian2cuda (GPL) are `[pretest]`-only. Core deps stay permissive (Apache/MIT/BSD).
* Raw data and `runs/` are gitignored. Only small derived tables (< 1 MB) with a source DOI in the file are committed (`src/flyevo/data/comparative/`).
* Code license: Apache-2.0.

## Gates

* M1 pre-test is a go/no-go (`scripts/pretest_shiu.py`). Do not start M3 before its result is recorded in `docs/M1_RESULT.md`.
* Before M3, write `docs/PREREG.md` (hypotheses, metrics, null model, seeds, stopping rule). It is frozen after that; changes go in a dated "Deviations" section.
* Golden test (`tests/golden/`) must stay green; tolerance changes require a note in the PR.

## Testing

* Unit: generator determinism, KC sparsity bounds, divisive normalization scale invariance, plasticity sign.
* Property (hypothesis): any in-bounds genotype → valid phenotype, finite fitness.
* Golden: `experiment=smoke` on CPU < 2 min → reference metrics ± tolerance.
* CI is CPU-only; Python matrix 3.13 (required) + 3.14 (`continue-on-error`); the CUDA image is build-only in CI.

## Windows / WSL2

* Everything runs in WSL2 (Ubuntu) or in Docker. JAX-CUDA is Linux-only; never attempt it on native Windows.
* Keep the repo on the WSL2 filesystem (`~/…`), not `/mnt/c/…`.
* `nvidia-smi` must work inside WSL2 before `make docker-cuda`.

## Do / Don't

* Do: small PRs, one milestone step each; tests with each feature; pin versions in `pyproject.toml`; update `docs/PLAN.md` when a decision changes.
* Don't: add topology evolution (architecture is fixed from connectome); evolve MBON weights (they are learned within a lifetime); add new dependencies without a one-line justification in the PR; write results into notebooks.

## First tasks (in order)

1. ~~Repo skeleton per `docs/PLAN.md` §3, `.python-version` = `3.13`, `pyproject.toml` with `requires-python = ">=3.13,<3.15"` and extras `cuda`, `pretest`, `dev`; Apache-2.0.~~ ✅
2. ~~`Makefile`, `docker/Dockerfile` (cpu/cuda), `docker/compose.yaml`, `.devcontainer/`, CI.~~ ✅ (`uv.lock` generated and verified on 3.13/CPU; Docker images not yet built)
3. `flyevo doctor`, `flyevo data fetch` (hemibrain olfactory subgraph, cached, checksummed).
4. `viz/rerun_log.py`: neurons in 3D from SWC, save `.rrd`, GIF for README.
5. `scripts/pretest_shiu.py` (M1).
