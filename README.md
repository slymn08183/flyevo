# flyevo

**In-silico evolution of the *Drosophila* olfactory circuit, tested against real comparative connectomics.**

We evolve developmental parameters of the *D. melanogaster* olfactory pathway (antennal lobe → mushroom body → MBON) under an ecological fitness, and ask: under strong artificial selection for a *D. sechellia*-like niche, does the circuit arrive at the **same mechanisms** observed in the real species (PN count vs. bouton count vs. E/I balance, per glomerulus)? The claim is about mechanism-level convergence against a neutral-drift null — not "replaying evolution".

> Status: **M0 — repo skeleton.** No scientific results yet. Roadmap: [docs/PLAN.md](docs/PLAN.md) §8.

<!-- GIF of the circuit in Rerun goes here (docs/assets/) once M0 step 4 lands -->

## Quick start

Linux / WSL2 / devcontainer (native Windows is CPU-only and unsupported for CUDA):

```bash
make setup                                # uv sync + pre-commit + minimal data
uv run flyevo doctor                      # is the environment healthy?
uv run flyevo evolve experiment=smoke     # 2-min CPU run
uv run flyevo analyze runs/<run>/
uv run flyevo viz runs/<run>/ --serve     # Rerun web viewer
```

Docker: `make docker-cpu` or `make docker-cuda` (needs NVIDIA Container Toolkit), then
`docker compose -f docker/compose.yaml --profile cuda run --rm flyevo-cuda evolve experiment=smoke`.

A free [neuPrint](https://neuprint.janelia.org) token is required for data fetch — copy `.env.example` to `.env`.

## Layout

```
src/flyevo/     the package — source of truth
  data/         neuprint fetch, cache, checksums, comparative tables (with DOIs)
  circuit/      connectome graph → cell-type groups → JAX arrays (fixed architecture)
  develop/      genotype → phenotype generator (stochastic development)
  models/       rate model (L1), LIF (L2), plasticity, normalization
  env/          odor panels, niches, lifetime trials
  fitness/      objectives + energetic cost
  evo/          evosax / QDax / pymoo runners, neutral-drift null
  analysis/     mechanism-level metrics, statistics
  viz/          Rerun logging, figures (the only place that plots)
  motifs/       pure reusable circuit motifs
configs/        Hydra config groups — every experiment is CLI + config
tests/          unit · property (hypothesis) · golden (2-min smoke evolution)
scripts/        standalone scripts (M1 pre-test; optional [pretest] extra)
docker/         Dockerfile (cpu / cuda targets) + compose
docs/           plan, pre-registration, milestone results
notebooks/      exploration only — never the source of a result
```

`data/` and `runs/` are created locally and gitignored. Every run writes its resolved config, git SHA and data checksums to `runs/<date>/<time>_<name>/`.

## Data & licenses

| What | License | Where it is used |
|---|---|---|
| Code | Apache-2.0 | — |
| hemibrain v1.2.1, MaleCNS v1.0 | CC BY 4.0 | core package, via neuPrint |
| FlyWire v783 | **CC BY-NC 4.0** | only `scripts/pretest_shiu.py` (`[pretest]` extra); never imported by `src/flyevo/` |
| Brian2 | CeCILL | `[pretest]` extra only |

Raw data is never committed. Small derived tables carry their source DOI in the file.

## Contributing

Small PRs, one step each, tests with every feature: `make lint && make test`. Working rules for AI coding assistants live in [.claude/CLAUDE.md](.claude/CLAUDE.md); they are not project documentation.
