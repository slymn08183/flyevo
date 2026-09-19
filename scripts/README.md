Standalone scripts outside the core package. Everything that touches **FlyWire data (CC BY-NC 4.0)**
lives here and is never imported from `src/flyevo/`. The simulator they use (`flyevo.models.lif`)
is data-agnostic and lives in the package.

| Script | What |
|---|---|
| `flywire_io.py` | Loads the FlyWire v783 connectome files shipped with the reference model into a `Network`. |
| `flywire_experiment.py` | One whole-brain activation/silencing experiment from a YAML in `configs/pretest/` → `runs/pretest/<name>/`. |
| `flywire_viz.py` | Renders a finished experiment: `viz.rrd` (interactive 3D, open with `rerun <file>`) and `activity.gif`. Needs `data/flywire/neuron_annotations.tsv` (32 MB, github.com/flyconnectome/flywire_annotations, Schlegel et al. 2024, doi:10.1038/s41586-024-07686-5). CPU only. |
| `pretest_shiu.py` | M1 go/no-go (not written yet): apply the known *D. sechellia* changes by hand, check the behavioural readout. |

Data (not committed, ~104 MB) goes to `data/flywire/`, from github.com/philshiu/Drosophila_brain_model:
`Connectivity_783.parquet`, `Completeness_783.csv`. Cite Dorkenwald et al. 2024
(doi:10.1038/s41586-024-07558-y) and Shiu et al. 2024 (doi:10.1038/s41586-024-07763-9).

Run on the GPU:

```bash
docker compose -f docker/compose.yaml run --rm --entrypoint "uv run --no-sync python" \
    flyevo scripts/flywire_experiment.py configs/pretest/sugar_grn.yaml
```
