"""`flyevo` command line: doctor | data | evolve | analyze | viz.

Heavy imports (jax, rerun, neuprint) stay inside commands so `flyevo --help` is instant.
"""

from __future__ import annotations

import os
import platform
import sys
from importlib import metadata
from pathlib import Path
from typing import Annotated

import typer

# JAX grabs 75% of GPU memory up front by default; on small or desktop-shared GPUs (e.g. 4 GB
# under WSL2) that fails with CUDA_ERROR_OUT_OF_MEMORY. Allocate on demand unless the user
# exported their own value. Must be set before jax is imported.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)
data_app = typer.Typer(no_args_is_help=True, help="Connectome data: fetch, cache, checksum.")
app.add_typer(data_app, name="data")

_VERSIONED = ("jax", "evosax", "qdax", "pymoo", "neuprint-python", "navis", "rerun-sdk", "mlflow")


def _not_yet(milestone: str) -> None:
    typer.secho(f"not implemented yet — scheduled for {milestone} (docs/PLAN.md §8)", fg="yellow")
    raise typer.Exit(code=2)


@app.command()
def doctor() -> None:
    """Environment health: Python, JAX backend, GPU, package versions, neuprint token."""
    ok = True
    typer.echo(f"python    {platform.python_version()}  ({sys.platform})")
    if sys.platform == "win32":
        typer.secho("warn      native Windows: CPU only. Use WSL2 or Docker for CUDA.", fg="yellow")

    try:
        import jax

        devices = ", ".join(str(d) for d in jax.devices())
        typer.echo(f"jax       backend={jax.default_backend()}  devices=[{devices}]")
        # listing devices is not enough: run a real op so allocation problems show up here
        probe = float((jax.numpy.ones((256, 256)) @ jax.numpy.ones((256, 256)))[0, 0])
        typer.echo(f"jax       compute ok ({probe:.0f})")
    except Exception as exc:  # doctor must report, not crash
        ok = False
        typer.secho(f"jax       FAILED: {exc}", fg="red")

    for name in _VERSIONED:
        try:
            typer.echo(f"{name:<16}{metadata.version(name)}")
        except metadata.PackageNotFoundError:
            ok = False
            typer.secho(f"{name:<16}MISSING", fg="red")

    if os.environ.get("NEUPRINT_APPLICATION_CREDENTIALS"):
        typer.echo("neuprint  token found")
    else:
        typer.secho("neuprint  token missing (see .env.example)", fg="yellow")

    # TODO(M0 step 3): verify data/ checksums once `flyevo data fetch` exists.
    raise typer.Exit(code=0 if ok else 1)


@data_app.command("fetch")
def data_fetch(
    minimal: Annotated[bool, typer.Option(help="Only what tests and smoke need.")] = False,
) -> None:
    """Fetch the hemibrain olfactory subgraph into data/ and write SHA256 checksums."""
    _not_yet("M0 step 3")


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def evolve(ctx: typer.Context) -> None:
    """Run an experiment. Arguments are Hydra overrides, e.g. `experiment=smoke seed=1`."""
    _not_yet("M2-M3")


@app.command()
def analyze(run_dir: Path) -> None:
    """Mechanism-level metrics + figures for a finished run."""
    _not_yet("M3")


@app.command()
def viz(
    run_dir: Path,
    serve: Annotated[bool, typer.Option(help="Serve the Rerun web viewer.")] = False,
) -> None:
    """Open a run's viz.rrd in Rerun."""
    _not_yet("M0 step 4")
