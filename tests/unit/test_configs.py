from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
EXPERIMENTS = sorted(p.stem for p in (CONFIG_DIR / "experiment").glob("*.yaml"))
EVOS = sorted(p.stem for p in (CONFIG_DIR / "evo").glob("*.yaml"))
ENVS = sorted(p.stem for p in (CONFIG_DIR / "env").glob("*.yaml"))


def _compose(*overrides: str):  # type: ignore[no-untyped-def]
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        return compose(config_name="config", overrides=list(overrides))


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_every_experiment_composes(experiment: str) -> None:
    cfg = _compose(f"experiment={experiment}")
    assert cfg.experiment.name == experiment


@pytest.mark.parametrize("evo", EVOS)
@pytest.mark.parametrize("env", ENVS)
def test_every_evo_env_pair_composes(evo: str, env: str) -> None:
    cfg = _compose(f"evo={evo}", f"env={env}")
    assert cfg.evo.name == evo
    assert cfg.env.name == env


def test_smoke_overrides_population() -> None:
    cfg = _compose("experiment=smoke")
    assert cfg.evo.population_size == 16
    assert cfg.evo.generations == 20
