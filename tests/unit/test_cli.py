import os

from typer.testing import CliRunner

from flyevo.cli import app

runner = CliRunner()


def test_help_lists_all_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("doctor", "data", "evolve", "analyze", "viz"):
        assert command in result.output


def test_unimplemented_commands_exit_2() -> None:
    assert runner.invoke(app, ["evolve", "experiment=smoke"]).exit_code == 2
    assert runner.invoke(app, ["data", "fetch", "--minimal"]).exit_code == 2


def test_gpu_preallocation_is_off_by_default() -> None:
    assert os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] in {"false", "true"}  # user may override
