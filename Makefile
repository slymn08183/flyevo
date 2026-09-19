.DEFAULT_GOAL := help
COMPOSE := docker compose -f docker/compose.yaml

.PHONY: help setup test lint format data evolve viz docker mlflow clean

help:  ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

setup:  ## uv sync + pre-commit + minimal data fetch
	uv sync --all-extras
	uv run pre-commit install
	uv run flyevo data fetch --minimal

test:  ## pytest (unit + property + golden)
	uv run pytest

lint:  ## ruff + mypy
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

format:  ## ruff autofix + format
	uv run ruff check --fix .
	uv run ruff format .

data:  ## fetch + checksum connectome data
	uv run flyevo data fetch

evolve:  ## 2-min CPU smoke run
	uv run flyevo evolve experiment=smoke

viz:  ## serve latest run in the Rerun web viewer (RUN=runs/<...>)
	uv run flyevo viz $(RUN) --serve

docker:  ## build the GPU image (Docker Desktop + NVIDIA driver)
	$(COMPOSE) build flyevo

mlflow:  ## start the MLflow UI on :5000
	$(COMPOSE) up -d mlflow

clean:  ## remove caches (keeps data/ and runs/)
	rm -rf .pytest_cache .mypy_cache .ruff_cache .hypothesis build dist
