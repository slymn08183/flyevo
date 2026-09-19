# syntax=docker/dockerfile:1
# Single GPU image. Build context is the repo root:
#   docker compose -f docker/compose.yaml build flyevo
#
# No nvidia/cuda base on purpose: the jax[cuda12] wheels ship their own CUDA + cuDNN libraries,
# so a slim Python base is enough (saves ~3 GB). Only the host driver is needed, injected by the
# NVIDIA container runtime (`--gpus all` / compose `devices`).
# uv's download cache lives in a build cache mount, not in the image (saves ~2 GB).

FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 \
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project --extra cuda
COPY src ./src
COPY configs ./configs
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --extra cuda
# Allocate GPU memory on demand: JAX's default (grab 75% up front) fails on small or
# desktop-shared GPUs. Set here, not only in cli.py, so scripts/ and `python -c` get it too.
# Kept last so changing it never invalidates the dependency layers. Override with `-e ...=true`.
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
ENTRYPOINT ["uv", "run", "--no-sync", "flyevo"]
