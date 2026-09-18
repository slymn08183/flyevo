# syntax=docker/dockerfile:1
# Multi-stage: base → cpu, and a separate cuda target. Build context is the repo root.
#   docker build -f docker/Dockerfile --target cpu  -t flyevo:cpu  .
#   docker build -f docker/Dockerfile --target cuda -t flyevo:cuda .
# uv's download cache lives in a build cache mount, not in the image (saves ~2 GB).

# ---- base: uv + project ----
FROM python:3.13-slim AS base
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project
COPY src ./src
COPY configs ./configs
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev
ENTRYPOINT ["uv", "run", "--no-sync", "flyevo"]

# ---- cpu ----
FROM base AS cpu

# ---- cuda ----
FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu24.04 AS cuda
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_PYTHON=3.13
# uv-managed Python; do not depend on apt Python
RUN uv python install 3.13
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project --extra cuda
COPY src ./src
COPY configs ./configs
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --extra cuda
ENTRYPOINT ["uv", "run", "--no-sync", "flyevo"]
