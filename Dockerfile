# syntax=docker/dockerfile:1
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable


FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    DATABASE_PATH=/data/forest.db \
    ARCHIVE_DIR=/app/data/archive \
    ARCHIVE_INDEX_PATH=/app/data/archive_index.json

# Tesseract з українською — для PDF-сканів без текстового шару.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-ukr \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 forest \
    && mkdir /data && chown forest /data

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY data ./data

USER forest
VOLUME ["/data"]
CMD ["forest-bot"]
