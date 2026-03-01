# ==============================================================================
# NyxClaw - CPU-Only Dockerfile
# ==============================================================================
# Multi-stage build optimized for ONNX CPU inference.
#
# Build:
#   docker build -t nyxclaw .
#   docker build --build-arg INSTALL_LOCAL_VOICE=true -t nyxclaw:local-voice .
#
# Run:
#   docker run -p 8080:8080 --env-file .env nyxclaw
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Base image with non-root user
# ------------------------------------------------------------------------------
FROM python:3.10-slim AS base

ARG INSTALL_LOCAL_VOICE=false

# Prevent interactive prompts and set Python to not buffer output
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies.
# libespeak-ng1  — espeak-ng shared library required by piper-tts for
#                  phonemization (piper-phonemize dlopen()s it at runtime).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libespeak-ng1 \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Create non-root user early so all subsequent files are owned by them
RUN useradd --create-home --shell /bin/bash appuser

# Set working directory and give ownership to appuser
WORKDIR /app
RUN chown appuser:appuser /app

# Install uv for fast package management (used throughout)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Download pretrained models during the build so the image starts cold-free.
# uv run --with keeps huggingface_hub isolated to a temporary env — no install/uninstall needed.
RUN mkdir -p pretrained_models/wav2arkit \
    && uv run --with huggingface_hub python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('myned-ai/wav2arkit_cpu', local_dir='pretrained_models/wav2arkit')" \
    && if [ "$INSTALL_LOCAL_VOICE" = "true" ]; then \
        mkdir -p pretrained_models/piper pretrained_models/faster_whisper_small_en \
        && uv run --with huggingface_hub python -c "\
import shutil; \
from huggingface_hub import hf_hub_download; \
[shutil.copy2( \
    hf_hub_download('rhasspy/piper-voices', f'en_US-hfc_female-medium.onnx{s}', \
        subfolder='en/en_US/hfc_female/medium'), \
    f'pretrained_models/piper/en_US-hfc_female-medium.onnx{s}') \
 for s in ('', '.json')]" \
        && uv run --with 'faster-whisper>=1.1.0' python -c "\
from faster_whisper.utils import download_model; \
download_model('small.en', output_dir='pretrained_models/faster_whisper_small_en')" \
        && rm -rf /root/.cache/huggingface \
        && python -c "\
import urllib.request, pathlib; \
url = 'https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx'; \
dest = pathlib.Path('pretrained_models/silero_vad.onnx'); \
urllib.request.urlretrieve(url, dest); \
print(f'silero_vad.onnx downloaded: {dest.stat().st_size} bytes')"; \
    fi \
    && chown -R appuser:appuser pretrained_models 2>/dev/null || true

# ------------------------------------------------------------------------------
# Stage 2: Dependencies installation with uv (as appuser)
# ------------------------------------------------------------------------------
FROM base AS dependencies

ARG INSTALL_LOCAL_VOICE=false

# uv already installed in base stage

# Switch to non-root user BEFORE installing dependencies
# This ensures .venv is owned by appuser from the start
USER appuser

# Copy dependency files first for better caching
COPY --chown=appuser:appuser pyproject.toml README.md ./

# Install Python dependencies and clean uv cache in same layer
RUN if [ "$INSTALL_LOCAL_VOICE" = "true" ]; then \
        (uv sync --frozen --no-dev --extra local_voice || uv sync --no-dev --extra local_voice); \
    else \
        (uv sync --frozen --no-dev || uv sync --no-dev); \
    fi \
    && rm -rf ~/.cache/uv

# ------------------------------------------------------------------------------
# Stage 3: Production image
# ------------------------------------------------------------------------------
FROM dependencies AS production

# Copy application code (already running as appuser)
COPY --chown=appuser:appuser ./src ./src

# Environment variables
ENV SERVER_HOST=0.0.0.0 \
    SERVER_PORT=8080 \
    PYTHONPATH=/app/src

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

# Start the server
CMD ["uv", "run", "--frozen", "--no-dev", "python", "src/main.py"]

# ------------------------------------------------------------------------------
# Stage 4: Development image (with hot reload)
# ------------------------------------------------------------------------------
FROM dependencies AS development

# Copy application code
COPY --chown=appuser:appuser ./src ./src

# Mount point for source code (overrides the COPY above when mounted)
VOLUME ["/app"]

# Environment variables for development
ENV DEBUG=true \
    SERVER_HOST=0.0.0.0 \
    SERVER_PORT=8080 \
    PYTHONPATH=/app/src

# Expose port
EXPOSE 8080

# Start with hot reload
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]
