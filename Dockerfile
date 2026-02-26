# ============================================================
# Project 12: Multi-Modal RAG Dashboard
# Multi-stage Docker build
# ============================================================

# ── Stage 1: Builder ─────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install uv for fast dependency resolution
RUN pip install uv --no-cache-dir

# Copy dependency files first (layer cache optimization)
COPY pyproject.toml .
COPY requirements.txt .

# Install all dependencies into /build/.venv
RUN uv venv .venv && \
    uv pip install -r requirements.txt

# ── Stage 2: Runtime ─────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Security: run as non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /build/.venv /app/.venv

# Copy application code
COPY config.py          .
COPY main.py            .
COPY ingestion/         ingestion/
COPY retrieval/         retrieval/
COPY generation/        generation/
COPY evaluation/        evaluation/
COPY dashboard/         dashboard/

# Create data directories
RUN mkdir -p data/sample_docs data/uploads data/eval_dataset && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Activate venv
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Port Azure Web App expects
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Start FastAPI with uvicorn
CMD ["python", "-m", "uvicorn", "dashboard.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info"]