# ────────────────────────────────────────────────────────────────────────────
# Biotech Clinical Trials Simulator — Dockerfile
#
# Multi-stage build:
#   Stage 1 (builder): Install Python deps into a venv
#   Stage 2 (runtime): Copy venv + app code — lean final image
#
# Build: docker build -t biotech-sim .
# Run:   docker run -p 8000:8000 -p 8265:8265 biotech-sim
# ────────────────────────────────────────────────────────────────────────────

# ── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps needed for scientific Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtualenv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Minimal runtime system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtualenv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application source
COPY src/ ./src/
COPY configs/ ./configs/
COPY pyproject.toml .

# Create non-root user for security
RUN useradd -m -u 1000 simuser && chown -R simuser:simuser /app
USER simuser

# ── Runtime configuration ─────────────────────────────────────────────────────

# Ray dashboard: 8265
# FastAPI:       8000
# MLflow:        5000 (separate service in docker-compose)
EXPOSE 8000 8265

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default: start FastAPI server
# Ray will auto-init in local mode on first request
CMD ["python", "-m", "uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--log-level", "info"]
