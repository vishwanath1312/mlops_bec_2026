# ══════════════════════════════════════════════════════════════════════
#  Dockerfile — Churn Prediction API (Session 6)
#
#  BUILD:   docker build -t churn-api:1.0 .
#  RUN:     docker run -p 8000:8000 -e MLFLOW_TRACKING_URI=http://host.docker.internal:5001 churn-api:1.0
#
#  LAYER ORDER MATTERS — see the workshop Session 6 document for the
#  full explanation of why dependencies are copied before application code.
# ══════════════════════════════════════════════════════════════════════

# ── Layer 1: Base image ──────────────────────────────────────────────────
# python:3.11-slim is a minimal Debian-based image — ~50MB vs ~1GB for
# the full python:3.11 image. Always prefer slim/alpine for production.
FROM python:3.11-slim

# ── Layer 2: Working directory ───────────────────────────────────────────
WORKDIR /app

# ── Layer 3: System dependencies ─────────────────────────────────────────
# libgomp1 is required by XGBoost for OpenMP multi-threaded execution.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Layer 4: Python dependencies (THE CACHING TRICK) ─────────────────────
# Copy requirements.txt BEFORE any application code. This layer is only
# rebuilt when requirements.txt changes — not on every code edit.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Layer 5: Application code ────────────────────────────────────────────
# Copied AFTER dependencies. Code changes only invalidate this layer.
COPY api/ ./api/
COPY src/ ./src/

# ── Layer 6: Environment variables ───────────────────────────────────────
# host.docker.internal lets the container reach services on the host
# machine (e.g. MLflow running locally during the workshop).
ENV MLFLOW_TRACKING_URI=http://98.130.129.135:5001
ENV PYTHONPATH=/app

# ── Layer 7: Port documentation ──────────────────────────────────────────
# Documents the port — does NOT open it. That happens via `docker run -p`.
EXPOSE 8000

# ── Layer 8: Startup command ─────────────────────────────────────────────
# --host 0.0.0.0 is REQUIRED — binds to all interfaces so requests from
# outside the container (your browser, curl, etc.) can reach the API.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
