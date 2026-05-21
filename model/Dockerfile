# =============================================================================
# Stage 1 — trainer
# Heavy image: downloads data, trains the model, produces app/model.pkl
# This stage is NOT shipped to production — only the artifact is copied out.
# =============================================================================
FROM python:3.11-slim AS trainer

WORKDIR /train

# Install deps first so this layer is cached unless requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train.py .

# Run training — produces app/model.pkl and app/model_metadata.json
RUN python train.py


# =============================================================================
# Stage 2 — runtime
# Lean image: only what is needed to serve predictions.
# Does NOT contain training code, raw data, or heavy build tools.
# =============================================================================
FROM python:3.11-slim AS runtime

# ── Security: run as non-root ─────────────────────────────────────────────────
RUN groupadd --gid 1001 appgroup \
    && useradd  --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

WORKDIR /app

# ── Dependencies ──────────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ──────────────────────────────────────────────────────────
COPY app/ ./app/

# ── Model artifacts from trainer stage ───────────────────────────────────────
COPY --from=trainer /train/app/model.pkl           ./app/model.pkl
COPY --from=trainer /train/app/model_metadata.json ./app/model_metadata.json

# ── Build-time metadata ───────────────────────────────────────────────────────
ARG MODEL_VERSION=dev
ARG BUILD_DATE=unknown
ARG GIT_SHA=unknown

LABEL org.opencontainers.image.version="${MODEL_VERSION}" \
    org.opencontainers.image.created="${BUILD_DATE}" \
    org.opencontainers.image.revision="${GIT_SHA}"

# ── Runtime environment ───────────────────────────────────────────────────────
ENV APP_ENV=production \
    ENABLE_METRICS=true \
    LOG_FORMAT=json \
    MODEL_VERSION=${MODEL_VERSION} \
    MODEL_PATH=app/model.pkl \
    MODEL_METADATA_PATH=app/model_metadata.json \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# ── Ownership + drop privileges ───────────────────────────────────────────────
RUN chown -R appuser:appgroup /app
USER appuser

EXPOSE 80

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c \
    "import urllib.request; urllib.request.urlopen('http://localhost:80/health')" \
    || exit 1

# ── Start server ──────────────────────────────────────────────────────────────
CMD ["uvicorn", "app.main:app", \
    "--host", "0.0.0.0", \
    "--port", "80", \
    "--workers", "1", \
    "--log-level", "warning"]