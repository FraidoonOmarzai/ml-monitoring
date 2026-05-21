from __future__ import annotations

import logging

from prometheus_client import REGISTRY, Counter, Gauge, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

from app.core.config import get_settings

log = logging.getLogger(__name__)


# ── Safe registration helpers (idempotent across test runs) ───────────────────


def _histogram(name: str, doc: str, buckets: list) -> Histogram:
    try:
        return Histogram(name, doc, buckets=buckets)
    except ValueError:
        return REGISTRY._names_to_collectors[name]


def _counter(name: str, doc: str, labels: list | None = None) -> Counter:
    try:
        return Counter(name, doc, labelnames=labels or [])
    except ValueError:
        return REGISTRY._names_to_collectors[name]


def _gauge(name: str, doc: str, labels: list | None = None) -> Gauge:
    try:
        return Gauge(name, doc, labelnames=labels or [])
    except ValueError:
        return REGISTRY._names_to_collectors[name]


# ── Metric definitions ────────────────────────────────────────────────────────


class MLMetrics:
    def __init__(self, prefix: str) -> None:

        # Output distribution — primary signal for model drift
        self.prediction_probability = _histogram(
            f"{prefix}_prediction_probability",
            "Distribution of model output probability scores",
            buckets=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
        )

        self.prediction_label_total = _counter(
            f"{prefix}_prediction_label_total",
            "Cumulative predictions split by class label",
            labels=["label"],  # "0" or "1"
        )

        # Input feature distributions — early warning of data drift
        self.feature_age = _histogram(
            f"{prefix}_feature_age_years",
            "Distribution of incoming patient age",
            buckets=[20, 30, 40, 50, 55, 60, 65, 70, 75, 80, 90],
        )
        self.feature_trestbps = _histogram(
            f"{prefix}_feature_trestbps_mmhg",
            "Distribution of resting blood pressure",
            buckets=[90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 200],
        )
        self.feature_chol = _histogram(
            f"{prefix}_feature_chol_mgdl",
            "Distribution of serum cholesterol",
            buckets=[150, 175, 200, 225, 250, 275, 300, 350, 400, 500],
        )
        self.feature_thalach = _histogram(
            f"{prefix}_feature_thalach_bpm",
            "Distribution of maximum heart rate achieved",
            buckets=[80, 100, 110, 120, 130, 140, 150, 160, 170, 180, 200],
        )

        # Model lifecycle
        self.model_load_seconds = _gauge(
            f"{prefix}_model_load_duration_seconds",
            "Time to load the model artifact from disk",
        )
        self.model_info = _gauge(
            f"{prefix}_model_info",
            "Static model metadata (always 1)",
            labels=["version", "env"],
        )


_metrics: MLMetrics | None = None


def get_metrics() -> MLMetrics:
    global _metrics
    if _metrics is None:
        _metrics = MLMetrics(prefix=get_settings().metrics_prefix)
    return _metrics


def setup_instrumentator(app) -> None:
    """Attach prometheus-fastapi-instrumentator — adds /metrics endpoint."""
    if not get_settings().enable_metrics:
        log.info("Metrics disabled — skipping instrumentator setup")
        return

    Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        should_respect_env_var=True,
        env_var_name="ENABLE_METRICS",
        excluded_handlers=["/health", "/ready", "/metrics"],
    ).instrument(app).expose(app, endpoint="/metrics")

    log.info("Prometheus metrics exposed at /metrics")


def record_prediction(request, probability: float, prediction: int) -> None:
    """Call once per successful /predict to update ML metrics."""
    m = get_metrics()
    m.prediction_probability.observe(probability)
    m.prediction_label_total.labels(label=str(prediction)).inc()
    m.feature_age.observe(request.age)
    m.feature_trestbps.observe(request.trestbps)
    m.feature_chol.observe(request.chol)
    m.feature_thalach.observe(request.thalach)
