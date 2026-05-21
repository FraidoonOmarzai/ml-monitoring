from __future__ import annotations

import json
import logging
import time
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from app.core.config import get_settings
from app.ml.schemas import PredictionRequest

log = logging.getLogger(__name__)

FEATURE_ORDER = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]


class ModelPredictor:
    def __init__(self) -> None:
        self._pipeline: Pipeline | None = None
        self._metadata: dict[str, Any] = {}
        self._load_duration: float = 0.0

    def load(self) -> None:
        settings = get_settings()

        if not settings.model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at '{settings.model_path}'. "
                "Run:  python train.py"
            )

        t0 = time.perf_counter()
        self._pipeline = joblib.load(settings.model_path)
        self._load_duration = time.perf_counter() - t0

        if settings.model_metadata_path.exists():
            self._metadata = json.loads(settings.model_metadata_path.read_text())

        log.info(
            "Model loaded",
            extra={
                "version": self.model_version,
                "load_s": round(self._load_duration, 4),
                "path": str(settings.model_path),
            },
        )

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    @property
    def model_version(self) -> str:
        return self._metadata.get("model_version", get_settings().model_version)

    @property
    def load_duration(self) -> float:
        return self._load_duration

    def predict(self, request: PredictionRequest) -> dict[str, Any]:
        if not self._pipeline:
            raise RuntimeError("Model is not loaded. Call load() during app startup.")

        # Named DataFrame prevents sklearn feature-name warnings in logs
        features = pd.DataFrame(
            [[getattr(request, col) for col in FEATURE_ORDER]],
            columns=FEATURE_ORDER,
        )

        prediction: int = int(self._pipeline.predict(features)[0])
        probability: float = float(self._pipeline.predict_proba(features)[0][1])

        return {
            "prediction": prediction,
            "probability": round(probability, 4),
            "model_version": self.model_version,
        }


# Module-level singleton — shared across all requests
_predictor = ModelPredictor()


def get_predictor() -> ModelPredictor:
    """FastAPI dependency — returns the loaded predictor."""
    return _predictor
