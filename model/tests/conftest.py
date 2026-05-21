from __future__ import annotations

import os
from pathlib import Path

# Resolve absolute paths so tests work from any working directory
_MODEL_DIR = Path(__file__).parent.parent  # → heart-disease-mlops/model/

os.environ.setdefault("MODEL_PATH", str(_MODEL_DIR / "app" / "model.pkl"))
os.environ.setdefault(
    "MODEL_METADATA_PATH", str(_MODEL_DIR / "app" / "model_metadata.json")
)
os.environ.setdefault("ENABLE_METRICS", "true")
os.environ.setdefault("APP_ENV", "development")
os.environ.setdefault("LOG_FORMAT", "text")

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture(scope="session")
def client():
    """Full FastAPI app with lifespan — model loads once for the whole session."""
    app = create_app()
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


@pytest.fixture
def valid_payload() -> dict:
    return {
        "age": 63,
        "sex": 1,
        "cp": 3,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 0,
        "ca": 0,
        "thal": 1,
    }


@pytest.fixture
def low_risk_payload() -> dict:
    return {
        "age": 35,
        "sex": 0,
        "cp": 0,
        "trestbps": 118,
        "chol": 195,
        "fbs": 0,
        "restecg": 0,
        "thalach": 180,
        "exang": 0,
        "oldpeak": 0.0,
        "slope": 2,
        "ca": 0,
        "thal": 2,
    }
