from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """13 clinical features from the UCI Heart Disease (Cleveland) dataset."""

    age: float = Field(..., ge=1, le=120, description="Age in years")
    sex: float = Field(..., ge=0, le=1, description="Sex — 1=male, 0=female")
    cp: float = Field(..., ge=0, le=3, description="Chest pain type (0–3)")
    trestbps: float = Field(
        ..., ge=80, le=220, description="Resting blood pressure mm Hg"
    )
    chol: float = Field(..., ge=100, le=600, description="Serum cholesterol mg/dl")
    fbs: float = Field(..., ge=0, le=1, description="Fasting blood sugar >120 mg/dl")
    restecg: float = Field(..., ge=0, le=2, description="Resting ECG results (0–2)")
    thalach: float = Field(..., ge=60, le=220, description="Max heart rate achieved")
    exang: float = Field(..., ge=0, le=1, description="Exercise-induced angina")
    oldpeak: float = Field(
        ..., ge=0, le=7, description="ST depression (exercise vs rest)"
    )
    slope: float = Field(
        ..., ge=0, le=2, description="Slope of peak exercise ST segment"
    )
    ca: float = Field(..., ge=0, le=4, description="Number of major vessels (0–4)")
    thal: float = Field(..., ge=0, le=3, description="Thalassemia type (0–3)")

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 = no disease, 1 = disease")
    probability: float = Field(..., ge=0.0, le=1.0)
    model_version: str
    request_id: str | None = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    app_env: str


class ReadinessResponse(BaseModel):
    ready: bool
    checks: dict[str, bool]
