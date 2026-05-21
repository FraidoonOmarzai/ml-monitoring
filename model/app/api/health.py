from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.config import Settings, get_settings
from app.ml.predictor import ModelPredictor, get_predictor
from app.ml.schemas import HealthResponse, ReadinessResponse

log = logging.getLogger(__name__)
router = APIRouter(tags=["ops"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    description="K8s livenessProbe target. Returns 200 while the process is alive.",
)
def health(
    settings: Settings = Depends(get_settings),
    predictor: ModelPredictor = Depends(get_predictor),
) -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=predictor.is_loaded,
        model_version=predictor.model_version,
        app_env=settings.app_env,
    )


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Readiness probe",
    description="K8s readinessProbe target. Returns 503 until model is loaded.",
)
def readiness(predictor: ModelPredictor = Depends(get_predictor)) -> ReadinessResponse:
    checks = {"model_loaded": predictor.is_loaded}
    all_ok = all(checks.values())

    if not all_ok:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"ready": False, "checks": checks},
        )

    return ReadinessResponse(ready=True, checks=checks)
