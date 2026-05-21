from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, status

from app.ml.predictor import ModelPredictor, get_predictor
from app.ml.schemas import PredictionRequest, PredictionResponse
from app.monitoring.metrics import record_prediction

log = logging.getLogger(__name__)
router = APIRouter(tags=["prediction"])


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict heart disease",
    description=(
        "Accepts 13 patient features, returns binary prediction (0/1) "
        "and a probability score. Metrics recorded to Prometheus."
    ),
)
def predict(
    payload: PredictionRequest,
    predictor: ModelPredictor = Depends(get_predictor),
) -> PredictionResponse:
    request_id = str(uuid.uuid4())

    try:
        result = predictor.predict(payload)
    except RuntimeError as exc:
        log.error("Model unavailable  request_id=%s  error=%s", request_id, exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not ready",
        ) from exc
    except Exception as exc:
        log.exception("Unexpected prediction error  request_id=%s", request_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed",
        ) from exc

    record_prediction(payload, result["probability"], result["prediction"])

    log.info(
        "Prediction served",
        extra={
            "request_id": request_id,
            "prediction": result["prediction"],
            "probability": result["probability"],
        },
    )

    return PredictionResponse(**result, request_id=request_id)
