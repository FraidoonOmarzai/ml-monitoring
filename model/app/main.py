from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.health import router as health_router
from app.api.predict import router as predict_router
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.ml.predictor import get_predictor
from app.monitoring.metrics import get_metrics, setup_instrumentator

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: logging → model load → Prometheus gauges
    Shutdown: log message
    Replaces the deprecated @app.on_event("startup") pattern.
    """
    setup_logging()
    settings = get_settings()
    log.info(
        "Starting up",
        extra={"env": settings.app_env, "version": settings.model_version},
    )

    predictor = get_predictor()
    predictor.load()

    if settings.enable_metrics:
        m = get_metrics()
        m.model_load_seconds.set(predictor.load_duration)
        m.model_info.labels(version=predictor.model_version, env=settings.app_env).set(
            1
        )

    log.info("Ready to serve", extra={"model_version": predictor.model_version})

    yield

    log.info("Shutting down")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.model_version,
        description=(
            "Binary classifier for heart disease risk. "
            "Monitored with Prometheus + Grafana, deployed on Kubernetes."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    origins = ["*"] if not settings.is_production else []
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    setup_instrumentator(app)

    app.include_router(health_router)
    app.include_router(predict_router)

    @app.exception_handler(404)
    async def not_found(request, exc):
        return JSONResponse(
            status_code=404,
            content={"detail": f"Route '{request.url.path}' not found"},
        )

    return app


app = create_app()
