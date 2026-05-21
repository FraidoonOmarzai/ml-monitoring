from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # App
    app_name: str = "Heart Disease Prediction API"
    app_env: Literal["development", "staging", "production"] = "development"
    model_version: str = "1.0.0"

    # Logging
    log_level: str = "INFO"
    log_format: Literal["json", "text"] = "json"

    # Model artifact paths
    model_path: Path = Path("app/model.pkl")
    model_metadata_path: Path = Path("app/model_metadata.json")

    # Prometheus
    enable_metrics: bool = True
    metrics_prefix: str = "heart_disease"

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()
