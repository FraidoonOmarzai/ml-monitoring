from __future__ import annotations

import pytest

from app.ml.predictor import ModelPredictor
from app.ml.schemas import PredictionRequest


def _make_request(**overrides) -> PredictionRequest:
    base = {
        "age": 52,
        "sex": 1,
        "cp": 0,
        "trestbps": 125,
        "chol": 212,
        "fbs": 0,
        "restecg": 1,
        "thalach": 168,
        "exang": 0,
        "oldpeak": 1.0,
        "slope": 2,
        "ca": 2,
        "thal": 3,
    }
    return PredictionRequest(**{**base, **overrides})


@pytest.fixture(scope="module")
def predictor():
    p = ModelPredictor()
    p.load()
    return p


class TestModelLoading:
    def test_is_loaded_after_load(self, predictor):
        assert predictor.is_loaded is True

    def test_model_version_is_non_empty_string(self, predictor):
        assert isinstance(predictor.model_version, str)
        assert len(predictor.model_version) > 0

    def test_load_duration_is_positive(self, predictor):
        assert predictor.load_duration > 0

    def test_unloaded_predictor_raises_on_predict(self):
        p = ModelPredictor()
        with pytest.raises(RuntimeError, match="not loaded"):
            p.predict(_make_request())

    def test_missing_model_file_raises_file_not_found(self, tmp_path, monkeypatch):
        from app.core.config import Settings

        fake_settings = Settings(model_path=tmp_path / "nonexistent.pkl")
        monkeypatch.setattr("app.ml.predictor.get_settings", lambda: fake_settings)
        p = ModelPredictor()
        with pytest.raises(FileNotFoundError):
            p.load()


class TestPrediction:
    def test_returns_required_keys(self, predictor):
        result = predictor.predict(_make_request())
        assert {"prediction", "probability", "model_version"} <= result.keys()

    def test_prediction_is_binary(self, predictor):
        assert predictor.predict(_make_request())["prediction"] in (0, 1)

    def test_probability_in_unit_interval(self, predictor):
        p = predictor.predict(_make_request())["probability"]
        assert 0.0 <= p <= 1.0

    def test_probability_is_rounded_to_4dp(self, predictor):
        p = predictor.predict(_make_request())["probability"]
        assert p == round(p, 4)

    def test_model_version_matches_metadata(self, predictor):
        result = predictor.predict(_make_request())
        assert result["model_version"] == predictor.model_version

    def test_different_inputs_can_produce_different_outputs(self, predictor):
        high = predictor.predict(_make_request(age=70, cp=0, ca=3, thal=3))
        low = predictor.predict(_make_request(age=30, cp=2, ca=0, thal=2))
        assert isinstance(high["prediction"], int)
        assert isinstance(low["prediction"], int)
