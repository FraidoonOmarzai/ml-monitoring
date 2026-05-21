from __future__ import annotations


class TestHealth:
    def test_returns_200(self, client):
        assert client.get("/health").status_code == 200

    def test_status_is_ok(self, client):
        assert client.get("/health").json()["status"] == "ok"

    def test_model_loaded_true(self, client):
        assert client.get("/health").json()["model_loaded"] is True

    def test_model_version_present(self, client):
        assert "model_version" in client.get("/health").json()

    def test_app_env_present(self, client):
        assert "app_env" in client.get("/health").json()


class TestReadiness:
    def test_returns_200_when_ready(self, client):
        assert client.get("/ready").status_code == 200

    def test_ready_true(self, client):
        assert client.get("/ready").json()["ready"] is True

    def test_checks_field_present(self, client):
        body = client.get("/ready").json()
        assert "checks" in body
        assert body["checks"]["model_loaded"] is True


class TestPredict:
    def test_returns_200(self, client, valid_payload):
        assert client.post("/predict", json=valid_payload).status_code == 200

    def test_response_has_required_fields(self, client, valid_payload):
        body = client.post("/predict", json=valid_payload).json()
        assert {
            "prediction",
            "probability",
            "model_version",
            "request_id",
        } <= body.keys()

    def test_prediction_is_binary(self, client, valid_payload):
        body = client.post("/predict", json=valid_payload).json()
        assert body["prediction"] in (0, 1)

    def test_probability_is_in_range(self, client, valid_payload):
        body = client.post("/predict", json=valid_payload).json()
        assert 0.0 <= body["probability"] <= 1.0

    def test_request_id_is_a_string(self, client, valid_payload):
        body = client.post("/predict", json=valid_payload).json()
        assert isinstance(body["request_id"], str)
        assert len(body["request_id"]) == 36

    def test_each_request_gets_unique_id(self, client, valid_payload):
        id1 = client.post("/predict", json=valid_payload).json()["request_id"]
        id2 = client.post("/predict", json=valid_payload).json()["request_id"]
        assert id1 != id2

    def test_missing_field_returns_422(self, client, valid_payload):
        payload = {k: v for k, v in valid_payload.items() if k != "age"}
        assert client.post("/predict", json=payload).status_code == 422

    def test_age_below_min_returns_422(self, client, valid_payload):
        assert (
            client.post("/predict", json={**valid_payload, "age": 0}).status_code == 422
        )

    def test_age_above_max_returns_422(self, client, valid_payload):
        assert (
            client.post("/predict", json={**valid_payload, "age": 999}).status_code
            == 422
        )

    def test_chol_below_min_returns_422(self, client, valid_payload):
        assert (
            client.post("/predict", json={**valid_payload, "chol": 10}).status_code
            == 422
        )

    def test_empty_body_returns_422(self, client):
        assert client.post("/predict", json={}).status_code == 422

    def test_low_risk_profile_accepted(self, client, low_risk_payload):
        r = client.post("/predict", json=low_risk_payload)
        assert r.status_code == 200
        assert r.json()["prediction"] in (0, 1)


class TestMetrics:
    def test_metrics_endpoint_returns_200(self, client):
        assert client.get("/metrics").status_code == 200

    def test_prediction_probability_histogram_present(self, client, valid_payload):
        client.post("/predict", json=valid_payload)
        assert "heart_disease_prediction_probability" in client.get("/metrics").text

    def test_prediction_label_counter_present(self, client, valid_payload):
        client.post("/predict", json=valid_payload)
        assert "heart_disease_prediction_label_total" in client.get("/metrics").text

    def test_feature_age_histogram_present(self, client, valid_payload):
        client.post("/predict", json=valid_payload)
        assert "heart_disease_feature_age_years" in client.get("/metrics").text

    def test_feature_chol_histogram_present(self, client, valid_payload):
        client.post("/predict", json=valid_payload)
        assert "heart_disease_feature_chol_mgdl" in client.get("/metrics").text

    def test_http_request_metrics_present(self, client):
        assert "http_requests_total" in client.get("/metrics").text


class TestNotFound:
    def test_unknown_route_returns_404(self, client):
        assert client.get("/does-not-exist").status_code == 404

    def test_404_body_has_detail(self, client):
        assert "detail" in client.get("/does-not-exist").json()
