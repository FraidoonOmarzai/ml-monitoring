<h1 align=center> ML-Monitoring (Prometheus+Grafana) </h1>


## 1 ML model + local API
```bash
Train, serve, and test the model locally — everything works before adding infrastructure
train.py  | FastAPI  | Pydantic v2  | Prometheus metrics  | pytest 18/18  | uv / venv  | smoke test
```

### Steps
```bash
- Define structure
- Add pyproject.toml, requirements.txt and create venv
- Add code to train.py
- Add code to core/config.py and core/logging.py
- Add code to ml/schemas.py and ml/predictor.py
- add code to monitoring/metrics.py
- Add code to api/health.py and api/predict.py
- Define main.py
- Add code to tests/conftest.py, tests/test_api.py and tests/test_predictor.py
```

### How To Run
```bash
# 1. Create the folder structure
mkdir -p heart-disease-mlops/model/app/{api,core,ml,monitoring}
mkdir -p heart-disease-mlops/model/tests

# 2. Copy each file above into its path

# 3. Create empty __init__.py files
touch heart-disease-mlops/model/app/__init__.py
touch heart-disease-mlops/model/app/api/__init__.py
touch heart-disease-mlops/model/app/core/__init__.py
touch heart-disease-mlops/model/app/ml/__init__.py
touch heart-disease-mlops/model/app/monitoring/__init__.py
touch heart-disease-mlops/model/tests/__init__.py

# 4. Install dependencies
cd heart-disease-mlops/model
pip install -r requirements.txt
pip install pytest httpx  # test-only deps

# 5. Train the model
python train.py

# 6. Run tests (from project root)
cd ..
python -m pytest model/tests/ -v

# 7. Start the API
cd model
uvicorn app.main:app --reload --port 8080

# 8. Try it
curl http://localhost:8080/health
curl http://localhost:8080/docs        # Swagger UI
curl http://localhost:8080/metrics     # Prometheus output
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'

```