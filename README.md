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

---

---

## 2 Docker

```bash
Containerise the service — multi-stage build, non-root user, local run + verify
Dockerfile  | multi-stage build  | docker compose  | HEALTHCHECK  | layer caching
```

### files

```bash
heart-disease-mlops/
├── model/
│   └── Dockerfile           ← multi-stage build
├── docker-compose.yml       ← run everything locally with one command
├── docker-compose.override.yml  ← dev overrides (hot reload, mounts)
├── .dockerignore            ← keep images lean
└── .env.example             ← document all env vars
```

### steps

```bash
- Add Dockerfile
- Add .dockerignore
- Add docker-compose.yml and docker-compose.override.yml
- Add monitoring/prometheus.yml, monitoring/grafana/provisioning/datasources/prometheus.yml and monitoring/grafana/provisioning/dashboards/default.yml
- Add dashboards/model.json
- Add .env.example
- Update gitignore

```

### How To Run

```bash
# From heart-disease-mlops/ root

# ── Option A: production mode (full build — trains model inside Docker) ───────
docker compose -f docker-compose.yml up --build

# ── Option B: dev mode (hot reload, uses your local .pkl) ─────────────────────
# First make sure you have already run: cd model && python train.py
docker compose up --build

# ── Verify everything is running ──────────────────────────────────────────────
docker compose ps
# model       → running  (port 8080)
# prometheus  → running  (port 9090)
# grafana     → running  (port 3000)

# ── Test the API inside Docker ────────────────────────────────────────────────
curl http://localhost:8080/health
curl http://localhost:8080/metrics
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,
       "restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,
       "slope":0,"ca":0,"thal":1}'

# ── Open Grafana ──────────────────────────────────────────────────────────────
open http://localhost:3000   # admin / admin

# ── Check Prometheus scraped the model ────────────────────────────────────────
open http://localhost:9090/targets
# Should show: heart-disease-model → UP

# ── Stop everything ───────────────────────────────────────────────────────────
docker compose down

# ── Stop and remove volumes (wipe Prometheus + Grafana data) ──────────────────
docker compose down -v
```

`Phase 2 in one sentence: Package the Phase 1 model into Docker so it runs identically anywhere, then use docker compose to start it alongside Prometheus (metrics collector) and Grafana (dashboard) with one command.`

### The three things you actually need to understand:

- Dockerfile has two stages. Stage 1 is a throwaway — it installs heavy libraries and trains the model. Stage 2 is what ships to production — it only has the model file and the serving code. The final image is lean and secure because it never contains training code or raw data.
- docker-compose.yml is the glue. It starts all three containers on the same internal network so Prometheus can reach the model by typing model:80 (the service name), and Grafana can reach Prometheus by typing prometheus:9090. You only expose ports 8080, 9090, and 3000 to your laptop.
- docker-compose.override.yml is dev mode. Docker automatically merges it when you run docker compose up. It mounts your source code as a live volume and enables --reload, so you save a file and the change is immediately live inside the container. Skip it with docker compose -f docker-compose.yml up when you want production behaviour.

### The one command to remember:

```bash
cd model && python train.py   # generate model.pkl first
cd ..
docker compose up --build     # starts everything
```

- Then open localhost:8080/docs to test the API, localhost:9090/targets to confirm Prometheus is scraping, and localhost:3000 for Grafana (admin / admin).
---
---
