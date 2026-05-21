#!/usr/bin/env bash
# =============================================================================
# kubernetes/scripts/setup.sh
# One-shot script: start minikube → build image → load → apply manifests
#
# Usage:
#   chmod +x kubernetes/scripts/setup.sh
#   ./kubernetes/scripts/setup.sh
# =============================================================================

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

IMAGE_NAME="heart-disease-model"
IMAGE_TAG="1.0.0"
NAMESPACE="model-serving"
DEPLOY_NAME="heart-disease-model"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ── Pre-flight checks ──────────────────────────────────────────────────────────
info "Running pre-flight checks..."
command -v minikube >/dev/null 2>&1 || error "minikube not found. Install: brew install minikube"
command -v kubectl  >/dev/null 2>&1 || error "kubectl not found.  Install: brew install kubectl"
command -v docker   >/dev/null 2>&1 || error "docker not found.   Install Docker Desktop"

MODEL_PKL="$PROJECT_ROOT/model/app/model.pkl"
if [ ! -f "$MODEL_PKL" ]; then
  error "Model artifact not found at $MODEL_PKL\nRun first:  cd model && python train.py"
fi
success "Pre-flight checks passed"

# ── Step 1: Start minikube ─────────────────────────────────────────────────────
info "Starting minikube..."
if minikube status | grep -q "Running"; then
  success "minikube already running"
else
  minikube start \
    --driver=docker \
    --memory=4096 \
    --cpus=4 \
    --nodes=2
  success "minikube started"
fi

info "Enabling metrics-server addon..."
minikube addons enable metrics-server
success "metrics-server enabled"

# ── Step 2: Build Docker image ─────────────────────────────────────────────────
info "Building Docker image ${IMAGE_NAME}:${IMAGE_TAG}..."
docker build \
  --build-arg MODEL_VERSION="${IMAGE_TAG}" \
  --build-arg BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --build-arg GIT_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)" \
  -t "${IMAGE_NAME}:${IMAGE_TAG}" \
  -f "$PROJECT_ROOT/model/Dockerfile" \
  "$PROJECT_ROOT/model/"
success "Image built: ${IMAGE_NAME}:${IMAGE_TAG}"

# ── Step 3: Load image into minikube ──────────────────────────────────────────
# minikube has its own Docker daemon — must load the image explicitly
info "Loading image into minikube..."
minikube image load "${IMAGE_NAME}:${IMAGE_TAG}"
success "Image loaded into minikube"

# ── Step 4: Apply Kubernetes manifests ────────────────────────────────────────
info "Applying Kubernetes manifests..."

kubectl apply -f "$PROJECT_ROOT/kubernetes/namespace.yaml"
success "Namespace: $NAMESPACE"

kubectl apply -f "$PROJECT_ROOT/kubernetes/model/deployment.yaml"
success "Deployment: $DEPLOY_NAME"

kubectl apply -f "$PROJECT_ROOT/kubernetes/model/service.yaml"
success "Service: $DEPLOY_NAME"

kubectl apply -f "$PROJECT_ROOT/kubernetes/model/hpa.yaml"
success "HPA: ${DEPLOY_NAME}-hpa"

# ServiceMonitor requires Prometheus Operator CRD (installed in Phase 4)
if kubectl get crd servicemonitors.monitoring.coreos.com >/dev/null 2>&1; then
  kubectl apply -f "$PROJECT_ROOT/kubernetes/model/service_monitor.yaml"
  success "ServiceMonitor: $DEPLOY_NAME"
else
  warn "ServiceMonitor CRD not found — skipping (will apply in Phase 4)"
fi

# ── Step 5: Wait for rollout ───────────────────────────────────────────────────
info "Waiting for deployment rollout (timeout: 120s)..."
kubectl rollout status deployment/"$DEPLOY_NAME" \
  -n "$NAMESPACE" \
  --timeout=120s
success "Deployment is ready"

# ── Step 6: Show status ────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}  Phase 3 deploy complete${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
kubectl get pods -n "$NAMESPACE"
echo ""
kubectl get svc   -n "$NAMESPACE"
echo ""
echo "  Port-forward the model service:"
echo "    kubectl port-forward svc/${DEPLOY_NAME} 8080:80 -n ${NAMESPACE}"
echo ""
echo "  Then run the smoke test:"
echo "    ./kubernetes/scripts/smoke_test.sh http://localhost:8080"
echo ""