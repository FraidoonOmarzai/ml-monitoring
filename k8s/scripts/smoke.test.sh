#!/usr/bin/env bash
# =============================================================================
# kubernetes/scripts/smoke_test.sh
# Post-deploy smoke test — validates the model API is responding correctly.
#
# Usage:
#   chmod +x kubernetes/scripts/smoke_test.sh
#   ./kubernetes/scripts/smoke_test.sh http://localhost:8080
# =============================================================================

set -euo pipefail

BASE_URL="${1:-http://localhost:8080}"
PASS=0; FAIL=0; ERRORS=()

GREEN='\033[0;32m'; RED='\033[0;31m'; NC='\033[0m'
pass() { echo -e "  ${GREEN}✓${NC}  $1"; ((PASS++)) || true; }
fail() { echo -e "  ${RED}✗${NC}  $1"; ((FAIL++)) || true; ERRORS+=("$1"); }

check_contains() {
  local name="$1" expected="$2" actual="$3"
  if echo "$actual" | grep -q "$expected"; then pass "$name"; else fail "$name — expected '$expected' not found"; fi
}
check_status() {
  local name="$1" expected="$2" actual="$3"
  if [ "$actual" = "$expected" ]; then pass "$name"; else fail "$name — expected HTTP $expected, got HTTP $actual"; fi
}

VALID_PAYLOAD='{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'
INVALID_PAYLOAD='{"age": -999}'

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Smoke test → $BASE_URL"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "GET /health"
HEALTH=$(curl -sf "$BASE_URL/health" 2>/dev/null || echo "CONNECTION_FAILED")
check_contains "status is ok"           '"status":"ok"'       "$HEALTH"
check_contains "model_loaded is true"   '"model_loaded":true' "$HEALTH"
check_contains "model_version present"  '"model_version"'     "$HEALTH"
echo ""

echo "GET /ready"
READY=$(curl -sf "$BASE_URL/ready" 2>/dev/null || echo "CONNECTION_FAILED")
check_contains "ready is true"         '"ready":true'        "$READY"
check_contains "model_loaded check"    '"model_loaded":true' "$READY"
echo ""

echo "POST /predict (valid payload)"
PRED_STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE_URL/predict" -H "Content-Type: application/json" -d "$VALID_PAYLOAD")
PRED_BODY=$(curl -sf -X POST "$BASE_URL/predict" -H "Content-Type: application/json" -d "$VALID_PAYLOAD" 2>/dev/null || echo "CONNECTION_FAILED")
check_status   "returns HTTP 200"           "200"             "$PRED_STATUS"
check_contains "prediction field present"   '"prediction"'    "$PRED_BODY"
check_contains "probability field present"  '"probability"'   "$PRED_BODY"
check_contains "request_id field present"   '"request_id"'    "$PRED_BODY"
echo ""

echo "POST /predict (invalid payload)"
BAD_STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE_URL/predict" -H "Content-Type: application/json" -d "$INVALID_PAYLOAD")
check_status "returns HTTP 422 for bad input" "422" "$BAD_STATUS"
echo ""

echo "GET /metrics"
METRICS=$(curl -sf "$BASE_URL/metrics" 2>/dev/null || echo "CONNECTION_FAILED")
check_contains "prometheus format"                  "# HELP"                               "$METRICS"
check_contains "prediction probability histogram"   "heart_disease_prediction_probability" "$METRICS"
check_contains "prediction label counter"           "heart_disease_prediction_label_total" "$METRICS"
check_contains "http requests counter"              "http_requests_total"                  "$METRICS"
echo ""

echo "GET /does-not-exist"
NOT_FOUND=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/does-not-exist")
check_status "returns HTTP 404" "404" "$NOT_FOUND"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "  ${GREEN}Passed: $PASS${NC}   $([ $FAIL -gt 0 ] && echo -e "${RED}Failed: $FAIL${NC}" || echo "Failed: 0")"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ "${#ERRORS[@]}" -gt 0 ]; then
  echo ""; echo "  Failures:"
  for err in "${ERRORS[@]}"; do echo -e "    ${RED}•${NC} $err"; done
  echo ""; exit 1
fi
echo ""; echo "  All checks passed ✓"; echo ""