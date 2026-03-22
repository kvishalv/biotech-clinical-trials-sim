#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────────────────
# start_cluster.sh
# One-click local Ray cluster startup + API server launch
#
# Usage:
#   chmod +x start_cluster.sh
#   ./start_cluster.sh                  # start everything
#   ./start_cluster.sh --api-only       # skip Ray, just start the API
#   ./start_cluster.sh --ray-only       # start Ray cluster only
#   ./start_cluster.sh --stop           # stop all services
# ────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

# ── Defaults ──────────────────────────────────────────────────────────────────
RAY_NUM_CPUS=${RAY_NUM_CPUS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}
API_PORT=${API_PORT:-8000}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8265}
MLFLOW_PORT=${MLFLOW_PORT:-5000}

# Parse args
API_ONLY=false
RAY_ONLY=false
STOP=false

for arg in "$@"; do
  case $arg in
    --api-only)  API_ONLY=true ;;
    --ray-only)  RAY_ONLY=true ;;
    --stop)      STOP=true ;;
    *) warn "Unknown argument: $arg" ;;
  esac
done

# ── Stop mode ─────────────────────────────────────────────────────────────────
if $STOP; then
  info "Stopping all services..."
  ray stop --force 2>/dev/null && success "Ray stopped" || warn "Ray was not running"
  pkill -f "uvicorn src.api.main:app" 2>/dev/null && success "API server stopped" || warn "API was not running"
  pkill -f "mlflow server" 2>/dev/null && success "MLflow stopped" || warn "MLflow was not running"
  exit 0
fi

# ── Pre-flight checks ─────────────────────────────────────────────────────────
info "Running pre-flight checks..."

command -v python3 >/dev/null 2>&1 || error "python3 not found. Install Python 3.11+"
command -v pip >/dev/null 2>&1 || error "pip not found"

PYTHON_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
info "Python version: $PYTHON_VER"

# Check required packages
python3 -c "import ray" 2>/dev/null || { warn "ray not installed. Installing..."; pip install "ray[default]>=2.10.0" -q; }
python3 -c "import fastapi" 2>/dev/null || { warn "fastapi not installed. Installing deps..."; pip install -r requirements.txt -q; }

success "Pre-flight checks passed"

# ── Start MLflow (background) ─────────────────────────────────────────────────
if ! $API_ONLY && ! $RAY_ONLY; then
  if lsof -i :${MLFLOW_PORT} >/dev/null 2>&1; then
    warn "MLflow already running on port ${MLFLOW_PORT}"
  else
    info "Starting MLflow tracking server on port ${MLFLOW_PORT}..."
    mkdir -p ./mlruns
    mlflow server \
      --backend-store-uri sqlite:///mlruns/mlflow.db \
      --default-artifact-root ./mlruns/artifacts \
      --host 0.0.0.0 \
      --port ${MLFLOW_PORT} \
      --workers 1 \
      &>/tmp/mlflow.log &
    MLFLOW_PID=$!
    sleep 2
    if kill -0 $MLFLOW_PID 2>/dev/null; then
      success "MLflow started (PID $MLFLOW_PID) → http://localhost:${MLFLOW_PORT}"
    else
      warn "MLflow failed to start. Check /tmp/mlflow.log"
    fi
  fi
fi

# ── Start Ray local cluster ───────────────────────────────────────────────────
if ! $API_ONLY; then
  if ray status >/dev/null 2>&1; then
    success "Ray cluster already running"
  else
    info "Starting Ray local cluster (${RAY_NUM_CPUS} CPUs)..."
    ray start \
      --head \
      --num-cpus="${RAY_NUM_CPUS}" \
      --dashboard-host="0.0.0.0" \
      --dashboard-port="${RAY_DASHBOARD_PORT}" \
      --include-dashboard=true \
      --disable-usage-stats

    sleep 3
    if ray status >/dev/null 2>&1; then
      success "Ray cluster started → dashboard: http://localhost:${RAY_DASHBOARD_PORT}"
    else
      error "Ray failed to start. Check logs with: ray logs"
    fi
  fi
fi

if $RAY_ONLY; then
  info "Ray-only mode — skipping API server."
  info ""
  info "Ray cluster is running:"
  ray status
  exit 0
fi

# ── Start FastAPI server ──────────────────────────────────────────────────────
info "Starting FastAPI simulation server on port ${API_PORT}..."

if lsof -i :${API_PORT} >/dev/null 2>&1; then
  warn "Port ${API_PORT} already in use. Stop existing service first or set API_PORT=<other>"
  exit 1
fi

# Set environment
export MLFLOW_TRACKING_URI="http://localhost:${MLFLOW_PORT}"
export RAY_ADDRESS="auto"

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Biotech Clinical Trials Simulator${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo -e "  API server:     ${BLUE}http://localhost:${API_PORT}${NC}"
echo -e "  API docs:       ${BLUE}http://localhost:${API_PORT}/docs${NC}"
echo -e "  Ray dashboard:  ${BLUE}http://localhost:${RAY_DASHBOARD_PORT}${NC}"
echo -e "  MLflow UI:      ${BLUE}http://localhost:${MLFLOW_PORT}${NC}"
echo ""
echo -e "  Quick test:"
echo -e "  ${YELLOW}curl -s -X POST http://localhost:${API_PORT}/simulate \\${NC}"
echo -e "  ${YELLOW}  -H 'Content-Type: application/json' \\${NC}"
echo -e "  ${YELLOW}  -d '{\"seed\":42,\"n_patients\":100,\"n_weeks\":12}' | python3 -m json.tool${NC}"
echo ""
echo -e "  Press Ctrl+C to stop the API server."
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo ""

# Start uvicorn in the foreground (so Ctrl+C stops it cleanly)
python3 -m uvicorn src.api.main:app \
  --host 0.0.0.0 \
  --port "${API_PORT}" \
  --workers 1 \
  --log-level info \
  --reload
