#!/usr/bin/env bash

set -euo pipefail

BASE_CONFIG="/Users/tianjian/Source/NetForge/.stress_runs/cr35_140d_joint_metadata_topology_only_no_layered_maxent_micro_parametric_refit_10runs_100rep_rerun_20260327_132542.json"
DAYS=280
REPLICATES=10
PARALLEL_JOBS=8
START_TS=""
OUTPUT_ROOT=""
CONDA_ENV="gt"
POSTERIOR_PARTITION_SWEEPS=""
POSTERIOR_PARTITION_SWEEP_NITER=""
POSTERIOR_PARTITION_BETA=""
SEED=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-config)
      BASE_CONFIG="$2"
      shift 2
      ;;
    --days)
      DAYS="$2"
      shift 2
      ;;
    --replicates)
      REPLICATES="$2"
      shift 2
      ;;
    --parallel-jobs)
      PARALLEL_JOBS="$2"
      shift 2
      ;;
    --start-ts)
      START_TS="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --conda-env)
      CONDA_ENV="$2"
      shift 2
      ;;
    --posterior-partition-sweeps)
      POSTERIOR_PARTITION_SWEEPS="$2"
      shift 2
      ;;
    --posterior-partition-sweep-niter)
      POSTERIOR_PARTITION_SWEEP_NITER="$2"
      shift 2
      ;;
    --posterior-partition-beta)
      POSTERIOR_PARTITION_BETA="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    -h|--help)
      cat <<'EOF'
Usage: run_cr35_partition_factor_grid_refresh.sh [options]

Options:
  --base-config PATH
  --days INT
  --replicates INT
  --parallel-jobs INT
  --start-ts INT
  --output-root PATH
  --conda-env NAME
  --posterior-partition-sweeps INT
  --posterior-partition-sweep-niter INT
  --posterior-partition-beta FLOAT
  --seed INT
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HELPER="$SCRIPT_DIR/partition_factor_grid_refresh.py"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/netforge-mpl}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/netforge-cache}"

PLAN_CMD=(
  conda run --no-capture-output -n "$CONDA_ENV" python "$HELPER" plan
  --base-config "$BASE_CONFIG"
  --days "$DAYS"
  --replicates "$REPLICATES"
)

if [[ -n "$OUTPUT_ROOT" ]]; then
  PLAN_CMD+=(--output-root "$OUTPUT_ROOT")
fi
if [[ -n "$START_TS" ]]; then
  PLAN_CMD+=(--start-ts "$START_TS")
fi
if [[ -n "$POSTERIOR_PARTITION_SWEEPS" ]]; then
  PLAN_CMD+=(--posterior-partition-sweeps "$POSTERIOR_PARTITION_SWEEPS")
fi
if [[ -n "$POSTERIOR_PARTITION_SWEEP_NITER" ]]; then
  PLAN_CMD+=(--posterior-partition-sweep-niter "$POSTERIOR_PARTITION_SWEEP_NITER")
fi
if [[ -n "$POSTERIOR_PARTITION_BETA" ]]; then
  PLAN_CMD+=(--posterior-partition-beta "$POSTERIOR_PARTITION_BETA")
fi
if [[ -n "$SEED" ]]; then
  PLAN_CMD+=(--seed "$SEED")
fi

RUN_ROOT="$("${PLAN_CMD[@]}")"
echo "Run root: $RUN_ROOT"

COMBO_LABELS="$RUN_ROOT/combo_labels.txt"

xargs -P "$PARALLEL_JOBS" -I {} \
  conda run --no-capture-output -n "$CONDA_ENV" python "$HELPER" run-combo --output-root "$RUN_ROOT" --combo "{}" \
  < "$COMBO_LABELS"

REPORT_PATH="$(
  conda run --no-capture-output -n "$CONDA_ENV" python "$HELPER" summarize --output-root "$RUN_ROOT"
)"

echo "Report: $REPORT_PATH"
