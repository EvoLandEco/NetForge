#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
HELPER="$SCRIPT_DIR/partition_factor_full_workflow.py"

BASE_CONFIG="/Users/tianjian/Source/NetForge/.stress_runs/cr35_140d_joint_metadata_topology_only_no_layered_maxent_micro_parametric_refit_10runs_100rep_rerun_20260327_132542.json"
SETTINGS_ROOT="$EXPERIMENT_ROOT/settings"
OUTPUT_ROOT=""
DAYS=280
START_TS=""
PARALLEL_JOBS=8
CONDA_ENV="gt"
NUM_SAMPLES=""
SIMULATION_REPLICATES=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-config)
      BASE_CONFIG="$2"
      shift 2
      ;;
    --settings-root)
      SETTINGS_ROOT="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --days)
      DAYS="$2"
      shift 2
      ;;
    --start-ts)
      START_TS="$2"
      shift 2
      ;;
    --parallel-jobs)
      PARALLEL_JOBS="$2"
      shift 2
      ;;
    --conda-env)
      CONDA_ENV="$2"
      shift 2
      ;;
    --num-samples)
      NUM_SAMPLES="$2"
      shift 2
      ;;
    --simulation-replicates)
      SIMULATION_REPLICATES="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

mkdir -p "$SETTINGS_ROOT"

PLAN_ARGS=(
  "$HELPER" plan
  --base-config "$BASE_CONFIG"
  --settings-root "$SETTINGS_ROOT"
  --days "$DAYS"
)

if [[ -n "$OUTPUT_ROOT" ]]; then
  PLAN_ARGS+=(--output-root "$OUTPUT_ROOT")
fi

if [[ -n "$START_TS" ]]; then
  PLAN_ARGS+=(--start-ts "$START_TS")
fi

if [[ -n "$NUM_SAMPLES" ]]; then
  PLAN_ARGS+=(--num-samples "$NUM_SAMPLES")
fi

if [[ -n "$SIMULATION_REPLICATES" ]]; then
  PLAN_ARGS+=(--simulation-replicates "$SIMULATION_REPLICATES")
fi

OUTPUT_ROOT="$(python "${PLAN_ARGS[@]}")"
export SETTINGS_ROOT
export HELPER
export CONDA_ENV

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/mpl
export XDG_CACHE_HOME=/tmp/xdg

echo "Full-workflow factor grid output root: $OUTPUT_ROOT"

< "$SETTINGS_ROOT/combo_labels.txt" xargs -P "$PARALLEL_JOBS" -I {} bash -lc '
  conda run --no-capture-output -n "$CONDA_ENV" python "$HELPER" run-combo --settings-root "$SETTINGS_ROOT" --combo "{}"
'

REPORT_PATH="$(conda run --no-capture-output -n "$CONDA_ENV" python "$HELPER" summarize --settings-root "$SETTINGS_ROOT")"
echo "Summary report: $REPORT_PATH"
