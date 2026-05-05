#!/bin/bash
#SBATCH --job-name=nextham_eva
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --output=evaluation_%j.out
#SBATCH --error=evaluation_%j.err

set -euo pipefail

source ~/.bashrc
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-}"
conda activate nextham

export PYTHONNOUSERSITE=True
export PYTHONUNBUFFERED=1

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
if [[ ! -f "${REPO_ROOT}/paper_table_metrics_nextham.py" ]]; then
  echo "ERROR: paper_table_metrics_nextham.py not found under REPO_ROOT=${REPO_ROOT}" >&2
  exit 2
fi
cd "${REPO_ROOT}"

echo "Starting job ${SLURM_JOB_ID} on $(hostname)  (paper_eps_ood_only)"
echo "REPO_ROOT=${REPO_ROOT}"

# -------------------- USER CONFIG (anonymized defaults) --------------------
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${REPO_ROOT}}"
CHECKPOINT_GLOB="${CHECKPOINT_GLOB:-${CHECKPOINT_ROOT}/res/model_range*_best.pth.tar}"
CHECKPOINT_GLOB_OOD="${CHECKPOINT_GLOB_OOD:-${CHECKPOINT_ROOT}/res/model_range*_best.pth.tar}"

DATA_ROOT="${DATA_ROOT:-/path/to/nextham_pth}"
# Default: do **not** run in-distribution. To also run ID, export TEST_LIST_ID explicitly, e.g.
#   sbatch --export=ALL,TEST_LIST_ID=${DATA_ROOT}/test_random_ratio.txt vision_jobs/paper_eps_ood_only.slurm
TEST_LIST_ID="${TEST_LIST_ID:-}"
TEST_LIST_OOD="${TEST_LIST_OOD:-${DATA_ROOT}/test_ood33.txt}"

MAT_HAM_LMDB="${MAT_HAM_LMDB:-/path/to/merged_lmdb/mat_ham.lmdb}"
DATABASE_INDICES_FILE="${DATABASE_INDICES_FILE:-/path/to/mat_ham_database_indices.pt}"

EPS_MODE="${EPS_MODE:-gamma}"
EPS_RANDOM_K_SEED="${EPS_RANDOM_K_SEED:-0}"
EPS_NO_GAUGE="${EPS_NO_GAUGE:-0}"
HIDE_EPS_FERMI_WINDOW="${HIDE_EPS_FERMI_WINDOW:-0}"
LATEX_EPSILON_METRIC="${LATEX_EPSILON_METRIC:-full_spectrum}"
EPS_ENERGY_UNITS="${EPS_ENERGY_UNITS:-ev}"
EPS_ZOOM_WINDOW="${EPS_ZOOM_WINDOW:-${EPS_ZOOM_WINDOW_HA:-}}"

NK_POINTS="${NK_POINTS:-10}"

SUBSAMPLE_RANDOM="${SUBSAMPLE_RANDOM:-1}"
SUBSAMPLE_SEED="${SUBSAMPLE_SEED:-42}"

OUTPUT_JSON="${OUTPUT_JSON:-${REPO_ROOT}/paper_metrics_eps_ood_only_${SLURM_JOB_ID}.json}"
# ----------------------------------------------------------------------------------

if [[ ! -f "${TEST_LIST_OOD:-}" ]]; then
  echo "ERROR: TEST_LIST_OOD not found: ${TEST_LIST_OOD:-}" >&2
  exit 2
fi

ARGS=(
  python "${REPO_ROOT}/paper_table_metrics_nextham.py"
  --checkpoint-glob "${CHECKPOINT_GLOB}"
  --device cuda:0
  --model-name graph_attention_transformer_nonlinear_materials_ham_soc
  --input-irreps 64x0e
  --orbital-layout sssppddf
  --no-spinful
  --with-trace
  --trace-out-len 81
  --mat-ham-lmdb "${MAT_HAM_LMDB}"
  --database-indices-file "${DATABASE_INDICES_FILE}"
  --eps-mode "${EPS_MODE}"
  --eps-random-k-seed "${EPS_RANDOM_K_SEED}"
  --nk-points "${NK_POINTS}"
  --eps-energy-units "${EPS_ENERGY_UNITS}"
  --eps-report-both-gauges
  --latex-epsilon-metric "${LATEX_EPSILON_METRIC}"
  --output-json "${OUTPUT_JSON}"
)
if [[ "${EPS_NO_GAUGE}" == "1" ]]; then
  ARGS+=(--no-eps-gauge)
fi
if [[ "${HIDE_EPS_FERMI_WINDOW}" == "1" ]]; then
  ARGS+=(--hide-eps-fermi-window)
fi
if [[ -n "${EPS_ZOOM_WINDOW}" ]]; then
  ARGS+=(--eps-zoom-window-ha "${EPS_ZOOM_WINDOW}")
fi
[[ -f "${TEST_LIST_ID:-}" ]] && ARGS+=(--test-list-id "${TEST_LIST_ID}")
ARGS+=(--test-list-ood "${TEST_LIST_OOD}")
[[ -n "${CHECKPOINT_GLOB_OOD:-}" ]] && ARGS+=(--checkpoint-glob-ood "${CHECKPOINT_GLOB_OOD}")

if [[ "${MAX_SAMPLES}" != "-1" ]]; then
  ARGS+=(--max-samples-per-split "${MAX_SAMPLES}")
  if [[ "${SUBSAMPLE_RANDOM}" == "1" ]]; then
    ARGS+=(--subsample-random --subsample-seed "${SUBSAMPLE_SEED}")
  fi
fi

echo "CHECKPOINT_GLOB=${CHECKPOINT_GLOB}"
echo "CHECKPOINT_GLOB_OOD=${CHECKPOINT_GLOB_OOD:-<unset, OOD would reuse ID>}"
echo "TEST_LIST_ID=${TEST_LIST_ID:-<empty>}"
echo "TEST_LIST_OOD=${TEST_LIST_OOD}"
echo "EPS_MODE=${EPS_MODE} (gamma=Γ-only ε) NK_POINTS=${NK_POINTS} EPS_ENERGY_UNITS=${EPS_ENERGY_UNITS} EPS_REPORT_BOTH_GAUGES=1"
echo "LATEX_EPSILON_METRIC=${LATEX_EPSILON_METRIC} EPS_ZOOM_WINDOW=${EPS_ZOOM_WINDOW:-<default ±10 in ${EPS_ENERGY_UNITS} units>}"
echo "EPS_NO_GAUGE=${EPS_NO_GAUGE} HIDE_EPS_FERMI_WINDOW=${HIDE_EPS_FERMI_WINDOW}"
echo "MAX_SAMPLES=${MAX_SAMPLES}  OUTPUT_JSON=${OUTPUT_JSON}"

srun "${ARGS[@]}"

echo "Done. JSON: ${OUTPUT_JSON}"
