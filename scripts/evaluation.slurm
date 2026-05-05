#!/bin/bash
#SBATCH --job-name=nextham_evaluation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --output=paper_evaluation_%j.out
#SBATCH --error=paper_evaluation_%j.err


set -euo pipefail

source ~/.bashrc
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-}"
conda activate nextham

export PYTHONNOUSERSITE=True
export PYTHONUNBUFFERED=1

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
if [[ ! -f "${REPO_ROOT}/paper_table_metrics_nextham.py" ]]; then
  REPO_ROOT="${NEXT_HAM_REPO:-}"
fi
if [[ ! -f "${REPO_ROOT}/paper_table_metrics_nextham.py" ]]; then
  echo "ERROR: paper_table_metrics_nextham.py not found. Submit from repo root or set NEXT_HAM_REPO (REPO_ROOT=${REPO_ROOT:-empty})" >&2
  exit 2
fi
cd "${REPO_ROOT}"

echo "Starting job ${SLURM_JOB_ID} on $(hostname)"
echo "REPO_ROOT=${REPO_ROOT}"

# -------------------- Paths: checkpoints & data --------------------
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${REPO_ROOT}}"
CHECKPOINTS="${CHECKPOINTS:-}"
CHECKPOINT_GLOB="${CHECKPOINT_GLOB:-}"
if [[ -z "${CHECKPOINTS}" && -z "${CHECKPOINT_GLOB}" ]]; then
  CHECKPOINT_GLOB="${CHECKPOINT_ROOT}/res/model_range*_best.pth.tar"
fi
# Second ensemble for OOD list only; leave empty to reuse ID checkpoints for OOD
CHECKPOINT_GLOB_OOD="${CHECKPOINT_GLOB_OOD:-}"

DATA_ROOT="${DATA_ROOT:-/path/to/nextham_pth}"
# ${VAR-default}: unset → default; TEST_LIST_ID= clears ID list for OOD-only (sbatch --export)
TEST_LIST_ID="${TEST_LIST_ID-${DATA_ROOT}/test_random_ratio.txt}"
TEST_LIST_OOD="${TEST_LIST_OOD-${DATA_ROOT}/test_ood33.txt}"

MAT_HAM_LMDB="${MAT_HAM_LMDB:-/path/to/merged_lmdb/mat_ham.lmdb}"
DATABASE_INDICES_FILE="${DATABASE_INDICES_FILE:-/path/to/mat_ham_database_indices.pt}"

# -------------------- Model (match training) --------------------
DEVICE="${DEVICE:-cuda:0}"
MODEL_NAME="${MODEL_NAME:-graph_attention_transformer_nonlinear_materials_ham_soc}"
INPUT_IRREPS="${INPUT_IRREPS:-64x0e}"
RADIUS="${RADIUS:-8.0}"
NUM_BASIS="${NUM_BASIS:-64}"
ORBITAL_LAYOUT="${ORBITAL_LAYOUT:-sssppddf}"
TRACE_OUT_LEN="${TRACE_OUT_LEN:-81}"
TARGET="${TARGET:-hamiltonian}"
TARGET_BLOCKS_TYPE="${TARGET_BLOCKS_TYPE:-all}"
SPINFUL="${SPINFUL:-0}"

EPS_MODE="${EPS_MODE:-mesh}"
EPS_RANDOM_K_SEED="${EPS_RANDOM_K_SEED:-0}"
EPS_REPORT_BOTH_GAUGES="${EPS_REPORT_BOTH_GAUGES:-1}"
EPS_NO_GAUGE="${EPS_NO_GAUGE:-0}"
HIDE_EPS_FERMI_WINDOW="${HIDE_EPS_FERMI_WINDOW:-0}"
LATEX_EPSILON_METRIC="${LATEX_EPSILON_METRIC:-full_spectrum}"
EPS_ENERGY_UNITS="${EPS_ENERGY_UNITS:-ev}"
EPS_ZOOM_WINDOW="${EPS_ZOOM_WINDOW:-${EPS_ZOOM_WINDOW_HA:-}}"

NK_POINTS="${NK_POINTS:-10}"

MAX_SAMPLES="${MAX_SAMPLES:--10000}"
SUBSAMPLE_RANDOM="${SUBSAMPLE_RANDOM:-1}"
SUBSAMPLE_SEED="${SUBSAMPLE_SEED:-42}"

OUTPUT_JSON="${OUTPUT_JSON:-${REPO_ROOT}/evaluation_${SLURM_JOB_ID}.json}"
# ----------------------------------------------------------------------------------

if [[ -n "${TEST_LIST_ID}" && ! -f "${TEST_LIST_ID}" ]]; then
  echo "ERROR: TEST_LIST_ID not found: ${TEST_LIST_ID}" >&2
  exit 2
fi
if [[ -n "${TEST_LIST_OOD}" && ! -f "${TEST_LIST_OOD}" ]]; then
  echo "ERROR: TEST_LIST_OOD not found: ${TEST_LIST_OOD}" >&2
  exit 2
fi
if [[ -z "${TEST_LIST_ID}" && -z "${TEST_LIST_OOD}" ]]; then
  echo "ERROR: Set at least one of TEST_LIST_ID / TEST_LIST_OOD (use TEST_LIST_ID= or TEST_LIST_OOD= to clear defaults)." >&2
  exit 2
fi

ARGS=(
  python "${REPO_ROOT}/paper_table_metrics_nextham.py"
  --device "${DEVICE}"
  --model-name "${MODEL_NAME}"
  --input-irreps "${INPUT_IRREPS}"
  --radius "${RADIUS}"
  --num-basis "${NUM_BASIS}"
  --orbital-layout "${ORBITAL_LAYOUT}"
  --with-trace
  --trace-out-len "${TRACE_OUT_LEN}"
  --target "${TARGET}"
  --target-blocks-type "${TARGET_BLOCKS_TYPE}"
  --mat-ham-lmdb "${MAT_HAM_LMDB}"
  --database-indices-file "${DATABASE_INDICES_FILE}"
  --eps-mode "${EPS_MODE}"
  --eps-random-k-seed "${EPS_RANDOM_K_SEED}"
  --nk-points "${NK_POINTS}"
  --eps-energy-units "${EPS_ENERGY_UNITS}"
  --latex-epsilon-metric "${LATEX_EPSILON_METRIC}"
  --output-json "${OUTPUT_JSON}"
)

if [[ "${SPINFUL}" == "1" ]]; then
  ARGS+=(--spinful)
else
  ARGS+=(--no-spinful)
fi

if [[ -n "${CHECKPOINTS}" ]]; then
  read -r -a _ckpts <<< "${CHECKPOINTS}"
  ARGS+=(--checkpoints "${_ckpts[@]}")
else
  ARGS+=(--checkpoint-glob "${CHECKPOINT_GLOB}")
fi

if [[ "${EPS_REPORT_BOTH_GAUGES}" == "1" ]]; then
  ARGS+=(--eps-report-both-gauges)
fi
if [[ "${EPS_NO_GAUGE}" == "1" ]]; then
  ARGS+=(--no-eps-gauge)
fi
if [[ "${HIDE_EPS_FERMI_WINDOW}" == "1" ]]; then
  ARGS+=(--hide-eps-fermi-window)
fi
if [[ -n "${EPS_ZOOM_WINDOW}" ]]; then
  ARGS+=(--eps-zoom-window-ha "${EPS_ZOOM_WINDOW}")
fi

[[ -n "${TEST_LIST_ID}" ]] && ARGS+=(--test-list-id "${TEST_LIST_ID}")
if [[ -n "${TEST_LIST_OOD}" ]]; then
  ARGS+=(--test-list-ood "${TEST_LIST_OOD}")
  [[ -n "${CHECKPOINT_GLOB_OOD}" ]] && ARGS+=(--checkpoint-glob-ood "${CHECKPOINT_GLOB_OOD}")
fi

if [[ "${MAX_SAMPLES}" != "-1" ]]; then
  ARGS+=(--max-samples-per-split "${MAX_SAMPLES}")
  if [[ "${SUBSAMPLE_RANDOM}" == "1" ]]; then
    ARGS+=(--subsample-random --subsample-seed "${SUBSAMPLE_SEED}")
  fi
fi

echo "CHECKPOINT_GLOB=${CHECKPOINT_GLOB}"
echo "CHECKPOINTS=${CHECKPOINTS:-<empty>}"
echo "CHECKPOINT_GLOB_OOD=${CHECKPOINT_GLOB_OOD:-<empty, reuse ID checkpoints for OOD>}"
echo "TEST_LIST_ID=${TEST_LIST_ID:-<empty>}"
echo "TEST_LIST_OOD=${TEST_LIST_OOD:-<empty>}"
echo "DEVICE=${DEVICE} SPINFUL=${SPINFUL} MODEL_NAME=${MODEL_NAME} INPUT_IRREPS=${INPUT_IRREPS} ORBITAL_LAYOUT=${ORBITAL_LAYOUT}"
echo "EPS_MODE=${EPS_MODE} NK_POINTS=${NK_POINTS} EPS_ENERGY_UNITS=${EPS_ENERGY_UNITS} EPS_REPORT_BOTH_GAUGES=${EPS_REPORT_BOTH_GAUGES} LATEX_EPSILON_METRIC=${LATEX_EPSILON_METRIC}"
echo "EPS_ZOOM_WINDOW=${EPS_ZOOM_WINDOW:-<default ±10 in ${EPS_ENERGY_UNITS} units>} EPS_NO_GAUGE=${EPS_NO_GAUGE} HIDE_EPS_FERMI_WINDOW=${HIDE_EPS_FERMI_WINDOW}"
echo "MAX_SAMPLES=${MAX_SAMPLES} OUTPUT_JSON=${OUTPUT_JSON}"

srun "${ARGS[@]}"

echo "Done. JSON: ${OUTPUT_JSON}"
