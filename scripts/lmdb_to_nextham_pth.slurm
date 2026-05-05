#!/bin/bash
#SBATCH --job-name=lmdb2pth
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --output=lmdb2pth_%j.out
#SBATCH --error=lmdb2pth_%j.err

set -euo pipefail

export PYTHONNOUSERSITE=True

# Set H0_DATASET_ROOT to the directory that contains merged LMDB + indices (see README).
# Optional overrides: MAT_HAM_LMDB, DATABASE_INDICES_FILE, LMDB_CONVERT_OUTPUT_DIR
: "${H0_DATASET_ROOT:?Set H0_DATASET_ROOT to the H0_dataset directory (merged_lmdb, indices, outputs)}"

MAT_HAM_LMDB="${MAT_HAM_LMDB:-${H0_DATASET_ROOT}/merged_lmdb/mat_ham.lmdb}"
DATABASE_INDICES_FILE="${DATABASE_INDICES_FILE:-${H0_DATASET_ROOT}/merged_mat_ham_database_indices_matched.pt}"
LMDB_CONVERT_OUTPUT_DIR="${LMDB_CONVERT_OUTPUT_DIR:-${H0_DATASET_ROOT}/nextham_pth}"

REPO_ROOT="${SLURM_SUBMIT_DIR:-}"
if [[ ! -f "${REPO_ROOT}/lmdb_to_nextham_pth.py" ]]; then
  REPO_ROOT="${NEXT_HAM_REPO:-}"
fi
if [[ ! -f "${REPO_ROOT}/lmdb_to_nextham_pth.py" ]]; then
  echo "ERROR: lmdb_to_nextham_pth.py not found. sbatch from this repository root or set NEXT_HAM_REPO (REPO_ROOT=${REPO_ROOT:-empty})" >&2
  exit 2
fi
cd "${REPO_ROOT}"

echo "Starting job ${SLURM_JOB_ID} on host $(hostname)"
echo "REPO_ROOT=${REPO_ROOT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not_set}"
nvidia-smi || true

srun python "${REPO_ROOT}/lmdb_to_nextham_pth.py" \
  --mat-ham-lmdb-path "${MAT_HAM_LMDB}" \
  --database-indices-file "${DATABASE_INDICES_FILE}" \
  --output-dir "${LMDB_CONVERT_OUTPUT_DIR}" \
  --orbital-layout "sssppddf" \
  --seed 42
