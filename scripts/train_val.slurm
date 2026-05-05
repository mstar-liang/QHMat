#!/bin/bash
#SBATCH --job-name=nextham_random
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --qos=standard
#SBATCH --gres=gpu:4
#SBATCH --output=train_val_random_%j.out
#SBATCH --error=train_val_random_%j.err

set -euo pipefail

source ~/.bashrc
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-}"
conda activate nextham

export PYTHONNOUSERSITE=True
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WORLD_SIZE=4
export PYTHONUNBUFFERED=1

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
if [[ ! -f "${REPO_ROOT}/train_val.py" ]]; then
  echo "ERROR: train_val.py not found under REPO_ROOT=${REPO_ROOT}" >&2
  exit 2
fi

# User-overridable paths (anonymized defaults).
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/res}"
DATA_PATH="${DATA_PATH:-/path/to/nextham_pth}"

cd "${REPO_ROOT}"

echo "Starting job ${SLURM_JOB_ID} on host $(hostname)"
echo "REPO_ROOT=${REPO_ROOT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not_set}"

# NOTE: no --wa-lmdb-eigen-fallback flag on purpose.
# This keeps the default WA fallback split: n//2.
srun python "${REPO_ROOT}/train_val.py" \
  --output-dir "${OUTPUT_DIR}" \
  --auto-resume \
  --model-name "graph_attention_transformer_nonlinear_materials_ham_soc" \
  --data-path "${DATA_PATH}" \
  --auto-split \
  --split-tag "random" \
  --split-train-ratio 0.8 \
  --split-val-ratio 0.1 \
  --split-seed 42 \
  --input-irreps "64x0e" \
  --radius 8.0 \
  --is-accurate-label \
  --trace-out-len 81 \
  --epochs 100 \
  --num-models 4 \
  --lr 5e-4 \
  --batch-size 1 \
  --eval-batch-size 1 \
  --weight-decay 0 \
  --num-basis 64 \
  --with-trace \
  --test-interval 10000 \
  --target "hamiltonian" \
  --target-blocks-type "all" \
  --no-spinful \
  --orbital-layout "sssppddf" \
  --use-wa-loss-lmdb \
  --wa-k-grid 4 4 4 \
  --print-freq 100
