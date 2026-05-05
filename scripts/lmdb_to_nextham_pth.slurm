#!/bin/bash
#SBATCH --job-name=vision_lmdb2pth
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --output=vision_lmdb2pth_%j.out
#SBATCH --error=vision_lmdb2pth_%j.err

set -euo pipefail

# If your cluster uses modules/conda, uncomment and customize these lines.
# source ~/.bashrc
# conda activate <your_env_name>

export PYTHONNOUSERSITE=True

# IMPORTANT: under sbatch, BASH_SOURCE can point to Slurm spool copy.
# Prefer submit directory, with explicit fallback to this project path.
REPO_ROOT="${SLURM_SUBMIT_DIR:-/scratch/user/cxliang_tamu.edu/NextHAM-main-v2}"
if [[ ! -f "${REPO_ROOT}/lmdb_to_nextham_pth.py" ]]; then
  REPO_ROOT="/scratch/user/cxliang_tamu.edu/NextHAM-main-v2"
fi
if [[ ! -f "${REPO_ROOT}/lmdb_to_nextham_pth.py" ]]; then
  echo "ERROR: lmdb_to_nextham_pth.py not found under REPO_ROOT=${REPO_ROOT}" >&2
  exit 2
fi
cd "${REPO_ROOT}"

echo "Starting job ${SLURM_JOB_ID} on host $(hostname)"
echo "REPO_ROOT=${REPO_ROOT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not_set}"
nvidia-smi || true

srun python "${REPO_ROOT}/lmdb_to_nextham_pth.py" \
  --mat-ham-lmdb-path "/scratch/project/prj-02-pi-shuiwang-ji/H0_dataset/merged_lmdb/mat_ham.lmdb" \
  --database-indices-file "/scratch/project/prj-02-pi-shuiwang-ji/H0_dataset/merged_mat_ham_database_indices_matched.pt" \
  --output-dir "/scratch/project/prj-02-pi-shuiwang-ji/H0_dataset/nextham_pth_noprecompute" \
  --nextham-root "${REPO_ROOT}" \
  --orbital-layout "sssppddf" \
  --seed 42 \
  --num-workers "${SLURM_CPUS_PER_TASK:-4}" \
  --resume-existing \
  --skip-split

