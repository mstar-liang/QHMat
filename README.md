# NextHAM Baseline Release

Minimal code to convert merged Hamiltonian LMDB data into NextHAM-style `.pth` samples, train the baseline model, and run paper-style metrics (Hamiltonian MAE and ε).

## End-to-end workflow

1. **LMDB → `.pth` dataset** — `lmdb_to_nextham_pth.py` reads your merged `mat_ham` LMDB and index tensor, writes `samples/*.pth` plus `train.txt`, `val.txt`, and `test.txt`.
2. **Training** — `scripts/train_val.slurm` runs `train_val.py` on the converted directory.
3. **Evaluation** — `scripts/evaluation.slurm` runs `paper_table_metrics_nextham.py` on saved checkpoints (and the same LMDB + indices used for ε).

Configure paths via environment variables (and optionally `sbatch --export=ALL,...`) so nothing site-specific needs to live in the repo.

### 1. Environment

Create and activate the conda environment from `environment.yml`.

### 2. Convert LMDB to `.pth`

**Inputs (typical layout under your dataset root):**

- `merged_lmdb/mat_ham.lmdb`
- `merged_mat_ham_database_indices_matched.pt` (or your matching indices `.pt`)

**Output directory** (default below): `samples/` with numbered `.pth` files and split list files.

From the repository root (where `lmdb_to_nextham_pth.py` lives):

```bash
export H0_DATASET_ROOT=/path/to/H0_dataset
# Optional: non-default filenames or output location
# export MAT_HAM_LMDB=...
# export DATABASE_INDICES_FILE=...
# export LMDB_CONVERT_OUTPUT_DIR="${H0_DATASET_ROOT}/nextham_pth"
sbatch scripts/lmdb_to_nextham_pth.slurm
```

Or run Python directly:

```bash
python lmdb_to_nextham_pth.py \
  --mat-ham-lmdb-path /path/to/mat_ham.lmdb \
  --database-indices-file /path/to/indices.pt \
  --output-dir /path/to/nextham_pth \
  --orbital-layout sssppddf
```

Use `--precompute-band-ref` and related flags only if you need full band-reference tensors in each `.pth` (larger files). Run `python lmdb_to_nextham_pth.py -h` for all CLI options.

### 3. Train

Point `DATA_PATH` at the **same directory** that contains `samples/` and the split `.txt` files (usually `LMDB_CONVERT_OUTPUT_DIR` from step 2). `train_val.slurm` defaults to `/path/to/nextham_pth`; override with `export` or `sbatch --export`.

```bash
export DATA_PATH=/path/to/nextham_pth
export OUTPUT_DIR=/path/to/checkpoints   # optional; default is ${REPO_ROOT}/res
sbatch scripts/train_val.slurm
```

Submit from this repository root, or set `REPO_ROOT` to the root that contains `train_val.py`.

### 4. Evaluate checkpoints

Set checkpoint glob(s), LMDB path, indices file, and test lists. `scripts/evaluation.slurm` defaults to paper-style list names (`test_random_ratio.txt`, `test_ood33.txt`) under `DATA_ROOT`. If you only have `test.txt` from step 2, point `TEST_LIST_ID` at it and clear the OOD list if needed, for example:

```bash
export DATA_ROOT=/path/to/nextham_pth
export CHECKPOINT_GLOB=/path/to/res/model_range*_best.pth.tar
export MAT_HAM_LMDB=/path/to/mat_ham.lmdb
export DATABASE_INDICES_FILE=/path/to/indices.pt
export TEST_LIST_ID="${DATA_ROOT}/test.txt"
export TEST_LIST_OOD=
sbatch scripts/evaluation.slurm
```

See the header comments in `scripts/evaluation.slurm` for ε options (`EPS_MODE`, `EPS_ENERGY_UNITS`, etc.).

## Included components

- `lmdb_to_nextham_pth.py`: LMDB → NextHAM `.pth` conversion.
- `train_val.py`: training entry point.
- `paper_table_metrics_nextham.py`: Hamiltonian + ε metrics and JSON output.
- `scripts/lmdb_to_nextham_pth.slurm`, `scripts/train_val.slurm`, `scripts/evaluation.slurm`: Slurm launchers (paths via env vars).
- `nets/`, `tg_src/`: model and tensor-graph code.
- `dataset_nano.py`, `engine.py`, `logger.py`, `output_data_convert.py`, `optim_factory.py`, `utils.py`: support modules.
- `environment.yml`, `LICENSE`.

## Not included

- Checkpoints (`res*`), logs, plots, and job outputs.
- Raw or merged LMDB datasets (you provide paths locally).

## Anonymization

Before sharing a copy of this tree, search for and replace any remaining local paths, cluster partitions, or account-specific settings in your own exports and copies of the Slurm scripts.
