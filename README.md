# NextHAM Baseline Release

This folder contains the minimal code needed to run the baseline training and paper metric evaluation workflows.

## Included components

- `train_val.py`: baseline training entry point.
- `paper_table_metrics_nextham.py`: paper metric aggregation and epsilon evaluation.
- `nets/`, `tg_src/`: model and tensor-graph dependencies used by training/evaluation.
- `dataset_nano.py`, `engine.py`, `logger.py`, `output_data_convert.py`, `optim_factory.py`, `utils.py`: runtime support modules imported by the entry points.
- `vision_jobs/train_val_noprecompute_random.slurm`: training launcher.
- `vision_jobs/paper_eps_ood_only.slurm`: OOD epsilon evaluation launcher.
- `environment.yml`: environment specification from the original project.
- `LICENSE`.

## Not included

Large artifacts and non-essential components are intentionally excluded, such as:

- checkpoints (`res*`), logs, generated plots, and job outputs.
- conversion/visualization utilities unrelated to baseline train/eval claims.
- dataset files (must be provided separately at your local paths).

## Quick start

1. Create/activate the conda environment from `environment.yml`.
2. Update dataset/checkpoint paths in the slurm scripts (or pass overrides via `sbatch --export`).
3. Submit jobs from this folder:
   - training: `sbatch vision_jobs/train_val_noprecompute_random.slurm`
   - OOD eval: `sbatch vision_jobs/paper_eps_ood_only.slurm`

## Anonymization reminder

Before anonymous submission, remove or replace any user names, internal paths, cluster hostnames, and project identifiers in scripts or docs.
