#!/bin/bash
#SBATCH --job-name=validation_eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=logs/jobs/%x-%j.out
#SBATCH --error=logs/jobs/%x-%j.err
#SBATCH -A rti

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"
CONTAINER=/opt/apptainer/vllm-ubuntu2404-uv.sif

srun --ntasks=1 apptainer exec \
  --overlay .overlay \
  --bind /storage/remote/DevStorage \
  "$CONTAINER" \
  /usr/bin/python3 -u src/validation/analyze.py \
  --dir data/validation/annotation/exports \
  --tasks data/validation/annotation/tasks_385.jsonl \
  "$@"