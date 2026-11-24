#!/bin/bash
#SBATCH --job-name=alignment_v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/jobs/extraction/alignment_v2_%j.out
#SBATCH --error=logs/jobs/extraction/alignment_v2_%j.err
#SBATCH --account=rti

set -euo pipefail

CONTAINER=/opt/apptainer/images/base/ubuntu-noble/20250415-py3.12-cuda12.8/ubuntu-noble-20250415-py3.12-cuda12.8-v1.0.sif
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

export SR4ALL_EXTRACTION_DIR="$REPO_DIR/data/extraction/repaired"
export SR4ALL_ALIGNED_DIR="$REPO_DIR/data/extraction/repaired_aligned"
export SR4ALL_MDS_DIR="$REPO_DIR/data/retrieval/merged/mds"
export SR4ALL_ALIGNMENT_LOG_DIR="$REPO_DIR/logs/extraction"
export PYTHONUNBUFFERED=1

apptainer exec \
  --fakeroot \
  --overlay "$REPO_DIR/.overlay-vllm" \
  --bind /storage/remote/DevStorage \
  "$CONTAINER" \
  bash -c "cd '$REPO_DIR' && python3 -u src/extraction_v2/2_alignment.py --processes '$SLURM_CPUS_PER_TASK'"