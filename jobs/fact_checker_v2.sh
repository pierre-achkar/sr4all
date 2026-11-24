#!/bin/bash
#SBATCH --job-name=fact_check_v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=h100
#SBATCH --gres=gpu:H100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=logs/jobs/extraction/factcheck_v2_%j.out
#SBATCH --error=logs/jobs/extraction/factcheck_v2_%j.err
#SBATCH --account=rti

set -euo pipefail

CONTAINER=/opt/apptainer/images/base/ubuntu-noble/20250415-py3.12-cuda12.8/ubuntu-noble-20250415-py3.12-cuda12.8-v1.0.sif
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
PYTHON="$REPO_DIR/.factcheck_venv/bin/python"

export HF_HOME="/home/fhg/pie65738/models_cache"
export NLTK_DATA="$REPO_DIR/.nltk_data"
# Defaults target the post-repair alignment pass. Override these with --export
# when reusing this launcher for another fact-check input/output pair.
export SR4ALL_ALIGNED_DIR="${SR4ALL_ALIGNED_DIR:-$REPO_DIR/data/extraction/repaired_aligned}"
export SR4ALL_FACT_CHECKED_DIR="${SR4ALL_FACT_CHECKED_DIR:-$REPO_DIR/data/extraction/repaired_fact_checked}"
export SR4ALL_FACT_CHECK_LOG_DIR="${SR4ALL_FACT_CHECK_LOG_DIR:-$REPO_DIR/logs/extraction/repaired_fact_checked}"
export VLLM_GPU_MEMORY_UTILIZATION=0.70
export VLLM_USE_V1=0
export PYTHONUNBUFFERED=1
export PYTHONSAFEPATH=1

mkdir -p "$NLTK_DATA" "$REPO_DIR/logs/jobs/extraction"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

run_worker() {
  local worker_name="$1"
  local shard_index="$2"
  local gpu_id="$3"

  CUDA_VISIBLE_DEVICES="$gpu_id" apptainer exec \
    --fakeroot \
    --nv \
    --overlay "$REPO_DIR/.overlay-vllm" \
    --bind /storage/remote/DevStorage \
    --env HF_HOME="$HF_HOME" \
    --env NLTK_DATA="$NLTK_DATA" \
    --env SR4ALL_ALIGNED_DIR="$SR4ALL_ALIGNED_DIR" \
    --env SR4ALL_FACT_CHECKED_DIR="$SR4ALL_FACT_CHECKED_DIR" \
    --env SR4ALL_FACT_CHECK_LOG_DIR="$SR4ALL_FACT_CHECK_LOG_DIR" \
    --env VLLM_GPU_MEMORY_UTILIZATION="$VLLM_GPU_MEMORY_UTILIZATION" \
    --env VLLM_USE_V1="$VLLM_USE_V1" \
    --env PYTHONUNBUFFERED=1 \
    --env PYTHONSAFEPATH=1 \
    --env CUDA_VISIBLE_DEVICES="$gpu_id" \
    --env SR4ALL_SHARD_INDEX="$shard_index" \
    --env SR4ALL_SHARD_COUNT=2 \
    --env SR4ALL_FACT_CHECK_WORKER="$worker_name" \
    "$CONTAINER" \
    bash -c "cd /tmp && PYTHONPATH='$REPO_DIR/src' '$PYTHON' -P -u '$REPO_DIR/src/extraction_v2/3_fact_checking.py'"
}

# Each worker owns a stable, disjoint modulo shard and a single H100.
run_worker shard_0 0 0 &
worker_0_pid=$!
run_worker shard_1 1 1 &
worker_1_pid=$!

wait "$worker_0_pid"
wait "$worker_1_pid"