#!/bin/bash
#SBATCH --job-name=repair_v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=h100
#SBATCH --gres=gpu:H100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=320G
#SBATCH --output=logs/jobs/extraction/repair_v2_%j.out
#SBATCH --error=logs/jobs/extraction/repair_v2_%j.err
#SBATCH --account=rti

set -euo pipefail

CONTAINER=/opt/apptainer/images/base/ubuntu-noble/20250415-py3.12-cuda12.8/ubuntu-noble-20250415-py3.12-cuda12.8-v1.0.sif
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

export HF_HOME="/home/fhg/pie65738/models_cache"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

# v2 repair-stage paths
export SR4ALL_PROJECT="$REPO_DIR"
export SR4ALL_FACT_CHECKED_DIR="${SR4ALL_FACT_CHECKED_DIR:-$REPO_DIR/data/extraction/fact_checked}"
export SR4ALL_REPAIRED_DIR="${SR4ALL_REPAIRED_DIR:-$REPO_DIR/data/extraction/repaired}"
export SR4ALL_MDS_DIR="${SR4ALL_MDS_DIR:-$REPO_DIR/data/retrieval/merged/mds}"
export SR4ALL_REPAIR_LOG_DIR="${SR4ALL_REPAIR_LOG_DIR:-$REPO_DIR/logs/extraction}"

# Model/runtime settings; each can be overridden with --export=ALL,...
export SR4ALL_REPAIR_MODEL="${SR4ALL_REPAIR_MODEL:-Qwen/Qwen3.6-27B}"
export SR4ALL_REPAIR_BATCH_SIZE="${SR4ALL_REPAIR_BATCH_SIZE:-20}"
export SR4ALL_MAX_REPAIR_ROUNDS="${SR4ALL_MAX_REPAIR_ROUNDS:-1}"

# Optional args, for example: --limit 100 --overwrite
REPAIR_ARGS="${REPAIR_ARGS:-}"

mkdir -p "$REPO_DIR/logs/jobs/extraction" "$SR4ALL_REPAIR_LOG_DIR" "$SR4ALL_REPAIRED_DIR"

echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "GPUs:      ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Repo:      $REPO_DIR"
echo "Input:     $SR4ALL_FACT_CHECKED_DIR"
echo "Output:    $SR4ALL_REPAIRED_DIR"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

run_worker() {
  local worker_name="$1"
  local shard_index="$2"
  local gpu_ids="$3"

  CUDA_VISIBLE_DEVICES="$gpu_ids" apptainer exec \
    --fakeroot \
    --nv \
    --overlay "$REPO_DIR/.overlay-vllm" \
    --bind /storage/remote/DevStorage \
    --env HF_HOME="$HF_HOME" \
    --env HF_HUB_OFFLINE="$HF_HUB_OFFLINE" \
    --env TRANSFORMERS_OFFLINE="$TRANSFORMERS_OFFLINE" \
    --env PYTHONUNBUFFERED=1 \
    --env OMP_NUM_THREADS="$OMP_NUM_THREADS" \
    --env SR4ALL_PROJECT="$SR4ALL_PROJECT" \
    --env SR4ALL_FACT_CHECKED_DIR="$SR4ALL_FACT_CHECKED_DIR" \
    --env SR4ALL_REPAIRED_DIR="$SR4ALL_REPAIRED_DIR" \
    --env SR4ALL_MDS_DIR="$SR4ALL_MDS_DIR" \
    --env SR4ALL_REPAIR_LOG_DIR="$SR4ALL_REPAIR_LOG_DIR" \
    --env SR4ALL_REPAIR_MODEL="$SR4ALL_REPAIR_MODEL" \
    --env SR4ALL_REPAIR_TP=2 \
    --env SR4ALL_REPAIR_BATCH_SIZE="$SR4ALL_REPAIR_BATCH_SIZE" \
    --env SR4ALL_MAX_REPAIR_ROUNDS="$SR4ALL_MAX_REPAIR_ROUNDS" \
    --env CUDA_VISIBLE_DEVICES="$gpu_ids" \
    --env SR4ALL_SHARD_INDEX="$shard_index" \
    --env SR4ALL_SHARD_COUNT=2 \
    --env SR4ALL_REPAIR_WORKER="$worker_name" \
    "$CONTAINER" \
    bash -c "cd '$REPO_DIR' && python3 -u src/extraction_v2/4_repair.py $REPAIR_ARGS"
}

# Each worker owns a stable, disjoint modulo shard and runs one TP=2 model on two H100s.
run_worker shard_0 0 0,1 &
worker_0_pid=$!
run_worker shard_1 1 2,3 &
worker_1_pid=$!

wait "$worker_0_pid"
wait "$worker_1_pid"