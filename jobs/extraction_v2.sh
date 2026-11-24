#!/bin/bash
#SBATCH --job-name=vllm_qwen36_extraction
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -p h100
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=320G
#SBATCH --output=logs/jobs/extraction/vllm_%x-%j.out
#SBATCH --error=logs/jobs/extraction/vllm_%x-%j.err
#SBATCH -A rti

set -euo pipefail

# --- CONFIGURATION ---
CONTAINER=/opt/apptainer/images/base/ubuntu-noble/20250415-py3.12-cuda12.8/ubuntu-noble-20250415-py3.12-cuda12.8-v1.0.sif
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

export HF_HOME="/home/fhg/pie65738/models_cache"

# Weights are already cached; keep the compute node from reaching for the hub.
# Drop these two lines the first time you pull a new model.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Logging and tqdm appear in the .out file immediately rather than at job end.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=16
# Empty means no cap in 1_extraction.py: process the full corpus.
export SR4ALL_MAX_DOCUMENTS=""

# NOTE: VLLM_USE_V1=0 is deliberately NOT set. The V0 engine forfeits automatic
# prefix caching and CUDA graphs, and every prompt in this run shares the same
# ~2.5k-token instruction prefix. If V1 turns out to be unstable on this vLLM
# build, re-add `export VLLM_USE_V1=0` here -- enable_prefix_caching=True is
# already set in the engine config, so caching survives; CUDA graphs do not.

# --- EXECUTION ---
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "GPUs:      ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Repo:      $REPO_DIR"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

run_worker() {
  local worker_name="$1"
  local shard_index="$2"
  local gpu_ids="$3"

  CUDA_VISIBLE_DEVICES="$gpu_ids" apptainer exec \
    --fakeroot \
    --nv \
    --overlay "$REPO_DIR/.overlay-vllm" \
    --env HF_HOME="$HF_HOME" \
    --env HF_HUB_OFFLINE="$HF_HUB_OFFLINE" \
    --env TRANSFORMERS_OFFLINE="$TRANSFORMERS_OFFLINE" \
    --env PYTHONUNBUFFERED=1 \
    --env OMP_NUM_THREADS="$OMP_NUM_THREADS" \
    --env CUDA_VISIBLE_DEVICES="$gpu_ids" \
    --env SR4ALL_MAX_DOCUMENTS="$SR4ALL_MAX_DOCUMENTS" \
    --env SR4ALL_MDS_DIR="$REPO_DIR/data/retrieval/merged/mds" \
    --env SR4ALL_EXTRACTION_DIR="$REPO_DIR/data/extraction/first_pass" \
    --env SR4ALL_EXTRACTION_LOG_DIR="$REPO_DIR/logs/extraction" \
    --env SR4ALL_SHARD_INDEX="$shard_index" \
    --env SR4ALL_SHARD_COUNT=2 \
    --env SR4ALL_EXTRACTION_WORKER="$worker_name" \
    --env SR4ALL_EXTRACTION_TP=2 \
    "$CONTAINER" \
    bash -c "cd '$REPO_DIR' && python3 -u src/extraction_v2/1_extraction.py"
}

# Two independent TP=2 engines consume GPU pairs 0,1 and 2,3. The modulo shards
# are disjoint and each output path is owned by exactly one worker.
run_worker shard_0 0 0,1 &
worker_0_pid=$!
run_worker shard_1 1 2,3 &
worker_1_pid=$!

wait "$worker_0_pid"
wait "$worker_1_pid"