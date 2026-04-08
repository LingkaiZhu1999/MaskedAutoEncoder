#!/bin/bash
#SBATCH --time=100:00:00
#SBATCH --mem=300G
#SBATCH --cpus-per-gpu=40
#SBATCH --gpus=4
#SBATCH --partition=gpu-h200-141g-ellis

set -euo pipefail

# module load scicomp-python-env
source .venv/bin/activate
module load triton/2025.1-gcc
module load gcc

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/scratch/shareddata/dldata/imagenet-1k-wds/imagenet-1k-wds/}"

# Global batch size across all nodes.
BATCH_SIZE="${BATCH_SIZE:-4096}"
EPOCHS="${EPOCHS:-800}"
WORKERS="${WORKERS:-20}"
PRINT_FREQ="${PRINT_FREQ:-10}"
MASTER_PORT="${MASTER_PORT:-10001}"

# Simple scaling knobs
BASE_LR="${BASE_LR:-0.00015}"
BASE_BATCH_SIZE="${BASE_BATCH_SIZE:-256}"
BASE_WARMUP_EPOCHS="${BASE_WARMUP_EPOCHS:-40}"

WORLD_SIZE="${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}}"
MASTER_NODE="${MASTER_NODE:-$(scontrol show hostnames "${SLURM_NODELIST}" | head -n 1)}"

# Per-node batch size passed to each launcher.
if (( BATCH_SIZE % WORLD_SIZE != 0 )); then
  echo "ERROR: BATCH_SIZE (${BATCH_SIZE}) must be divisible by WORLD_SIZE (${WORLD_SIZE})." >&2
  exit 1
fi
PER_NODE_BATCH_SIZE=$(( BATCH_SIZE / WORLD_SIZE ))

LR=$("${PYTHON_BIN}" - <<PY
base_lr = float(${BASE_LR})
base_bs = float(${BASE_BATCH_SIZE})
eff_bs = float(${BATCH_SIZE})
print(f"{base_lr * eff_bs / base_bs:.8f}")
PY
)

# main.py uses step-based warmup, so convert warmup epochs -> warmup steps.
IMAGENET_TRAIN_SAMPLES=1281167
TRAIN_BATCHES=$(( IMAGENET_TRAIN_SAMPLES / BATCH_SIZE ))
T_WARM_UP=$(( BASE_WARMUP_EPOCHS * TRAIN_BATCHES ))

echo "Launching MAE pretraining with:"
echo "  python=${PYTHON_BIN}, data=${DATA_ROOT}"
echo "  batch_size(global)=${BATCH_SIZE}, batch_size(per-node)=${PER_NODE_BATCH_SIZE}"
echo "  base_lr=${BASE_LR}, computed_lr=${LR}, t_warm_up=${T_WARM_UP}"
echo "  world_size(nodes)=${WORLD_SIZE}, master_node=${MASTER_NODE}:${MASTER_PORT}"

# Launch one Python launcher process per node. Each launcher then mp.spawn's local GPU workers.
srun --ntasks="${WORLD_SIZE}" --ntasks-per-node=1 --kill-on-bad-exit=1 \
  bash -lc '
    export RANK="${SLURM_PROCID}"
    export WORLD_SIZE="${SLURM_NTASKS}"
    export MASTER_ADDR="'"${MASTER_NODE}"'"
    export MASTER_PORT="'"${MASTER_PORT}"'"
    exec "'"${PYTHON_BIN}"'" -u main.py "'"${DATA_ROOT}"'" \
      --epochs "'"${EPOCHS}"'" \
      --batch-size "'"${PER_NODE_BATCH_SIZE}"'" \
      --workers "'"${WORKERS}"'" \
      --print-freq "'"${PRINT_FREQ}"'" \
      --image_size 224 \
      --patch_size 16 \
      --in_channels 3 \
      --d_model 1024 \
      --num_heads 16 \
      --d_ff 4096 \
      --num_layers 24 \
      --decoder-d-model 512 \
      --decoder-num-heads 16 \
      --decoder-d-ff 2048 \
      --decoder-num-layers 8 \
      --mask-ratio 0.75 \
      --wd 0.05 \
      --lr "'"${LR}"'" \
      --min_lr 0.0 \
      --t_warm_up "'"${T_WARM_UP}"'" \
      --dist-url env:// \
      --multiprocessing-distributed \
      --world-size -1 \
      --rank -1 \
      --dist-backend nccl \
      --compile \
      --bf16 \
      --use_zero
  '
