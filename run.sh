#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --mem=300G
#SBATCH --cpus-per-gpu=40
#SBATCH --gpus=2
#SBATCH --partition=gpu-h200-141g-ellis

set -euo pipefail

# module load scicomp-python-env
source .venv/bin/activate
module load triton/2025.1-gcc
module load gcc

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/scratch/shareddata/dldata/imagenet-1k-wds/imagenet-1k-wds/}"

# main.py treats --batch-size as total batch when using --multiprocessing-distributed
BATCH_SIZE="${BATCH_SIZE:-4096}"
EPOCHS="${EPOCHS:-800}"
WORKERS="${WORKERS:-20}"
PRINT_FREQ="${PRINT_FREQ:-10}"
MASTER_PORT="${MASTER_PORT:-10001}"

# Simple scaling knobs
BASE_LR="${BASE_LR:-0.00015}"
BASE_BATCH_SIZE="${BASE_BATCH_SIZE:-256}"
BASE_WARMUP_EPOCHS="${BASE_WARMUP_EPOCHS:-40}"

WORLD_SIZE="${SLURM_NNODES:-1}"
NODE_RANK="${SLURM_NODEID:-0}"
MASTER_NODE="${MASTER_NODE:-$(scontrol show hostnames "${SLURM_NODELIST}" | head -n 1)}"

# With this script's DDP launch style, global/effective batch = total batch / number_of_nodes.
EFFECTIVE_BATCH_SIZE=$(( BATCH_SIZE / WORLD_SIZE ))

LR=$("${PYTHON_BIN}" - <<PY
base_lr = float(${BASE_LR})
base_bs = float(${BASE_BATCH_SIZE})
eff_bs = float(${EFFECTIVE_BATCH_SIZE})
print(f"{base_lr * eff_bs / base_bs:.8f}")
PY
)

# main.py uses step-based warmup, so convert warmup epochs -> warmup steps.
IMAGENET_TRAIN_SAMPLES=1281167
TRAIN_BATCHES=$(( IMAGENET_TRAIN_SAMPLES / BATCH_SIZE ))
T_WARM_UP=$(( BASE_WARMUP_EPOCHS * TRAIN_BATCHES ))

echo "Launching MAE pretraining with:"
echo "  python=${PYTHON_BIN}, data=${DATA_ROOT}"
echo "  batch_size(per-node)=${BATCH_SIZE}, effective_batch_size=${EFFECTIVE_BATCH_SIZE}"
echo "  base_lr=${BASE_LR}, computed_lr=${LR}, t_warm_up=${T_WARM_UP}"
echo "  world_size=${WORLD_SIZE}, node_rank=${NODE_RANK}, master_node=${MASTER_NODE}:${MASTER_PORT}"

"${PYTHON_BIN}" -u main.py "${DATA_ROOT}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --workers "${WORKERS}" \
  --print-freq "${PRINT_FREQ}" \
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
  --lr "${LR}" \
  --min_lr 0.0 \
  --t_warm_up "${T_WARM_UP}" \
  --dist-url "tcp://${MASTER_NODE}:${MASTER_PORT}" \
  --multiprocessing-distributed \
  --world-size "${WORLD_SIZE}" \
  --rank "${NODE_RANK}" \
  --dist-backend nccl \
  --compile \
  --bf16 \
  --use_zero
