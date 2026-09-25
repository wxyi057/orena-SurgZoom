#!/usr/bin/env bash
# Fine-tune Qwen3.5-9B with LoRA exactly as for SurgZoom (one input regime per run).
#
#   bash scripts/train.sh <train.jsonl> <output_dir> [extra ms-swift args ...]
#
# Environment (defaults in brackets):
#   REGIME [3]            input regime, loads configs/regime${REGIME}.env
#   NPROC_PER_NODE [4]    GPUs per node       NNODES [1]  NODE_RANK [0]
#   MASTER_ADDR / MASTER_PORT                 for multi-node runs (standard torchrun vars)
#   GLOBAL_BATCH [32]     gradient accumulation is derived from it
#   EPOCHS [15]  LR [1e-4]  ATTN [flash_attn]  BASE_MODEL [Qwen/Qwen3.5-9B]
#
# Smoke test on the example data (1 GPU, 2 steps):
#   NPROC_PER_NODE=1 GLOBAL_BATCH=1 SAVE_STRATEGY=steps bash scripts/train.sh \
#       examples/demo/train_smoke.jsonl runs/smoke --max_steps 2 --save_steps 2
set -euo pipefail
DATA=${1:?usage: train.sh <train.jsonl> <output_dir> [extra args]}
OUT=${2:?usage: train.sh <train.jsonl> <output_dir> [extra args]}
shift 2
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

REGIME=${REGIME:-3}
# shellcheck disable=SC1090
source "${HERE}/configs/regime${REGIME}.env"
export FORCE_QWENVL_VIDEO_READER=${FORCE_QWENVL_VIDEO_READER:-torchvision}
export USE_HF=1                                 # resolve model ids on the Hugging Face Hub
export NPROC_PER_NODE=${NPROC_PER_NODE:-4} NNODES=${NNODES:-1} NODE_RANK=${NODE_RANK:-0}

GLOBAL_BATCH=${GLOBAL_BATCH:-32}
PDBS=1
GAS=$(( GLOBAL_BATCH / (NNODES * NPROC_PER_NODE * PDBS) ))
[ "$GAS" -lt 1 ] && GAS=1

echo "[train] regime ${REGIME}  world $((NNODES * NPROC_PER_NODE))  grad-accum ${GAS}  max_length ${MAX_LENGTH}"
swift sft \
    --model "${BASE_MODEL:-Qwen/Qwen3.5-9B}" --model_type qwen3_5 \
    --dataset "$DATA" --split_dataset_ratio 0 \
    --tuner_type lora --lora_rank 64 --lora_alpha 128 --lora_dropout 0.05 \
    --target_modules all-linear \
    --freeze_vit false --freeze_aligner false --freeze_llm false \
    --torch_dtype bfloat16 --attn_impl "${ATTN:-flash_attn}" \
    --num_train_epochs "${EPOCHS:-15}" --learning_rate "${LR:-1e-4}" \
    --lr_scheduler_type cosine --warmup_ratio 0.03 \
    --per_device_train_batch_size "$PDBS" --gradient_accumulation_steps "$GAS" \
    --gradient_checkpointing true --max_length "$MAX_LENGTH" \
    --eval_strategy no --save_strategy "${SAVE_STRATEGY:-epoch}" --save_total_limit 100 \
    --logging_steps 5 --logging_first_step true \
    --dataloader_num_workers "${WORKERS:-4}" --seed 42 \
    --output_dir "$OUT" "$@"
