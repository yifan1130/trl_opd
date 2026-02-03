#!/bin/bash
# Shell script to run GKD training on Countdown task
# Usage: bash scripts/run_gkd.sh

# Set visible GPUs (modify as needed)
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Wandb API key (set your key here to skip wandb login)
export WANDB_API_KEY="a83edc41aa70fba425289d46bef8a884c9a694a0"

# Model configuration
STUDENT_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
TEACHER_MODEL="Qwen/Qwen3-4B-Instruct-2507"

# Dataset configuration
DATASET_NAME="HuggingFaceTB/Countdown-Task-GOLD"
DATASET_CONFIG="verified_Qwen3-4B-Instruct-2507"

# Output directory
OUTPUT_DIR="./outputs/gkd-countdown-qwen3-4b-to-qwen2.5-1.5b"

# Run training
python scripts/train_countdown_gkd.py \
    --student_model "$STUDENT_MODEL" \
    --teacher_model "$TEACHER_MODEL" \
    --dataset_name "$DATASET_NAME" \
    --dataset_config "$DATASET_CONFIG" \
    --output_dir "$OUTPUT_DIR" \
    --learning_rate 5e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 3 \
    --max_completion_length 512 \
    --temperature 1.0 \
    --use_uld_loss false \
    --lmbda 1.0 \
    --beta 1.0 \
    --bf16 true \
    --logging_steps 1 \
    --save_strategy steps \
    --save_steps 150 \
    --save_total_limit 3 \
    --log_completions true \
    --log_completions_steps 100 \
    --report_to wandb \
    --wandb_project countdown-distillation \
    --run_name gkd-countdown-qwen3-4b-to-qwen2.5-1.5b
