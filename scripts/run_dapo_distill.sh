#!/bin/bash
# Shell script to run on-policy distillation on DAPO-Math dataset
# Run this AFTER starting the vLLM server (run_vllm_server_deepseek.sh)
#
# Student: DeepSeek-R1-Distill-Qwen-1.5B
# Teacher: DeepSeek-R1-Distill-Qwen-7B
#
# Usage:
#   Terminal 1: bash scripts/run_vllm_server_deepseek.sh
#   Terminal 2: bash scripts/run_dapo_distill.sh

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Change to repo directory
cd "$REPO_DIR"

# Training uses GPUs 4-7 (vLLM server uses 0-3)
export CUDA_VISIBLE_DEVICES=4,5,6,7

# Wandb API key
export WANDB_API_KEY="a83edc41aa70fba425289d46bef8a884c9a694a0"

# Model configuration
STUDENT_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
TEACHER_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

# Dataset configuration
DATASET_PATH="/data/cxu/tree_grpo/YYF42_data_train/train_DAPO-Math-17k_sampled.parquet"

# Output directory
OUTPUT_DIR="./outputs/dapo-math-distill-deepseek-7b-to-1.5b"

# Run training with vLLM server mode and DeepSpeed
accelerate launch \
    --num_processes 4 \
    --use_deepspeed \
    --deepspeed_config_file scripts/ds_config_zero2.json \
    scripts/train_dapo_math_distill.py \
    --student_model "$STUDENT_MODEL" \
    --teacher_model "$TEACHER_MODEL" \
    --dataset_path "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --learning_rate 5e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 3 \
    --max_length 32768 \
    --max_completion_length 4096 \
    --temperature 1.0 \
    --use_uld_loss false \
    --lmbda 1.0 \
    --beta 1.0 \
    --bf16 true \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host "localhost" \
    --vllm_server_port 8001 \
    --vllm_server_timeout 300 \
    --logging_steps 1 \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 10 \
    --log_completions true \
    --log_completions_steps 50 \
    --report_to wandb \
    --wandb_project dapo-math-distillation \
    --run_name dapo-distill-deepseek-7b-to-1.5b
