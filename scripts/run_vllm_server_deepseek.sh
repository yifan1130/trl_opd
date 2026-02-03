#!/bin/bash
# vLLM server for DeepSeek-R1-Distill-Qwen-1.5B (student model)
# This server handles generation during on-policy distillation training
#
# Uses GPUs 0-3 with data_parallel_size=4 for 4x generation throughput
# Training script uses GPUs 4-7
#
# Usage:
#   Terminal 1: bash scripts/run_vllm_server_deepseek.sh
#   Terminal 2: bash scripts/run_dapo_distill.sh

# Use GPUs 0-3 for vLLM server
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Student model for generation
STUDENT_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# Start vLLM server with TRL
# data_parallel_size=4 creates 4 instances for parallel generation
# max_model_len=32768 to support 16k generation + prompt
trl vllm-serve \
    --model "$STUDENT_MODEL" \
    --tensor_parallel_size 1 \
    --data_parallel_size 4 \
    --max_model_len 32768 \
    --dtype bfloat16 \
    --port 8001 \
    --host 0.0.0.0 \
    --gpu_memory_utilization 0.90
