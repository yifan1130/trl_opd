#!/bin/bash
# Start vLLM server with data parallelism for GOLD/GKD training
# Run this FIRST in Terminal 1, then run training in Terminal 2
#
# Usage: bash scripts/run_vllm_server.sh

# vLLM server uses GPUs 0-3 (4 data parallel instances)
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Student model (generates responses during training)
MODEL="Qwen/Qwen2.5-1.5B-Instruct"

# Server configuration
HOST="0.0.0.0"
PORT=8001

# Start vLLM server with 4 data parallel instances
# Each instance runs on 1 GPU, giving 4x generation throughput
trl vllm-serve \
    --model "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --tensor_parallel_size 1 \
    --data_parallel_size 4 \
    --gpu_memory_utilization 0.9 \
    --dtype bfloat16 \
    --max_model_len 8192 \
    --enable_prefix_caching true \
    --trust_remote_code true
