#!/bin/bash

# MultiArith 실험 실행 스크립트
# Router + Node Merge를 사용한 MultiArith 데이터셋 실험

echo "========================================"
echo "MultiArith Experiment with Router + Node Merge"
echo "========================================"
echo "Model: Meta-Llama-3.1-8B-Instruct"
echo "Dataset: MultiArith"
echo "Mode: FullConnected"
echo "========================================"

# vLLM 서버 확인
echo "Checking vLLM server..."
python3 check_vllm_server.py || {
    echo "ERROR: vLLM server is not running properly!"
    echo "Please ensure vLLM server is running:"
    echo "  vllm serve ${LLAMA_MODEL_ID:-meta-llama/Llama-3.1-8B-Instruct} --host 0.0.0.0 --port 6789 --max-model-len 8192"
    echo "Continuing anyway..."
}

echo "vLLM server is running."
echo ""

# 실험 실행
python run_experiment_multiarith.py \
    --router_config Router/config/gsm8k_config.yaml \
    --test_samples ${1:-600} \
    --mode FullConnected \
    --llm_name Meta-Llama-3.1-8B-Instruct \
    --num_iterations 2 \
    --merge_iterations 1 \
    --num_rounds 2 \
    --pruning_rate 0.1 \
    --lr 0.1 \
    --batch_size 4 \
    2>&1 | tee multiarith_experiment_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "========================================"
echo "Experiment completed!"
echo "Check the logs and outputs/ directory for results"
echo "========================================"
