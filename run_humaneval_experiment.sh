#!/bin/bash

# HumanEval 실험 실행 스크립트
# Router + Node Merge를 사용한 HumanEval 데이터셋 실험

echo "========================================"
echo "HumanEval Experiment with Router + Node Merge"
echo "========================================"
echo "Model: Meta-Llama-3.1-8B-Instruct"
echo "Dataset: HumanEval"
echo "Mode: FullConnected"
echo "========================================"

# vLLM 서버 확인
echo "Checking vLLM server..."
curl -s http://localhost:6789/health || {
    echo "ERROR: vLLM server is not running!"
    echo "Please start vLLM server first:"
    echo "  vllm serve ${LLAMA_MODEL_ID:-meta-llama/Llama-3.1-8B-Instruct} --host 0.0.0.0 --port 6789 --max-model-len 8192"
    exit 1
}

echo "vLLM server is running."
echo ""

# 실험 실행
python run_experiment_humaneval.py \
    --router_config Router/config/humaneval_config.yaml \
    --test_samples 164 \
    --mode FullConnected \
    --llm_name Meta-Llama-3.1-8B-Instruct \
    --num_iterations 2 \
    --merge_iterations 1 \
    --num_rounds 2 \
    --pruning_rate 0.1 \
    --lr 0.1 \
    --batch_size 4 \
    2>&1 | tee humaneval_experiment_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "========================================"
echo "Experiment completed!"
echo "Check the logs and outputs/ directory for results"
echo "========================================"
