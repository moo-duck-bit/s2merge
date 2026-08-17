#!/bin/bash

# GSM8K 전체 데이터셋 실험 스크립트
# Router + NodeMerge 통합 실험

echo "=========================================="
echo "GSM8K Full Dataset Experiment"
echo "   Router + NodeMerge with Llama-3.1-8B"
echo "=========================================="
echo ""

# 작업 디렉토리 설정 (스크립트가 놓인 위치를 저장소 루트로 사용)
WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WORKSPACE"

# vLLM 서버가 서빙 중인 모델 id (vllm serve 에 넘긴 --model 값과 같아야 함)
LLAMA_MODEL_ID="${LLAMA_MODEL_ID:-meta-llama/Llama-3.1-8B-Instruct}"

# 로그 파일 설정
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="gsm8k_full_experiment_${TIMESTAMP}.log"
echo "Log file: $LOG_FILE"
echo ""

# Python 환경 확인
echo "Python version:"
python --version
echo ""

# vLLM 서버 상태 확인
echo "Checking vLLM server status..."
if python -c "import requests; requests.get('http://localhost:6789/v1/models', timeout=2)" 2>/dev/null; then
    echo "vLLM server is running (port 6789)"
else
    echo "vLLM server is NOT running"
    echo "Please start the vLLM server first:"
    echo "    python -m vllm.entrypoints.openai.api_server \\"
    echo "        --model $LLAMA_MODEL_ID \\"
    echo "        --port 6789 \\"
    echo "        --max-model-len 8192 \\"
    echo "        --gpu-memory-utilization 0.9"
    exit 1
fi
echo ""

# GSM8K 데이터셋 확인
if [ ! -f "datasets/gsm8k/train.jsonl" ] || [ ! -f "datasets/gsm8k/gsm8k.jsonl" ]; then
    echo "GSM8K dataset not found"
    echo "   Run: python convert_to_jsonl.py"
    exit 1
else
    echo "GSM8K dataset exists"
    echo ""
fi

# 실험 시작 시간 기록
START_TIME=$(date +%s)
echo "⏰ Experiment started at: $(date)"
echo ""

# 실험 설정 출력
echo "Experiment Configuration:"
echo "   - Model: Meta-Llama-3.1-8B-Instruct"
echo "   - Mode: FullConnected"
echo "   - Test Samples: ALL (1319 samples)"
echo "   - Training Samples: 40"
echo "   - Batch Size: 4"
echo "   - Iterations: 10"
echo "   - Merge Iterations: 5"
echo "   - Num Rounds: 2"
echo "   - Learning Rate: 0.1"
echo "   - Pruning Rate: 0.10"
echo "   - Process Timeout: 3600s (1 hour per batch)"
echo ""
echo "This will take several hours to complete!"
echo ""

# 실험 실행
echo "Starting GSM8K full dataset experiment..."
echo "=========================================="
echo ""

# timeout을 각 배치가 아닌 전체 실험에 적용
# 전체 실험 타임아웃: 10800초 (3시간)
timeout 10800 python run_experiment_gsm8k.py \
    --llm_name "Meta-Llama-3.1-8B-Instruct" \
    --mode "FullConnected" \
    --num_iterations 10 \
    --merge_iterations 5 \
    --pruning_rate 0.10 \
    --num_rounds 2 \
    --lr 0.1 \
    --batch_size 4 \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=$?

# 실험 종료 시간 기록
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 124 ]; then
    echo "Experiment timed out after 3 hours"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "Experiment completed successfully!"
else
    echo "Experiment failed with exit code: $EXIT_CODE"
fi
echo "⏱Total duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "Log saved to: $LOG_FILE"
echo "Results saved to: outputs/"
echo "Conversation logs: result/Router-NodeMerge_GSM8K_*/logs/"
echo "=========================================="
