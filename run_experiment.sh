#!/bin/bash
# 14번째 라인이 테스트 데이터 샘플링 코드

export PYTHONPATH="/Users/shy/Documents/LAMDA Lab/WWW 2026+ICML 2026/AgentDropout:$PYTHONPATH"

python experiments/run_humaneval.py \
  --agent_nums 5 \
  --mode FullConnected \
  --batch_size 10 \
  --num_iterations 2 \
  --imp_per_iterations 1 \
  --pruning_rate 0.10 \
  --num_rounds 2 \
  --num_samples 100 \
  --llm_name gpt-4o-mini \
  --optimized_spatial \
  --optimized_temporal \
  --diff \
  --dec