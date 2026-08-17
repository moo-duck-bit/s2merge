#!/bin/bash
# Node Merge 알고리즘을 사용한 MMLU 학습

export PYTHONPATH="/Users/shy/Documents/LAMDA Lab/WWW 2026+ICML 2026/AgentDropout:$PYTHONPATH"

python experiments/run_mmlu_with_merge.py \
  --agent_nums 5 \
  --mode FullConnected \
  --batch_size 10 \
  --num_iterations 2 \
  --merge_iterations 1 \
  --pruning_rate 0.10 \
  --num_rounds 2 \
  --num_samples 100 \
  --llm_name gpt-4o-mini \
  --optimized_spatial \
  --optimized_temporal \
  --diff \
  --dec \
  --use_node_merge