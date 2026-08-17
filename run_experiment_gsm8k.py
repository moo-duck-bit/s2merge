"""
Router + AgentDropout 통합 실험 스크립트 (GSM8K)

Router로 에이전트를 선정한 후, AgentDropout의 merge 알고리즘으로 실행
"""

import sys
import os

# NodeMerge 프로젝트 루트를 sys.path에 추가
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)
sys.stdout.reconfigure(encoding='utf-8')

# HTTP 요청 로그 비활성화
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

import argparse
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional

from Router.src.router import Router
from AgentDropout.graph.graph_merge import Graph
from AgentDropout.tools.reader.readers import JSONLReader
from dataset_load.gsm8k_dataset import gsm_data_process, gsm_get_predict
from AgentDropout.utils.const import AgentPrune_ROOT
from AgentDropout.utils.globals import (
    PromptTokens, CompletionTokens, Cost,
    SunkPromptTokens, SunkCompletionTokens, SunkCost
)
from AgentDropout.utils.log import initialize_log_file, agent_conversation_log
from experiments.accuracy import Accuracy
from experiments.run_gsm8k_with_merge import (
    get_kwargs,  # GSM8K용 kwargs 생성 함수
    find_merge_candidate,  # Node merge 함수들
    find_merge_partner,
    merge_nodes_in_graph
)
from AgentDropout.utils.utils import nuclear_norm, frobenius_norm
import torch
import torch.nn.functional as F
import asyncio
import copy
import time

# GSM8K prompt set merge import (15개 role 지원)
from AgentDropout.prompt import gsm8k_prompt_set_merge


def dataloader(dataset, batch_size, i_batch):
    """Simple dataloader for GSM8K dataset"""
    start_idx = i_batch * batch_size
    end_idx = start_idx + batch_size
    return dataset[start_idx:end_idx]


def convert_router_agents_to_agentdropout(selected_agents: List[Dict[str, Any]]) -> tuple[List[str], List[Dict]]:
    """
    Router에서 선정된 에이전트를 AgentDropout 형식으로 변환

    Args:
        selected_agents: Router에서 반환한 final_agents
            예: [{'agent': 'Algebra_Expert', 'probability': 0.28}, ...]

    Returns:
        (agent_names, node_kwargs) 튜플
        - agent_names: AgentDropout 에이전트 타입 리스트 (모두 'MathSolver')
        - node_kwargs: 각 에이전트의 kwargs (role 포함)
    """
    # Router role 이름 → GSM8K prompt set role 이름 매핑
    # (Router는 underscore 사용, GSM8K는 공백 사용)
    ROLE_MAPPING = {
        'Mathematical_Analyst': 'Mathematical Analyst',
        'Math_Solver': 'Math Solver',
        'Programming_Expert': 'Programming Expert',
        'Inspector': 'Inspector',
        'Problem_Decomposer': 'Problem Decomposer',
        'Pattern_Recognizer': 'Pattern Recognizer',
        'Reverse_Engineer': 'Reverse Engineer',
        'Logical_Critic': 'Logical Critic',
        'Visualizer': 'Visualizer',
        'Axiomatic_Purist': 'Axiomatic Purist',
        'Unit_Checker': 'Unit Checker',
        'Step-back_Abstractionist': 'Step-back Abstractionist',
        'Edge_Case_Hunter': 'Edge Case Hunter',
        'Heuristic_Estimator': 'Heuristic Estimator',
        'Literal_Translator': 'Literal Translator',
    }

    agent_names = []
    node_kwargs = []

    for agent_info in selected_agents:
        router_role = agent_info['agent']  # Router에서 선정한 역할명

        # 매핑된 role 이름 사용, 매핑이 없으면 'Math Solver'를 기본값으로
        mapped_role = ROLE_MAPPING.get(router_role, 'Math Solver')

        # MathSolver에 role 파라미터로 전달
        agent_names.append('MathSolver')
        node_kwargs.append({'role': mapped_role})

    return agent_names, node_kwargs


async def run_with_router_and_train(
    router_config: str,
    dataset_train: list,
    dataset_val: list,
    mode: str,
    llm_name: str,
    decision_method: str,
    use_llm_summary: bool,
    num_iterations: int,
    merge_iterations: int,
    pruning_rate: float,
    num_rounds: int,
    lr: float,
    batch_size: int,
    eval_batch_size: int,
    limit_questions: int,
    sample_question: str = None
) -> Dict[str, Any]:
    """
    Router로 에이전트를 선정한 후 AgentDropout의 전체 학습+평가 파이프라인 실행

    Args:
        router_config: Router config 파일 경로
        dataset_train: 학습용 데이터셋 (train)
        dataset_val: 평가용 데이터셋 (test)
        sample_question: Router가 분석할 샘플 질문
        mode: Graph mode
        llm_name: LLM 모델 이름
        decision_method: 최종 결정 방법
        use_llm_summary: Router에서 LLM summary 사용 여부
        num_iterations: 총 최적화 iteration 수
        merge_iterations: Merge stage iteration 수
        pruning_rate: Edge pruning rate
        num_rounds: 최적화/추론 라운드 수
        lr: Learning rate
        batch_size: Batch size
        limit_questions: 평가 시 최대 질문 수

    Returns:
        결과 딕셔너리
    """
    print("="*80)
    print("Router + AgentDropout integrated pipeline (Training + Evaluation)")
    print("="*80)

    # 대화 로그 파일 초기화
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    experiment_name = f"Router-NodeMerge_GSM8K_{timestamp}"
    conversation_log_path = AgentPrune_ROOT / f'result/{experiment_name}/logs/agent_conversations.txt'
    conversation_log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(conversation_log_path, 'w', encoding='utf-8') as f:
        f.write(f"{'='*80}\n")
        f.write(f"Agent Conversation Log - {experiment_name}\n")
        f.write(f"{'='*80}\n\n")

    # 비용 측정 시작
    router_cost_start = Cost.instance().value
    router_prompt_start = PromptTokens.instance().value
    router_completion_start = CompletionTokens.instance().value

    # Sunk cost 초기화 (Router 단계 전)
    SunkPromptTokens.instance().reset()
    SunkCompletionTokens.instance().reset()
    SunkCost.instance().reset()

    # 1. Router로 에이전트 선정
    print("\nStage 1: Agent selection with Router")
    print("-" * 80)

    # 샘플 질문이 없으면 test 데이터셋의 첫 질문 사용
    if sample_question is None:
        sample_question = dataset_val[0]['task']

    print(f"Query: {sample_question[:150]}...")

    router = Router(router_config)
    router_result = router.route(sample_question, use_llm_summary=use_llm_summary)

    selected_agents = router_result['final_agents']
    print(f"\n{len(selected_agents)} agents selected:")
    for agent_info in selected_agents:
        print(f"   - {agent_info['agent']} (확률: {agent_info['probability']:.3f})")
    print(f"   선정된 Blocks: {router_result['selected_blocks']}")
    print(f"   Block Uncertainty: {router_result['uncertainty']['block_uncertainty']:.3f}")
    print(f"   Role Uncertainty: {router_result['uncertainty']['role_uncertainty']:.3f}")

    # Router 비용 측정 (T_sunk)
    router_cost = Cost.instance().value - router_cost_start
    router_prompt = PromptTokens.instance().value - router_prompt_start
    router_completion = CompletionTokens.instance().value - router_completion_start

    # T_sunk 측정 (Router 단계에서 사용된 토큰)
    t_sunk_prompt = SunkPromptTokens.instance().value
    t_sunk_completion = SunkCompletionTokens.instance().value
    t_sunk_total = t_sunk_prompt + t_sunk_completion

    print(f"\nT_sunk (Router Stage):")
    print(f"   Prompt Tokens: {t_sunk_prompt:,.0f}")
    print(f"   Completion Tokens: {t_sunk_completion:,.0f}")
    print(f"   Total: {t_sunk_total:,.0f}")

    # 대화 로그 기록
    with open(conversation_log_path, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"ROUTER STAGE (T_sunk)\n")
        f.write(f"{'='*80}\n")
        f.write(f"Query: {sample_question[:200]}...\n")
        f.write(f"\nSelected Agents:\n")
        for agent_info in selected_agents:
            f.write(f"  - {agent_info['agent']} (probability: {agent_info['probability']:.3f})\n")
        f.write(f"\nT_sunk Tokens:\n")
        f.write(f"  Prompt: {t_sunk_prompt:,.0f}\n")
        f.write(f"  Completion: {t_sunk_completion:,.0f}\n")
        f.write(f"  Total: {t_sunk_total:,.0f}\n")
        f.write(f"{'='*80}\n\n")

    # 2. Router 에이전트를 AgentDropout 형식으로 변환
    print("\nStage 2: Convert agents to AgentDropout format")
    print("-" * 80)
    agent_names, node_kwargs = convert_router_agents_to_agentdropout(selected_agents)
    print(f"AgentDropout format conversion completed:")
    for name, kwargs in zip(agent_names, node_kwargs):
        print(f"   - {name} (role: {kwargs['role']})")

    # 3. Graph 생성
    print("\nStage 3: Graph initialization")
    print("-" * 80)

    # get_kwargs()로 Graph 파라미터 생성
    graph_kwargs = get_kwargs(mode, len(agent_names))

    print(f"\nDEBUG: get_kwargs() returned:")
    for k, v in graph_kwargs.items():
        if k != 'fixed_spatial_masks' and k != 'fixed_temporal_masks':
            print(f"  {k}: {v}")

    # node_kwargs 병합: get_kwargs()의 기본값과 Router의 role 정보 결합
    default_node_kwargs = graph_kwargs.get('node_kwargs')
    if default_node_kwargs is None:
        # get_kwargs()가 node_kwargs를 생성하지 않은 경우 (일반 mode)
        default_node_kwargs = [{} for _ in range(len(agent_names))]

    for i, router_kwargs in enumerate(node_kwargs):
        if i < len(default_node_kwargs):
            # Router의 role 정보를 기본값에 추가 (덮어쓰지 않고 병합)
            default_node_kwargs[i].update(router_kwargs)
    graph_kwargs['node_kwargs'] = default_node_kwargs

    print(f"\nDEBUG: After merging node_kwargs:")
    print(f"  graph_kwargs['node_kwargs']: {graph_kwargs['node_kwargs']}")

    # Args 객체 먼저 생성 (Graph 초기화 전에 필요)
    class Args:
        """train 함수에 필요한 args 객체"""
        def __init__(self):
            self.use_node_merge = True
            self.domain = 'gsm8k'
            self.llm_name = llm_name
            self.mode = mode
            self.dec = True  # Node merge를 위해 True로 설정
            self.diff = True  # dec=True와 함께 사용
            self.optimized_spatial = True  # Node merge 사용
            self.optimized_temporal = True  # diff=True와 함께 사용
            self.merge_iterations = merge_iterations
            self.edge_iterations = merge_iterations  # Edge stage iterations (same as merge)
            self.num_iterations = num_iterations
            self.pruning_rate = pruning_rate  # Edge Dropout을 위한 pruning rate (Stage 2)
            self.delta = 0.1  # Frobenius norm threshold
            self.num_rounds = num_rounds  # 파라미터로 받은 값 사용 (최소 2 권장)

    args = Args()

    # num_rounds 검증 (diff=True일 때 최소 2 필요)
    if args.diff and args.num_rounds < 2:
        print(f"WARNING: num_rounds={args.num_rounds} but diff=True requires num_rounds >= 2")
        print(f"         Automatically setting num_rounds to 2")
        args.num_rounds = 2

    # graph_kwargs에서 명시적으로 덮어쓸 파라미터 제거
    # (args 값이 우선되도록)
    graph_kwargs_filtered = {k: v for k, v in graph_kwargs.items()
                            if k not in ['optimized_spatial', 'optimized_temporal',
                                        'rounds', 'diff', 'dec']}

    print(f"\nDEBUG: graph_kwargs_filtered:")
    print(f"  node_kwargs in filtered: {'node_kwargs' in graph_kwargs_filtered}")
    if 'node_kwargs' in graph_kwargs_filtered:
        print(f"  node_kwargs: {graph_kwargs_filtered['node_kwargs']}")

    graph = Graph(
        domain='gsm8k',
        llm_name=llm_name,
        agent_names=agent_names,
        decision_method=decision_method,
        optimized_spatial=args.optimized_spatial,  # args에서 가져옴
        optimized_temporal=args.optimized_temporal,  # args에서 가져옴
        rounds=args.num_rounds,
        diff=args.diff,
        dec=args.dec,
        **graph_kwargs_filtered  # 필터링된 파라미터만 사용
    )

    print(f"Graph creation completed: {len(graph.nodes)} nodes")
    print(f"  dec={args.dec}, diff={args.diff}")
    print(f"  optimized_spatial={args.optimized_spatial}, optimized_temporal={args.optimized_temporal}")
    print(f"  rounds={args.num_rounds}")

    # Set conversation log path for the graph
    graph.conversation_log_path = conversation_log_path
    print(f"Conversation logging enabled: {conversation_log_path}")

    # Graph 객체 내부 상태 확인
    print(f"\nDEBUG: Graph internal state:")
    print(f"  graph.diff: {graph.diff}")
    print(f"  graph.optimized_spatial: {graph.optimized_spatial}")
    print(f"  graph.optimized_temporal: {graph.optimized_temporal}")
    print(f"  graph.rounds: {graph.rounds}")
    if hasattr(graph, 'temporal_logits_1'):
        if isinstance(graph.temporal_logits_1, torch.nn.ParameterList):
            print(f"  graph.temporal_logits_1: ParameterList with length {len(graph.temporal_logits_1)}")
        else:
            print(f"  graph.temporal_logits_1: {type(graph.temporal_logits_1)}")
    else:
        print(f"  graph.temporal_logits_1: NOT FOUND")



    # 4. Training (Node Merge + Edge Optimization)
    print(f"\nStage 4: Training (Node Merge Algorithm)")
    print("="*80)

    # Training 비용 측정 시작
    train_cost_start = Cost.instance().value
    train_prompt_start = PromptTokens.instance().value
    train_completion_start = CompletionTokens.instance().value

    # === GSM8K Training Logic (from run_gsm8k_with_merge.py) ===
    graph.optimized_spatial = False
    graph.optimized_temporal = False

    if not graph.diff:
        optimizer = torch.optim.Adam([graph.spatial_logits_1, graph.temporal_logits_1], lr=lr)
    else:
        optimizer = torch.optim.Adam(
            list(graph.spatial_logits_1.parameters()) + list(graph.temporal_logits_1.parameters()),
            lr=lr
        )

    # MERGE STAGE: Training
    for i_batch in range(merge_iterations):
        print(f"\n{'='*80}")
        print(f"Merge Stage - Train Iteration {i_batch}")
        print(f"{'='*80}")
        start_ts = time.time()
        answer_log_probs = []
        answers = []
        add_losses = []

        current_batch = dataloader(dataset_train, batch_size, i_batch)
        if not current_batch:
            break

        for record in current_batch:
            realized_graph = copy.deepcopy(graph)
            realized_graph.spatial_logits_1 = graph.spatial_logits_1
            realized_graph.temporal_logits_1 = graph.temporal_logits_1

            n_nodes = len(list(graph.nodes.values()))
            if not graph.diff:
                spatial_matrix_train = realized_graph.spatial_logits_1.reshape((n_nodes, n_nodes))
                temporal_matrix_train = realized_graph.temporal_logits_1.reshape((n_nodes, n_nodes))
            else:
                spatial_matrix_train = [param.reshape((n_nodes, n_nodes)) for param in realized_graph.spatial_logits_1]
                temporal_matrix_train = [param.reshape((n_nodes, n_nodes)) for param in realized_graph.temporal_logits_1]

            spatial_matrix_fixed = torch.tensor(graph_kwargs["fixed_spatial_masks"], dtype=torch.float32).reshape((n_nodes, n_nodes))
            temporal_matrix_fixed = torch.tensor(graph_kwargs["fixed_temporal_masks"], dtype=torch.float32).reshape((n_nodes, n_nodes))

            if not graph.diff:
                loss_s = nuclear_norm(spatial_matrix_train)
                loss_t = nuclear_norm(temporal_matrix_train)
                frob_loss_s = frobenius_norm(spatial_matrix_fixed, spatial_matrix_train)
                frob_loss_t = frobenius_norm(temporal_matrix_fixed, temporal_matrix_train)
            else:
                loss_s = torch.mean(torch.stack([nuclear_norm(matrix) for matrix in spatial_matrix_train]))
                loss_t = torch.mean(torch.stack([nuclear_norm(matrix) for matrix in temporal_matrix_train]))
                frob_loss_s = torch.mean(torch.stack([frobenius_norm(spatial_matrix_fixed, matrix) for matrix in spatial_matrix_train]))
                frob_loss_t = torch.mean(torch.stack([frobenius_norm(temporal_matrix_fixed, matrix) for matrix in temporal_matrix_train]))

            add_loss = loss_s + loss_t + F.relu(frob_loss_s - args.delta) + F.relu(frob_loss_t - args.delta)
            add_loss = 0

            task = record["task"]
            answer = record["answer"]
            answers.append(answer)
            input_dict = {"task": task}
            answer_log_probs.append(asyncio.create_task(realized_graph.arun(input_dict, num_rounds, skip=True)))
            add_losses.append(add_loss)

        raw_results = await asyncio.gather(*answer_log_probs)
        raw_answers, log_probs = zip(*raw_results)
        loss_list = []
        utilities = []

        for answer_pair, log_prob, add_loss, true_answer in zip(raw_answers, log_probs, add_losses, answers):
            predict_answer = gsm_get_predict(answer_pair[0])
            try:
                is_solved = float(predict_answer) == float(true_answer)
            except:
                is_solved = False
            utility = float(is_solved)
            utilities.append(utility)
            single_loss = -log_prob * utility
            loss_list.append(single_loss + add_loss)

        total_loss = torch.mean(torch.stack(loss_list))
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        print(f"Batch time: {time.time() - start_ts:.3f}s")
        print(f"Average utility: {sum(utilities)/len(utilities):.3f}, Loss: {total_loss.item():.4f}")

    # MERGE STAGE: Node Merge
    if len(list(graph.nodes.values())) > 2:
        print(f"\n{'='*80}")
        print("PERFORMING NODE MERGE")
        print(f"{'='*80}\n")

        with torch.no_grad():
            if not graph.diff:
                weight_matrix = torch.sigmoid(graph.spatial_logits_1).reshape(
                    len(list(graph.nodes.values())), len(list(graph.nodes.values()))
                )
            else:
                weight_matrix = torch.mean(torch.stack([
                    torch.sigmoid(logit) for logit in graph.spatial_logits_1
                ]), dim=0).reshape(len(list(graph.nodes.values())), len(list(graph.nodes.values())))

        merge_candidate_id, min_delta_w = find_merge_candidate(graph, weight_matrix)
        execution_order = [node.id for node in list(graph.nodes.values())]
        partner_id, partner_weight = find_merge_partner(graph, weight_matrix, merge_candidate_id, execution_order)

        if partner_id is not None:
            print(f"Merging nodes: {partner_id} ← {merge_candidate_id}")
            graph = await merge_nodes_in_graph(graph, partner_id, merge_candidate_id, llm_name)
            print(f"After merge: {len(graph.nodes)} nodes remaining")

    # EDGE STAGE: Training (if more than 2 nodes remain)
    if len(list(graph.nodes.values())) > 2:
        for i_batch in range(args.edge_iterations):
            print(f"\n{'='*80}")
            print(f"Edge Stage - Train Iteration {i_batch}")
            print(f"{'='*80}")
            start_ts = time.time()
            answer_log_probs = []
            answers = []
            add_losses = []

            current_batch = dataloader(dataset_train, batch_size, i_batch)
            if not current_batch:
                break

            for record in current_batch:
                realized_graph = copy.deepcopy(graph)
                realized_graph.spatial_logits_1 = graph.spatial_logits_1
                realized_graph.temporal_logits_1 = graph.temporal_logits_1

                task = record["task"]
                answer = record["answer"]
                answers.append(answer)
                input_dict = {"task": task}
                answer_log_probs.append(asyncio.create_task(realized_graph.arun(input_dict, num_rounds, skip=True)))
                add_losses.append(0)

            raw_results = await asyncio.gather(*answer_log_probs)
            raw_answers, log_probs = zip(*raw_results)
            loss_list = []
            utilities = []

            for answer_pair, log_prob, add_loss, true_answer in zip(raw_answers, log_probs, add_losses, answers):
                predict_answer = gsm_get_predict(answer_pair[0])
                try:
                    is_solved = float(predict_answer) == float(true_answer)
                except:
                    is_solved = False
                utility = float(is_solved)
                utilities.append(utility)
                single_loss = -log_prob * utility
                loss_list.append(single_loss + add_loss)

            total_loss = torch.mean(torch.stack(loss_list))
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            print(f"Batch time: {time.time() - start_ts:.3f}s")
            print(f"Average utility: {sum(utilities)/len(utilities):.3f}, Loss: {total_loss.item():.4f}")

        # Edge Stage: Edge Dropout
        print(f"\n{'='*80}")
        print(f"EDGE STAGE: EDGE DROPOUT (rate={args.pruning_rate})")
        print(f"{'='*80}\n")

        if not graph.diff:
            spatial_masks, temporal_masks = graph.update_masks(args.pruning_rate)
        else:
            spatial_masks, temporal_masks = graph.update_masks_diff(args.pruning_rate)

        if not graph.diff:
            spatial_sparsity = sum(spatial_masks) / len(spatial_masks) if len(spatial_masks) > 0 else 0
            temporal_sparsity = sum(temporal_masks) / len(temporal_masks) if len(temporal_masks) > 0 else 0
        else:
            spatial_sparsity = sum(spatial_masks[0]) / len(spatial_masks[0]) if len(spatial_masks[0]) > 0 else 0
            temporal_sparsity = sum(temporal_masks[0]) / len(temporal_masks[0]) if len(temporal_masks[0]) > 0 else 0

        print(f"Spatial sparsity: {spatial_sparsity:.4f}")
        print(f"Temporal sparsity: {temporal_sparsity:.4f}")

    graph.optimized_spatial = True
    graph.optimized_temporal = True

    # === End of Training Logic ===

    # Training 비용 측정
    train_cost = Cost.instance().value - train_cost_start
    train_prompt = PromptTokens.instance().value - train_prompt_start
    train_completion = CompletionTokens.instance().value - train_completion_start

    # T_inference 계산 (Training 단계)
    train_t_inference_prompt = train_prompt  # Training에서는 Router를 사용하지 않으므로 전체가 inference
    train_t_inference_completion = train_completion
    train_t_inference_total = train_t_inference_prompt + train_t_inference_completion

    print(f"\n{'='*80}")
    print("TRAINING COMPLETED")
    print(f"{'='*80}")
    print(f"Training Cost: ${train_cost:.4f}")
    print(f"Training Tokens - Prompt: {train_prompt:,}, Completion: {train_completion:,}")
    print(f"Training T_inference: {train_t_inference_total:,.0f} tokens")
    print(f"Final Graph: {len(graph.nodes)} nodes")
    if not hasattr(graph.spatial_masks, '__iter__'):
        print(f"Spatial sparsity: {graph.spatial_masks.sum()/graph.spatial_masks.numel():.3f}")

    # 대화 로그 기록
    with open(conversation_log_path, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"TRAINING STAGE (Merge Algorithm)\n")
        f.write(f"{'='*80}\n")
        f.write(f"Iterations: {num_iterations} (Merge: {merge_iterations})\n")
        f.write(f"Final Graph: {len(graph.nodes)} nodes\n")
        f.write(f"\nTraining Tokens (T_inference):\n")
        f.write(f"  Prompt: {train_t_inference_prompt:,.0f}\n")
        f.write(f"  Completion: {train_t_inference_completion:,.0f}\n")
        f.write(f"  Total: {train_t_inference_total:,.0f}\n")
        f.write(f"{'='*80}\n\n")

    # 5. 평가 (GSM8K Evaluation)
    print(f"\nStage 5: Evaluation")
    print("="*80)

    eval_cost_start = Cost.instance().value
    eval_prompt_start = PromptTokens.instance().value
    eval_completion_start = CompletionTokens.instance().value

    # GSM8K evaluation logic (from run_gsm8k_with_merge.py)
    total_solved = 0
    total_executed = 0

    # Limit questions if specified
    eval_dataset = dataset_val[:limit_questions] if limit_questions else dataset_val

    num_batches = (len(eval_dataset) + eval_batch_size - 1) // eval_batch_size
    for i_batch in range(num_batches):
        print(f"Eval Batch {i_batch+1}/{num_batches}")
        start_ts = time.time()
        answer_log_probs = []
        answers = []

        current_batch = dataloader(eval_dataset, eval_batch_size, i_batch)
        if not current_batch:
            break

        for record in current_batch:
            realized_graph = copy.deepcopy(graph)
            realized_graph.spatial_logits = graph.spatial_logits
            realized_graph.temporal_logits = graph.temporal_logits

            task = record["task"]
            answer = record["answer"]
            answers.append(answer)
            input_dict = {"task": task}
            answer_log_probs.append(asyncio.create_task(realized_graph.arun(input_dict, num_rounds)))

        raw_results = await asyncio.gather(*answer_log_probs)
        raw_answers, log_probs = zip(*raw_results)

        for answer_pair, true_answer in zip(raw_answers, answers):
            predict_answer = gsm_get_predict(answer_pair[0])
            try:
                is_solved = float(predict_answer) == float(true_answer)
            except:
                is_solved = False
            total_solved += is_solved
            total_executed += 1

        batch_accuracy = total_solved / total_executed if total_executed > 0 else 0
        print(f"Batch time: {time.time() - start_ts:.3f}s, Cumulative Accuracy: {batch_accuracy:.4f}")

    score = total_solved / total_executed if total_executed > 0 else 0

    eval_cost = Cost.instance().value - eval_cost_start
    eval_prompt = PromptTokens.instance().value - eval_prompt_start
    eval_completion = CompletionTokens.instance().value - eval_completion_start

    # T_inference 계산 (Evaluation 단계)
    eval_t_inference_prompt = eval_prompt  # Evaluation에서도 Router를 사용하지 않으므로 전체가 inference
    eval_t_inference_completion = eval_completion
    eval_t_inference_total = eval_t_inference_prompt + eval_t_inference_completion

    # 전체 T_inference 계산 (Training + Evaluation)
    total_t_inference_prompt = train_t_inference_prompt + eval_t_inference_prompt
    total_t_inference_completion = train_t_inference_completion + eval_t_inference_completion
    total_t_inference = total_t_inference_prompt + total_t_inference_completion

    print(f"\n{'='*80}")
    print("EVALUATION COMPLETED")
    print(f"{'='*80}")
    print(f"Final Accuracy: {score:.4f} ({total_solved}/{total_executed})")
    print(f"Evaluation Cost: ${eval_cost:.4f}")
    print(f"Evaluation Tokens - Prompt: {eval_prompt:,}, Completion: {eval_completion:,}")
    print(f"Evaluation T_inference: {eval_t_inference_total:,.0f} tokens")

    print(f"\n{'='*80}")
    print("TOKEN USAGE SUMMARY")
    print(f"{'='*80}")
    print(f"T_sunk (Router): {t_sunk_total:,.0f} tokens")
    print(f"   - Prompt: {t_sunk_prompt:,.0f}")
    print(f"   - Completion: {t_sunk_completion:,.0f}")
    print(f"\nT_inference (Merge + Evaluation): {total_t_inference:,.0f} tokens")
    print(f"   - Training: {train_t_inference_total:,.0f}")
    print(f"   - Evaluation: {eval_t_inference_total:,.0f}")
    print(f"\nTotal Tokens: {PromptTokens.instance().value + CompletionTokens.instance().value:,.0f}")
    print(f"{'='*80}")

    # 대화 로그 기록
    with open(conversation_log_path, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"EVALUATION STAGE\n")
        f.write(f"{'='*80}\n")
        f.write(f"Accuracy: {score:.4f} ({total_solved}/{total_executed})\n")
        f.write(f"Samples: {limit_questions if limit_questions else 'all'}\n")
        f.write(f"\nEvaluation Tokens (T_inference):\n")
        f.write(f"  Prompt: {eval_t_inference_prompt:,.0f}\n")
        f.write(f"  Completion: {eval_t_inference_completion:,.0f}\n")
        f.write(f"  Total: {eval_t_inference_total:,.0f}\n")
        f.write(f"\n{'='*80}\n")
        f.write(f"FINAL TOKEN SUMMARY\n")
        f.write(f"{'='*80}\n")
        f.write(f"T_sunk (Router): {t_sunk_total:,.0f}\n")
        f.write(f"T_inference (Training): {train_t_inference_total:,.0f}\n")
        f.write(f"T_inference (Evaluation): {eval_t_inference_total:,.0f}\n")
        f.write(f"T_inference (Total): {total_t_inference:,.0f}\n")
        f.write(f"Grand Total: {PromptTokens.instance().value + CompletionTokens.instance().value:,.0f}\n")
        f.write(f"{'='*80}\n\n")

    print(f"\nConversation log saved: {conversation_log_path}")

    # 결과 반환
    return {
        'router_selection': {
            'selected_agents': selected_agents,
            'num_agents': len(selected_agents),
            'selected_blocks': router_result['selected_blocks'],
            'block_uncertainty': router_result['uncertainty']['block_uncertainty'],
            'role_uncertainty': router_result['uncertainty']['role_uncertainty'],
            'router_cost': router_cost,
            'router_prompt_tokens': router_prompt,
            'router_completion_tokens': router_completion,
            'T_sunk_Prompt_Tokens': float(t_sunk_prompt),
            'T_sunk_Completion_Tokens': float(t_sunk_completion),
            'T_sunk_Total_Tokens': float(t_sunk_total)
        },
        'training': {
            'num_iterations': num_iterations,
            'merge_iterations': merge_iterations,
            'final_num_nodes': len(graph.nodes),
            'train_cost': train_cost,
            'train_prompt_tokens': train_prompt,
            'train_completion_tokens': train_completion,
            'T_inference_Prompt_Tokens': float(train_t_inference_prompt),
            'T_inference_Completion_Tokens': float(train_t_inference_completion),
            'T_inference_Total_Tokens': float(train_t_inference_total)
        },
        'evaluation': {
            'accuracy': score,
            'eval_cost': eval_cost,
            'eval_prompt_tokens': eval_prompt,
            'eval_completion_tokens': eval_completion,
            'T_inference_Prompt_Tokens': float(eval_t_inference_prompt),
            'T_inference_Completion_Tokens': float(eval_t_inference_completion),
            'T_inference_Total_Tokens': float(eval_t_inference_total)
        },
        'total': {
            'total_cost': Cost.instance().value,
            'total_prompt_tokens': PromptTokens.instance().value,
            'total_completion_tokens': CompletionTokens.instance().value,
            'T_sunk_Total': float(t_sunk_total),
            'T_inference_Total': float(total_t_inference)
        },
        'metadata': {
            'mode': mode,
            'llm_name': llm_name,
            'decision_method': decision_method,
            'conversation_log_path': str(conversation_log_path)
        }
    }


async def run_gsm8k_experiment(
    router_config: str,
    test_samples: int = None,
    mode: str = 'FullConnected',
    llm_name: str = 'gpt-4o-mini',
    num_iterations: int = 10,
    merge_iterations: int = 5,
    pruning_rate: float = 0.10,
    num_rounds: int = 2,
    lr: float = 0.1,
    batch_size: int = 20,
    save_results: bool = True
) -> Dict[str, Any]:
    """
    GSM8K 데이터셋에 대해 Router + AgentDropout 통합 실험 실행 (학습+평가 포함)

    Args:
        router_config: Router config 파일 경로
        test_samples: 평가에 사용할 샘플 수
        mode: Graph mode
        llm_name: LLM 모델 이름
        num_iterations: 총 최적화 iteration 수
        merge_iterations: Merge stage iteration 수
        num_rounds: 라운드 수
        lr: Learning rate
        batch_size: Batch size
        save_results: 결과 저장 여부

    Returns:
        실험 결과 딕셔너리
    """
    print("="*80)
    print("GSM8K Router + AgentDropout integrated experiment (Training + Evaluation)")
    print("="*80)
    print(f"Mode: {mode}")
    print(f"LLM: {llm_name}")
    print(f"Iterations: {num_iterations} (Merge: {merge_iterations})")
    print(f"Test Samples: {test_samples}")
    print()

    # GSM8K 데이터 로드
    dataset_train = JSONLReader.parse_file('datasets/gsm8k/train.jsonl')
    dataset_train = gsm_data_process(dataset_train)

    dataset_test = JSONLReader.parse_file('datasets/gsm8k/gsm8k.jsonl')
    dataset_test = gsm_data_process(dataset_test)

    # 평가 데이터셋 샘플링
    if test_samples and test_samples < len(dataset_test):
        original_test_len = len(dataset_test)
        dataset_test = dataset_test[:test_samples]
        print(f"Evaluation data: {test_samples} samples (original: {original_test_len})")

    print(f"Training data: {len(dataset_train)} samples (full train dataset)")
    print()

    # Router + Train + Evaluate 파이프라인 실행
    result = await run_with_router_and_train(
        router_config=router_config,
        dataset_train=dataset_train,
        dataset_val=dataset_test,
        sample_question=None,  # test 데이터셋의 첫 질문 사용
        mode=mode,
        llm_name=llm_name,
        decision_method='FinalRefer',
        use_llm_summary=True,
        num_iterations=num_iterations,
        merge_iterations=merge_iterations,
        pruning_rate=pruning_rate,
        num_rounds=num_rounds,
        lr=lr,
        batch_size=batch_size,
        eval_batch_size=10,  # Evaluation batch size
        limit_questions=test_samples
    )

    # 결과 저장
    if save_results:
        os.makedirs('outputs', exist_ok=True)
        output_file = f"outputs/router_agentdropout_gsm8k_train.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nSave Results: {output_file}")

    # 요약 출력
    print("\n" + "="*80)
    print("Final Results Summary")
    print("="*80)
    print(f"Router selected agents: {result['router_selection']['num_agents']}")
    print(f"   → Number of nodes after training: {result['training']['final_num_nodes']}")
    print(f"\nAccuracy: {result['evaluation']['accuracy']*100:.1f}%")
    print(f"\nTotal Cost: ${result['total']['total_cost']:.4f}")
    print(f"   Router Cost: ${result['router_selection']['router_cost']:.4f}")
    print(f"   Training Cost: ${result['training']['train_cost']:.4f}")
    print(f"   Evaluation Cost: ${result['evaluation']['eval_cost']:.4f}")
    print(f"\nPrompt Tokens: {result['total']['total_prompt_tokens']:,.0f}")
    print(f"   Router: {result['router_selection']['router_prompt_tokens']:,.0f}")
    print(f"   Training: {result['training']['train_prompt_tokens']:,.0f}")
    print(f"   Evaluation: {result['evaluation']['eval_prompt_tokens']:,.0f}")
    print(f"\nCompletion Tokens: {result['total']['total_completion_tokens']:,.0f}")
    print(f"   Router: {result['router_selection']['router_completion_tokens']:,.0f}")
    print(f"   Training: {result['training']['train_completion_tokens']:,.0f}")
    print(f"   Evaluation: {result['evaluation']['eval_completion_tokens']:,.0f}")
    print("="*80)

    return result


def main():
    parser = argparse.ArgumentParser(description='Router + AgentDropout 통합 실험 (GSM8K 학습+평가)')
    parser.add_argument('--router_config', type=str, default='Router/config/gsm8k_config.yaml',
                        help='Router config 파일 경로')
    parser.add_argument('--test_samples', type=int, default=None,
                        help='평가에 사용할 샘플 수 (None = 전체 데이터 사용)')
    parser.add_argument('--mode', type=str, default='FullConnected',
                        choices=['DirectAnswer', 'FullConnected', 'Random', 'Chain', 'Debate', 'Layered', 'Star', 'Mesh'],
                        help='Graph mode')
    parser.add_argument('--llm_name', type=str, default='gpt-4o-mini',
                        help='LLM 모델 이름')
    parser.add_argument('--num_iterations', type=int, default=2,
                        help='총 최적화 iteration 수')
    parser.add_argument('--merge_iterations', type=int, default=1,
                        help='Merge stage iteration 수')
    parser.add_argument('--pruning_rate', type=float, default=0.1,
                        help='Edge pruning rate (Edge Dropout 비율)')
    parser.add_argument('--num_rounds', type=int, default=2,
                        help='라운드 수')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Batch size')
    parser.add_argument('--no_save', action='store_true',
                        help='결과 저장 안 함')

    args = parser.parse_args()

    # 비동기 실험 실행
    summary = asyncio.run(run_gsm8k_experiment(
        router_config=args.router_config,
        test_samples=args.test_samples,
        mode=args.mode,
        llm_name=args.llm_name,
        num_iterations=args.num_iterations,
        merge_iterations=args.merge_iterations,
        pruning_rate=args.pruning_rate,
        num_rounds=args.num_rounds,
        lr=args.lr,
        batch_size=args.batch_size,
        save_results=not args.no_save
    ))


if __name__ == "__main__":
    main()
