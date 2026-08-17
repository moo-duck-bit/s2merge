"""
Router + AgentDropout 통합 실험 스크립트 (HumanEval)

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
from AgentDropout.tools.coding.python_executor import PyExecutor
from AgentDropout.utils.const import AgentPrune_ROOT
from AgentDropout.utils.globals import (
    PromptTokens, CompletionTokens, Cost,
    SunkPromptTokens, SunkCompletionTokens, SunkCost,
    MergePromptTokens, MergeCompletionTokens, MergeCost,
    Time
)
from experiments.accuracy import Accuracy
from experiments.run_humaneval_with_merge import (
    get_kwargs,  # HumanEval용 kwargs 생성 함수
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

# HumanEval prompt set import
from AgentDropout.prompt import humaneval_prompt_set_merge


def dataloader(dataset, batch_size, i_batch):
    """Simple dataloader for HumanEval dataset"""
    start_idx = i_batch * batch_size
    end_idx = start_idx + batch_size
    return dataset[start_idx:end_idx]


def convert_router_agents_to_agentdropout(selected_agents: List[Dict[str, Any]]) -> tuple[List[str], List[Dict]]:
    """
    Router에서 선정된 에이전트를 AgentDropout 형식으로 변환
    
    Args:
        selected_agents: Router에서 반환한 final_agents
    
    Returns:
        (agent_names, node_kwargs) 튜플
        - agent_names: AgentDropout 에이전트 타입 리스트 (모두 'CodeWriting')
        - node_kwargs: 각 에이전트의 kwargs
    """
    agent_names = []
    node_kwargs = []
    
    for agent_info in selected_agents:
        # HumanEval은 코딩 에이전트 사용
        agent_names.append('CodeWriting')
        node_kwargs.append({})
    
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
    num_rounds: int,
    pruning_rate: float,
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
        num_rounds: 최적화/추론 라운드 수
        lr: Learning rate
        batch_size: Batch size
        eval_batch_size: Evaluation batch size
        limit_questions: 평가 시 최대 질문 수
    
    Returns:
        결과 딕셔너리
    """
    print("="*80)
    print("Router + AgentDropout integrated pipeline (Training + Evaluation)")
    print("="*80)
    
    # 전역 토큰 카운터 초기화
    PromptTokens.instance().reset()
    CompletionTokens.instance().reset()
    Cost.instance().reset()
    SunkPromptTokens.instance().reset()
    SunkCompletionTokens.instance().reset()
    SunkCost.instance().reset()
    MergePromptTokens.instance().reset()
    MergeCompletionTokens.instance().reset()
    MergeCost.instance().reset()
    
    # T_sunk (Router 비용) 측정 시작
    router_cost_start = Cost.instance().value
    router_prompt_start = PromptTokens.instance().value
    router_completion_start = CompletionTokens.instance().value
    
    # 1. Router로 에이전트 선정
    print("\nStage 1: Agent selection with Router (T_sunk)")
    print("-" * 80)
    
    # 샘플 질문이 없으면 test 데이터셋의 첫 질문 사용
    if sample_question is None:
        sample_question = dataset_val[0]['prompt']
    
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
    
    # T_sunk: Router 비용 측정 및 누적
    router_cost = Cost.instance().value - router_cost_start
    router_prompt = PromptTokens.instance().value - router_prompt_start
    router_completion = CompletionTokens.instance().value - router_completion_start
    
    SunkCost.instance().add(router_cost)
    SunkPromptTokens.instance().add(router_prompt)
    SunkCompletionTokens.instance().add(router_completion)
    
    print(f"\n[T_sunk] Router Cost: ${router_cost:.4f}")
    print(f"[T_sunk] Router Tokens - Prompt: {router_prompt:,}, Completion: {router_completion:,}")
    
    # 2. Router 에이전트를 AgentDropout 형식으로 변환
    print("\nStage 2: Convert agents to AgentDropout format")
    print("-" * 80)
    agent_names, node_kwargs = convert_router_agents_to_agentdropout(selected_agents)
    print(f"AgentDropout format conversion completed:")
    for name, kwargs in zip(agent_names, node_kwargs):
        print(f"   - {name}")
    
    # 3. Graph 생성
    print("\nStage 3: Graph initialization")
    print("-" * 80)
    
    # get_kwargs()로 Graph 파라미터 생성
    graph_kwargs = get_kwargs(mode, len(agent_names))
    
    # node_kwargs 병합
    default_node_kwargs = graph_kwargs.get('node_kwargs')
    if default_node_kwargs is None:
        default_node_kwargs = [{} for _ in range(len(agent_names))]
    
    for i, router_kwargs in enumerate(node_kwargs):
        if i < len(default_node_kwargs):
            default_node_kwargs[i].update(router_kwargs)
    graph_kwargs['node_kwargs'] = default_node_kwargs
    
    # Args 객체 생성
    class Args:
        def __init__(self):
            self.use_node_merge = True
            self.domain = 'humaneval'
            self.llm_name = llm_name
            self.mode = mode
            self.dec = True
            self.diff = True
            self.optimized_spatial = True
            self.optimized_temporal = True
            self.merge_iterations = merge_iterations
            self.edge_iterations = merge_iterations
            self.num_iterations = num_iterations
            self.pruning_rate = pruning_rate  # Edge Dropout을 위한 pruning rate (Stage 2)
            self.delta = 0.1
            self.num_rounds = num_rounds
    
    args = Args()
    
    if args.diff and args.num_rounds < 2:
        print(f"WARNING: num_rounds={args.num_rounds} but diff=True requires num_rounds >= 2")
        print(f"         Automatically setting num_rounds to 2")
        args.num_rounds = 2
    
    graph_kwargs_filtered = {k: v for k, v in graph_kwargs.items() 
                            if k not in ['optimized_spatial', 'optimized_temporal', 
                                        'rounds', 'diff', 'dec']}
    
    graph = Graph(
        domain='humaneval',
        llm_name=llm_name,
        agent_names=agent_names,
        decision_method=decision_method,
        optimized_spatial=args.optimized_spatial,
        optimized_temporal=args.optimized_temporal,
        rounds=args.num_rounds,
        diff=args.diff,
        dec=args.dec,
        **graph_kwargs_filtered
    )
    
    print(f"Graph creation completed: {len(graph.nodes)} nodes")
    print(f"  dec={args.dec}, diff={args.diff}")
    print(f"  optimized_spatial={args.optimized_spatial}, optimized_temporal={args.optimized_temporal}")
    print(f"  rounds={args.num_rounds}")
    
    # 에이전트 대화 로그 설정 (토큰에 영향 없이 로깅)
    current_time = Time.instance().value or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    Time.instance().value = current_time
    log_dir = Path(f"{AgentPrune_ROOT}/result/Router-NodeMerge_HumanEval_{current_time}/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    conversation_log_path = log_dir / "agent_conversations.txt"
    
    # 그래프에 로그 경로 설정
    graph.conversation_log_path = conversation_log_path
    
    print(f"  Agent conversation log: {conversation_log_path}")
    
    # Initialize PyExecutor for code execution
    executor = PyExecutor()
    
    # 4. Training (Node Merge + Edge Optimization) - T_merge 단계
    print(f"\nStage 4: Training (Node Merge Algorithm - T_merge)")
    print("="*80)
    
    # T_merge: Merge Stage 비용 측정 시작 (기존 T_sunk 제외)
    merge_stage_cost_start = Cost.instance().value
    merge_stage_prompt_start = PromptTokens.instance().value
    merge_stage_completion_start = CompletionTokens.instance().value
    
    # === HumanEval Training Logic ===
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
        test_cases_list = []
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
            
            prompt = record["prompt"]
            test_cases = record["test"]
            test_cases_list.append(test_cases)
            input_dict = {"task": prompt}
            answer_log_probs.append(asyncio.create_task(realized_graph.arun(input_dict, num_rounds, skip=True)))
            add_losses.append(add_loss)
        
        raw_results = await asyncio.gather(*answer_log_probs)
        raw_answers, log_probs = zip(*raw_results)
        loss_list = []
        utilities = []
        
        for answer_pair, log_prob, add_loss, test_cases in zip(raw_answers, log_probs, add_losses, test_cases_list):
            code = answer_pair[0].lstrip("```python\n").rstrip("\n```")
            try:
                is_solved, _, _ = executor.execute(code, [test_cases], timeout=100)
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
    
    # EDGE STAGE: Training
    if len(list(graph.nodes.values())) > 2:
        for i_batch in range(args.edge_iterations):
            print(f"\n{'='*80}")
            print(f"Edge Stage - Train Iteration {i_batch}")
            print(f"{'='*80}")
            start_ts = time.time()
            answer_log_probs = []
            test_cases_list = []
            add_losses = []
            
            current_batch = dataloader(dataset_train, batch_size, i_batch)
            if not current_batch:
                break
            
            for record in current_batch:
                realized_graph = copy.deepcopy(graph)
                realized_graph.spatial_logits_1 = graph.spatial_logits_1
                realized_graph.temporal_logits_1 = graph.temporal_logits_1
                
                prompt = record["prompt"]
                test_cases = record["test"]
                test_cases_list.append(test_cases)
                input_dict = {"task": prompt}
                answer_log_probs.append(asyncio.create_task(realized_graph.arun(input_dict, num_rounds, skip=True)))
                add_losses.append(0)
            
            raw_results = await asyncio.gather(*answer_log_probs)
            raw_answers, log_probs = zip(*raw_results)
            loss_list = []
            utilities = []
            
            for answer_pair, log_prob, add_loss, test_cases in zip(raw_answers, log_probs, add_losses, test_cases_list):
                code = answer_pair[0].lstrip("```python\n").rstrip("\n```")
                try:
                    is_solved, _, _ = executor.execute(code, [test_cases], timeout=100)
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
    
    # T_merge: Merge Stage 비용 계산 (Router 비용 제외)
    merge_stage_cost = Cost.instance().value - merge_stage_cost_start
    merge_stage_prompt = PromptTokens.instance().value - merge_stage_prompt_start
    merge_stage_completion = CompletionTokens.instance().value - merge_stage_completion_start
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETED (T_merge)")
    print(f"{'='*80}")
    print(f"[T_merge] Total Training Cost: ${merge_stage_cost:.4f}")
    print(f"[T_merge] Training Tokens - Prompt: {merge_stage_prompt:,}, Completion: {merge_stage_completion:,}")
    print(f"[T_merge] LLM Merge Operations - Cost: ${MergeCost.instance().value:.4f}, "
          f"Prompt: {MergePromptTokens.instance().value:,.0f}, "
          f"Completion: {MergeCompletionTokens.instance().value:,.0f}")
    print(f"Final Graph: {len(graph.nodes)} nodes")
    
    # 5. 평가 (HumanEval Evaluation) - T_inference 단계
    print(f"\nStage 5: Evaluation (T_inference)")
    print("="*80)
    
    # T_inference: 추론 비용 측정 시작 (Router + Merge 비용 제외)
    inference_cost_start = Cost.instance().value
    inference_prompt_start = PromptTokens.instance().value
    inference_completion_start = CompletionTokens.instance().value
    
    # HumanEval evaluation logic
    total_solved = 0
    total_executed = 0
    
    eval_dataset = dataset_val[:limit_questions] if limit_questions else dataset_val
    
    num_batches = (len(eval_dataset) + eval_batch_size - 1) // eval_batch_size
    for i_batch in range(num_batches):
        print(f"Eval Batch {i_batch+1}/{num_batches}")
        start_ts = time.time()
        answer_log_probs = []
        test_cases_list = []
        
        current_batch = dataloader(eval_dataset, eval_batch_size, i_batch)
        if not current_batch:
            break
        
        for record in current_batch:
            realized_graph = copy.deepcopy(graph)
            realized_graph.spatial_logits = graph.spatial_logits
            realized_graph.temporal_logits = graph.temporal_logits
            
            prompt = record["prompt"]
            test_cases = record["test"]
            test_cases_list.append(test_cases)
            input_dict = {"task": prompt}
            answer_log_probs.append(asyncio.create_task(realized_graph.arun(input_dict, num_rounds)))
        
        raw_results = await asyncio.gather(*answer_log_probs)
        raw_answers, log_probs = zip(*raw_results)
        
        for answer_pair, test_cases in zip(raw_answers, test_cases_list):
            code = answer_pair[0].lstrip("```python\n").rstrip("\n```")
            try:
                is_solved, _, _ = executor.execute(code, [test_cases], timeout=100)
            except:
                is_solved = False
            total_solved += is_solved
            total_executed += 1
        
        batch_accuracy = total_solved / total_executed if total_executed > 0 else 0
        print(f"Batch time: {time.time() - start_ts:.3f}s, Cumulative Pass@1: {batch_accuracy:.4f}")
    
    score = total_solved / total_executed if total_executed > 0 else 0
    
    # T_inference: 추론 비용 계산
    inference_cost = Cost.instance().value - inference_cost_start
    inference_prompt = PromptTokens.instance().value - inference_prompt_start
    inference_completion = CompletionTokens.instance().value - inference_completion_start
    
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETED (T_inference)")
    print(f"{'='*80}")
    print(f"Final Pass@1: {score:.4f} ({total_solved}/{total_executed})")
    print(f"[T_inference] Evaluation Cost: ${inference_cost:.4f}")
    print(f"[T_inference] Evaluation Tokens - Prompt: {inference_prompt:,}, Completion: {inference_completion:,}")
    
    # 결과 반환 (T_sunk, T_merge, T_inference 분리)
    return {
        'router_selection': {
            'selected_agents': selected_agents,
            'num_agents': len(selected_agents),
            'selected_blocks': router_result['selected_blocks'],
            'block_uncertainty': router_result['uncertainty']['block_uncertainty'],
            'role_uncertainty': router_result['uncertainty']['role_uncertainty']
        },
        'cost_breakdown': {
            'T_sunk': {
                'description': 'Router stage (sunk cost)',
                'cost': SunkCost.instance().value,
                'prompt_tokens': SunkPromptTokens.instance().value,
                'completion_tokens': SunkCompletionTokens.instance().value,
                'total_tokens': SunkPromptTokens.instance().value + SunkCompletionTokens.instance().value
            },
            'T_merge': {
                'description': 'Node Merge stage (structure optimization)',
                'cost': merge_stage_cost,
                'prompt_tokens': merge_stage_prompt,
                'completion_tokens': merge_stage_completion,
                'total_tokens': merge_stage_prompt + merge_stage_completion,
                'llm_merge_operations': {
                    'cost': MergeCost.instance().value,
                    'prompt_tokens': MergePromptTokens.instance().value,
                    'completion_tokens': MergeCompletionTokens.instance().value
                }
            },
            'T_inference': {
                'description': 'Final problem solving (inference)',
                'cost': inference_cost,
                'prompt_tokens': inference_prompt,
                'completion_tokens': inference_completion,
                'total_tokens': inference_prompt + inference_completion
            }
        },
        'training': {
            'num_iterations': num_iterations,
            'merge_iterations': merge_iterations,
            'final_num_nodes': len(graph.nodes)
        },
        'evaluation': {
            'pass_at_1': score,
            'total_solved': total_solved,
            'total_executed': total_executed
        },
        'total': {
            'total_cost': Cost.instance().value,
            'total_prompt_tokens': PromptTokens.instance().value,
            'total_completion_tokens': CompletionTokens.instance().value,
            'total_tokens': PromptTokens.instance().value + CompletionTokens.instance().value
        },
        'metadata': {
            'mode': mode,
            'llm_name': llm_name,
            'decision_method': decision_method
        }
    }


async def run_humaneval_experiment(
    router_config: str,
    test_samples: int = None,
    mode: str = 'FullConnected',
    llm_name: str = 'gpt-4o-mini',
    num_iterations: int = 2,
    merge_iterations: int = 1,
    num_rounds: int = 2,
    pruning_rate: float = 0.10,
    lr: float = 0.1,
    batch_size: int = 10,
    save_results: bool = True
) -> Dict[str, Any]:
    """
    HumanEval 데이터셋에 대해 Router + AgentDropout 통합 실험 실행 (학습+평가 포함)
    
    Args:
        router_config: Router config 파일 경로
        test_samples: 평가에 사용할 샘플 수
        mode: Graph mode
        llm_name: LLM 모델 이름
        num_iterations: 총 최적화 iteration 수
        merge_iterations: Merge stage iteration 수
        num_rounds: 라운드 수
        pruning_rate: Edge pruning rate
        lr: Learning rate
        batch_size: Batch size
        save_results: 결과 저장 여부
    
    Returns:
        실험 결과 딕셔너리
    """
    print("="*80)
    print("HumanEval Router + AgentDropout integrated experiment (Training + Evaluation)")
    print("="*80)
    print(f"Mode: {mode}")
    print(f"LLM: {llm_name}")
    print(f"Iterations: {num_iterations} (Merge: {merge_iterations})")
    print(f"Test Samples: {test_samples}")
    print()
    
    # HumanEval 데이터 로드
    dataset = JSONLReader.parse_file('dataset_load/humaneval/humaneval-py.jsonl')
    
    # 평가 데이터셋 샘플링
    if test_samples and test_samples < len(dataset):
        original_test_len = len(dataset)
        dataset_test = dataset[:test_samples]
        print(f"Evaluation data: {test_samples} samples (original: {original_test_len})")
    else:
        dataset_test = dataset
    
    # 전체 데이터를 학습에 사용
    dataset_train = dataset
    print(f"Training data: {len(dataset_train)} samples (full dataset)")
    print()
    
    # Router + Train + Evaluate 파이프라인 실행
    result = await run_with_router_and_train(
        router_config=router_config,
        dataset_train=dataset_train,
        dataset_val=dataset_test,
        sample_question=None,
        mode=mode,
        llm_name=llm_name,
        decision_method='FinalWriteCode',
        use_llm_summary=True,
        num_iterations=num_iterations,
        merge_iterations=merge_iterations,
        num_rounds=num_rounds,
        pruning_rate=pruning_rate,
        lr=lr,
        batch_size=batch_size,
        eval_batch_size=10,
        limit_questions=test_samples
    )
    
    # 결과 저장
    if save_results:
        os.makedirs('outputs', exist_ok=True)
        output_file = f"outputs/router_agentdropout_humaneval_train.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nSave Results: {output_file}")
    
    # 요약 출력
    print("\n" + "="*80)
    print("Final Results Summary")
    print("="*80)
    print(f"Router selected agents: {result['router_selection']['num_agents']}")
    print(f"   → Number of nodes after training: {result['training']['final_num_nodes']}")
    print(f"\nPass@1: {result['evaluation']['pass_at_1']*100:.1f}% ({result['evaluation']['total_solved']}/{result['evaluation']['total_executed']})")
    
    print(f"\n{'='*80}")
    print("Token Cost Breakdown")
    print(f"{'='*80}")
    
    # T_sunk
    print(f"\n[T_sunk] Router Stage (Sunk Cost)")
    print(f"   Cost: ${result['cost_breakdown']['T_sunk']['cost']:.4f}")
    print(f"   Tokens: {result['cost_breakdown']['T_sunk']['total_tokens']:,.0f} "
          f"(Prompt: {result['cost_breakdown']['T_sunk']['prompt_tokens']:,.0f}, "
          f"Completion: {result['cost_breakdown']['T_sunk']['completion_tokens']:,.0f})")
    
    # T_merge
    print(f"\n[T_merge] Node Merge Stage (Structure Optimization)")
    print(f"   Cost: ${result['cost_breakdown']['T_merge']['cost']:.4f}")
    print(f"   Tokens: {result['cost_breakdown']['T_merge']['total_tokens']:,.0f} "
          f"(Prompt: {result['cost_breakdown']['T_merge']['prompt_tokens']:,.0f}, "
          f"Completion: {result['cost_breakdown']['T_merge']['completion_tokens']:,.0f})")
    print(f"   ├─ LLM Merge Operations: ${result['cost_breakdown']['T_merge']['llm_merge_operations']['cost']:.4f}")
    print(f"   └─ Tokens: {result['cost_breakdown']['T_merge']['llm_merge_operations']['prompt_tokens']:,.0f} prompt + "
          f"{result['cost_breakdown']['T_merge']['llm_merge_operations']['completion_tokens']:,.0f} completion")
    
    # T_inference
    print(f"\n[T_inference] Inference Stage (Final Problem Solving)")
    print(f"   Cost: ${result['cost_breakdown']['T_inference']['cost']:.4f}")
    print(f"   Tokens: {result['cost_breakdown']['T_inference']['total_tokens']:,.0f} "
          f"(Prompt: {result['cost_breakdown']['T_inference']['prompt_tokens']:,.0f}, "
          f"Completion: {result['cost_breakdown']['T_inference']['completion_tokens']:,.0f})")
    
    # Total
    print(f"\n{'─'*80}")
    print(f"[Total] Overall Cost")
    print(f"   Cost: ${result['total']['total_cost']:.4f}")
    print(f"   Tokens: {result['total']['total_tokens']:,.0f} "
          f"(Prompt: {result['total']['total_prompt_tokens']:,.0f}, "
          f"Completion: {result['total']['total_completion_tokens']:,.0f})")
    print("="*80)
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Router + AgentDropout 통합 실험 (HumanEval 학습+평가)')
    parser.add_argument('--router_config', type=str, default='Router/config/humaneval_config.yaml',
                        help='Router config 파일 경로 (HumanEval은 GSM8K config 사용)')
    parser.add_argument('--test_samples', type=int, default=None,
                        help='평가에 사용할 샘플 수 (None = 전체 데이터 사용)')
    parser.add_argument('--mode', type=str, default='FullConnected',
                        choices=['DirectAnswer', 'FullConnected', 'Random', 'Chain', 'Debate', 'Layered', 'Star', 'Mesh'],
                        help='Graph mode')
    parser.add_argument('--llm_name', type=str, default='Meta-Llama-3.1-8B-Instruct',
                        help='LLM 모델 이름 (Meta-Llama-3.1-8B-Instruct)')
    parser.add_argument('--num_iterations', type=int, default=2,
                        help='총 최적화 iteration 수')
    parser.add_argument('--merge_iterations', type=int, default=1,
                        help='Merge stage iteration 수')
    parser.add_argument('--num_rounds', type=int, default=2,
                        help='라운드 수')
    parser.add_argument('--pruning_rate', type=float, default=0.1,
                        help='Edge dropout pruning rate')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--no_save', action='store_true',
                        help='결과 저장 안 함')
    
    args = parser.parse_args()
    
    # 비동기 실험 실행
    summary = asyncio.run(run_humaneval_experiment(
        router_config=args.router_config,
        test_samples=args.test_samples,
        mode=args.mode,
        llm_name=args.llm_name,
        num_iterations=args.num_iterations,
        merge_iterations=args.merge_iterations,
        num_rounds=args.num_rounds,
        pruning_rate=args.pruning_rate,
        lr=args.lr,
        batch_size=args.batch_size,
        save_results=not args.no_save
    ))


if __name__ == "__main__":
    main()
