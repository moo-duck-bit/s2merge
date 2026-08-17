"""
Router + AgentDropout 통합 실험 스크립트

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
from dataset_load.mmlu_dataset import MMLUDataset
from dataset_download import download_mmlu
from AgentDropout.utils.const import AgentPrune_ROOT
from AgentDropout.utils.globals import PromptTokens, CompletionTokens, Cost
from experiments.accuracy import Accuracy
from experiments.train_mmlu_with_merge import train
from experiments.evaluate_mmlu import evaluate
from experiments.run_mmlu_with_merge import get_kwargs  # kwargs 생성 함수 임포트
import torch


def convert_router_agents_to_agentdropout(selected_agents: List[Dict[str, Any]]) -> tuple[List[str], List[Dict]]:
    """
    Router에서 선정된 에이전트를 AgentDropout 형식으로 변환
    
    Args:
        selected_agents: Router에서 반환한 final_agents
            예: [{'agent': 'Sociology_Philosophy_Ethics', 'probability': 0.28}, ...]
    
    Returns:
        (agent_names, node_kwargs) 튜플
        - agent_names: AgentDropout 에이전트 타입 리스트 (모두 'AnalyzeAgent')
        - node_kwargs: 각 에이전트의 kwargs (role 포함)
    """
    agent_names = []
    node_kwargs = []
    
    for agent_info in selected_agents:
        role_name = agent_info['agent']  # Router에서 선정한 역할명을 그대로 사용
        
        # AnalyzeAgent에 role 파라미터로 전달
        agent_names.append('AnalyzeAgent')
        node_kwargs.append({'role': role_name})
    
    return agent_names, node_kwargs


async def run_with_router_and_train(
    router_config: str,
    dataset_train: MMLUDataset,
    dataset_val: MMLUDataset,
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
    limit_questions: int,
    sample_question: str = None
) -> Dict[str, Any]:
    """
    Router로 에이전트를 선정한 후 AgentDropout의 전체 학습+평가 파이프라인 실행
    
    Args:
        router_config: Router config 파일 경로
        dataset_train: 학습용 데이터셋 (dev)
        dataset_val: 평가용 데이터셋 (val)
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
    
    # 비용 측정 시작
    router_cost_start = Cost.instance().value
    router_prompt_start = PromptTokens.instance().value
    router_completion_start = CompletionTokens.instance().value
    
    # 1. Router로 에이전트 선정
    print("\nStage 1: Agent selection with Router")
    print("-" * 80)
    
    # 샘플 질문이 없으면 val 데이터셋의 첫 질문 사용
    if sample_question is None:
        sample_record = dataset_val[0]
        sample_input = dataset_val.record_to_input(sample_record)
        sample_question = sample_input['task']
    
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
    
    # Router 비용 측정
    router_cost = Cost.instance().value - router_cost_start
    router_prompt = PromptTokens.instance().value - router_prompt_start
    router_completion = CompletionTokens.instance().value - router_completion_start
    
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
    
    # get_kwargs()로 Graph 파라미터 생성 (run_mmlu_with_merge.py와 동일)
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
            self.domain = 'mmlu'
            self.llm_name = llm_name
            self.mode = mode
            self.dec = True  # Node merge를 위해 True로 설정
            self.diff = True  # dec=True와 함께 사용
            self.optimized_spatial = True  # Node merge 사용
            self.optimized_temporal = True  # diff=True와 함께 사용
            self.merge_iterations = merge_iterations
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
        domain='mmlu',
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
    
    await train(
        graph=graph,
        dataset=dataset_train,
        num_iters=num_iterations,
        num_rounds=num_rounds,
        lr=lr,
        batch_size=batch_size,
        imp_per_iters=merge_iterations,
        pruning_rate=0.1,  # Not used in node merge (only for edge dropout)
        args=args,
        kwargs=graph_kwargs  # get_kwargs()에서 생성한 파라미터 전달
    )
    
    # Training 비용 측정
    train_cost = Cost.instance().value - train_cost_start
    train_prompt = PromptTokens.instance().value - train_prompt_start
    train_completion = CompletionTokens.instance().value - train_completion_start
    
    print("\nTraining completed!")
    print(f"   Final number of nodes: {len(graph.nodes)}")
    if not hasattr(graph.spatial_masks, '__iter__'):
        print(f"   Spatial sparsity: {graph.spatial_masks.sum()/graph.spatial_masks.numel():.3f}")
    
    # 5. 평가
    print(f"\nStage 5: Evaluation")
    print("="*80)
    
    eval_cost_start = Cost.instance().value
    eval_prompt_start = PromptTokens.instance().value
    eval_completion_start = CompletionTokens.instance().value
    
    score = await evaluate(
        graph=graph,
        dataset=dataset_val,
        num_rounds=num_rounds,
        limit_questions=limit_questions,
        eval_batch_size=10,
        dec=False,
        args=args
    )
    
    eval_cost = Cost.instance().value - eval_cost_start
    eval_prompt = PromptTokens.instance().value - eval_prompt_start
    eval_completion = CompletionTokens.instance().value - eval_completion_start
    
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
            'router_completion_tokens': router_completion
        },
        'training': {
            'num_iterations': num_iterations,
            'merge_iterations': merge_iterations,
            'final_num_nodes': len(graph.nodes),
            'train_cost': train_cost,
            'train_prompt_tokens': train_prompt,
            'train_completion_tokens': train_completion
        },
        'evaluation': {
            'accuracy': score,
            'eval_cost': eval_cost,
            'eval_prompt_tokens': eval_prompt,
            'eval_completion_tokens': eval_completion
        },
        'total': {
            'total_cost': Cost.instance().value,
            'total_prompt_tokens': PromptTokens.instance().value,
            'total_completion_tokens': CompletionTokens.instance().value
        },
        'metadata': {
            'mode': mode,
            'llm_name': llm_name,
            'decision_method': decision_method
        }
    }


async def run_mmlu_experiment(
    router_config: str,
    num_samples: int = None,
    val_samples: int = None,
    mode: str = 'FullConnected',
    llm_name: str = 'gpt-4o-mini',
    num_iterations: int = 10,
    merge_iterations: int = 5,
    pruning_rate: float = 0.10,
    num_rounds: int = 2,
    lr: float = 0.1,
    batch_size: int = 10,
    save_results: bool = True
) -> Dict[str, Any]:
    """
    MMLU 데이터셋에 대해 Router + AgentDropout 통합 실험 실행 (학습+평가 포함)
    
    Args:
        router_config: Router config 파일 경로
        num_samples: 학습에 사용할 샘플 수 (None이면 전체 dev 데이터 사용)
        val_samples: 평가에 사용할 샘플 수
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
    print("MMLU Router + AgentDropout integrated experiment (Training + Evaluation)")
    print("="*80)
    print(f"Mode: {mode}")
    print(f"LLM: {llm_name}")
    print(f"Iterations: {num_iterations} (Merge: {merge_iterations})")
    print(f"Val Samples: {val_samples}")
    print()
    
    # MMLU 데이터 로드 (이미 존재하면 download_mmlu() 스킵)
    try:
        dataset_train = MMLUDataset('dev')
        dataset_val = MMLUDataset('val')
    except:
        download_mmlu()
        dataset_train = MMLUDataset('dev')
        dataset_val = MMLUDataset('val')
    
    # 평가 데이터셋 샘플링
    if val_samples and val_samples < len(dataset_val):
        original_val_len = len(dataset_val)
        dataset_val._total_df = dataset_val._total_df.iloc[:val_samples]
        print(f"Evaluation data: {val_samples} samples (original: {original_val_len})")
    
    print(f"Training data: {len(dataset_train)} samples (full dev dataset)")
    print()
    
    # Router + Train + Evaluate 파이프라인 실행
    result = await run_with_router_and_train(
        router_config=router_config,
        dataset_train=dataset_train,
        dataset_val=dataset_val,
        sample_question=None,  # val 데이터셋의 첫 질문 사용
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
        limit_questions=val_samples
    )
    
    # 결과 저장
    if save_results:
        os.makedirs('outputs', exist_ok=True)
        output_file = f"outputs/router_agentdropout_mmlu_train.json"
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
    parser = argparse.ArgumentParser(description='Router + AgentDropout 통합 실험 (MMLU 학습+평가)')
    parser.add_argument('--router_config', type=str, default='Router/config/mmlu_config.yaml',
                        help='Router config 파일 경로')
    parser.add_argument('--val_samples', type=int, default=None,
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
    parser.add_argument('--batch_size', type=int, default=20,
                        help='Batch size')
    parser.add_argument('--no_save', action='store_true',
                        help='결과 저장 안 함')
    
    args = parser.parse_args()
    
    # 비동기 실험 실행
    summary = asyncio.run(run_mmlu_experiment(
        router_config=args.router_config,
        num_samples=None,  # 전체 dev 데이터 사용
        val_samples=args.val_samples,
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
