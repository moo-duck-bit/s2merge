import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

import asyncio
from typing import Union, Literal, List
import argparse
import random

from AgentDropout.graph.graph_merge import Graph
from dataset_load.mmlu_dataset import MMLUDataset
from dataset_download import download_mmlu as download
from experiments.train_mmlu_with_merge import train
from experiments.evaluate_mmlu import evaluate
from AgentDropout.utils.const import AgentPrune_ROOT
from AgentDropout.utils.globals import PromptTokens, CompletionTokens, Cost
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="MMLU with Node Merge Algorithm")

    parser.add_argument('--mode', type=str, default='FullConnected',
                        choices=['DirectAnswer', 'FullConnected', 'Random', 'Chain', 'Debate', 'Layered','Star', 'Mesh',
                                 'FakeFullConnected','FakeRandom','FakeChain','FakeStar','FakeMesh','FakeAGRandom','FakeAGFull'],
                        help="Mode of operation. Default is 'FullConnected'.")
    parser.add_argument('--lr', type=float, default=0.1,
                        help="learning rate")
    parser.add_argument('--delta', type=float, default=0.1,
                        help="noise level")
    parser.add_argument('--batch_size', type=int, default=4,
                        help="batch size")
    parser.add_argument('--agent_names', nargs='+', type=str, default=['AnalyzeAgent'],
                        help='Specify agent names as a list of strings')
    parser.add_argument('--agent_nums', nargs='+', type=int, default=[5],
                        help='Specify the number of agents for each name in agent_names')
    parser.add_argument('--num_iterations', type=int, default=10,
                        help="Total number of optimization iterations (merge + edge optimization). Default 10.")
    parser.add_argument('--merge_iterations', type=int, default=5,
                        help="Number of iterations for node merge stage. Default 5.")
    parser.add_argument('--pruning_rate', type=float, default=0.10,
                        help="The rate of edge pruning in Stage 2. Default 0.10.")
    parser.add_argument('--num_rounds',type=int,default=1,
                        help="Number of optimization/inference rounds for one query")
    parser.add_argument('--llm_name', type=str, default="gpt-3.5-turbo",
                        help="Model name, None runs the default ChatGPT4")
    parser.add_argument('--domain', type=str, default="mmlu",
                        help="Domain (the same as dataset name), default 'MMLU'")
    parser.add_argument('--decision_method', type=str, default="FinalRefer",
                        help="the decision method of the final node")
    parser.add_argument('--optimized_spatial',action='store_true')
    parser.add_argument('--optimized_temporal',action='store_true')
    parser.add_argument('--diff',action='store_true')
    parser.add_argument('--dec',action='store_true')
    parser.add_argument('--cot',action='store_true')
    parser.add_argument('--use_node_merge',action='store_true',
                        help='Use node merge algorithm instead of edge pruning')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to use from dev/val dataset. If not specified, use all data.')
    args = parser.parse_args()
    result_path = AgentPrune_ROOT / "result"
    os.makedirs(result_path, exist_ok=True)
    if len(args.agent_names) != len(args.agent_nums):
        parser.error("The number of agent names must match the number of agent counts.")
        
    return args

async def main():
    args = parse_args()
    
    mode = args.mode
    decision_method = args.decision_method
    agent_names = [name for name,num in zip(args.agent_names,args.agent_nums) for _ in range(num)]
    kwargs = get_kwargs(mode,len(agent_names))
    limit_questions = 153
    
    graph = Graph(domain=args.domain,
                  llm_name=args.llm_name,
                  agent_names=agent_names,
                  decision_method=decision_method,
                  optimized_spatial=args.optimized_spatial,
                  optimized_temporal=args.optimized_temporal,
                  rounds=args.num_rounds,
                  diff=args.diff,
                  dec=args.dec,
                  **kwargs)
    download()
    dataset_train = MMLUDataset('dev')
    dataset_val = MMLUDataset('val')
    
    # Sample validation dataset only for faster evaluation
    # Training uses full dev dataset (like GSM8K)
    if args.num_samples:
        original_val_len = len(dataset_val)
        if args.num_samples < len(dataset_val):
            dataset_val._total_df = dataset_val._total_df.iloc[:args.num_samples]
            print(f"Sampled {args.num_samples} samples from val dataset for evaluation (original: {original_val_len})")
        print(f"Using full dev dataset for training: {len(dataset_train)} samples")
    
    if args.optimized_spatial or args.optimized_temporal:
        print(f"\n{'='*80}")
        print(f"Training with Two-Stage Node Merge Algorithm")
        print(f"  Merge Stage: Node Optimization ({args.merge_iterations} iterations) → Node Merge")
        print(f"  Edge Stage: Edge Optimization ({args.num_iterations - args.merge_iterations} iterations) → Edge Dropout (rate: {args.pruning_rate})")
        print(f"{'='*80}\n")
        
        await train(graph=graph,
                   dataset=dataset_train,
                   num_iters=args.num_iterations,
                   num_rounds=args.num_rounds,
                   lr=args.lr,
                   batch_size=20,
                   imp_per_iters=args.merge_iterations,
                   pruning_rate=0,  # Not used in node merge
                   args=args,
                   kwargs=kwargs)
        
    print("Final spatial logits: ",graph.spatial_logits)
    print("Final temporal logits: ",graph.temporal_logits)
    print("Final spatial masks: ",graph.spatial_masks)
    print("Final temporal masks: ",graph.temporal_masks)
    print(f"Final number of nodes: {len(graph.nodes)}")
    
    if not args.diff:
        if isinstance(graph.spatial_masks, list):
            graph.spatial_masks = torch.tensor(graph.spatial_masks)
        if isinstance(graph.temporal_masks, list):
            graph.temporal_masks = torch.tensor(graph.temporal_masks)
        print("Final spatial sparsity:",graph.spatial_masks.sum()/graph.spatial_masks.numel())
        print("Final temporal sparsity:",graph.temporal_masks.sum()/graph.temporal_masks.numel())
    else:
        # Convert list elements to tensors if needed
        if isinstance(graph.spatial_masks[0], list):
            spatial_masks_tensors = [torch.tensor(mask) for mask in graph.spatial_masks]
        else:
            spatial_masks_tensors = graph.spatial_masks
        spatial_sparsity = torch.mean(torch.stack([mask.sum() / mask.numel() for mask in spatial_masks_tensors]))
        print("Spatial sparsity (mean):", spatial_sparsity)

        if isinstance(graph.temporal_masks[0], list):
            temporal_masks_tensors = [torch.tensor(mask) for mask in graph.temporal_masks]
        else:
            temporal_masks_tensors = graph.temporal_masks
        temporal_sparsity = torch.mean(torch.stack([mask.sum() / mask.numel() for mask in temporal_masks_tensors]))
        print("Temporal sparsity (mean):", temporal_sparsity)

    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    # Track evaluation cost separately (but don't reset - keep cumulative cost)
    eval_cost_start = Cost.instance().value
    eval_prompt_start = PromptTokens.instance().value
    eval_completion_start = CompletionTokens.instance().value
    
    if args.dec:
        score = await evaluate(graph=graph,dataset=dataset_val,num_rounds=args.num_rounds,
                             limit_questions=limit_questions,eval_batch_size=args.batch_size,dec=True,args=args)
    else:
        score = await evaluate(graph=graph,dataset=dataset_val,num_rounds=args.num_rounds,
                             limit_questions=limit_questions,eval_batch_size=args.batch_size,args=args)
    
    eval_cost = Cost.instance().value - eval_cost_start
    eval_prompt = PromptTokens.instance().value - eval_prompt_start
    eval_completion = CompletionTokens.instance().value - eval_completion_start
    
    print(f"\nAccuracy: {score:.4f}")
    print(f"Total Cost (Training + Evaluation): ${Cost.instance().value:.4f}")
    print(f"Total Prompt Tokens: {PromptTokens.instance().value:.0f}")
    print(f"Total Completion Tokens: {CompletionTokens.instance().value:.0f}")
    print(f"Evaluation Cost: ${eval_cost:.4f}")
    print(f"Evaluation Prompt Tokens: {eval_prompt:.0f}")
    print(f"Evaluation Completion Tokens: {eval_completion:.0f}")


def get_kwargs(mode:Union[Literal['DirectAnswer'],Literal['FullConnected'],Literal['Random'],Literal['Chain'],Literal['Debate'],Literal['Layered'],Literal['Star'],Literal['Mesh'],
                          Literal['FakeFullConnected'],Literal['FakeRandom'],Literal['FakeChain'],Literal['FakeStar'],Literal['FakeMesh'],Literal['FakeAGRandom'],Literal['FakeAGFull']],
               N:int):
    initial_spatial_probability: float = 0.5
    fixed_spatial_masks:List[List[int]] = None
    initial_temporal_probability: float = 0.5
    fixed_temporal_masks:List[List[int]] = None
    node_kwargs = None
    
    def generate_layered_graph(N,layer_num=2):
        adj_matrix = [[0]*N for _ in range(N)]
        base_size = N // layer_num
        remainder = N % layer_num
        layers = []
        for i in range(layer_num):
            size = base_size + (1 if i < remainder else 0)
            layers.extend([i] * size)
        for i in range(N):
            current_layer = layers[i]
            for j in range(N):
                if layers[j] == current_layer + 1:
                    adj_matrix[i][j] = 1
        return adj_matrix
    
    def generate_mesh_graph(N):
        adj_matrix = [[0] * N for _ in range(N)]
        for i in range(0, N):
            for j in range(i+1,N):
                adj_matrix[i][j] = 1
        return adj_matrix
    
    def generate_star_graph(N):
        adj_matrix = [[0] * N for _ in range(N)]
        for i in range(1,N):
            adj_matrix[0][i] = 1
        return adj_matrix
    
    if mode=='DirectAnswer':
        fixed_spatial_masks = [[0]]
        fixed_temporal_masks = [[0]]
        node_kwargs = [{'role':'Normal'}]
    elif mode=='FullConnected' or mode == 'FakeFullConnected' or mode=='FakeAGFull':
        fixed_spatial_masks = [[1 if i!=j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]
    elif mode=='Random' or mode == 'FakeRandom' or mode == 'FakeAGRandom':
        fixed_spatial_masks = [[random.randint(0, 1)  if i!=j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[random.randint(0, 1) for _ in range(N)] for _ in range(N)]
    elif mode=='Chain' or mode == 'FakeChain':
        fixed_spatial_masks = [[1 if i==j+1 else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 if i==0 and j==N-1 else 0 for i in range(N)] for j in range(N)]
    elif mode == 'Debate':
        fixed_spatial_masks = [[0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    elif mode == 'Layered':
        fixed_spatial_masks = generate_layered_graph(N)
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    elif mode == 'Mesh' or mode=='FakeMesh':
        fixed_spatial_masks = generate_mesh_graph(N)
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    elif mode == 'Star' or mode=='FakeStar':
        fixed_spatial_masks = generate_star_graph(N)
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    
    if 'Fake' in mode and 'AG' not in mode:
        node_kwargs = [{'role':'Fake'} if i % 2 == N % 2 else {'role':'Normal'} for i in range(N)]
    elif 'Fake' in mode and 'AG' in mode:
        node_kwargs = [{'role':'Fake'} if i % 2 == N % 2 else {'role':None} for i in range(N)]
        
    return {"initial_spatial_probability": initial_spatial_probability,
            "fixed_spatial_masks": fixed_spatial_masks,
            "initial_temporal_probability": initial_temporal_probability,
            "fixed_temporal_masks": fixed_temporal_masks,
            "node_kwargs":node_kwargs}    

if __name__ == "__main__":
    asyncio.run(main())
