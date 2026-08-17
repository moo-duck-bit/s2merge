import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

import asyncio
from typing import Union, Literal, List, Dict, Tuple
import argparse
import random
import time
import copy
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from AgentDropout.graph.graph_merge import Graph
from dataset_load.multiarith_dataset import multiarith_data_process
from dataset_load.gsm8k_dataset import gsm_get_predict
from AgentDropout.tools.reader.readers import JSONReader
from AgentDropout.utils.const import AgentPrune_ROOT
from AgentDropout.utils.globals import PromptTokens, CompletionTokens, Cost, Time
from AgentDropout.utils.utils import nuclear_norm, frobenius_norm
from AgentDropout.llm.llm_registry import LLMRegistry


def load_result(result_file):
    if not result_file.exists():
        with open(result_file, 'w',encoding='utf-8') as file:
            json.dump([], file)

    with open(result_file, 'r',encoding='utf-8') as file:
        data = json.load(file)
    return data


async def merge_prompts_with_llm(node1, node2, llm_name: str = "gpt-4o-mini") -> str:
    """Use LLM to intelligently merge two agent prompts."""
    llm = LLMRegistry.get(llm_name)
    
    node1_constraint = node1.constraint if hasattr(node1, 'constraint') else ""
    node2_constraint = node2.constraint if hasattr(node2, 'constraint') else ""
    node1_role = node1.role if hasattr(node1, 'role') else "Agent"
    node2_role = node2.role if hasattr(node2, 'role') else "Agent"
    
    merge_system_prompt = """You are an expert at merging agent instructions while preserving role clarity.
Your task is to combine two agent prompts intelligently:
1. Maintain the PRIMARY agent's core role and responsibilities
2. Extract only the key, non-redundant insights from the SECONDARY agent
3. Integrate secondary insights naturally without compromising primary role
4. Keep the result concise and actionable
5. Output ONLY the merged prompt text, no explanations."""

    merge_user_prompt = f"""Merge these two agent prompts:

PRIMARY AGENT (role: {node1_role}):
{node1_constraint}

SECONDARY AGENT (role: {node2_role}):
{node2_constraint}

Create a merged prompt that:
- Keeps the PRIMARY agent's role as the main focus
- Adds valuable context from SECONDARY agent if relevant
- Removes redundant information
- Maintains clarity and conciseness

Output the merged prompt:"""

    message = [
        {'role': 'system', 'content': merge_system_prompt},
        {'role': 'user', 'content': merge_user_prompt}
    ]
    
    try:
        merged_constraint = await llm.agen(message)
        return merged_constraint.strip()
    except Exception as e:
        print(f"  Warning: LLM merge failed ({e}), using simple concatenation")
        return f"{node1_constraint}\n\nAdditional context from merged agent: {node2_constraint}"


def calculate_transition_efficiency(graph, weight_matrix: torch.Tensor) -> Dict[str, float]:
    """Calculate transition efficiency ΔW for each node."""
    delta_w = {}
    for i, node in enumerate(list(graph.nodes.values())):
        out_weights = weight_matrix[i, :]
        in_weights = weight_matrix[:, i]
        max_out = torch.max(out_weights).item() if out_weights.numel() > 0 else 0.0
        max_in = torch.max(in_weights).item() if in_weights.numel() > 0 else 0.0
        delta_w[node.id] = max_out - max_in
    return delta_w


def find_merge_candidate(graph, weight_matrix: torch.Tensor) -> Tuple[str, float]:
    """Find the node with minimum transition efficiency."""
    delta_w = calculate_transition_efficiency(graph, weight_matrix)
    min_node_id = min(delta_w, key=delta_w.get)
    min_delta_w = delta_w[min_node_id]
    print(f"Transition efficiency (ΔW): {delta_w}")
    print(f"Selected merge candidate: {min_node_id} with ΔW = {min_delta_w:.4f}")
    return min_node_id, min_delta_w


def find_merge_partner(graph, weight_matrix: torch.Tensor, target_node_id: str, 
                       execution_order: List[str]) -> Tuple[str, float]:
    """Find the best partner node to merge with."""
    target_idx = None
    for i, node in enumerate(list(graph.nodes.values())):
        if node.id == target_node_id:
            target_idx = i
            break
    
    if target_idx is None:
        raise ValueError(f"Node {target_node_id} not found")
    
    candidates = {}
    n_nodes = len(list(graph.nodes.values()))
    
    for j in range(n_nodes):
        if j == target_idx:
            continue
        in_weight = weight_matrix[j, target_idx].item()
        out_weight = weight_matrix[target_idx, j].item()
        if in_weight > 0 or out_weight > 0:
            max_weight = max(in_weight, out_weight)
            candidates[list(graph.nodes.values())[j].id] = {
                'weight': max_weight,
                'in_weight': in_weight,
                'out_weight': out_weight,
                'idx': j
            }
    
    if not candidates:
        print(f"Warning: No connected nodes found for {target_node_id}")
        return None, 0.0
    
    max_weight = max(c['weight'] for c in candidates.values())
    top_candidates = {nid: c for nid, c in candidates.items() if c['weight'] == max_weight}
    
    if len(top_candidates) == 1:
        partner_id = list(top_candidates.keys())[0]
    else:
        in_edge_candidates = {nid: c for nid, c in top_candidates.items() 
                             if c['in_weight'] >= c['out_weight']}
        if in_edge_candidates:
            candidates_to_check = in_edge_candidates
        else:
            candidates_to_check = top_candidates
        
        partner_id = min(candidates_to_check.keys(), 
                        key=lambda nid: execution_order.index(nid) if nid in execution_order else float('inf'))
    
    print(f"Selected merge partner: {partner_id} with weight {top_candidates[partner_id]['weight']:.4f}")
    return partner_id, top_candidates[partner_id]['weight']


async def merge_nodes_in_graph(graph, node1_id: str, node2_id: str, llm_name: str):
    """Merge two nodes in the graph with LLM-based prompt merging."""
    pre_merge_cost = Cost.instance().value
    pre_merge_prompt = PromptTokens.instance().value
    pre_merge_completion = CompletionTokens.instance().value
    
    node1 = graph.nodes[node1_id]
    node2 = graph.nodes[node2_id]
    
    print(f"\n=== Merging nodes: {node2_id} -> {node1_id} ===")
    print(f"Node1 ({node1_id}) role: {node1.role if hasattr(node1, 'role') else 'N/A'}")
    print(f"Node2 ({node2_id}) role: {node2.role if hasattr(node2, 'role') else 'N/A'}")
    
    merged_constraint = await merge_prompts_with_llm(node1, node2, llm_name)
    node1.constraint = merged_constraint
    print(f"Merged constraint: {merged_constraint[:200]}...")
    
    for succ_id in node2.spatial_successors:
        if succ_id != node1_id and succ_id not in node1.spatial_successors:
            node1.add_successor(succ_id, edge_type='spatial')
    
    for succ_id in node2.temporal_successors:
        if succ_id != node1_id and succ_id not in node1.temporal_successors:
            node1.add_successor(succ_id, edge_type='temporal')
    
    for other_node in graph.nodes.values():
        if other_node.id == node1_id or other_node.id == node2_id:
            continue
        if node2_id in other_node.spatial_successors:
            other_node.spatial_successors.remove(node2_id)
            if node1_id not in other_node.spatial_successors:
                other_node.add_successor(node1_id, edge_type='spatial')
        if node2_id in other_node.temporal_successors:
            other_node.temporal_successors.remove(node2_id)
            if node1_id not in other_node.temporal_successors:
                other_node.add_successor(node1_id, edge_type='temporal')
    
    del graph.nodes[node2_id]
    
    graph.potential_spatial_edges = [(src, tgt) for src, tgt in graph.potential_spatial_edges 
                                     if src != node2_id and tgt != node2_id]
    graph.potential_temporal_edges = [(src, tgt) for src, tgt in graph.potential_temporal_edges 
                                      if src != node2_id and tgt != node2_id]
    
    new_size = len(graph.nodes)
    n_edges = len(graph.potential_spatial_edges)
    n_temporal_edges = len(graph.potential_temporal_edges)
    
    if graph.diff:
        # Store old lengths
        num_spatial = len(graph.spatial_logits_1)
        num_temporal = len(graph.temporal_logits_1)
        
        # Update spatial logits and masks
        new_spatial_logits = []
        new_spatial_masks = []
        for round_idx in range(num_spatial):
            new_spatial_logits.append(
                torch.nn.Parameter(torch.full((n_edges,), 0.5, dtype=torch.float32))
            )
            new_spatial_masks.append([1.0] * n_edges)
        
        graph.spatial_logits_1 = torch.nn.ParameterList(new_spatial_logits)
        graph.spatial_masks = new_spatial_masks
        
        # Update temporal logits and masks (may have different length)
        new_temporal_logits = []
        new_temporal_masks = []
        for round_idx in range(num_temporal):
            new_temporal_logits.append(
                torch.nn.Parameter(torch.full((n_temporal_edges,), 0.5, dtype=torch.float32))
            )
            new_temporal_masks.append([1.0] * n_temporal_edges)
        
        graph.temporal_logits_1 = torch.nn.ParameterList(new_temporal_logits)
        graph.temporal_masks = new_temporal_masks
    else:
        graph.spatial_logits_1 = torch.nn.Parameter(
            torch.full((n_edges,), 0.5, dtype=torch.float32)
        )
        graph.temporal_logits_1 = torch.nn.Parameter(
            torch.full((n_temporal_edges,), 0.5, dtype=torch.float32)
        )
        graph.spatial_masks = [1.0] * n_edges
        graph.temporal_masks = [1.0] * n_temporal_edges
    
    post_merge_cost = Cost.instance().value
    post_merge_prompt = PromptTokens.instance().value
    post_merge_completion = CompletionTokens.instance().value
    
    merge_cost = post_merge_cost - pre_merge_cost
    merge_prompt = post_merge_prompt - pre_merge_prompt
    merge_completion = post_merge_completion - pre_merge_completion
    
    print(f"Merge operation cost: ${merge_cost:.4f}")
    print(f"Merge tokens: {merge_prompt} prompt + {merge_completion} completion")
    print(f"Graph now has {new_size} nodes\n")
    
    return graph


def dataloader(data_list, batch_size, i_batch):
    """Simple dataloader."""
    return data_list[i_batch*batch_size:i_batch*batch_size + batch_size]


def parse_args():
    parser = argparse.ArgumentParser(description="MultiArith with Node Merge Algorithm")

    parser.add_argument('--dataset_json', type=str, default='datasets/MultiArith/test.json',
                        help='Path to MultiArith test dataset')
    parser.add_argument('--train_json', type=str, default='datasets/MultiArith/train.json',
                        help='Path to MultiArith train dataset')
    parser.add_argument('--result_file', type=str, default=None,
                        help='Path to result file')
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
    parser.add_argument('--agent_names', nargs='+', type=str, default=['MathSolver'],
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
    parser.add_argument('--domain', type=str, default="gsm8k",
                        help="Domain (the same as dataset name), default 'gsm8k'")
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
                        help='Number of samples to use from test dataset. If not specified, use all data.')
    args = parser.parse_args()
    result_path = AgentPrune_ROOT / "result"
    os.makedirs(result_path, exist_ok=True)
    if len(args.agent_names) != len(args.agent_nums):
        parser.error("The number of agent names must match the number of agent counts.")
        
    return args


async def main():
    args = parse_args()
    
    # Setup result file
    current_time = Time.instance().value or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    Time.instance().value = current_time
    result_dir = Path(f"{AgentPrune_ROOT}/result/MultiArith")
    result_dir.mkdir(parents=True, exist_ok=True)
    result_file = result_dir / f"{args.domain}_llama3_{current_time}.json"
    
    mode = args.mode
    decision_method = args.decision_method
    agent_names = [name for name,num in zip(args.agent_names,args.agent_nums) for _ in range(num)]
    kwargs = get_kwargs(mode,len(agent_names))
    
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
    
    # Load datasets
    dataset = JSONReader.parse_file(args.dataset_json)
    dataset = multiarith_data_process(dataset)
    train_dataset = JSONReader.parse_file(args.train_json)
    train_dataset = multiarith_data_process(train_dataset)
    
    # Sample test dataset if specified
    if args.num_samples and args.num_samples < len(dataset):
        dataset = dataset[:args.num_samples]
        print(f"Sampled {args.num_samples} samples from test dataset (total: {len(dataset)})")
    
    print(f"Using training dataset: {len(train_dataset)} samples")
    print(f"Using test dataset: {len(dataset)} samples")
    
    if args.use_node_merge and args.dec:
        # === TWO-STAGE TRAINING WITH NODE MERGE ===
        print(f"\n{'='*80}")
        print(f"Training with Two-Stage Node Merge Algorithm")
        print(f"  Merge Stage: Node Optimization ({args.merge_iterations} iterations) → Node Merge")
        print(f"  Edge Stage: Edge Optimization ({args.num_iterations - args.merge_iterations} iterations) → Edge Dropout (rate: {args.pruning_rate})")
        print(f"{'='*80}\n")
        
        print(f"\n{'='*80}")
        print("STAGE 1: MERGE STAGE")
        print(f"{'='*80}\n")
        
        graph.optimized_spatial = False
        graph.optimized_temporal = False
        
        if not graph.diff:
            optimizer = torch.optim.Adam([graph.spatial_logits_1, graph.temporal_logits_1], lr=args.lr)
        else:
            optimizer = torch.optim.Adam(
                list(graph.spatial_logits_1.parameters()) + list(graph.temporal_logits_1.parameters()),
                lr=args.lr
            )
        
        # MERGE STAGE: Training
        for i_batch in range(args.merge_iterations):
            print(f"\nMerge Stage - Iteration {i_batch}/{args.merge_iterations}")
            print("-" * 80)
            start_ts = time.time()
            answer_log_probs = []
            answers = []
            add_losses = []
            
            current_batch = dataloader(train_dataset, args.batch_size, i_batch)
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
                
                spatial_matrix_fixed = torch.tensor(kwargs["fixed_spatial_masks"], dtype=torch.float32).reshape((n_nodes, n_nodes))
                temporal_matrix_fixed = torch.tensor(kwargs["fixed_temporal_masks"], dtype=torch.float32).reshape((n_nodes, n_nodes))
                
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
                answer_log_probs.append(asyncio.create_task(realized_graph.arun(input_dict, args.num_rounds, skip=True)))
                add_losses.append(add_loss)
            
            raw_results = await asyncio.gather(*answer_log_probs)
            raw_answers, log_probs = zip(*raw_results)
            loss_list = []
            utilities = []
            
            for answer_pair, log_prob, add_loss, true_answer in zip(raw_answers, log_probs, add_losses, answers):
                predict_answer = gsm_get_predict(answer_pair[0])
                is_solved = float(predict_answer) == float(true_answer)
                utility = is_solved
                utilities.append(utility)
                single_loss = -log_prob * utility
                loss_list.append(single_loss + add_loss)
            
            total_loss = torch.mean(torch.stack(loss_list))
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            print(f"Batch time: {time.time() - start_ts:.3f}s")
            print(f"Utilities: {utilities}, Loss: {total_loss.item():.4f}")
            print(f"Cost: ${Cost.instance().value:.4f}, Tokens: {PromptTokens.instance().value}/{CompletionTokens.instance().value}")
        
        # MERGE STAGE: Node Merge
        if len(list(graph.nodes.values())) > 2:
            print(f"\n{'='*80}")
            print("MERGE STAGE: PERFORMING NODE MERGE")
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
                graph = await merge_nodes_in_graph(graph, partner_id, merge_candidate_id, args.llm_name)
                
                # Update fixed masks for new graph size
                n_nodes = len(list(graph.nodes.values()))
                old_n = len(kwargs["fixed_spatial_masks"])
                if old_n > n_nodes:
                    merge_idx = old_n - n_nodes
                    kwargs["fixed_spatial_masks"] = [
                        [kwargs["fixed_spatial_masks"][i][j] for j in range(old_n) if j != merge_idx]
                        for i in range(old_n) if i != merge_idx
                    ]
                    kwargs["fixed_temporal_masks"] = [
                        [kwargs["fixed_temporal_masks"][i][j] for j in range(old_n) if j != merge_idx]
                        for i in range(old_n) if i != merge_idx
                    ]
        
        # === STAGE 2: EDGE STAGE ===
        print(f"\n{'='*80}")
        print("STAGE 2: EDGE STAGE")
        print(f"{'='*80}\n")
        
        # Re-initialize weights
        n_nodes = len(list(graph.nodes.values()))
        n_edges = len(graph.potential_spatial_edges)
        n_temporal_edges = len(graph.potential_temporal_edges)
        init_logit = torch.log(torch.tensor(0.5 / (1 - 0.5)))
        
        if not graph.diff:
            graph.spatial_logits_1 = torch.nn.Parameter(
                torch.ones(n_edges, requires_grad=True) * init_logit
            )
            graph.temporal_logits_1 = torch.nn.Parameter(
                torch.ones(n_temporal_edges, requires_grad=True) * init_logit
            )
            graph.spatial_masks = [1.0] * n_edges
            graph.temporal_masks = [1.0] * n_temporal_edges
            optimizer = torch.optim.Adam([graph.spatial_logits_1, graph.temporal_logits_1], lr=args.lr)
        else:
            num_spatial = len(graph.spatial_logits_1)
            num_temporal = len(graph.temporal_logits_1)
            
            graph.spatial_logits_1 = torch.nn.ParameterList([
                torch.nn.Parameter(torch.ones(n_edges) * init_logit)
                for _ in range(num_spatial)
            ])
            graph.temporal_logits_1 = torch.nn.ParameterList([
                torch.nn.Parameter(torch.ones(n_temporal_edges) * init_logit)
                for _ in range(num_temporal)
            ])
            graph.spatial_masks = [[1.0] * n_edges for _ in range(num_spatial)]
            graph.temporal_masks = [[1.0] * n_temporal_edges for _ in range(num_temporal)]
            optimizer = torch.optim.Adam(
                list(graph.spatial_logits_1.parameters()) + list(graph.temporal_logits_1.parameters()),
                lr=args.lr
            )
        
        # Edge Stage: Training
        edge_iters = args.num_iterations - args.merge_iterations
        for i_batch in range(edge_iters):
            print(f"\nEdge Stage - Iteration {i_batch}/{edge_iters}")
            print("-" * 80)
            start_ts = time.time()
            answer_log_probs = []
            answers = []
            add_losses = []
            
            current_batch = dataloader(train_dataset, args.batch_size, i_batch)
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
                
                spatial_matrix_fixed = torch.tensor(kwargs["fixed_spatial_masks"], dtype=torch.float32).reshape((n_nodes, n_nodes))
                temporal_matrix_fixed = torch.tensor(kwargs["fixed_temporal_masks"], dtype=torch.float32).reshape((n_nodes, n_nodes))
                
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
                answer_log_probs.append(asyncio.create_task(realized_graph.arun(input_dict, args.num_rounds, skip=True)))
                add_losses.append(add_loss)
            
            raw_results = await asyncio.gather(*answer_log_probs)
            raw_answers, log_probs = zip(*raw_results)
            loss_list = []
            utilities = []
            
            for answer_pair, log_prob, add_loss, true_answer in zip(raw_answers, log_probs, add_losses, answers):
                predict_answer = gsm_get_predict(answer_pair[0])
                is_solved = float(predict_answer) == float(true_answer)
                utility = is_solved
                utilities.append(utility)
                single_loss = -log_prob * utility
                loss_list.append(single_loss + add_loss)
            
            total_loss = torch.mean(torch.stack(loss_list))
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            print(f"Batch time: {time.time() - start_ts:.3f}s")
            print(f"Utilities: {utilities}, Loss: {total_loss.item():.4f}")
            print(f"Cost: ${Cost.instance().value:.4f}, Tokens: {PromptTokens.instance().value}/{CompletionTokens.instance().value}")
        
        # Edge Stage: Edge Dropout
        print(f"\n{'='*80}")
        print(f"EDGE STAGE: EDGE DROPOUT (rate={args.pruning_rate})")
        print(f"{'='*80}\n")
        
        if not graph.diff:
            spatial_masks, temporal_masks = graph.update_masks(args.pruning_rate)
        else:
            spatial_masks, temporal_masks = graph.update_masks_diff(args.pruning_rate)
        
        print("Spatial masks:", spatial_masks)
        print("Temporal masks:", temporal_masks)
        
        if not graph.diff:
            print(f"Spatial sparsity: {sum(spatial_masks) / len(spatial_masks):.4f}")
            print(f"Temporal sparsity: {sum(temporal_masks) / len(temporal_masks):.4f}")
        else:
            print(f"Spatial sparsity: {sum(spatial_masks[0]) / len(spatial_masks[0]):.4f}")
            print(f"Temporal sparsity: {sum(temporal_masks[0]) / len(temporal_masks[0]):.4f}")
    
    # Print final graph state
    print("\n" + "="*80)
    print("FINAL GRAPH STATE")
    print("="*80)
    print("Final spatial logits: ", graph.spatial_logits)
    print("Final temporal logits: ", graph.temporal_logits)
    print("Final spatial masks: ", graph.spatial_masks)
    print("Final temporal masks: ", graph.temporal_masks)
    print(f"Final number of nodes: {len(graph.nodes)}")
    
    if not graph.diff:
        if isinstance(graph.spatial_masks, list):
            print(f"Final spatial sparsity: {sum(graph.spatial_masks) / len(graph.spatial_masks):.4f}")
            print(f"Final temporal sparsity: {sum(graph.temporal_masks) / len(graph.temporal_masks):.4f}")
        else:
            print(f"Final spatial sparsity: {graph.spatial_masks.sum() / graph.spatial_masks.numel():.4f}")
            print(f"Final temporal sparsity: {graph.temporal_masks.sum() / graph.temporal_masks.numel():.4f}")
    else:
        if isinstance(graph.spatial_masks[0], list):
            print(f"Final spatial sparsity (mean): {sum(graph.spatial_masks[0]) / len(graph.spatial_masks[0]):.4f}")
            print(f"Final temporal sparsity (mean): {sum(graph.temporal_masks[0]) / len(graph.temporal_masks[0]):.4f}")
        else:
            spatial_sparsity = torch.mean(torch.stack([mask.sum() / mask.numel() for mask in graph.spatial_masks]))
            print(f"Final spatial sparsity (mean): {spatial_sparsity:.4f}")
            temporal_sparsity = torch.mean(torch.stack([mask.sum() / mask.numel() for mask in graph.temporal_masks]))
            print(f"Final temporal sparsity (mean): {temporal_sparsity:.4f}")
    
    # === EVALUATION ===
    print(f"\n{'='*80}")
    print("EVALUATION RESULTS")
    print(f"{'='*80}\n")
    
    test_cost_start = Cost.instance().value
    test_prompt_start = PromptTokens.instance().value
    test_completion_start = CompletionTokens.instance().value
    
    total_solved = 0
    total_executed = 0
    
    num_batches = len(dataset) // args.batch_size
    for i_batch in range(num_batches):
        print(f"Test Batch {i_batch}/{num_batches}")
        start_ts = time.time()
        answer_log_probs = []
        answers = []
        
        current_batch = dataloader(dataset, args.batch_size, i_batch)
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
            answer_log_probs.append(asyncio.create_task(realized_graph.arun(input_dict, args.num_rounds)))
        
        raw_results = await asyncio.gather(*answer_log_probs)
        raw_answers, log_probs = zip(*raw_results)
        
        for answer_pair, true_answer in zip(raw_answers, answers):
            predict_answer = gsm_get_predict(answer_pair[0])
            is_solved = float(predict_answer) == float(true_answer)
            total_solved += is_solved
            total_executed += 1
        
        accuracy = total_solved / total_executed
        print(f"Batch time: {time.time() - start_ts:.3f}s, Accuracy: {accuracy:.4f}")
    
    test_cost = Cost.instance().value - test_cost_start
    test_prompt = PromptTokens.instance().value - test_prompt_start
    test_completion = CompletionTokens.instance().value - test_completion_start
    
    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}")
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Total Cost (Training + Evaluation): ${Cost.instance().value:.4f}")
    print(f"Total Prompt Tokens: {PromptTokens.instance().value:.0f}")
    print(f"Total Completion Tokens: {CompletionTokens.instance().value:.0f}")
    print(f"Evaluation Cost: ${test_cost:.4f}")
    print(f"Evaluation Prompt Tokens: {test_prompt:.0f}")
    print(f"Evaluation Completion Tokens: {test_completion:.0f}")


def get_kwargs(mode:Union[Literal['DirectAnswer'],Literal['FullConnected'],Literal['Random'],
                           Literal['Chain'],Literal['Debate'],Literal['Layered'],Literal['Star']], N:int):
    initial_spatial_probability: float = 0.5
    fixed_spatial_masks:List[List[int]] = None
    initial_temporal_probability: float = 0.5
    fixed_temporal_masks:List[List[int]] = None
    node_kwargs = None
    
    def generate_layered_graph(N,layer_num=2):
        adj_matrix = [[0 for _ in range(N)] for _ in range(N)]
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
    
    def generate_star_graph(n):
        matrix = [[0] * n for _ in range(n)]
        for i in range(0, n):
            for j in range(i+1,n):
                matrix[i][j] = 1
        return matrix
    
    if mode=='DirectAnswer':
        fixed_spatial_masks = [[0]]
        fixed_temporal_masks = [[0]]
        node_kwargs = [{'role':'Math Solver'}]
    elif mode=='FullConnected':
        fixed_spatial_masks = [[1 if i!=j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]
    elif mode=='Random':
        fixed_spatial_masks = [[random.randint(0, 1)  if i!=j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[random.randint(0, 1) for _ in range(N)] for _ in range(N)]
    elif mode=='Chain':
        fixed_spatial_masks = [[1 if i==j+1 else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 if i==0 and j==N-1 else 0 for i in range(N)] for j in range(N)]
    elif mode == 'Debate':
        fixed_spatial_masks = [[0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    elif mode == 'Layered':
        fixed_spatial_masks = generate_layered_graph(N)
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    elif mode == 'Star':
        fixed_spatial_masks = generate_star_graph(N)
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    
    return {"initial_spatial_probability": initial_spatial_probability,
            "fixed_spatial_masks": fixed_spatial_masks,
            "initial_temporal_probability": initial_temporal_probability,
            "fixed_temporal_masks": fixed_temporal_masks,
            "node_kwargs":node_kwargs}


if __name__ == '__main__':
    asyncio.run(main())
