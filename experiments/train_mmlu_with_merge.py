import torch
import torch.nn.functional as F
from typing import Iterator, List, Tuple, Dict
import pandas as pd
import numpy as np
import time
import asyncio
import copy
import random

from AgentDropout.graph.graph_merge import Graph
from experiments.accuracy import Accuracy
from AgentDropout.utils.globals import Cost, PromptTokens, CompletionTokens
from AgentDropout.utils.utils import nuclear_norm, frobenius_norm
from AgentDropout.llm.llm_registry import LLMRegistry


async def merge_prompts_with_llm(node1, node2, llm_name: str = "gpt-4o-mini") -> str:
    """
    Use LLM to intelligently merge two agent prompts.

    Extracts key information from node2's prompt and integrates it into node1's prompt
    while maintaining node1's core role and responsibilities.

    Args:
        node1: The main node (will keep its role)
        node2: The node to be absorbed
        llm_name: LLM to use for prompt merging

    Returns:
        Merged constraint/system prompt for node1
    """
    llm = LLMRegistry.get(llm_name)

    # Get original constraints/prompts
    node1_constraint = node1.constraint if hasattr(node1, 'constraint') else ""
    node2_constraint = node2.constraint if hasattr(node2, 'constraint') else ""

    node1_role = node1.role if hasattr(node1, 'role') else "Agent"
    node2_role = node2.role if hasattr(node2, 'role') else "Agent"

    # Create prompt for LLM to merge the two prompts
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


def calculate_transition_efficiency(graph: Graph, weight_matrix: torch.Tensor) -> Dict[str, float]:
    """
    Calculate transition efficiency ΔW for each node.

    ΔW(c) = max(Ã_out) - max(Ã_in)

    Args:
        graph: The graph containing nodes
        weight_matrix: Adjacency matrix with edge weights

    Returns:
        Dictionary mapping node_id to ΔW value
    """
    n_nodes = len(list(graph.nodes.values()))
    delta_w = {}

    for i, node in enumerate(list(graph.nodes.values())):
        # Get outgoing weights (row i)
        out_weights = weight_matrix[i, :]
        # Get incoming weights (column i)
        in_weights = weight_matrix[:, i]

        # Calculate max of outgoing and incoming weights
        max_out = torch.max(out_weights).item() if out_weights.numel() > 0 else 0.0
        max_in = torch.max(in_weights).item() if in_weights.numel() > 0 else 0.0

        # Calculate transition efficiency
        delta_w[node.id] = max_out - max_in

    return delta_w


def find_merge_candidate(graph: Graph, weight_matrix: torch.Tensor) -> Tuple[str, float]:
    """
    Find the node with minimum transition efficiency for merging.

    c' = arg min_{v ∈ V} ΔW(v)

    Args:
        graph: The graph containing nodes
        weight_matrix: Adjacency matrix with edge weights

    Returns:
        Tuple of (node_id, delta_w_value)
    """
    delta_w = calculate_transition_efficiency(graph, weight_matrix)

    # Find node with minimum ΔW
    min_node_id = min(delta_w, key=delta_w.get)
    min_delta_w = delta_w[min_node_id]

    print(f"Transition efficiency (ΔW) for all nodes: {delta_w}")
    print(f"Selected merge candidate: {min_node_id} with ΔW = {min_delta_w:.4f}")

    return min_node_id, min_delta_w


def find_merge_partner(graph: Graph, weight_matrix: torch.Tensor, target_node_id: str,
                       execution_order: List[str]) -> Tuple[str, float]:
    """
    Find the best partner node to merge with the target node.

    Steps:
    1. Candidate Search: Find all connected nodes N(c)
    2. Weight Comparison: Calculate S(c,j) = max(Ã[j,c], Ã[c,j])
    3. Tie-breaker: If multiple nodes have same weight, prefer in-edge, then execution order

    Args:
        graph: The graph containing nodes
        weight_matrix: Adjacency matrix with edge weights
        target_node_id: ID of the node to merge
        execution_order: List of node IDs in execution order

    Returns:
        Tuple of (partner_node_id, weight)
    """
    # Find target node index
    target_idx = None
    for i, node in enumerate(list(graph.nodes.values())):
        if node.id == target_node_id:
            target_idx = i
            break

    if target_idx is None:
        raise ValueError(f"Node {target_node_id} not found in graph")

    # Step 1: Candidate Search - Find all connected nodes
    candidates = {}
    n_nodes = len(list(graph.nodes.values()))

    for j in range(n_nodes):
        if j == target_idx:
            continue

        # Check if there's an edge (either direction)
        in_weight = weight_matrix[j, target_idx].item()
        out_weight = weight_matrix[target_idx, j].item()

        if in_weight > 0 or out_weight > 0:
            # Step 2: Weight Comparison - S(c,j) = max(Ã[j,c], Ã[c,j])
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

    # Find maximum weight
    max_weight = max(c['weight'] for c in candidates.values())

    # Get all candidates with maximum weight
    top_candidates = {nid: c for nid, c in candidates.items() if c['weight'] == max_weight}

    # Step 3: Tie-breaker
    if len(top_candidates) == 1:
        partner_id = list(top_candidates.keys())[0]
    else:
        # Check if weights are approximately equal
        tie_candidates = []
        for nid, c in top_candidates.items():
            if abs(c['in_weight'] - c['out_weight']) < 1e-6:  # Approximately equal
                tie_candidates.append(nid)

        if tie_candidates:
            # Priority 1: Prefer in-edge nodes (nodes that feed into target)
            in_edge_candidates = [nid for nid in tie_candidates
                                 if top_candidates[nid]['in_weight'] >= top_candidates[nid]['out_weight']]

            if in_edge_candidates:
                # Priority 2: Among in-edge nodes, select by execution order (most recent input)
                # Find the one that appears latest in execution order
                partner_id = max(in_edge_candidates,
                               key=lambda nid: execution_order.index(nid) if nid in execution_order else -1)
            else:
                # Fallback: select by execution order
                partner_id = max(tie_candidates,
                               key=lambda nid: execution_order.index(nid) if nid in execution_order else -1)
        else:
            # No exact ties, just select the first one
            partner_id = list(top_candidates.keys())[0]

    partner_weight = candidates[partner_id]['weight']

    print(f"Merge partner selection:")
    print(f"  Target node: {target_node_id}")
    print(f"  Candidates: {[(nid, c['weight']) for nid, c in candidates.items()]}")
    print(f"  Selected partner: {partner_id} with weight {partner_weight:.4f}")

    return partner_id, partner_weight


async def merge_nodes_in_graph(graph: Graph, node1_id: str, node2_id: str, llm_name: str = "gpt-4o-mini") -> Graph:
    """
    Merge two nodes in the graph by combining their prompts and reconnecting edges.

    The main agent's prompt is augmented with a summary of the absorbed agent's prompt.
    The merged node is placed at the position of node1.

    Args:
        graph: The graph containing nodes
        node1_id: ID of the main node (will be kept)
        node2_id: ID of the node to be absorbed
        llm_name: LLM to use for intelligent prompt merging

    Returns:
        Modified graph with merged nodes
    """
    # Find the nodes
    node1, node1_idx = None, None
    node2, node2_idx = None, None

    for i, node in enumerate(list(graph.nodes.values())):
        if node.id == node1_id:
            node1, node1_idx = node, i
        if node.id == node2_id:
            node2, node2_idx = node, i

    if node1 is None or node2 is None:
        raise ValueError(f"Nodes {node1_id} or {node2_id} not found in graph")

    print(f"\nMerging nodes: {node1_id} (main) ← {node2_id} (absorbed)")
    print(f"  Node1 agent: {node1.agent_name}, role: {node1.role if hasattr(node1, 'role') else 'N/A'}")
    print(f"  Node2 agent: {node2.agent_name}, role: {node2.role if hasattr(node2, 'role') else 'N/A'}")

    # Track tokens/cost before merge
    pre_merge_cost = Cost.instance().value
    pre_merge_prompt = PromptTokens.instance().value
    pre_merge_completion = CompletionTokens.instance().value

    # Merge prompts intelligently using LLM
    print(f"  Merging prompts with LLM...")
    if hasattr(node1, 'constraint') and hasattr(node2, 'constraint'):
        try:
            # Directly await since we're already in async context
            merged_constraint = await merge_prompts_with_llm(node1, node2, llm_name)

            node1.constraint = merged_constraint

            # Calculate merge cost
            merge_cost = Cost.instance().value - pre_merge_cost
            merge_prompt = PromptTokens.instance().value - pre_merge_prompt
            merge_completion = CompletionTokens.instance().value - pre_merge_completion

            print(f"  Prompts merged successfully")
            print(f"  Merge LLM cost: ${merge_cost:.4f}, Prompt tokens: {merge_prompt:.0f}, Completion tokens: {merge_completion:.0f}")
            print(f"  New constraint preview: {merged_constraint[:100]}...")
        except Exception as e:
            print(f"  Warning: Async merge failed ({e}), using simple merge")
            node1.constraint = f"{node1.constraint}\n\nAdditional context from merged agent: {node2.constraint}"

    # Update agent name to reflect merge
    merged_name = f"{node1.agent_name}_merged_with_{node2.agent_name}"
    node1.agent_name = merged_name
    print(f"  Merged agent name: {merged_name}")

    # Reconnect edges: node2's connections are transferred to node1
    # Spatial predecessors of node2 → node1
    for pred in node2.spatial_predecessors:
        if pred.id != node1_id and node1 not in pred.spatial_successors:
            pred.add_successor(node1, st='spatial')

    # Spatial successors of node2 → node1
    for succ in node2.spatial_successors:
        if succ.id != node1_id and node1 not in succ.spatial_predecessors:
            node1.add_successor(succ, st='spatial')

    # Temporal predecessors of node2 → node1
    for pred in node2.temporal_predecessors:
        if pred.id != node1_id and node1 not in pred.temporal_successors:
            pred.add_successor(node1, st='temporal')

    # Temporal successors of node2 → node1
    for succ in node2.temporal_successors:
        if succ.id != node1_id and node1 not in succ.temporal_predecessors:
            node1.add_successor(succ, st='temporal')

    # Remove node2 from the graph
    del graph.nodes[node2.id]

    # Update potential_edges lists to remove edges involving deleted node
    graph.potential_spatial_edges = [
        edge for edge in graph.potential_spatial_edges
        if edge[0] != node2_id and edge[1] != node2_id
    ]
    graph.potential_temporal_edges = [
        edge for edge in graph.potential_temporal_edges
        if edge[0] != node2_id and edge[1] != node2_id
    ]

    # Update logits dimensions for new graph size
    n_nodes = len(list(graph.nodes.values()))
    old_size = n_nodes + 1  # Size before removing node2

    # node2_idx should be valid for the old size
    if node2_idx >= old_size:
        print(f"Warning: node2_idx {node2_idx} >= old_size {old_size}, skipping logits update")
        return graph

    if not graph.diff:
        # Check current logits size
        current_size = int(graph.spatial_logits_1.numel() ** 0.5)
        if current_size != old_size:
            print(f"Warning: logits size mismatch. Expected {old_size}x{old_size}, got {current_size}x{current_size}")
            old_size = current_size

        # Reshape to matrix, remove row/column, then flatten
        # Update spatial_logits_1
        spatial_matrix = graph.spatial_logits_1.reshape(old_size, old_size)
        mask = torch.ones(old_size, old_size, dtype=torch.bool)
        mask[node2_idx, :] = False
        mask[:, node2_idx] = False
        graph.spatial_logits_1 = torch.nn.Parameter(
            spatial_matrix[mask].reshape(n_nodes, n_nodes).flatten(),
            requires_grad=True
        )

        # Update temporal_logits_1
        temporal_matrix = graph.temporal_logits_1.reshape(old_size, old_size)
        graph.temporal_logits_1 = torch.nn.Parameter(
            temporal_matrix[mask].reshape(n_nodes, n_nodes).flatten(),
            requires_grad=True
        )
    else:
        # For diff mode, update each logit in the list
        current_size = int(graph.spatial_logits_1[0].numel() ** 0.5)
        old_size = n_nodes + 1

        if current_size != old_size:
            print(f"Warning: logits size mismatch in diff mode. Expected {old_size}x{old_size}, got {current_size}x{current_size}")
            old_size = current_size

        if node2_idx >= old_size:
            print(f"Warning: node2_idx {node2_idx} >= old_size {old_size}, skipping logits update")
            return graph

        new_spatial_logits = []
        new_temporal_logits = []

        # Update spatial logits
        for i in range(len(graph.spatial_logits_1)):
            spatial_matrix = graph.spatial_logits_1[i].reshape(old_size, old_size)
            mask = torch.ones(old_size, old_size, dtype=torch.bool)
            mask[node2_idx, :] = False
            mask[:, node2_idx] = False
            new_spatial_logits.append(
                torch.nn.Parameter(spatial_matrix[mask].reshape(n_nodes, n_nodes).flatten())
            )

        # Update temporal logits (length may be different, e.g., rounds-1)
        for i in range(len(graph.temporal_logits_1)):
            temporal_matrix = graph.temporal_logits_1[i].reshape(old_size, old_size)
            mask = torch.ones(old_size, old_size, dtype=torch.bool)
            mask[node2_idx, :] = False
            mask[:, node2_idx] = False
            new_temporal_logits.append(
                torch.nn.Parameter(temporal_matrix[mask].reshape(n_nodes, n_nodes).flatten())
            )

        graph.spatial_logits_1 = torch.nn.ParameterList(new_spatial_logits)
        graph.temporal_logits_1 = torch.nn.ParameterList(new_temporal_logits)

    # Update spatial and temporal masks (remove row/column for node2)
    # NOTE: Skipping masks update as they will be re-initialized in Edge Stage
    # The complexity of updating masks in diff mode (different lengths for spatial/temporal)
    # is not worth it since we re-initialize weights and masks after merge anyway
    if False and hasattr(graph, 'spatial_masks'):
        if not graph.diff:
            # Check if masks are flattened
            if graph.spatial_masks.dim() == 1:
                # Reshape to matrix first
                spatial_mask_matrix = graph.spatial_masks.reshape(old_size, old_size)
                temporal_mask_matrix = graph.temporal_masks.reshape(old_size, old_size)

                # Create boolean mask
                mask = torch.ones(old_size, old_size, dtype=torch.bool)
                mask[node2_idx, :] = False
                mask[:, node2_idx] = False

                # Apply mask and flatten
                graph.spatial_masks = torch.nn.Parameter(
                    spatial_mask_matrix[mask].reshape(n_nodes, n_nodes).flatten(),
                    requires_grad=False
                )
                graph.temporal_masks = torch.nn.Parameter(
                    temporal_mask_matrix[mask].reshape(n_nodes, n_nodes).flatten(),
                    requires_grad=False
                )
            else:
                # Already 2D
                mask = torch.ones(old_size, old_size, dtype=torch.bool)
                mask[node2_idx, :] = False
                mask[:, node2_idx] = False
                graph.spatial_masks = torch.nn.Parameter(
                    graph.spatial_masks[mask].reshape(n_nodes, n_nodes),
                    requires_grad=False
                )
                graph.temporal_masks = torch.nn.Parameter(
                    graph.temporal_masks[mask].reshape(n_nodes, n_nodes),
                    requires_grad=False
                )
        else:
            # For diff mode, need to handle list of masks
            new_spatial_masks = []
            new_temporal_masks = []

            # Process spatial masks
            for i in range(len(graph.spatial_masks)):
                # Check if masks are flattened
                if graph.spatial_masks[i].dim() == 1:
                    # Reshape to matrix first
                    spatial_mask_matrix = graph.spatial_masks[i].reshape(old_size, old_size)

                    # Create boolean mask
                    mask = torch.ones(old_size, old_size, dtype=torch.bool)
                    mask[node2_idx, :] = False
                    mask[:, node2_idx] = False

                    # Apply mask and flatten (convert to float)
                    new_spatial_masks.append(
                        spatial_mask_matrix[mask].reshape(n_nodes, n_nodes).flatten().float()
                    )
                else:
                    # Already 2D
                    mask = torch.ones(old_size, old_size, dtype=torch.bool)
                    mask[node2_idx, :] = False
                    mask[:, node2_idx] = False
                    new_spatial_masks.append(
                        graph.spatial_masks[i][mask].reshape(n_nodes, n_nodes).float()
                    )

            # Process temporal masks (may have different length)
            for i in range(len(graph.temporal_masks)):
                # Check if masks are flattened
                if graph.temporal_masks[i].dim() == 1:
                    # Reshape to matrix first
                    temporal_mask_matrix = graph.temporal_masks[i].reshape(old_size, old_size)

                    # Create boolean mask
                    mask = torch.ones(old_size, old_size, dtype=torch.bool)
                    mask[node2_idx, :] = False
                    mask[:, node2_idx] = False

                    # Apply mask and flatten (convert to float)
                    new_temporal_masks.append(
                        temporal_mask_matrix[mask].reshape(n_nodes, n_nodes).flatten().float()
                    )
                else:
                    # Already 2D
                    mask = torch.ones(old_size, old_size, dtype=torch.bool)
                    mask[node2_idx, :] = False
                    mask[:, node2_idx] = False
                    new_temporal_masks.append(
                        graph.temporal_masks[i][mask].reshape(n_nodes, n_nodes).float()
                    )

            graph.spatial_masks = new_spatial_masks
            graph.temporal_masks = new_temporal_masks

    print(f"  Node {node2_id} merged into {node1_id}")
    print(f"  Remaining nodes: {len(list(graph.nodes.values()))}")

    return graph


async def train_with_node_merge(
    graph: Graph,
    dataset,
    num_iters: int = 100,
    num_rounds: int = 1,
    lr: float = 0.1,
    batch_size: int = 4,
    merge_iters: int = 5,
    edge_iters: int = 5,
    pruning_rate: float = 0.05,
    initial_probability: float = 0.5,
    args=None,
    kwargs=None,
) -> None:
    """
    Train the graph with node merging based on transition efficiency.

    Two-stage optimization:
    Stage 1: Node Optimization
        1. Initialize edge weights (0.5)
        2. Train with policy gradient
        3. Node Merge (replaces node dropout)
        4. Graph structure changes

    Stage 2: Edge Optimization
        1. Re-initialize edge weights (0.5) for new structure
        2. Train again with policy gradient
        3. Edge Dropout
        4. Finalize graph

    Args:
        graph: The computation graph
        dataset: Training dataset
        num_iters: Total number of training iterations (merge + edge)
        num_rounds: Number of rounds per query
        lr: Learning rate
        batch_size: Batch size
        merge_iters: Number of iterations for Merge Stage (before node merge)
        edge_iters: Number of iterations for Edge Stage (after node merge, before edge dropout)
        pruning_rate: Edge pruning rate
        initial_probability: Initial edge probability (0.5)
        args: Additional arguments
        kwargs: Additional keyword arguments
    """

    def infinite_data_loader() -> Iterator[pd.DataFrame]:
        while True:
            for idx in range(len(dataset)):
                record = dataset[idx]
                yield record

    loader = infinite_data_loader()

    print(f"\n{'='*80}")
    print(f"STAGE 1: NODE OPTIMIZATION")
    print(f"{'='*80}\n")

    if args.dec:
        graph.optimized_spatial = False
        graph.optimized_temporal = False

        if not graph.diff:
            optimizer = torch.optim.Adam([graph.spatial_logits_1, graph.temporal_logits_1], lr=lr)
        else:
            optimizer = torch.optim.Adam(
                list(graph.spatial_logits_1.parameters()) + list(graph.temporal_logits_1.parameters()),
                lr=lr
            )

        # Merge Stage: Train and then Node Merge
        for i_iter in range(merge_iters):
            print(f"\n{'='*80}")
            print(f"Merge Stage - Train Iteration {i_iter}")
            print(f"{'='*80}")
            start_ts = time.time()
            correct_answers = []
            answer_log_probs = []
            add_losses = []
            execution_orders = []  # Track execution order for tie-breaking

            for i_record, record in zip(range(batch_size), loader):
                realized_graph = copy.deepcopy(graph)
                realized_graph.spatial_logits_1 = graph.spatial_logits_1
                realized_graph.temporal_logits_1 = graph.temporal_logits_1

                # Get weight matrices
                if not graph.diff:
                    spatial_matrix_train = realized_graph.spatial_logits_1.reshape(
                        (len(list(graph.nodes.values())), len(list(graph.nodes.values())))
                    )
                    temporal_matrix_train = realized_graph.temporal_logits_1.reshape(
                        (len(list(graph.nodes.values())), len(list(graph.nodes.values())))
                    )
                else:
                    spatial_matrix_train = [
                        param.reshape((len(list(graph.nodes.values())), len(list(graph.nodes.values()))))
                        for param in realized_graph.spatial_logits_1
                    ]
                    temporal_matrix_train = [
                        param.reshape((len(list(graph.nodes.values())), len(list(graph.nodes.values()))))
                        for param in realized_graph.temporal_logits_1
                    ]

                # Calculate regularization losses (same as before)
                spatial_matrix_fixed = torch.tensor(
                    kwargs["fixed_spatial_masks"], dtype=torch.float32
                ).reshape((len(list(graph.nodes.values())), len(list(graph.nodes.values()))))
                temporal_matrix_fixed = torch.tensor(
                    kwargs["fixed_temporal_masks"], dtype=torch.float32
                ).reshape((len(list(graph.nodes.values())), len(list(graph.nodes.values()))))

                if not graph.diff:
                    loss_s = nuclear_norm(spatial_matrix_train)
                    loss_t = nuclear_norm(temporal_matrix_train)
                    frob_loss_s = frobenius_norm(spatial_matrix_fixed, spatial_matrix_train)
                    frob_loss_t = frobenius_norm(temporal_matrix_fixed, temporal_matrix_train)
                else:
                    loss_s = torch.mean(torch.stack([nuclear_norm(matrix) for matrix in spatial_matrix_train]))
                    loss_t = torch.mean(torch.stack([nuclear_norm(matrix) for matrix in temporal_matrix_train]))
                    frob_loss_s = torch.mean(torch.stack([
                        frobenius_norm(spatial_matrix_fixed, matrix) for matrix in spatial_matrix_train
                    ]))
                    frob_loss_t = torch.mean(torch.stack([
                        frobenius_norm(temporal_matrix_fixed, matrix) for matrix in temporal_matrix_train
                    ]))

                add_loss = loss_s + loss_t + F.relu(frob_loss_s - args.delta) + F.relu(frob_loss_t - args.delta)
                add_loss = 0  # Set to 0 as in original code

                input_dict = dataset.record_to_input(record)

                if args.dec:
                    answer_log_probs.append(asyncio.create_task(realized_graph.arun(input_dict, num_rounds, skip=True)))
                else:
                    answer_log_probs.append(asyncio.create_task(realized_graph.arun(input_dict, num_rounds)))

                correct_answer = dataset.record_to_target_answer(record)
                correct_answers.append(correct_answer)
                add_losses.append(add_loss)

                # Track execution order
                execution_orders.append([node.id for node in list(realized_graph.nodes.values())])

            raw_results = await asyncio.gather(*answer_log_probs)
            raw_answers, log_probs = zip(*raw_results)
            loss_list: List[torch.Tensor] = []
            utilities: List[float] = []
            answers: List[str] = []

            for raw_answer, log_prob, add_loss, correct_answer in zip(
                raw_answers, log_probs, add_losses, correct_answers
            ):
                answer = dataset.postprocess_answer(raw_answer)
                answers.append(answer)
                assert isinstance(correct_answer, str), \
                    f"String expected but got {correct_answer} of type {type(correct_answer)} (1)"
                accuracy = Accuracy()
                accuracy.update(answer, correct_answer)
                utility = accuracy.get()
                utilities.append(utility)
                single_loss = -log_prob * utility
                loss_list.append(single_loss + add_loss)

            total_loss = torch.mean(torch.stack(loss_list))
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if not graph.diff:
                spatial_probs = torch.sigmoid(graph.spatial_logits_1)
                temporal_probs = torch.sigmoid(graph.temporal_logits_1)
            else:
                spatial_probs = [torch.sigmoid(logit) for logit in graph.spatial_logits_1]
                temporal_probs = [torch.sigmoid(logit) for logit in graph.temporal_logits_1]

            print(f"\nBatch time {time.time() - start_ts:.3f}")
            print("Utilities:", utilities)
            print("Loss:", total_loss.item())
            print(f"Cost {Cost.instance().value}")
            print(f"PromptTokens {PromptTokens.instance().value}")
            print(f"CompletionTokens {CompletionTokens.instance().value}")

        # After Merge Stage training: Node Merge
        if len(list(graph.nodes.values())) > 2:
            print(f"\n{'='*80}")
            print(f"MERGE STAGE: NODE MERGE")
            print(f"{'='*80}")

            # Get current weight matrix
            with torch.no_grad():
                if not graph.diff:
                    weight_matrix = torch.sigmoid(graph.spatial_logits_1).reshape(
                        (len(list(graph.nodes.values())), len(list(graph.nodes.values())))
                    )
                else:
                    # Use mean of all weight matrices
                    weight_matrix = torch.mean(torch.stack([
                        torch.sigmoid(logit) for logit in graph.spatial_logits_1
                    ]), dim=0).reshape((len(list(graph.nodes.values())), len(list(graph.nodes.values()))))

            # Step 2: Find merge candidate (node with minimum ΔW)
            merge_candidate_id, min_delta_w = find_merge_candidate(graph, weight_matrix)

            # Always perform merge (no threshold check in original design)
            print(f"Proceeding with merge (ΔW = {min_delta_w:.4f})")

            # Step 3: Find merge partner
            if execution_orders:
                execution_order = execution_orders[-1]  # Use most recent execution order
            else:
                # Fallback: use node order from graph
                execution_order = [node.id for node in list(graph.nodes.values())]

            partner_id, partner_weight = find_merge_partner(
                graph, weight_matrix, merge_candidate_id, execution_order
            )

            if partner_id is not None:
                # Merge nodes (pass llm_name from graph or use default)
                llm_name = graph.llm_name if hasattr(graph, 'llm_name') else "gpt-4o-mini"
                graph = await merge_nodes_in_graph(graph, partner_id, merge_candidate_id, llm_name)

                # Update fixed masks for new graph structure
                n_nodes = len(list(graph.nodes.values()))
                # Find index to remove (this is approximate, ideally should track properly)
                old_n = len(kwargs["fixed_spatial_masks"])
                if old_n > n_nodes:
                    # Remove row and column corresponding to merged node
                    merge_idx = old_n - n_nodes  # Simplified assumption
                    kwargs["fixed_spatial_masks"] = [
                        [kwargs["fixed_spatial_masks"][i][j] for j in range(old_n) if j != merge_idx]
                        for i in range(old_n) if i != merge_idx
                    ]
                    kwargs["fixed_temporal_masks"] = [
                        [kwargs["fixed_temporal_masks"][i][j] for j in range(old_n) if j != merge_idx]
                        for i in range(old_n) if i != merge_idx
                    ]

        # EDGE STAGE: EDGE OPTIMIZATION
        print(f"\n{'='*80}")
        print(f"EDGE STAGE: EDGE OPTIMIZATION")
        print(f"Re-initializing edge weights to {initial_probability}")
        print(f"{'='*80}\n")

        # Re-initialize edge weights for new graph structure
        n_nodes = len(list(graph.nodes.values()))
        n_edges = len(graph.potential_spatial_edges)
        n_temporal_edges = len(graph.potential_temporal_edges)
        init_logit = torch.log(torch.tensor(initial_probability / (1 - initial_probability)))

        if not graph.diff:
            graph.spatial_logits_1 = torch.nn.Parameter(
                torch.ones(n_edges, requires_grad=True) * init_logit,
                requires_grad=True
            )
            graph.temporal_logits_1 = torch.nn.Parameter(
                torch.ones(n_temporal_edges, requires_grad=True) * init_logit,
                requires_grad=True
            )
            graph.spatial_masks = [1.0] * n_edges
            graph.temporal_masks = [1.0] * n_temporal_edges
            optimizer = torch.optim.Adam([graph.spatial_logits_1, graph.temporal_logits_1], lr=lr)
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
                lr=lr
            )

        # Edge Stage: Train with new graph structure
        for i_iter in range(edge_iters):
            print(f"\n{'='*80}")
            print(f"Edge Stage - Train Iteration {i_iter}")
            print(f"{'='*80}")
            start_ts = time.time()
            correct_answers = []
            answer_log_probs = []
            add_losses = []

            for i_record, record in zip(range(batch_size), loader):
                realized_graph = copy.deepcopy(graph)
                realized_graph.spatial_logits_1 = graph.spatial_logits_1
                realized_graph.temporal_logits_1 = graph.temporal_logits_1

                # Get weight matrices
                if not graph.diff:
                    spatial_matrix_train = realized_graph.spatial_logits_1.reshape(
                        (len(list(graph.nodes.values())), len(list(graph.nodes.values())))
                    )
                    temporal_matrix_train = realized_graph.temporal_logits_1.reshape(
                        (len(list(graph.nodes.values())), len(list(graph.nodes.values())))
                    )
                else:
                    spatial_matrix_train = [
                        param.reshape((len(list(graph.nodes.values())), len(list(graph.nodes.values()))))
                        for param in realized_graph.spatial_logits_1
                    ]
                    temporal_matrix_train = [
                        param.reshape((len(list(graph.nodes.values())), len(list(graph.nodes.values()))))
                        for param in realized_graph.temporal_logits_1
                    ]

                # Calculate regularization losses
                spatial_matrix_fixed = torch.tensor(
                    kwargs["fixed_spatial_masks"], dtype=torch.float32
                ).reshape((len(list(graph.nodes.values())), len(list(graph.nodes.values()))))
                temporal_matrix_fixed = torch.tensor(
                    kwargs["fixed_temporal_masks"], dtype=torch.float32
                ).reshape((len(list(graph.nodes.values())), len(list(graph.nodes.values()))))

                if not graph.diff:
                    loss_s = nuclear_norm(spatial_matrix_train)
                    loss_t = nuclear_norm(temporal_matrix_train)
                    frob_loss_s = frobenius_norm(spatial_matrix_fixed, spatial_matrix_train)
                    frob_loss_t = frobenius_norm(temporal_matrix_fixed, temporal_matrix_train)
                else:
                    loss_s = torch.mean(torch.stack([nuclear_norm(matrix) for matrix in spatial_matrix_train]))
                    loss_t = torch.mean(torch.stack([nuclear_norm(matrix) for matrix in temporal_matrix_train]))
                    frob_loss_s = torch.mean(torch.stack([
                        frobenius_norm(spatial_matrix_fixed, matrix) for matrix in spatial_matrix_train
                    ]))
                    frob_loss_t = torch.mean(torch.stack([
                        frobenius_norm(temporal_matrix_fixed, matrix) for matrix in temporal_matrix_train
                    ]))

                add_loss = loss_s + loss_t + F.relu(frob_loss_s - args.delta) + F.relu(frob_loss_t - args.delta)
                add_loss = 0

                input_dict = dataset.record_to_input(record)

                if args.dec:
                    answer_log_probs.append(asyncio.create_task(realized_graph.arun(input_dict, num_rounds, skip=True)))
                else:
                    answer_log_probs.append(asyncio.create_task(realized_graph.arun(input_dict, num_rounds)))

                correct_answer = dataset.record_to_target_answer(record)
                correct_answers.append(correct_answer)
                add_losses.append(add_loss)

            raw_results = await asyncio.gather(*answer_log_probs)
            raw_answers, log_probs = zip(*raw_results)
            loss_list: List[torch.Tensor] = []
            utilities: List[float] = []
            answers: List[str] = []

            for raw_answer, log_prob, add_loss, correct_answer in zip(
                raw_answers, log_probs, add_losses, correct_answers
            ):
                answer = dataset.postprocess_answer(raw_answer)
                answers.append(answer)
                assert isinstance(correct_answer, str), \
                    f"String expected but got {correct_answer} of type {type(correct_answer)} (1)"
                accuracy = Accuracy()
                accuracy.update(answer, correct_answer)
                utility = accuracy.get()
                utilities.append(utility)
                single_loss = -log_prob * utility
                loss_list.append(single_loss + add_loss)

            total_loss = torch.mean(torch.stack(loss_list))
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if not graph.diff:
                spatial_probs = torch.sigmoid(graph.spatial_logits_1)
                temporal_probs = torch.sigmoid(graph.temporal_logits_1)
            else:
                spatial_probs = [torch.sigmoid(logit) for logit in graph.spatial_logits_1]
                temporal_probs = [torch.sigmoid(logit) for logit in graph.temporal_logits_1]

            print(f"\nBatch time {time.time() - start_ts:.3f}")
            print("Utilities:", utilities)
            print("Loss:", total_loss.item())
            print(f"Cost {Cost.instance().value}")
            print(f"PromptTokens {PromptTokens.instance().value}")
            print(f"CompletionTokens {CompletionTokens.instance().value}")

        # After Edge Stage training: Edge Dropout
        print(f"\n{'='*80}")
        print(f"EDGE STAGE: EDGE DROPOUT (rate={pruning_rate})")
        print(f"{'='*80}")

        if not graph.diff:
            spatial_masks, temporal_masks = graph.update_masks(pruning_rate)
        else:
            spatial_masks, temporal_masks = graph.update_masks_diff(pruning_rate)

        print("Spatial masks:", spatial_masks)
        print("Temporal masks:", temporal_masks)

        if not graph.diff:
            if isinstance(spatial_masks, list):
                spatial_masks = torch.tensor(spatial_masks)
            if isinstance(temporal_masks, list):
                temporal_masks = torch.tensor(temporal_masks)
            print("Spatial sparsity:", spatial_masks.sum() / spatial_masks.numel())
            print("Temporal sparsity:", temporal_masks.sum() / temporal_masks.numel())
        else:
            if isinstance(spatial_masks[0], list):
                spatial_masks = [torch.tensor(m) for m in spatial_masks]
            if isinstance(temporal_masks[0], list):
                temporal_masks = [torch.tensor(m) for m in temporal_masks]
            print("Spatial sparsity:", spatial_masks[0].sum() / spatial_masks[0].numel())
            print("Temporal sparsity:", temporal_masks[0].sum() / temporal_masks[0].numel())


async def train(graph: Graph, dataset, num_iters: int = 100, num_rounds: int = 1,
                lr: float = 0.1, batch_size: int = 4, imp_per_iters: int = 1,
                pruning_rate: float = 0.05, args=None, kwargs=None) -> None:
    """
    Wrapper function to use two-stage node merge training.
    """
    # Split iterations into Merge Stage and Edge Stage
    merge_iters = imp_per_iters  # Use imp_per_iters for Merge Stage
    edge_iters = num_iters - merge_iters if num_iters > merge_iters else imp_per_iters

    initial_probability = 0.5  # Default initial probability

    await train_with_node_merge(
        graph=graph,
        dataset=dataset,
        num_iters=num_iters,
        num_rounds=num_rounds,
        lr=lr,
        batch_size=batch_size,
        merge_iters=merge_iters,
        edge_iters=edge_iters,
        pruning_rate=pruning_rate,
        initial_probability=initial_probability,
        args=args,
        kwargs=kwargs
    )
