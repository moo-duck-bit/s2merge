"""Absorption-based node merge.

One merge step is: pick the bottleneck node, pick the neighbour it is most
strongly tied to, fold its role into that neighbour's prompt, then delete the
node and every edge touching it. No new node is created and no edge of the
absorbed node is rewired — the graph simply gets one node smaller while the
role information survives in the partner's prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from s2rmerge.mas.graph import Graph
from s2rmerge.merge.criterion import select_merge_partner, select_merge_target
from s2rmerge.merge.fusion import concatenate, fuse_prompts


@dataclass(frozen=True)
class MergeOutcome:
    """What one merge step did, for the run report."""

    absorbed_id: str
    absorbed_role: str
    partner_id: str
    partner_role: str
    delta_w: float
    partner_score: float
    nodes_before: int
    nodes_after: int

    def __str__(self) -> str:
        return (
            f"merged {self.absorbed_role} ({self.absorbed_id}) into "
            f"{self.partner_role} ({self.partner_id}); "
            f"dW={self.delta_w:.4f}, S={self.partner_score:.4f}, "
            f"{self.nodes_before} -> {self.nodes_after} nodes"
        )


async def merge_once(
    graph: Graph,
    llm_name: Optional[str] = None,
    use_fusion: bool = True,
) -> Optional[MergeOutcome]:
    """Perform one absorption on ``graph`` in place.

    Returns ``None`` when the graph is too small to merge, or when the target
    turns out to have no connected neighbour to absorb it.

    With ``use_fusion=False`` the partner's prompt is extended by plain
    concatenation instead of the LLM fusion, which is the ablation in the paper.
    """
    if graph.num_nodes < 2:
        return None

    adjacency = graph.spatial_adjacency().numpy()
    node_ids = graph.node_ids

    target = select_merge_target(adjacency, node_ids)
    partner = select_merge_partner(adjacency, node_ids, target.node_id, execution_order=node_ids)
    if partner is None:
        return None

    absorbed = graph.find_node(target.node_id)
    primary = graph.find_node(partner.node_id)

    if use_fusion:
        primary.constraint = await fuse_prompts(primary, absorbed, llm_name)
    else:
        primary.constraint = concatenate(primary, absorbed)

    nodes_before = graph.num_nodes
    graph.remove_node(target.node_id)
    # The learned weights were sized for the old topology, so they are discarded
    # and relearned on the merged graph before edge pruning.
    graph.reset_parameters()

    return MergeOutcome(
        absorbed_id=target.node_id,
        absorbed_role=absorbed.role,
        partner_id=partner.node_id,
        partner_role=primary.role,
        delta_w=target.delta_w,
        partner_score=partner.score,
        nodes_before=nodes_before,
        nodes_after=graph.num_nodes,
    )


def merge_summary(outcomes: Dict[int, MergeOutcome]) -> str:
    return "\n".join(f"  iteration {i}: {outcome}" for i, outcome in sorted(outcomes.items()))
