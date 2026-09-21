"""The merge criterion picks the node the paper's definition says it should."""

import numpy as np
import pytest

from s2rmerge.merge.criterion import (
    select_merge_partner,
    select_merge_target,
    transfer_efficiency,
)

NODE_IDS = ["a", "b", "c"]


def test_delta_w_is_stable_in_minus_stable_out():
    # b receives two strong, identical edges and sends two weak, identical ones,
    # so its incoming term dominates and dW(b) must be positive.
    adjacency = np.array(
        [
            [0.0, 0.9, 0.1],
            [0.1, 0.0, 0.1],
            [0.1, 0.9, 0.0],
        ]
    )
    scores = transfer_efficiency(adjacency, NODE_IDS)

    incoming = np.array([0.9, 0.9])
    outgoing = np.array([0.1, 0.1])
    expected = incoming.sum() / (incoming.var() + 1e-8) - outgoing.sum() / (outgoing.var() + 1e-8)

    assert scores["b"] == pytest.approx(expected)
    assert scores["b"] > 0


def test_variance_penalises_erratic_signal():
    # Same incoming total, but one node's edges are lopsided. The erratic node
    # must score lower, which is what dividing by the variance buys.
    steady = np.array([[0.0, 0.5, 0.0], [0.0, 0.0, 0.0], [0.0, 0.5, 0.0]])
    erratic = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    assert (
        transfer_efficiency(steady, NODE_IDS)["b"]
        > transfer_efficiency(erratic, NODE_IDS)["b"]
    )


def test_target_is_the_argmax():
    adjacency = np.array(
        [
            [0.0, 0.9, 0.1],
            [0.1, 0.0, 0.1],
            [0.1, 0.9, 0.0],
        ]
    )
    target = select_merge_target(adjacency, NODE_IDS)

    assert target.node_id == "b"
    assert target.delta_w == max(target.scores.values())


def test_target_needs_two_nodes():
    with pytest.raises(ValueError):
        select_merge_target(np.zeros((1, 1)), ["a"])


def test_partner_is_the_strongest_interaction_in_either_direction():
    # c -> a is the single strongest edge touching a, even though a sends more
    # to b, so c must win on max(A[j,a], A[a,j]).
    adjacency = np.array(
        [
            [0.0, 0.4, 0.1],
            [0.2, 0.0, 0.1],
            [0.8, 0.1, 0.0],
        ]
    )
    partner = select_merge_partner(adjacency, NODE_IDS, "a")

    assert partner.node_id == "c"
    assert partner.score == pytest.approx(0.8)


def test_partner_ties_prefer_an_incoming_edge():
    # b and c both score 0.5 against a, but only c sends to a.
    adjacency = np.array(
        [
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
        ]
    )
    partner = select_merge_partner(adjacency, NODE_IDS, "a")

    assert partner.node_id == "c"


def test_partner_ties_then_prefer_the_most_recent_sender():
    # b and c both send to a with the same weight; c comes later in the
    # execution order, so it spoke most recently.
    adjacency = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.5, 0.0, 0.0],
        ]
    )
    partner = select_merge_partner(adjacency, NODE_IDS, "a", execution_order=["b", "c", "a"])

    assert partner.node_id == "c"


def test_isolated_target_has_no_partner():
    adjacency = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.7],
            [0.0, 0.7, 0.0],
        ]
    )
    assert select_merge_partner(adjacency, NODE_IDS, "a") is None
