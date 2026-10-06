"""Unit tests for Grounded MPC Debate Engine with Theory of Mind Belief Books."""

import pytest
import numpy as np
from arc3sdk.mpc_grounded_debate import (
    MPCGroundedDebateEngine,
    BeliefBook,
    CommunicativeAct,
    CommunicativeActType
)


def test_belief_book_entropy_and_updates():
    book = BeliefBook(agent_id="test_agent", persona="Analyst")
    book.action_beliefs = {1: 0.5, 2: 0.5}
    ent_init = book.get_entropy()
    assert ent_init > 0.0

    # Increase belief in action 1
    book.update_belief(action=1, delta_evidence=0.4)
    assert book.action_beliefs[1] > book.action_beliefs[2]


def test_mpc_grounded_debate_quorum_convergence():
    engine = MPCGroundedDebateEngine(max_debate_rounds=3, quorum_threshold=0.6)
    grid = np.zeros((10, 10), dtype=np.int32)

    # Simulator where action 2 is winning, action 1 is fatal
    def mock_verifier(g: np.ndarray, action: int):
        if action == 1:
            return 0.0, True  # Fatal
        elif action == 2:
            return 0.95, False # Highly positive
        return 0.5, False

    chosen_act, conf, transcript = engine.deliberate(
        grid=grid,
        available_actions=[1, 2, 3],
        verifier_fn=mock_verifier
    )

    assert chosen_act == 2
    assert conf >= 0.70
    assert len(transcript) > 0

    # Ensure critique acts were registered for action 1
    critiques = [t for t in transcript if t.act_type == CommunicativeActType.CRITIQUE]
    assert len(critiques) >= 1
    assert critiques[0].action_candidate == 1
