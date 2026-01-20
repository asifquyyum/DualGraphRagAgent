"""Tests for epistemic state tracking (world state, world model agent, sufficiency checker)."""

import pytest
from unittest.mock import MagicMock

from src.agents.world_model_agent import (
    WorldModelAgent,
    WorldModelUpdate,
    BeliefUpdate,
    UncertaintyUpdate,
)
from src.agents.sufficiency_checker import SufficiencyChecker, SufficiencyAssessment
from src.schema.world_state import (
    Belief,
    BeliefStatus,
    EvidenceSource,
    Uncertainty,
    UncertaintyType,
    WorldState,
)


@pytest.fixture
def mock_llm():
    """Create a mock LLM provider."""
    llm = MagicMock()
    llm.generate.return_value = "Mock response"
    return llm


@pytest.fixture
def world_state():
    """Create a test world state."""
    state = WorldState(
        original_query="What happens to a straddle when VIX spikes?",
        max_iterations=10,
    )
    return state


class TestWorldState:
    """Tests for WorldState."""

    def test_add_belief(self, world_state):
        belief = Belief(
            id="b1",
            content="Straddles profit from volatility spikes",
            status=BeliefStatus.HYPOTHETICAL,
            confidence=0.7,
        )
        world_state.add_belief(belief)

        assert "b1" in world_state.beliefs
        assert world_state.get_belief("b1") == belief

    def test_add_uncertainty(self, world_state):
        uncertainty = Uncertainty(
            id="u1",
            description="What is the delta exposure?",
            uncertainty_type=UncertaintyType.MISSING_DATA,
            priority=0.8,
        )
        world_state.add_uncertainty(uncertainty)

        assert "u1" in world_state.uncertainties
        assert world_state.get_uncertainty("u1") == uncertainty

    def test_get_confirmed_beliefs(self, world_state):
        world_state.add_belief(
            Belief(id="b1", content="Confirmed belief", status=BeliefStatus.CONFIRMED)
        )
        world_state.add_belief(
            Belief(id="b2", content="Hypothetical belief", status=BeliefStatus.HYPOTHETICAL)
        )

        confirmed = world_state.get_confirmed_beliefs()
        assert len(confirmed) == 1
        assert confirmed[0].id == "b1"

    def test_get_high_confidence_beliefs(self, world_state):
        world_state.add_belief(
            Belief(id="b1", content="High confidence", confidence=0.9)
        )
        world_state.add_belief(
            Belief(id="b2", content="Low confidence", confidence=0.3)
        )

        high_conf = world_state.get_high_confidence_beliefs(0.7)
        assert len(high_conf) == 1
        assert high_conf[0].id == "b1"

    def test_get_unresolved_uncertainties(self, world_state):
        u1 = Uncertainty(id="u1", description="Unresolved", uncertainty_type=UncertaintyType.MISSING_DATA)
        u2 = Uncertainty(id="u2", description="Exhausted", uncertainty_type=UncertaintyType.MISSING_DATA, max_attempts=1)
        u2.record_attempt()  # Exhaust attempts

        world_state.add_uncertainty(u1)
        world_state.add_uncertainty(u2)

        unresolved = world_state.get_unresolved_uncertainties()
        assert len(unresolved) == 1
        assert unresolved[0].id == "u1"

    def test_has_counterfactual_uncertainty(self, world_state):
        assert not world_state.has_counterfactual_uncertainty()

        world_state.add_uncertainty(
            Uncertainty(
                id="u1",
                description="What if VIX doubles?",
                uncertainty_type=UncertaintyType.COUNTERFACTUAL,
            )
        )
        assert world_state.has_counterfactual_uncertainty()

    def test_has_sufficient_evidence(self, world_state):
        # Not sufficient initially
        assert not world_state.has_sufficient_evidence()

        # Add confirmed belief and high completeness
        world_state.add_belief(
            Belief(id="b1", content="Test", status=BeliefStatus.CONFIRMED, confidence=0.9)
        )
        world_state.answer_completeness = 0.9

        assert world_state.has_sufficient_evidence()

    def test_should_continue(self, world_state):
        # Should continue with unresolved uncertainties
        world_state.add_uncertainty(
            Uncertainty(id="u1", description="Test", uncertainty_type=UncertaintyType.MISSING_DATA)
        )
        assert world_state.should_continue()

        # Should not continue after max iterations
        world_state.iteration_count = 10
        assert not world_state.should_continue()

    def test_to_context_string(self, world_state):
        world_state.add_belief(
            Belief(id="b1", content="Test belief", status=BeliefStatus.CONFIRMED, confidence=0.8)
        )
        world_state.add_uncertainty(
            Uncertainty(id="u1", description="Test uncertainty", uncertainty_type=UncertaintyType.MISSING_DATA, priority=0.7)
        )

        context = world_state.to_context_string()

        assert "straddle" in context.lower() or world_state.original_query in context
        assert "Test belief" in context
        assert "Test uncertainty" in context


class TestBelief:
    """Tests for Belief evidence handling."""

    def test_add_supporting_evidence(self):
        belief = Belief(id="b1", content="Test", status=BeliefStatus.HYPOTHETICAL, confidence=0.5)

        evidence = EvidenceSource(
            source_type="document",
            source_id="doc1",
            confidence=0.9,
        )
        belief.add_supporting_evidence(evidence)

        assert len(belief.supporting_evidence) == 1
        assert belief.status == BeliefStatus.CONFIRMED

    def test_add_contradicting_evidence(self):
        belief = Belief(id="b1", content="Test", status=BeliefStatus.CONFIRMED, confidence=0.8)

        # Add more contradicting than supporting evidence
        for i in range(3):
            belief.add_contradicting_evidence(
                EvidenceSource(source_type="document", source_id=f"doc{i}", confidence=0.9)
            )

        assert belief.status == BeliefStatus.CONTRADICTED
        assert belief.confidence < 0.5


class TestWorldModelAgent:
    """Tests for WorldModelAgent."""

    def test_basic_evidence_processing(self, mock_llm):
        agent = WorldModelAgent(mock_llm)
        world_state = WorldState(original_query="Test query")

        evidence = [
            {"source_type": "graph", "content": "Strategy X profits from volatility"},
            {"source_type": "document", "content": "VIX measures market fear"},
        ]

        # Make LLM fail to test fallback
        mock_llm.generate_structured.side_effect = ValueError("Mock error")

        update = agent._basic_evidence_processing(world_state, evidence)

        assert len(update.belief_updates) > 0
        assert update.completeness_estimate > 0

    def test_update_world_state(self, mock_llm):
        agent = WorldModelAgent(mock_llm)
        world_state = WorldState(original_query="Test query")

        # Mock successful LLM response
        mock_llm.generate_structured.return_value = WorldModelUpdate(
            belief_updates=[
                BeliefUpdate(
                    belief_id="b1",
                    content="VIX indicates market volatility",
                    status="new",
                    confidence=0.8,
                )
            ],
            uncertainty_updates=[],
            new_questions=[],
            completeness_estimate=0.5,
            reasoning="Evidence supports this belief",
        )

        evidence = [{"source_type": "graph", "content": "VIX data"}]
        updated_state = agent.update_world_state(world_state, evidence)

        assert "b1" in updated_state.beliefs
        assert updated_state.answer_completeness == 0.5

    def test_identify_gaps(self, mock_llm):
        agent = WorldModelAgent(mock_llm)
        world_state = WorldState(original_query="Test")

        world_state.remaining_questions = ["What is the delta?"]
        world_state.add_uncertainty(
            Uncertainty(id="u1", description="Missing gamma data", uncertainty_type=UncertaintyType.MISSING_DATA)
        )
        world_state.add_belief(
            Belief(id="b1", content="Low confidence belief", confidence=0.3)
        )

        gaps = agent.identify_gaps(world_state)

        assert "What is the delta?" in gaps
        assert "Missing gamma data" in gaps
        assert any("Low confidence belief" in g for g in gaps)


class TestSufficiencyChecker:
    """Tests for SufficiencyChecker."""

    def test_check_max_iterations(self, mock_llm):
        checker = SufficiencyChecker(mock_llm)
        world_state = WorldState(original_query="Test", max_iterations=5)
        world_state.iteration_count = 5

        assessment = checker.check(world_state)

        assert assessment.is_sufficient is True
        assert assessment.suggested_action == "synthesize"

    def test_check_high_completeness(self, mock_llm):
        checker = SufficiencyChecker(mock_llm, completeness_threshold=0.8)
        world_state = WorldState(original_query="Test")
        world_state.answer_completeness = 0.9
        world_state.add_belief(
            Belief(id="b1", content="Test", status=BeliefStatus.CONFIRMED)
        )

        assessment = checker.check(world_state)

        assert assessment.is_sufficient is True

    def test_check_critical_uncertainty(self, mock_llm):
        checker = SufficiencyChecker(mock_llm)
        world_state = WorldState(original_query="Test")
        world_state.answer_completeness = 0.5
        world_state.add_uncertainty(
            Uncertainty(
                id="u1",
                description="Critical missing data",
                uncertainty_type=UncertaintyType.MISSING_DATA,
                priority=0.9,
            )
        )

        assessment = checker.check(world_state)

        assert assessment.is_sufficient is False
        assert assessment.suggested_action == "traverse"

    def test_check_counterfactual_uncertainty(self, mock_llm):
        checker = SufficiencyChecker(mock_llm)
        world_state = WorldState(original_query="What if VIX spikes?")
        world_state.answer_completeness = 0.5
        world_state.add_uncertainty(
            Uncertainty(
                id="u1",
                description="What if VIX doubles?",
                uncertainty_type=UncertaintyType.COUNTERFACTUAL,
                priority=0.9,
            )
        )

        assessment = checker.check(world_state)

        assert assessment.suggested_action == "simulate"

    def test_estimate_additional_value(self, mock_llm):
        checker = SufficiencyChecker(mock_llm)
        world_state = WorldState(original_query="Test")
        world_state.answer_completeness = 0.5

        # Traverse value with uncertainties
        world_state.add_uncertainty(
            Uncertainty(id="u1", description="Test", uncertainty_type=UncertaintyType.MISSING_DATA)
        )
        value = checker.estimate_additional_value(world_state, "traverse")
        assert value > 0.5

        # Simulate value with counterfactual
        world_state.add_uncertainty(
            Uncertainty(id="u2", description="What if?", uncertainty_type=UncertaintyType.COUNTERFACTUAL)
        )
        value = checker.estimate_additional_value(world_state, "simulate")
        assert value > 0.5

    def test_should_terminate(self, mock_llm):
        checker = SufficiencyChecker(mock_llm)

        # Should not terminate initially with critical uncertainty
        world_state = WorldState(original_query="Test")
        world_state.add_uncertainty(
            Uncertainty(id="u1", description="Test", uncertainty_type=UncertaintyType.MISSING_DATA, priority=0.9)
        )
        assert not checker.should_terminate(world_state)

        # Should terminate with sufficient evidence
        world_state.answer_completeness = 0.9
        world_state.add_belief(Belief(id="b1", content="Test", status=BeliefStatus.CONFIRMED))
        assert checker.should_terminate(world_state)
