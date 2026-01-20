"""Tests for the query loop agents."""

import pytest
from unittest.mock import MagicMock, patch

from src.agents.semantic_aligner import SemanticAligner, AlignedQuery, AlignedTerm
from src.agents.cypher_agent import CypherAgent, CypherGenerationResult
from src.agents.traversal_agent import TraversalAgent, TraversalResult
from src.agents.synthesis_agent import SynthesisAgent, SynthesizedAnswer
from src.graph.networkx_store import NetworkXStore
from src.schema.ontology import (
    DomainOntology,
    Entity,
    Relationship,
    create_quant_finance_ontology,
)
from src.schema.world_state import (
    Belief,
    BeliefStatus,
    Uncertainty,
    UncertaintyType,
    WorldState,
)


@pytest.fixture
def ontology() -> DomainOntology:
    """Create test ontology."""
    return create_quant_finance_ontology()


@pytest.fixture
def store(ontology: DomainOntology) -> NetworkXStore:
    """Create test graph store with sample data."""
    store = NetworkXStore()
    store.initialize(ontology)

    # Add sample entities
    entities = [
        Entity(
            id="ent_spx",
            node_type="Instrument",
            properties={"symbol": "SPX", "name": "S&P 500 Index", "asset_class": "equity_index"},
        ),
        Entity(
            id="ent_vix",
            node_type="Instrument",
            properties={"symbol": "VIX", "name": "CBOE Volatility Index", "asset_class": "volatility_index"},
        ),
        Entity(
            id="ent_straddle",
            node_type="Strategy",
            properties={"name": "Long Straddle", "strategy_type": "volatility", "risk_profile": "long_vol"},
        ),
        Entity(
            id="ent_high_vol",
            node_type="MarketCondition",
            properties={"name": "High Volatility Regime", "vix_range": "25-40"},
        ),
    ]

    for entity in entities:
        store.add_entity(entity)

    # Add relationships
    relationships = [
        Relationship(
            id="rel_1",
            edge_type="TRADES",
            source_id="ent_straddle",
            target_id="ent_spx",
            properties={"direction": "long"},
        ),
        Relationship(
            id="rel_2",
            edge_type="PERFORMS_IN",
            source_id="ent_straddle",
            target_id="ent_high_vol",
            properties={"expected_pnl": "positive", "win_rate": 0.65},
        ),
    ]

    for rel in relationships:
        store.add_relationship(rel)

    return store


@pytest.fixture
def mock_llm():
    """Create a mock LLM provider."""
    llm = MagicMock()
    llm.generate.return_value = "Mock response"
    llm.generate_structured.return_value = AlignedQuery(
        original_query="test query",
        aligned_terms=[AlignedTerm(original="vol", canonical="volatility", term_type="concept")],
        entity_references=["Instrument", "Strategy"],
        relationship_references=["TRADES"],
        query_intent="retrieval",
    )
    return llm


class TestSemanticAligner:
    """Tests for SemanticAligner."""

    def test_basic_alignment(self, mock_llm, ontology):
        aligner = SemanticAligner(mock_llm, ontology)

        # Test basic alignment (falls back when LLM fails)
        mock_llm.generate_structured.side_effect = ValueError("Mock error")
        result = aligner._basic_alignment("What happens to vol when VIX spikes?")

        assert result.original_query == "What happens to vol when VIX spikes?"
        # Should detect domain terms
        assert any(t.original == "vol" for t in result.aligned_terms)

    def test_expand_query(self, mock_llm, ontology):
        aligner = SemanticAligner(mock_llm, ontology)

        aligned = AlignedQuery(
            original_query="test query",
            aligned_terms=[AlignedTerm(original="vol", canonical="volatility", term_type="concept")],
            entity_references=["Instrument"],
        )

        expanded = aligner.expand_query(aligned)
        assert "test query" in expanded
        assert "volatility" in expanded
        assert "Instrument" in expanded

    def test_generate_cypher_hints(self, mock_llm, ontology):
        aligner = SemanticAligner(mock_llm, ontology)

        aligned = AlignedQuery(
            original_query="test",
            entity_references=["Strategy", "Instrument"],
            relationship_references=["TRADES"],
            property_filters={"symbol": "SPX"},
        )

        hints = aligner.generate_cypher_hints(aligned)
        assert "Strategy" in hints["suggested_labels"]
        assert "TRADES" in hints["suggested_relationships"]


class TestCypherAgent:
    """Tests for CypherAgent."""

    def test_clean_cypher(self, mock_llm, ontology, store):
        agent = CypherAgent(mock_llm, ontology, store)

        # Test cleaning markdown
        raw = "```cypher\nMATCH (n) RETURN n\n```"
        cleaned = agent._clean_cypher(raw)
        assert cleaned == "MATCH (n) RETURN n"

    def test_validate_cypher(self, mock_llm, ontology, store):
        agent = CypherAgent(mock_llm, ontology, store)

        # Valid query
        is_valid, error = agent.validate_cypher("MATCH (n) RETURN n")
        assert is_valid is True
        assert error is None

        # Invalid - missing RETURN
        is_valid, error = agent.validate_cypher("MATCH (n)")
        assert is_valid is False
        assert "RETURN" in error

    def test_generate_cypher_fallback(self, mock_llm, ontology, store):
        """Test that fallback search works when Cypher fails."""
        agent = CypherAgent(mock_llm, ontology, store)

        # Make LLM return invalid Cypher
        mock_llm.generate.return_value = "INVALID QUERY"

        result = agent.generate_cypher("Find instruments")

        # Should use fallback
        assert result.fallback_used is True or len(result.attempts) > 0


class TestTraversalAgent:
    """Tests for TraversalAgent."""

    def test_explore_from_entities(self, mock_llm, ontology, store):
        agent = TraversalAgent(mock_llm, ontology, store)

        world_state = WorldState(original_query="test")

        result = agent.explore_from_entities(
            ["ent_straddle"],
            world_state,
            max_depth=1,
        )

        assert len(result.steps) > 0
        # Should find connected entities (SPX, High Vol regime)
        assert len(result.entities_discovered) > 0

    def test_explore_for_uncertainty(self, mock_llm, ontology, store):
        agent = TraversalAgent(mock_llm, ontology, store)

        world_state = WorldState(original_query="test")
        uncertainty = Uncertainty(
            id="u1",
            description="What is the straddle strategy?",
            uncertainty_type=UncertaintyType.MISSING_DATA,
        )

        result = agent.explore_for_uncertainty(uncertainty, world_state)

        # Should find straddle-related entities
        assert isinstance(result, TraversalResult)

    def test_get_entity_context(self, mock_llm, ontology, store):
        agent = TraversalAgent(mock_llm, ontology, store)

        context = agent.get_entity_context("ent_straddle")

        assert "entity" in context
        assert "neighbors" in context
        assert context["entity"]["id"] == "ent_straddle"


class TestSynthesisAgent:
    """Tests for SynthesisAgent."""

    def test_synthesize_with_beliefs(self, mock_llm):
        agent = SynthesisAgent(mock_llm)

        # Set up world state with beliefs
        world_state = WorldState(original_query="How does straddle perform in high vol?")
        world_state.add_belief(
            Belief(
                id="b1",
                content="Straddle strategy profits from high volatility",
                status=BeliefStatus.CONFIRMED,
                confidence=0.9,
            )
        )
        world_state.answer_completeness = 0.8

        mock_llm.generate.return_value = "The straddle strategy performs well in high volatility conditions."

        result = agent.synthesize(world_state)

        assert isinstance(result, SynthesizedAnswer)
        assert result.answer != ""
        assert result.confidence > 0

    def test_confidence_calculation(self, mock_llm):
        agent = SynthesisAgent(mock_llm)

        world_state = WorldState(original_query="test")

        # No beliefs - should have low confidence
        confirmed_beliefs: list[Belief] = []
        confidence = agent._calculate_confidence(world_state, confirmed_beliefs)
        assert confidence == 0.3

        # With confirmed beliefs - higher confidence
        world_state.answer_completeness = 0.8
        confirmed_beliefs = [
            Belief(id="b1", content="Test", status=BeliefStatus.CONFIRMED, confidence=0.9)
        ]
        confidence = agent._calculate_confidence(world_state, confirmed_beliefs)
        assert confidence > 0.5


class TestWorldState:
    """Tests for WorldState routing logic."""

    def test_routing_to_synthesize(self):
        state = WorldState(original_query="test")
        state.answer_completeness = 0.9
        state.add_belief(
            Belief(id="b1", content="Test", status=BeliefStatus.CONFIRMED, confidence=0.9)
        )

        assert state.get_routing_decision() == "synthesize"

    def test_routing_to_simulate(self):
        state = WorldState(original_query="What if VIX spikes?")
        state.answer_completeness = 0.5
        state.add_uncertainty(
            Uncertainty(
                id="u1",
                description="What if VIX spikes 20%?",
                uncertainty_type=UncertaintyType.COUNTERFACTUAL,
            )
        )

        assert state.get_routing_decision() == "simulate"

    def test_routing_to_traverse(self):
        state = WorldState(original_query="test")
        state.answer_completeness = 0.3
        state.add_uncertainty(
            Uncertainty(
                id="u1",
                description="Missing information about strategy",
                uncertainty_type=UncertaintyType.MISSING_DATA,
            )
        )

        assert state.get_routing_decision() == "traverse"

    def test_max_iterations(self):
        state = WorldState(original_query="test", max_iterations=3)
        state.iteration_count = 3

        # Should route to synthesize after max iterations
        assert state.get_routing_decision() == "synthesize"
        assert not state.should_continue()
