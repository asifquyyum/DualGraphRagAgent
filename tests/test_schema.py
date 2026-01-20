"""Tests for schema models (ontology and world state)."""

import pytest
from datetime import datetime

from src.schema.ontology import (
    DomainOntology,
    EdgeSchema,
    Entity,
    NodeSchema,
    PropertySchema,
    PropertyType,
    Relationship,
    create_quant_finance_ontology,
)
from src.schema.world_state import (
    Belief,
    BeliefStatus,
    EvidenceSource,
    Uncertainty,
    UncertaintyType,
    WorldState,
)


class TestPropertySchema:
    """Tests for PropertySchema."""

    def test_create_property(self):
        prop = PropertySchema(
            name="test_prop",
            property_type=PropertyType.STRING,
            required=True,
            description="A test property",
        )
        assert prop.name == "test_prop"
        assert prop.property_type == PropertyType.STRING
        assert prop.required is True

    def test_default_values(self):
        prop = PropertySchema(name="prop", property_type=PropertyType.INTEGER)
        assert prop.required is False
        assert prop.default is None


class TestNodeSchema:
    """Tests for NodeSchema."""

    def test_create_node_schema(self):
        schema = NodeSchema(
            name="Instrument",
            description="A financial instrument",
            properties=[
                PropertySchema(name="symbol", property_type=PropertyType.STRING, required=True),
                PropertySchema(name="price", property_type=PropertyType.FLOAT),
            ],
        )
        assert schema.name == "Instrument"
        assert len(schema.properties) == 2

    def test_get_required_properties(self):
        schema = NodeSchema(
            name="Test",
            properties=[
                PropertySchema(name="required_prop", property_type=PropertyType.STRING, required=True),
                PropertySchema(name="optional_prop", property_type=PropertyType.STRING),
            ],
        )
        required = schema.get_required_properties()
        assert required == ["required_prop"]

    def test_validate_properties_missing_required(self):
        schema = NodeSchema(
            name="Test",
            properties=[
                PropertySchema(name="name", property_type=PropertyType.STRING, required=True),
            ],
        )
        errors = schema.validate_properties({})
        assert len(errors) == 1
        assert "Missing required property: name" in errors[0]

    def test_validate_properties_unknown_property(self):
        schema = NodeSchema(
            name="Test",
            properties=[
                PropertySchema(name="name", property_type=PropertyType.STRING),
            ],
        )
        errors = schema.validate_properties({"name": "test", "unknown": "value"})
        assert any("Unknown property: unknown" in e for e in errors)


class TestEdgeSchema:
    """Tests for EdgeSchema."""

    def test_create_edge_schema(self):
        schema = EdgeSchema(
            name="TRADES",
            description="Trading relationship",
            source_types=["Strategy"],
            target_types=["Instrument"],
        )
        assert schema.name == "TRADES"
        assert schema.directed is True

    def test_is_valid_connection(self):
        schema = EdgeSchema(
            name="TRADES",
            source_types=["Strategy"],
            target_types=["Instrument"],
        )
        assert schema.is_valid_connection("Strategy", "Instrument") is True
        assert schema.is_valid_connection("Instrument", "Strategy") is False
        assert schema.is_valid_connection("Unknown", "Instrument") is False


class TestEntity:
    """Tests for Entity."""

    def test_create_entity(self):
        entity = Entity(
            id="ent_001",
            node_type="Instrument",
            properties={"symbol": "SPX", "name": "S&P 500"},
        )
        assert entity.id == "ent_001"
        assert entity.node_type == "Instrument"
        assert entity.properties["symbol"] == "SPX"

    def test_entity_to_dict(self):
        entity = Entity(
            id="ent_001",
            node_type="Instrument",
            properties={"symbol": "SPX"},
            source_document="doc_001",
            extraction_confidence=0.95,
        )
        d = entity.to_dict()
        assert d["id"] == "ent_001"
        assert d["node_type"] == "Instrument"
        assert d["symbol"] == "SPX"
        assert d["_source_document"] == "doc_001"


class TestRelationship:
    """Tests for Relationship."""

    def test_create_relationship(self):
        rel = Relationship(
            id="rel_001",
            edge_type="TRADES",
            source_id="ent_001",
            target_id="ent_002",
            properties={"direction": "long"},
        )
        assert rel.id == "rel_001"
        assert rel.edge_type == "TRADES"
        assert rel.source_id == "ent_001"
        assert rel.target_id == "ent_002"


class TestDomainOntology:
    """Tests for DomainOntology."""

    def test_create_ontology(self):
        ontology = DomainOntology(
            name="TestOntology",
            description="A test ontology",
        )
        assert ontology.name == "TestOntology"
        assert len(ontology.node_schemas) == 0

    def test_add_node_schema(self):
        ontology = DomainOntology(name="Test")
        schema = NodeSchema(name="Entity", description="Generic entity")
        ontology.add_node_schema(schema)
        assert "Entity" in ontology.node_schemas

    def test_add_edge_schema(self):
        ontology = DomainOntology(name="Test")
        schema = EdgeSchema(
            name="RELATES_TO",
            source_types=["Entity"],
            target_types=["Entity"],
        )
        ontology.add_edge_schema(schema)
        assert "RELATES_TO" in ontology.edge_schemas

    def test_get_canonical_term(self):
        ontology = DomainOntology(name="Test")
        ontology.domain_terms = {"vol": "volatility", "iv": "implied_volatility"}
        assert ontology.get_canonical_term("vol") == "volatility"
        assert ontology.get_canonical_term("unknown") == "unknown"

    def test_validate_entity(self):
        ontology = DomainOntology(name="Test")
        ontology.add_node_schema(
            NodeSchema(
                name="Instrument",
                properties=[
                    PropertySchema(name="symbol", property_type=PropertyType.STRING, required=True),
                ],
            )
        )

        valid_entity = Entity(
            id="ent_001",
            node_type="Instrument",
            properties={"symbol": "SPX"},
        )
        assert ontology.validate_entity(valid_entity) == []

        invalid_entity = Entity(
            id="ent_002",
            node_type="Unknown",
            properties={},
        )
        errors = ontology.validate_entity(invalid_entity)
        assert len(errors) > 0

    def test_to_cypher_schema(self):
        ontology = create_quant_finance_ontology()
        cypher = ontology.to_cypher_schema()
        assert "// Graph Schema" in cypher
        assert "Instrument" in cypher


class TestQuantFinanceOntology:
    """Tests for the pre-built quant finance ontology."""

    def test_create_ontology(self):
        ontology = create_quant_finance_ontology()
        assert ontology.name == "QuantFinanceOntology"

    def test_has_expected_node_types(self):
        ontology = create_quant_finance_ontology()
        expected_types = ["Instrument", "Strategy", "MarketCondition", "RiskFactor", "Event", "Concept"]
        for node_type in expected_types:
            assert node_type in ontology.node_schemas

    def test_has_expected_edge_types(self):
        ontology = create_quant_finance_ontology()
        expected_types = ["TRADES", "HEDGES", "PERFORMS_IN", "AFFECTED_BY", "INDICATES", "RELATED_TO"]
        for edge_type in expected_types:
            assert edge_type in ontology.edge_schemas

    def test_has_domain_terms(self):
        ontology = create_quant_finance_ontology()
        assert "vol" in ontology.domain_terms
        assert "vix" in ontology.domain_terms
        assert ontology.domain_terms["vol"] == "volatility"


class TestBelief:
    """Tests for Belief."""

    def test_create_belief(self):
        belief = Belief(
            id="belief_001",
            content="VIX indicates market fear",
            status=BeliefStatus.HYPOTHETICAL,
            confidence=0.5,
        )
        assert belief.id == "belief_001"
        assert belief.status == BeliefStatus.HYPOTHETICAL

    def test_add_supporting_evidence(self):
        belief = Belief(
            id="belief_001",
            content="Test belief",
            status=BeliefStatus.HYPOTHETICAL,
            confidence=0.5,
        )
        evidence = EvidenceSource(
            source_type="document",
            source_id="doc_001",
            confidence=0.9,
        )
        belief.add_supporting_evidence(evidence)
        assert len(belief.supporting_evidence) == 1
        assert belief.status == BeliefStatus.CONFIRMED

    def test_add_contradicting_evidence(self):
        belief = Belief(
            id="belief_001",
            content="Test belief",
            status=BeliefStatus.CONFIRMED,
            confidence=0.8,
        )
        # Add strong contradicting evidence
        for _ in range(3):
            belief.add_contradicting_evidence(
                EvidenceSource(source_type="document", source_id="doc", confidence=0.9)
            )
        assert belief.status == BeliefStatus.CONTRADICTED


class TestUncertainty:
    """Tests for Uncertainty."""

    def test_create_uncertainty(self):
        uncertainty = Uncertainty(
            id="unc_001",
            description="Missing VIX data for 2020",
            uncertainty_type=UncertaintyType.MISSING_DATA,
            priority=0.8,
        )
        assert uncertainty.id == "unc_001"
        assert uncertainty.uncertainty_type == UncertaintyType.MISSING_DATA

    def test_should_explore(self):
        uncertainty = Uncertainty(
            id="unc_001",
            description="Test",
            uncertainty_type=UncertaintyType.AMBIGUOUS,
            max_attempts=3,
        )
        assert uncertainty.should_explore() is True
        uncertainty.record_attempt()
        uncertainty.record_attempt()
        uncertainty.record_attempt()
        assert uncertainty.should_explore() is False


class TestWorldState:
    """Tests for WorldState."""

    def test_create_world_state(self):
        state = WorldState(original_query="What happens when VIX spikes?")
        assert state.original_query == "What happens when VIX spikes?"
        assert state.iteration_count == 0

    def test_add_belief(self):
        state = WorldState()
        belief = Belief(id="b1", content="Test belief")
        state.add_belief(belief)
        assert "b1" in state.beliefs

    def test_add_uncertainty(self):
        state = WorldState()
        uncertainty = Uncertainty(
            id="u1",
            description="Test uncertainty",
            uncertainty_type=UncertaintyType.MISSING_DATA,
        )
        state.add_uncertainty(uncertainty)
        assert "u1" in state.uncertainties

    def test_mark_node_explored(self):
        state = WorldState()
        state.mark_node_explored("node_001")
        assert state.is_node_explored("node_001")
        assert not state.is_node_explored("node_002")

    def test_get_confirmed_beliefs(self):
        state = WorldState()
        state.add_belief(Belief(id="b1", content="Confirmed", status=BeliefStatus.CONFIRMED))
        state.add_belief(Belief(id="b2", content="Hypothetical", status=BeliefStatus.HYPOTHETICAL))
        confirmed = state.get_confirmed_beliefs()
        assert len(confirmed) == 1
        assert confirmed[0].id == "b1"

    def test_has_sufficient_evidence(self):
        state = WorldState()
        state.answer_completeness = 0.9
        state.add_belief(Belief(id="b1", content="Test", status=BeliefStatus.CONFIRMED))
        assert state.has_sufficient_evidence()

        state.answer_completeness = 0.5
        assert not state.has_sufficient_evidence()

    def test_should_continue(self):
        state = WorldState(max_iterations=10)
        state.add_uncertainty(
            Uncertainty(id="u1", description="Test", uncertainty_type=UncertaintyType.MISSING_DATA)
        )
        assert state.should_continue()

        state.iteration_count = 10
        assert not state.should_continue()

    def test_get_routing_decision(self):
        state = WorldState()

        # Default should be synthesize when no uncertainties
        state.answer_completeness = 0.9
        state.add_belief(Belief(id="b1", content="Test", status=BeliefStatus.CONFIRMED))
        assert state.get_routing_decision() == "synthesize"

        # With counterfactual uncertainty, should simulate
        state.answer_completeness = 0.5
        state.add_uncertainty(
            Uncertainty(
                id="u1",
                description="What if VIX spikes?",
                uncertainty_type=UncertaintyType.COUNTERFACTUAL,
            )
        )
        assert state.get_routing_decision() == "simulate"

    def test_to_context_string(self):
        state = WorldState(original_query="Test query")
        state.add_belief(Belief(id="b1", content="Test belief", status=BeliefStatus.CONFIRMED, confidence=0.9))
        context = state.to_context_string()
        assert "Test query" in context
        assert "Test belief" in context

    def test_to_summary_dict(self):
        state = WorldState(original_query="Test")
        summary = state.to_summary_dict()
        assert "query" in summary
        assert "iteration" in summary
        assert "routing_decision" in summary
