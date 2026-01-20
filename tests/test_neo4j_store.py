"""Tests for Neo4j graph store.

Note: These tests require a running Neo4j instance.
Set NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD environment variables.
"""

import os

import pytest

from src.graph.neo4j_store import Neo4jStore
from src.schema.ontology import (
    Entity,
    Relationship,
    create_quant_finance_ontology,
)
from src.tools.neo4j_tools import Neo4jSettings, Neo4jTools


# Skip all tests if Neo4j is not available
def neo4j_available() -> bool:
    """Check if Neo4j is available for testing."""
    try:
        settings = Neo4jSettings()
        tools = Neo4jTools(settings)
        result = tools.verify_connection()
        tools.close()
        return result
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not neo4j_available(),
    reason="Neo4j not available for testing",
)


@pytest.fixture
def neo4j_settings() -> Neo4jSettings:
    """Create Neo4j settings from environment."""
    return Neo4jSettings()


@pytest.fixture
def neo4j_tools(neo4j_settings: Neo4jSettings) -> Neo4jTools:
    """Create Neo4j tools instance."""
    tools = Neo4jTools(neo4j_settings)
    yield tools
    tools.close()


@pytest.fixture
def store(neo4j_settings: Neo4jSettings) -> Neo4jStore:
    """Create Neo4j store with test ontology."""
    store = Neo4jStore(neo4j_settings)
    ontology = create_quant_finance_ontology()
    store.initialize(ontology)

    # Clean up before test
    store.clear()

    yield store

    # Clean up after test
    store.clear()
    store.close()


@pytest.fixture
def sample_entities() -> list[Entity]:
    """Create sample entities for testing."""
    return [
        Entity(
            id="neo_ent_001",
            node_type="Instrument",
            properties={"symbol": "SPX", "name": "S&P 500 Index"},
        ),
        Entity(
            id="neo_ent_002",
            node_type="Instrument",
            properties={"symbol": "VIX", "name": "CBOE Volatility Index"},
        ),
        Entity(
            id="neo_ent_003",
            node_type="Strategy",
            properties={"name": "Long Straddle", "strategy_type": "volatility"},
        ),
    ]


@pytest.fixture
def sample_relationships(sample_entities: list[Entity]) -> list[Relationship]:
    """Create sample relationships for testing."""
    return [
        Relationship(
            id="neo_rel_001",
            edge_type="TRADES",
            source_id="neo_ent_003",
            target_id="neo_ent_001",
            properties={"direction": "long"},
        ),
    ]


class TestNeo4jTools:
    """Tests for Neo4j tools."""

    def test_verify_connection(self, neo4j_tools: Neo4jTools):
        assert neo4j_tools.verify_connection() is True

    def test_get_schema_info(self, neo4j_tools: Neo4jTools):
        schema = neo4j_tools.get_schema_info()
        assert "labels" in schema
        assert "relationship_types" in schema

    def test_execute_query(self, neo4j_tools: Neo4jTools):
        result = neo4j_tools.execute_read("RETURN 1 AS value")
        assert result.success
        assert result.records[0]["value"] == 1


class TestNeo4jStore:
    """Tests for Neo4j store basic operations."""

    def test_add_entity(self, store: Neo4jStore, sample_entities: list[Entity]):
        entity = sample_entities[0]
        entity_id = store.add_entity(entity)
        assert entity_id == "neo_ent_001"

    def test_get_entity(self, store: Neo4jStore, sample_entities: list[Entity]):
        store.add_entity(sample_entities[0])
        retrieved = store.get_entity("neo_ent_001")
        assert retrieved is not None
        assert retrieved.properties["symbol"] == "SPX"

    def test_get_entity_not_found(self, store: Neo4jStore):
        result = store.get_entity("nonexistent")
        assert result is None

    def test_add_relationship(
        self,
        store: Neo4jStore,
        sample_entities: list[Entity],
        sample_relationships: list[Relationship],
    ):
        # Add entities first
        store.add_entity(sample_entities[0])
        store.add_entity(sample_entities[2])

        rel_id = store.add_relationship(sample_relationships[0])
        assert rel_id == "neo_rel_001"

    def test_get_relationship(
        self,
        store: Neo4jStore,
        sample_entities: list[Entity],
        sample_relationships: list[Relationship],
    ):
        store.add_entity(sample_entities[0])
        store.add_entity(sample_entities[2])
        store.add_relationship(sample_relationships[0])

        retrieved = store.get_relationship("neo_rel_001")
        assert retrieved is not None
        assert retrieved.edge_type == "TRADES"


class TestNeo4jStoreQueries:
    """Tests for Neo4j store query operations."""

    def test_find_entities_by_type(self, store: Neo4jStore, sample_entities: list[Entity]):
        for entity in sample_entities:
            store.add_entity(entity)

        instruments = store.find_entities_by_type("Instrument")
        assert len(instruments) == 2

    def test_find_entities_by_property(self, store: Neo4jStore, sample_entities: list[Entity]):
        for entity in sample_entities:
            store.add_entity(entity)

        results = store.find_entities_by_property("symbol", "SPX")
        assert len(results) == 1

    def test_get_neighbors(
        self,
        store: Neo4jStore,
        sample_entities: list[Entity],
        sample_relationships: list[Relationship],
    ):
        for entity in sample_entities:
            store.add_entity(entity)
        for rel in sample_relationships:
            store.add_relationship(rel)

        neighbors = store.get_neighbors("neo_ent_003", direction="outgoing")
        assert len(neighbors) == 1

    def test_execute_cypher_query(self, store: Neo4jStore, sample_entities: list[Entity]):
        for entity in sample_entities:
            store.add_entity(entity)

        results = store.execute_query("MATCH (n:Instrument) RETURN n.symbol AS symbol")
        assert len(results) == 2
        symbols = {r["symbol"] for r in results}
        assert "SPX" in symbols
        assert "VIX" in symbols


class TestNeo4jStoreModifications:
    """Tests for Neo4j store modification operations."""

    def test_update_entity(self, store: Neo4jStore, sample_entities: list[Entity]):
        store.add_entity(sample_entities[0])

        updated = store.update_entity("neo_ent_001", {"price": 4500.0})
        assert updated is not None
        assert updated.properties["price"] == 4500.0

    def test_delete_entity(
        self,
        store: Neo4jStore,
        sample_entities: list[Entity],
        sample_relationships: list[Relationship],
    ):
        for entity in sample_entities:
            store.add_entity(entity)
        for rel in sample_relationships:
            store.add_relationship(rel)

        result = store.delete_entity("neo_ent_003")
        assert result is True
        assert store.get_entity("neo_ent_003") is None

    def test_clear(self, store: Neo4jStore, sample_entities: list[Entity]):
        for entity in sample_entities:
            store.add_entity(entity)

        store.clear()
        stats = store.get_statistics()
        assert stats["num_entities"] == 0


class TestNeo4jStoreBulk:
    """Tests for Neo4j store bulk operations."""

    def test_bulk_add_entities(self, store: Neo4jStore, sample_entities: list[Entity]):
        ids = store.bulk_add_entities(sample_entities)
        assert len(ids) == 3

    def test_bulk_add_relationships(
        self,
        store: Neo4jStore,
        sample_entities: list[Entity],
        sample_relationships: list[Relationship],
    ):
        store.bulk_add_entities(sample_entities)
        ids = store.bulk_add_relationships(sample_relationships)
        assert len(ids) == 1


class TestNeo4jStoreStatistics:
    """Tests for Neo4j store statistics."""

    def test_get_statistics(
        self,
        store: Neo4jStore,
        sample_entities: list[Entity],
        sample_relationships: list[Relationship],
    ):
        store.bulk_add_entities(sample_entities)
        store.bulk_add_relationships(sample_relationships)

        stats = store.get_statistics()
        assert stats["num_entities"] == 3
        assert stats["num_relationships"] == 1
        assert "Instrument" in stats["node_types"]
