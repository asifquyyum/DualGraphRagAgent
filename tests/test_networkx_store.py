"""Tests for NetworkX graph store."""

import pytest

from src.graph.networkx_store import NetworkXStore
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


@pytest.fixture
def ontology() -> DomainOntology:
    """Create a test ontology."""
    return create_quant_finance_ontology()


@pytest.fixture
def store(ontology: DomainOntology) -> NetworkXStore:
    """Create a NetworkX store with test ontology."""
    store = NetworkXStore()
    store.initialize(ontology)
    return store


@pytest.fixture
def sample_entities() -> list[Entity]:
    """Create sample entities for testing."""
    return [
        Entity(
            id="ent_001",
            node_type="Instrument",
            properties={"symbol": "SPX", "name": "S&P 500 Index", "asset_class": "equity_index"},
        ),
        Entity(
            id="ent_002",
            node_type="Instrument",
            properties={"symbol": "VIX", "name": "CBOE Volatility Index", "asset_class": "volatility_index"},
        ),
        Entity(
            id="ent_003",
            node_type="Strategy",
            properties={"name": "Long Straddle", "strategy_type": "volatility", "risk_profile": "long_vol"},
        ),
        Entity(
            id="ent_004",
            node_type="MarketCondition",
            properties={"name": "High Volatility Regime", "vix_range": "25-40"},
        ),
    ]


@pytest.fixture
def sample_relationships() -> list[Relationship]:
    """Create sample relationships for testing."""
    return [
        Relationship(
            id="rel_001",
            edge_type="TRADES",
            source_id="ent_003",
            target_id="ent_001",
            properties={"direction": "long"},
        ),
        Relationship(
            id="rel_002",
            edge_type="PERFORMS_IN",
            source_id="ent_003",
            target_id="ent_004",
            properties={"expected_pnl": "positive"},
        ),
    ]


class TestNetworkXStoreBasics:
    """Test basic store operations."""

    def test_initialize(self, store: NetworkXStore, ontology: DomainOntology):
        assert store.ontology == ontology
        assert len(store.graph) == 0

    def test_add_entity(self, store: NetworkXStore, sample_entities: list[Entity]):
        entity = sample_entities[0]
        entity_id = store.add_entity(entity)
        assert entity_id == "ent_001"
        assert "ent_001" in store.graph.nodes

    def test_add_entity_invalid_type(self, store: NetworkXStore):
        entity = Entity(id="bad", node_type="InvalidType", properties={})
        with pytest.raises(ValueError, match="validation failed"):
            store.add_entity(entity)

    def test_get_entity(self, store: NetworkXStore, sample_entities: list[Entity]):
        store.add_entity(sample_entities[0])
        retrieved = store.get_entity("ent_001")
        assert retrieved is not None
        assert retrieved.properties["symbol"] == "SPX"

    def test_get_entity_not_found(self, store: NetworkXStore):
        assert store.get_entity("nonexistent") is None

    def test_add_relationship(
        self, store: NetworkXStore, sample_entities: list[Entity], sample_relationships: list[Relationship]
    ):
        # Add entities first
        store.add_entity(sample_entities[0])  # ent_001
        store.add_entity(sample_entities[2])  # ent_003

        rel = sample_relationships[0]
        rel_id = store.add_relationship(rel)
        assert rel_id == "rel_001"

    def test_add_relationship_missing_source(self, store: NetworkXStore, sample_relationships: list[Relationship]):
        with pytest.raises(ValueError, match="Source entity not found"):
            store.add_relationship(sample_relationships[0])

    def test_get_relationship(
        self, store: NetworkXStore, sample_entities: list[Entity], sample_relationships: list[Relationship]
    ):
        store.add_entity(sample_entities[0])
        store.add_entity(sample_entities[2])
        store.add_relationship(sample_relationships[0])

        retrieved = store.get_relationship("rel_001")
        assert retrieved is not None
        assert retrieved.edge_type == "TRADES"


class TestNetworkXStoreQueries:
    """Test query operations."""

    def test_find_entities_by_type(self, store: NetworkXStore, sample_entities: list[Entity]):
        for entity in sample_entities:
            store.add_entity(entity)

        instruments = store.find_entities_by_type("Instrument")
        assert len(instruments) == 2

        strategies = store.find_entities_by_type("Strategy")
        assert len(strategies) == 1

    def test_find_entities_by_property(self, store: NetworkXStore, sample_entities: list[Entity]):
        for entity in sample_entities:
            store.add_entity(entity)

        results = store.find_entities_by_property("symbol", "SPX")
        assert len(results) == 1
        assert results[0].id == "ent_001"

    def test_get_neighbors(
        self, store: NetworkXStore, sample_entities: list[Entity], sample_relationships: list[Relationship]
    ):
        # Set up graph
        for entity in sample_entities:
            store.add_entity(entity)
        for rel in sample_relationships:
            store.add_relationship(rel)

        # Get neighbors of strategy
        neighbors = store.get_neighbors("ent_003", direction="outgoing")
        assert len(neighbors) == 2

        # Get neighbors with edge type filter
        neighbors = store.get_neighbors("ent_003", edge_types=["TRADES"], direction="outgoing")
        assert len(neighbors) == 1
        assert neighbors[0][1].edge_type == "TRADES"

    def test_get_path(
        self, store: NetworkXStore, sample_entities: list[Entity], sample_relationships: list[Relationship]
    ):
        for entity in sample_entities:
            store.add_entity(entity)
        for rel in sample_relationships:
            store.add_relationship(rel)

        path = store.get_path("ent_003", "ent_001")
        assert path is not None
        assert len(path) == 1

    def test_get_path_not_found(self, store: NetworkXStore, sample_entities: list[Entity]):
        store.add_entity(sample_entities[0])
        store.add_entity(sample_entities[1])  # No relationship between them

        path = store.get_path("ent_001", "ent_002")
        assert path is None

    def test_search_entities(self, store: NetworkXStore, sample_entities: list[Entity]):
        for entity in sample_entities:
            store.add_entity(entity)

        results = store.search_entities("S&P")
        assert len(results) >= 1
        assert any(e.properties.get("symbol") == "SPX" for e in results)

        results = store.search_entities("volatility", node_types=["Strategy"])
        assert len(results) >= 1


class TestNetworkXStoreModifications:
    """Test modification operations."""

    def test_update_entity(self, store: NetworkXStore, sample_entities: list[Entity]):
        store.add_entity(sample_entities[0])

        updated = store.update_entity("ent_001", {"price": 4500.0})
        assert updated is not None
        assert updated.properties["price"] == 4500.0

    def test_update_entity_not_found(self, store: NetworkXStore):
        result = store.update_entity("nonexistent", {"key": "value"})
        assert result is None

    def test_delete_entity(
        self, store: NetworkXStore, sample_entities: list[Entity], sample_relationships: list[Relationship]
    ):
        for entity in sample_entities:
            store.add_entity(entity)
        for rel in sample_relationships:
            store.add_relationship(rel)

        # Delete entity should also remove relationships
        result = store.delete_entity("ent_003")
        assert result is True
        assert store.get_entity("ent_003") is None
        assert store.get_relationship("rel_001") is None

    def test_delete_relationship(
        self, store: NetworkXStore, sample_entities: list[Entity], sample_relationships: list[Relationship]
    ):
        store.add_entity(sample_entities[0])
        store.add_entity(sample_entities[2])
        store.add_relationship(sample_relationships[0])

        result = store.delete_relationship("rel_001")
        assert result is True
        assert store.get_relationship("rel_001") is None

    def test_clear(self, store: NetworkXStore, sample_entities: list[Entity]):
        for entity in sample_entities:
            store.add_entity(entity)

        store.clear()
        assert len(store.graph) == 0
        assert len(store._entities) == 0


class TestNetworkXStoreBulk:
    """Test bulk operations."""

    def test_bulk_add_entities(self, store: NetworkXStore, sample_entities: list[Entity]):
        ids = store.bulk_add_entities(sample_entities)
        assert len(ids) == 4

    def test_bulk_add_relationships(
        self, store: NetworkXStore, sample_entities: list[Entity], sample_relationships: list[Relationship]
    ):
        store.bulk_add_entities(sample_entities)
        ids = store.bulk_add_relationships(sample_relationships)
        assert len(ids) == 2


class TestNetworkXStoreStatistics:
    """Test statistics and export."""

    def test_get_statistics(
        self, store: NetworkXStore, sample_entities: list[Entity], sample_relationships: list[Relationship]
    ):
        store.bulk_add_entities(sample_entities)
        store.bulk_add_relationships(sample_relationships)

        stats = store.get_statistics()
        assert stats["num_entities"] == 4
        assert stats["num_relationships"] == 2
        assert "Instrument" in stats["node_types"]
        assert stats["node_types"]["Instrument"] == 2

    def test_export_import(
        self, store: NetworkXStore, sample_entities: list[Entity], sample_relationships: list[Relationship]
    ):
        store.bulk_add_entities(sample_entities)
        store.bulk_add_relationships(sample_relationships)

        # Export
        data = store.export_to_dict()
        assert len(data["entities"]) == 4
        assert len(data["relationships"]) == 2

        # Import to new store
        new_store = NetworkXStore()
        new_store.import_from_dict(data)
        assert new_store.get_entity("ent_001") is not None

    def test_get_subgraph(
        self, store: NetworkXStore, sample_entities: list[Entity], sample_relationships: list[Relationship]
    ):
        store.bulk_add_entities(sample_entities)
        store.bulk_add_relationships(sample_relationships)

        entities, relationships = store.get_subgraph(["ent_003"], include_neighbors=True, max_depth=1)
        assert len(entities) >= 1
        assert any(e.id == "ent_003" for e in entities)


class TestNetworkXStoreQuery:
    """Test query execution."""

    def test_execute_query_by_type(self, store: NetworkXStore, sample_entities: list[Entity]):
        store.bulk_add_entities(sample_entities)

        results = store.execute_query("MATCH (n:Instrument) RETURN n")
        assert len(results) == 2

    def test_execute_query_all_nodes(self, store: NetworkXStore, sample_entities: list[Entity]):
        store.bulk_add_entities(sample_entities)

        results = store.execute_query("MATCH (n) RETURN n")
        assert len(results) == 4
