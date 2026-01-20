"""NetworkX-based graph store for prototyping.

Provides an in-memory graph store using NetworkX for rapid development
and testing before moving to production Neo4j.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from datetime import datetime
from typing import Any

import networkx as nx

from src.graph.graph_interface import GraphStore
from src.schema.ontology import DomainOntology, Entity, Relationship


class NetworkXStore(GraphStore):
    """In-memory graph store using NetworkX.

    Suitable for prototyping and small-scale testing. All data is stored
    in memory and not persisted across sessions.
    """

    def __init__(self) -> None:
        """Initialize an empty NetworkX graph."""
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self.ontology: DomainOntology | None = None
        self._entities: dict[str, Entity] = {}
        self._relationships: dict[str, Relationship] = {}

    def initialize(self, ontology: DomainOntology) -> None:
        """Initialize the graph store with a domain ontology."""
        self.ontology = ontology
        self.clear()

    def add_entity(self, entity: Entity) -> str:
        """Add an entity to the graph."""
        # Validate against ontology if available
        if self.ontology:
            errors = self.ontology.validate_entity(entity)
            if errors:
                raise ValueError(f"Entity validation failed: {errors}")

        # Store entity
        self._entities[entity.id] = entity

        # Add node to NetworkX graph
        self.graph.add_node(
            entity.id,
            node_type=entity.node_type,
            **entity.properties,
            _source_document=entity.source_document,
            _source_chunk_id=entity.source_chunk_id,
            _extraction_confidence=entity.extraction_confidence,
        )

        return entity.id

    def add_relationship(self, relationship: Relationship) -> str:
        """Add a relationship between entities."""
        # Check source and target exist
        if relationship.source_id not in self._entities:
            raise ValueError(f"Source entity not found: {relationship.source_id}")
        if relationship.target_id not in self._entities:
            raise ValueError(f"Target entity not found: {relationship.target_id}")

        # Validate against ontology if available
        if self.ontology:
            errors = self.ontology.validate_relationship(relationship)
            if errors:
                raise ValueError(f"Relationship validation failed: {errors}")

        # Store relationship
        self._relationships[relationship.id] = relationship

        # Add edge to NetworkX graph
        self.graph.add_edge(
            relationship.source_id,
            relationship.target_id,
            key=relationship.id,
            edge_type=relationship.edge_type,
            **relationship.properties,
            _source_document=relationship.source_document,
            _source_chunk_id=relationship.source_chunk_id,
            _extraction_confidence=relationship.extraction_confidence,
        )

        return relationship.id

    def get_entity(self, entity_id: str) -> Entity | None:
        """Retrieve an entity by ID."""
        return self._entities.get(entity_id)

    def get_relationship(self, relationship_id: str) -> Relationship | None:
        """Retrieve a relationship by ID."""
        return self._relationships.get(relationship_id)

    def find_entities_by_type(self, node_type: str) -> list[Entity]:
        """Find all entities of a given type."""
        return [e for e in self._entities.values() if e.node_type == node_type]

    def find_entities_by_property(
        self, property_name: str, property_value: Any
    ) -> list[Entity]:
        """Find entities by a property value."""
        results = []
        for entity in self._entities.values():
            if property_name in entity.properties:
                if entity.properties[property_name] == property_value:
                    results.append(entity)
        return results

    def get_neighbors(
        self,
        entity_id: str,
        edge_types: list[str] | None = None,
        direction: str = "both",
    ) -> list[tuple[Entity, Relationship]]:
        """Get neighboring entities connected by relationships."""
        if entity_id not in self._entities:
            return []

        results = []

        # Get outgoing edges
        if direction in ("outgoing", "both"):
            for _, target_id, key in self.graph.out_edges(entity_id, keys=True):
                rel = self._relationships.get(key)
                if rel is None:
                    continue
                if edge_types and rel.edge_type not in edge_types:
                    continue
                target = self._entities.get(target_id)
                if target:
                    results.append((target, rel))

        # Get incoming edges
        if direction in ("incoming", "both"):
            for source_id, _, key in self.graph.in_edges(entity_id, keys=True):
                rel = self._relationships.get(key)
                if rel is None:
                    continue
                if edge_types and rel.edge_type not in edge_types:
                    continue
                source = self._entities.get(source_id)
                if source:
                    results.append((source, rel))

        return results

    def get_path(
        self,
        source_id: str,
        target_id: str,
        max_hops: int = 3,
    ) -> list[tuple[Entity, Relationship]] | None:
        """Find a path between two entities."""
        if source_id not in self._entities or target_id not in self._entities:
            return None

        try:
            # Find shortest path
            path_nodes = nx.shortest_path(
                self.graph, source_id, target_id, weight=None
            )

            if len(path_nodes) - 1 > max_hops:
                return None

            # Build result with entities and relationships
            result = []
            for i in range(len(path_nodes) - 1):
                current_id = path_nodes[i]
                next_id = path_nodes[i + 1]

                # Get edge between nodes
                edge_data = self.graph.get_edge_data(current_id, next_id)
                if edge_data:
                    # Get first edge (there might be multiple)
                    first_key = next(iter(edge_data.keys()))
                    rel = self._relationships.get(first_key)
                    entity = self._entities.get(next_id)

                    if rel and entity:
                        result.append((entity, rel))

            return result

        except nx.NetworkXNoPath:
            return None

    def execute_query(self, query: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Execute a simple query language for NetworkX.

        Supports basic patterns like:
        - MATCH (n:Type) RETURN n
        - MATCH (n:Type {prop: value}) RETURN n
        - MATCH (n)-[r:TYPE]->(m) RETURN n, r, m
        """
        params = params or {}
        query = query.strip()

        results: list[dict[str, Any]] = []

        # Pattern: Find nodes by type
        type_match = re.search(r"MATCH\s*\(\s*(\w+)\s*:\s*(\w+)\s*\)", query)
        if type_match:
            var_name = type_match.group(1)
            node_type = type_match.group(2)

            entities = self.find_entities_by_type(node_type)
            for entity in entities:
                results.append({var_name: entity.to_dict()})

            return results

        # Pattern: Find all nodes
        all_match = re.search(r"MATCH\s*\(\s*(\w+)\s*\)", query)
        if all_match:
            var_name = all_match.group(1)
            for entity in self._entities.values():
                results.append({var_name: entity.to_dict()})
            return results

        return results

    def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity and its relationships."""
        if entity_id not in self._entities:
            return False

        # Remove associated relationships
        rels_to_remove = [
            rel_id
            for rel_id, rel in self._relationships.items()
            if rel.source_id == entity_id or rel.target_id == entity_id
        ]

        for rel_id in rels_to_remove:
            del self._relationships[rel_id]

        # Remove from NetworkX
        self.graph.remove_node(entity_id)

        # Remove from entities dict
        del self._entities[entity_id]

        return True

    def delete_relationship(self, relationship_id: str) -> bool:
        """Delete a relationship."""
        rel = self._relationships.get(relationship_id)
        if rel is None:
            return False

        # Remove from NetworkX
        try:
            self.graph.remove_edge(rel.source_id, rel.target_id, key=relationship_id)
        except nx.NetworkXError:
            pass

        # Remove from dict
        del self._relationships[relationship_id]

        return True

    def clear(self) -> None:
        """Remove all data from the graph."""
        self.graph.clear()
        self._entities.clear()
        self._relationships.clear()

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about the graph."""
        node_types: dict[str, int] = {}
        for entity in self._entities.values():
            node_types[entity.node_type] = node_types.get(entity.node_type, 0) + 1

        edge_types: dict[str, int] = {}
        for rel in self._relationships.values():
            edge_types[rel.edge_type] = edge_types.get(rel.edge_type, 0) + 1

        return {
            "num_entities": len(self._entities),
            "num_relationships": len(self._relationships),
            "node_types": node_types,
            "edge_types": edge_types,
            "is_connected": nx.is_weakly_connected(self.graph)
            if len(self.graph) > 0
            else True,
            "num_components": nx.number_weakly_connected_components(self.graph)
            if len(self.graph) > 0
            else 0,
        }

    def update_entity(self, entity_id: str, properties: dict[str, Any]) -> Entity | None:
        """Update an entity's properties."""
        entity = self._entities.get(entity_id)
        if entity is None:
            return None

        # Update properties
        entity.properties.update(properties)
        entity.updated_at = datetime.utcnow()

        # Update NetworkX node
        self.graph.nodes[entity_id].update(properties)

        return entity

    def update_relationship(
        self, relationship_id: str, properties: dict[str, Any]
    ) -> Relationship | None:
        """Update a relationship's properties."""
        rel = self._relationships.get(relationship_id)
        if rel is None:
            return None

        # Update properties
        rel.properties.update(properties)
        rel.updated_at = datetime.utcnow()

        # Update NetworkX edge
        edge_data = self.graph.get_edge_data(rel.source_id, rel.target_id, key=relationship_id)
        if edge_data:
            edge_data.update(properties)

        return rel

    def bulk_add_entities(self, entities: list[Entity]) -> list[str]:
        """Add multiple entities efficiently."""
        ids = []
        for entity in entities:
            try:
                entity_id = self.add_entity(entity)
                ids.append(entity_id)
            except ValueError:
                pass  # Skip invalid entities
        return ids

    def bulk_add_relationships(self, relationships: list[Relationship]) -> list[str]:
        """Add multiple relationships efficiently."""
        ids = []
        for rel in relationships:
            try:
                rel_id = self.add_relationship(rel)
                ids.append(rel_id)
            except ValueError:
                pass  # Skip invalid relationships
        return ids

    def search_entities(
        self,
        query: str,
        node_types: list[str] | None = None,
        limit: int = 10,
    ) -> list[Entity]:
        """Search entities by text query."""
        query_lower = query.lower()
        results = []

        for entity in self._entities.values():
            if node_types and entity.node_type not in node_types:
                continue

            # Search in properties
            found = False
            for value in entity.properties.values():
                if isinstance(value, str) and query_lower in value.lower():
                    found = True
                    break

            if found:
                results.append(entity)
                if len(results) >= limit:
                    break

        return results

    def get_all_entities(self) -> Iterator[Entity]:
        """Iterate over all entities."""
        yield from self._entities.values()

    def get_all_relationships(self) -> Iterator[Relationship]:
        """Iterate over all relationships."""
        yield from self._relationships.values()

    def export_to_dict(self) -> dict[str, Any]:
        """Export graph to dictionary format."""
        return {
            "entities": [e.model_dump() for e in self._entities.values()],
            "relationships": [r.model_dump() for r in self._relationships.values()],
            "ontology": self.ontology.model_dump() if self.ontology else None,
        }

    def import_from_dict(self, data: dict[str, Any]) -> None:
        """Import graph from dictionary format."""
        self.clear()

        if data.get("ontology"):
            self.ontology = DomainOntology.model_validate(data["ontology"])

        for entity_data in data.get("entities", []):
            entity = Entity.model_validate(entity_data)
            self._entities[entity.id] = entity
            self.graph.add_node(
                entity.id,
                node_type=entity.node_type,
                **entity.properties,
            )

        for rel_data in data.get("relationships", []):
            rel = Relationship.model_validate(rel_data)
            self._relationships[rel.id] = rel
            self.graph.add_edge(
                rel.source_id,
                rel.target_id,
                key=rel.id,
                edge_type=rel.edge_type,
                **rel.properties,
            )
