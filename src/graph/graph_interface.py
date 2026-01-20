"""Abstract interface for graph storage backends.

Defines the contract that both NetworkX and Neo4j implementations must follow,
enabling seamless swapping between prototyping and production stores.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.schema.ontology import DomainOntology, Entity, Relationship


class GraphStore(ABC):
    """Abstract base class for graph storage implementations.

    Provides a consistent interface for graph operations across different
    backends (NetworkX for prototyping, Neo4j for production).
    """

    @abstractmethod
    def initialize(self, ontology: DomainOntology) -> None:
        """Initialize the graph store with a domain ontology.

        Args:
            ontology: The domain ontology defining valid node and edge types
        """
        pass

    @abstractmethod
    def add_entity(self, entity: Entity) -> str:
        """Add an entity to the graph.

        Args:
            entity: The entity to add

        Returns:
            The ID of the added entity
        """
        pass

    @abstractmethod
    def add_relationship(self, relationship: Relationship) -> str:
        """Add a relationship between entities.

        Args:
            relationship: The relationship to add

        Returns:
            The ID of the added relationship
        """
        pass

    @abstractmethod
    def get_entity(self, entity_id: str) -> Entity | None:
        """Retrieve an entity by ID.

        Args:
            entity_id: The entity's unique identifier

        Returns:
            The entity if found, None otherwise
        """
        pass

    @abstractmethod
    def get_relationship(self, relationship_id: str) -> Relationship | None:
        """Retrieve a relationship by ID.

        Args:
            relationship_id: The relationship's unique identifier

        Returns:
            The relationship if found, None otherwise
        """
        pass

    @abstractmethod
    def find_entities_by_type(self, node_type: str) -> list[Entity]:
        """Find all entities of a given type.

        Args:
            node_type: The type of entities to find

        Returns:
            List of matching entities
        """
        pass

    @abstractmethod
    def find_entities_by_property(
        self, property_name: str, property_value: Any
    ) -> list[Entity]:
        """Find entities by a property value.

        Args:
            property_name: Name of the property to search
            property_value: Value to match

        Returns:
            List of matching entities
        """
        pass

    @abstractmethod
    def get_neighbors(
        self,
        entity_id: str,
        edge_types: list[str] | None = None,
        direction: str = "both",
    ) -> list[tuple[Entity, Relationship]]:
        """Get neighboring entities connected by relationships.

        Args:
            entity_id: ID of the entity to find neighbors for
            edge_types: Optional filter for relationship types
            direction: 'outgoing', 'incoming', or 'both'

        Returns:
            List of (neighbor_entity, connecting_relationship) tuples
        """
        pass

    @abstractmethod
    def get_path(
        self,
        source_id: str,
        target_id: str,
        max_hops: int = 3,
    ) -> list[tuple[Entity, Relationship]] | None:
        """Find a path between two entities.

        Args:
            source_id: Starting entity ID
            target_id: Target entity ID
            max_hops: Maximum number of hops to search

        Returns:
            List of (entity, relationship) tuples forming the path, or None if no path
        """
        pass

    @abstractmethod
    def execute_query(self, query: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Execute a query in the store's native query language.

        Args:
            query: The query string (Cypher for Neo4j, custom for NetworkX)
            params: Optional query parameters

        Returns:
            List of result dictionaries
        """
        pass

    @abstractmethod
    def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity and its relationships.

        Args:
            entity_id: ID of the entity to delete

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    def delete_relationship(self, relationship_id: str) -> bool:
        """Delete a relationship.

        Args:
            relationship_id: ID of the relationship to delete

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Remove all data from the graph."""
        pass

    @abstractmethod
    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about the graph.

        Returns:
            Dictionary with counts and other metrics
        """
        pass

    @abstractmethod
    def update_entity(self, entity_id: str, properties: dict[str, Any]) -> Entity | None:
        """Update an entity's properties.

        Args:
            entity_id: ID of the entity to update
            properties: New property values to merge

        Returns:
            Updated entity or None if not found
        """
        pass

    @abstractmethod
    def update_relationship(
        self, relationship_id: str, properties: dict[str, Any]
    ) -> Relationship | None:
        """Update a relationship's properties.

        Args:
            relationship_id: ID of the relationship to update
            properties: New property values to merge

        Returns:
            Updated relationship or None if not found
        """
        pass

    @abstractmethod
    def bulk_add_entities(self, entities: list[Entity]) -> list[str]:
        """Add multiple entities efficiently.

        Args:
            entities: List of entities to add

        Returns:
            List of added entity IDs
        """
        pass

    @abstractmethod
    def bulk_add_relationships(self, relationships: list[Relationship]) -> list[str]:
        """Add multiple relationships efficiently.

        Args:
            relationships: List of relationships to add

        Returns:
            List of added relationship IDs
        """
        pass

    @abstractmethod
    def search_entities(
        self,
        query: str,
        node_types: list[str] | None = None,
        limit: int = 10,
    ) -> list[Entity]:
        """Search entities by text query.

        Args:
            query: Search query (matches against name/description properties)
            node_types: Optional filter for node types
            limit: Maximum results to return

        Returns:
            List of matching entities
        """
        pass

    def get_subgraph(
        self,
        entity_ids: list[str],
        include_neighbors: bool = True,
        max_depth: int = 1,
    ) -> tuple[list[Entity], list[Relationship]]:
        """Extract a subgraph containing specified entities.

        Args:
            entity_ids: IDs of entities to include
            include_neighbors: Whether to include neighboring entities
            max_depth: Depth of neighbor expansion

        Returns:
            Tuple of (entities, relationships) in the subgraph
        """
        entities = []
        relationships = []
        visited_entities = set()
        visited_rels = set()

        to_visit = list(entity_ids)
        current_depth = 0

        while to_visit and current_depth <= max_depth:
            next_level = []

            for entity_id in to_visit:
                if entity_id in visited_entities:
                    continue

                visited_entities.add(entity_id)
                entity = self.get_entity(entity_id)

                if entity:
                    entities.append(entity)

                    if include_neighbors and current_depth < max_depth:
                        for neighbor, rel in self.get_neighbors(entity_id):
                            if rel.id not in visited_rels:
                                relationships.append(rel)
                                visited_rels.add(rel.id)

                            if neighbor.id not in visited_entities:
                                next_level.append(neighbor.id)

            to_visit = next_level
            current_depth += 1

        return entities, relationships
