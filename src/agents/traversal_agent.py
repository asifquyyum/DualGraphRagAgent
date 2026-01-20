"""Graph traversal agent for exploring neighbors and paths.

Intelligently explores the graph to gather additional context
based on current knowledge state and uncertainties.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.graph.graph_interface import GraphStore
from src.llm.gemini_provider import GeminiProvider
from src.schema.ontology import DomainOntology, Entity, Relationship
from src.schema.world_state import Uncertainty, WorldState


class TraversalStep(BaseModel):
    """A single step in graph traversal."""

    source_entity_id: str
    entities_found: list[dict[str, Any]] = Field(default_factory=list)
    relationships_found: list[dict[str, Any]] = Field(default_factory=list)
    relevance_score: float = Field(default=0.5)


class TraversalResult(BaseModel):
    """Result of a graph traversal exploration."""

    steps: list[TraversalStep] = Field(default_factory=list)
    entities_discovered: list[dict[str, Any]] = Field(default_factory=list)
    relationships_discovered: list[dict[str, Any]] = Field(default_factory=list)
    new_evidence: list[str] = Field(default_factory=list)
    suggested_next_entities: list[str] = Field(default_factory=list)


class TraversalAgent:
    """Agent for intelligent graph exploration.

    Explores the graph based on current knowledge state, focusing on
    resolving uncertainties and gathering relevant evidence.
    """

    def __init__(
        self,
        llm: GeminiProvider,
        ontology: DomainOntology,
        store: GraphStore,
    ) -> None:
        """Initialize the traversal agent.

        Args:
            llm: The LLM provider
            ontology: The domain ontology
            store: The graph store
        """
        self.llm = llm
        self.ontology = ontology
        self.store = store

    def explore_from_entities(
        self,
        entity_ids: list[str],
        world_state: WorldState,
        max_depth: int = 2,
        max_neighbors: int = 10,
    ) -> TraversalResult:
        """Explore graph starting from given entities.

        Args:
            entity_ids: Starting entity IDs
            world_state: Current world state
            max_depth: Maximum traversal depth
            max_neighbors: Maximum neighbors per entity

        Returns:
            TraversalResult with discovered entities and relationships
        """
        result = TraversalResult()
        visited = set(world_state.explored_nodes)

        # Get priority uncertainties to guide exploration
        priority_uncertainty = world_state.get_highest_priority_uncertainty()

        for depth in range(max_depth):
            next_entities = []

            for entity_id in entity_ids:
                if entity_id in visited:
                    continue

                visited.add(entity_id)
                step = self._explore_entity(
                    entity_id, priority_uncertainty, max_neighbors
                )

                result.steps.append(step)
                result.entities_discovered.extend(step.entities_found)
                result.relationships_discovered.extend(step.relationships_found)

                # Add high-relevance neighbors to next level
                for entity_data in step.entities_found:
                    neighbor_id = entity_data.get("id")
                    if neighbor_id and neighbor_id not in visited:
                        next_entities.append(neighbor_id)

            entity_ids = next_entities[:max_neighbors]

        # Generate evidence from discoveries
        result.new_evidence = self._extract_evidence(result, world_state)

        # Suggest next entities to explore
        result.suggested_next_entities = self._suggest_next_exploration(
            result, world_state
        )

        return result

    def _explore_entity(
        self,
        entity_id: str,
        priority_uncertainty: Uncertainty | None,
        max_neighbors: int,
    ) -> TraversalStep:
        """Explore a single entity's neighborhood."""
        step = TraversalStep(source_entity_id=entity_id)

        # Get entity
        entity = self.store.get_entity(entity_id)
        if not entity:
            return step

        # Get neighbors
        neighbors = self.store.get_neighbors(entity_id)

        for neighbor, rel in neighbors[:max_neighbors]:
            # Calculate relevance based on uncertainty
            relevance = self._calculate_relevance(
                neighbor, rel, priority_uncertainty
            )

            step.entities_found.append(
                {
                    "id": neighbor.id,
                    "type": neighbor.node_type,
                    "properties": neighbor.properties,
                    "relevance": relevance,
                }
            )

            step.relationships_found.append(
                {
                    "id": rel.id,
                    "type": rel.edge_type,
                    "source_id": rel.source_id,
                    "target_id": rel.target_id,
                    "properties": rel.properties,
                }
            )

        # Calculate overall relevance
        if step.entities_found:
            step.relevance_score = sum(
                e["relevance"] for e in step.entities_found
            ) / len(step.entities_found)

        return step

    def _calculate_relevance(
        self,
        entity: Entity,
        relationship: Relationship,
        uncertainty: Uncertainty | None,
    ) -> float:
        """Calculate relevance score for a discovered entity."""
        relevance = 0.5

        if not uncertainty:
            return relevance

        uncertainty_lower = uncertainty.description.lower()

        # Check if entity type matches uncertainty keywords
        if entity.node_type.lower() in uncertainty_lower:
            relevance += 0.2

        # Check properties for relevant terms
        for prop_value in entity.properties.values():
            if isinstance(prop_value, str) and prop_value.lower() in uncertainty_lower:
                relevance += 0.15

        # Check relationship type relevance
        rel_type_words = relationship.edge_type.lower().replace("_", " ").split()
        for word in rel_type_words:
            if word in uncertainty_lower:
                relevance += 0.1

        return min(1.0, relevance)

    def _extract_evidence(
        self,
        result: TraversalResult,
        world_state: WorldState,
    ) -> list[str]:
        """Extract evidence statements from traversal results."""
        evidence = []

        # Group entities by type
        by_type: dict[str, list[dict[str, Any]]] = {}
        for entity in result.entities_discovered:
            by_type.setdefault(entity["type"], []).append(entity)

        # Generate evidence statements
        for entity_type, entities in by_type.items():
            names = [e["properties"].get("name", e["id"]) for e in entities[:5]]
            if names:
                evidence.append(
                    f"Found {len(entities)} {entity_type} entities: {', '.join(names)}"
                )

        # Summarize relationships
        rel_types: dict[str, int] = {}
        for rel in result.relationships_discovered:
            rel_types[rel["type"]] = rel_types.get(rel["type"], 0) + 1

        for rel_type, count in rel_types.items():
            evidence.append(f"Discovered {count} {rel_type} relationships")

        return evidence

    def _suggest_next_exploration(
        self,
        result: TraversalResult,
        world_state: WorldState,
    ) -> list[str]:
        """Suggest entities for further exploration."""
        suggestions = []

        # Find high-relevance entities not yet explored
        for step in result.steps:
            for entity in step.entities_found:
                if (
                    entity["relevance"] > 0.6
                    and entity["id"] not in world_state.explored_nodes
                ):
                    suggestions.append(entity["id"])

        return suggestions[:5]

    def explore_for_uncertainty(
        self,
        uncertainty: Uncertainty,
        world_state: WorldState,
        max_entities: int = 10,
    ) -> TraversalResult:
        """Explore graph specifically to resolve an uncertainty.

        Args:
            uncertainty: The uncertainty to resolve
            world_state: Current world state
            max_entities: Maximum entities to explore

        Returns:
            TraversalResult focused on the uncertainty
        """
        result = TraversalResult()

        # Search for entities related to the uncertainty
        search_results = self.store.search_entities(
            uncertainty.description, limit=max_entities
        )

        for entity in search_results:
            if entity.id in world_state.explored_nodes:
                continue

            step = self._explore_entity(entity.id, uncertainty, max_neighbors=5)
            result.steps.append(step)
            result.entities_discovered.extend(step.entities_found)
            result.relationships_discovered.extend(step.relationships_found)

        result.new_evidence = self._extract_evidence(result, world_state)

        return result

    def find_connecting_paths(
        self,
        source_ids: list[str],
        target_ids: list[str],
        max_hops: int = 3,
    ) -> list[dict[str, Any]]:
        """Find paths connecting source and target entities.

        Args:
            source_ids: Source entity IDs
            target_ids: Target entity IDs
            max_hops: Maximum path length

        Returns:
            List of paths found
        """
        paths = []

        for source_id in source_ids:
            for target_id in target_ids:
                if source_id == target_id:
                    continue

                path = self.store.get_path(source_id, target_id, max_hops)
                if path:
                    paths.append(
                        {
                            "source_id": source_id,
                            "target_id": target_id,
                            "length": len(path),
                            "path": [
                                {
                                    "entity": e.to_dict(),
                                    "relationship": r.to_dict(),
                                }
                                for e, r in path
                            ],
                        }
                    )

        return paths

    def get_entity_context(
        self,
        entity_id: str,
        context_depth: int = 1,
    ) -> dict[str, Any]:
        """Get rich context for an entity.

        Args:
            entity_id: The entity ID
            context_depth: Depth of context to gather

        Returns:
            Dictionary with entity and surrounding context
        """
        entity = self.store.get_entity(entity_id)
        if not entity:
            return {}

        context = {
            "entity": entity.to_dict(),
            "neighbors": [],
            "paths_summary": [],
        }

        # Get immediate neighbors
        neighbors = self.store.get_neighbors(entity_id)
        for neighbor, rel in neighbors[:20]:
            context["neighbors"].append(
                {
                    "entity": neighbor.to_dict(),
                    "relationship": rel.to_dict(),
                    "direction": "outgoing"
                    if rel.source_id == entity_id
                    else "incoming",
                }
            )

        return context
