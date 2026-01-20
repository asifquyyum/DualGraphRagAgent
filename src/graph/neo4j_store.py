"""Neo4j-based graph store for production use.

Implements the GraphStore interface using Neo4j as the backend,
providing persistence, indexing, and scalable graph queries.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from src.graph.graph_interface import GraphStore
from src.schema.ontology import DomainOntology, Entity, Relationship
from src.tools.neo4j_tools import Neo4jSettings, Neo4jTools


class Neo4jStore(GraphStore):
    """Neo4j-based graph store for production deployments.

    Provides persistent storage with full Cypher query support,
    indexing, and constraints.
    """

    def __init__(self, settings: Neo4jSettings | None = None) -> None:
        """Initialize the Neo4j store.

        Args:
            settings: Optional Neo4j settings. If not provided, loads from environment.
        """
        self.tools = Neo4jTools(settings)
        self.ontology: DomainOntology | None = None
        self._initialized = False

    def initialize(self, ontology: DomainOntology) -> None:
        """Initialize the graph store with a domain ontology.

        Creates necessary constraints and indexes based on the ontology.
        """
        self.ontology = ontology

        # Create constraints for each node type
        for node_type in ontology.node_schemas.keys():
            self.tools.create_constraints(node_type, "id")

            # Create index on name property if it exists
            schema = ontology.node_schemas[node_type]
            if any(p.name == "name" for p in schema.properties):
                self.tools.create_index(node_type, ["name"])

        # Create fulltext index for text search
        node_types = list(ontology.node_schemas.keys())
        if node_types:
            self.tools.create_fulltext_index(
                "entity_search",
                node_types,
                ["name", "description"],
            )

        self._initialized = True

    def add_entity(self, entity: Entity) -> str:
        """Add an entity to the graph."""
        if self.ontology:
            errors = self.ontology.validate_entity(entity)
            if errors:
                raise ValueError(f"Entity validation failed: {errors}")

        # Build node properties
        props = {
            "id": entity.id,
            "node_type": entity.node_type,
            **entity.properties,
            "_source_document": entity.source_document,
            "_source_chunk_id": entity.source_chunk_id,
            "_extraction_confidence": entity.extraction_confidence,
            "_created_at": entity.created_at.isoformat(),
            "_updated_at": entity.updated_at.isoformat(),
        }

        # Create node with label
        query = f"""
        CREATE (n:{entity.node_type} $props)
        RETURN n.id AS id
        """

        result = self.tools.execute_write(query, {"props": props})
        if not result.success:
            raise RuntimeError(f"Failed to create entity: {result.error}")

        return entity.id

    def add_relationship(self, relationship: Relationship) -> str:
        """Add a relationship between entities."""
        if self.ontology:
            errors = self.ontology.validate_relationship(relationship)
            if errors:
                raise ValueError(f"Relationship validation failed: {errors}")

        props = {
            "id": relationship.id,
            "edge_type": relationship.edge_type,
            **relationship.properties,
            "_source_document": relationship.source_document,
            "_source_chunk_id": relationship.source_chunk_id,
            "_extraction_confidence": relationship.extraction_confidence,
            "_created_at": relationship.created_at.isoformat(),
            "_updated_at": relationship.updated_at.isoformat(),
        }

        query = f"""
        MATCH (source {{id: $source_id}})
        MATCH (target {{id: $target_id}})
        CREATE (source)-[r:{relationship.edge_type} $props]->(target)
        RETURN r.id AS id
        """

        result = self.tools.execute_write(
            query,
            {
                "source_id": relationship.source_id,
                "target_id": relationship.target_id,
                "props": props,
            },
        )

        if not result.success:
            raise RuntimeError(f"Failed to create relationship: {result.error}")

        if not result.records:
            raise ValueError(
                f"Source ({relationship.source_id}) or target ({relationship.target_id}) not found"
            )

        return relationship.id

    def get_entity(self, entity_id: str) -> Entity | None:
        """Retrieve an entity by ID."""
        result = self.tools.execute_read(
            "MATCH (n {id: $id}) RETURN n, labels(n) AS labels",
            {"id": entity_id},
        )

        if not result.success or not result.records:
            return None

        record = result.records[0]
        node_data = dict(record.get("n", {}))
        labels = record.get("labels", [])

        # Determine node_type from labels (exclude generic labels)
        node_type = next((l for l in labels if l != "Node"), labels[0] if labels else "Unknown")

        return self._node_to_entity(node_data, node_type)

    def get_relationship(self, relationship_id: str) -> Relationship | None:
        """Retrieve a relationship by ID."""
        result = self.tools.execute_read(
            """
            MATCH (s)-[r {id: $id}]->(t)
            RETURN r, type(r) AS rel_type, s.id AS source_id, t.id AS target_id
            """,
            {"id": relationship_id},
        )

        if not result.success or not result.records:
            return None

        record = result.records[0]
        return self._record_to_relationship(record)

    def find_entities_by_type(self, node_type: str) -> list[Entity]:
        """Find all entities of a given type."""
        result = self.tools.execute_read(
            f"MATCH (n:{node_type}) RETURN n",
        )

        if not result.success:
            return []

        return [self._node_to_entity(dict(r.get("n", {})), node_type) for r in result.records]

    def find_entities_by_property(
        self, property_name: str, property_value: Any
    ) -> list[Entity]:
        """Find entities by a property value."""
        result = self.tools.execute_read(
            f"MATCH (n) WHERE n.{property_name} = $value RETURN n, labels(n) AS labels",
            {"value": property_value},
        )

        if not result.success:
            return []

        entities = []
        for r in result.records:
            node_data = dict(r.get("n", {}))
            labels = r.get("labels", [])
            node_type = next((l for l in labels if l != "Node"), "Unknown")
            entities.append(self._node_to_entity(node_data, node_type))

        return entities

    def get_neighbors(
        self,
        entity_id: str,
        edge_types: list[str] | None = None,
        direction: str = "both",
    ) -> list[tuple[Entity, Relationship]]:
        """Get neighboring entities connected by relationships."""
        rel_filter = ""
        if edge_types:
            rel_filter = ":" + "|".join(edge_types)

        if direction == "outgoing":
            pattern = f"(n)-[r{rel_filter}]->(m)"
        elif direction == "incoming":
            pattern = f"(n)<-[r{rel_filter}]-(m)"
        else:
            pattern = f"(n)-[r{rel_filter}]-(m)"

        query = f"""
        MATCH {pattern}
        WHERE n.id = $entity_id
        RETURN m, r, type(r) AS rel_type, labels(m) AS labels,
               startNode(r).id AS source_id, endNode(r).id AS target_id
        """

        result = self.tools.execute_read(query, {"entity_id": entity_id})

        if not result.success:
            return []

        neighbors = []
        for record in result.records:
            node_data = dict(record.get("m", {}))
            labels = record.get("labels", [])
            node_type = next((l for l in labels if l != "Node"), "Unknown")

            entity = self._node_to_entity(node_data, node_type)
            rel = self._record_to_relationship(record)

            neighbors.append((entity, rel))

        return neighbors

    def get_path(
        self,
        source_id: str,
        target_id: str,
        max_hops: int = 3,
    ) -> list[tuple[Entity, Relationship]] | None:
        """Find a path between two entities."""
        path_data = self.tools.find_path(source_id, target_id, max_hops)

        if not path_data:
            return None

        result = []
        nodes = path_data.get("nodes", [])
        relationships = path_data.get("relationships", [])

        for i, rel_data in enumerate(relationships):
            if i + 1 < len(nodes):
                node_data = nodes[i + 1]
                node_type = node_data.get("node_type", "Unknown")

                entity = self._node_to_entity(node_data, node_type)

                # Create relationship object
                rel = Relationship(
                    id=rel_data.get("props", {}).get("id", f"rel_{i}"),
                    edge_type=rel_data.get("type", "UNKNOWN"),
                    source_id=nodes[i].get("id", ""),
                    target_id=node_data.get("id", ""),
                    properties=rel_data.get("props", {}),
                )

                result.append((entity, rel))

        return result if result else None

    def execute_query(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a Cypher query."""
        result = self.tools.execute_read(query, params)
        if result.success:
            return result.records
        return []

    def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity and its relationships."""
        result = self.tools.execute_write(
            "MATCH (n {id: $id}) DETACH DELETE n RETURN count(n) AS deleted",
            {"id": entity_id},
        )
        if result.success and result.records:
            return result.records[0].get("deleted", 0) > 0
        return False

    def delete_relationship(self, relationship_id: str) -> bool:
        """Delete a relationship."""
        result = self.tools.execute_write(
            "MATCH ()-[r {id: $id}]-() DELETE r RETURN count(r) AS deleted",
            {"id": relationship_id},
        )
        if result.success and result.records:
            return result.records[0].get("deleted", 0) > 0
        return False

    def clear(self) -> None:
        """Remove all data from the graph."""
        self.tools.clear_database()

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about the graph."""
        schema_info = self.tools.get_schema_info()

        node_counts = {}
        for label in schema_info.get("labels", []):
            node_counts[label] = self.tools.get_node_count(label)

        rel_counts = {}
        for rel_type in schema_info.get("relationship_types", []):
            rel_counts[rel_type] = self.tools.get_relationship_count(rel_type)

        return {
            "num_entities": self.tools.get_node_count(),
            "num_relationships": self.tools.get_relationship_count(),
            "node_types": node_counts,
            "edge_types": rel_counts,
            "labels": schema_info.get("labels", []),
            "relationship_types": schema_info.get("relationship_types", []),
        }

    def update_entity(self, entity_id: str, properties: dict[str, Any]) -> Entity | None:
        """Update an entity's properties."""
        properties["_updated_at"] = datetime.utcnow().isoformat()

        result = self.tools.execute_write(
            """
            MATCH (n {id: $id})
            SET n += $props
            RETURN n, labels(n) AS labels
            """,
            {"id": entity_id, "props": properties},
        )

        if not result.success or not result.records:
            return None

        record = result.records[0]
        node_data = dict(record.get("n", {}))
        labels = record.get("labels", [])
        node_type = next((l for l in labels if l != "Node"), "Unknown")

        return self._node_to_entity(node_data, node_type)

    def update_relationship(
        self, relationship_id: str, properties: dict[str, Any]
    ) -> Relationship | None:
        """Update a relationship's properties."""
        properties["_updated_at"] = datetime.utcnow().isoformat()

        result = self.tools.execute_write(
            """
            MATCH (s)-[r {id: $id}]->(t)
            SET r += $props
            RETURN r, type(r) AS rel_type, s.id AS source_id, t.id AS target_id
            """,
            {"id": relationship_id, "props": properties},
        )

        if not result.success or not result.records:
            return None

        return self._record_to_relationship(result.records[0])

    def bulk_add_entities(self, entities: list[Entity]) -> list[str]:
        """Add multiple entities efficiently."""
        ids = []

        # Group by node type for efficient batch creation
        by_type: dict[str, list[Entity]] = {}
        for entity in entities:
            by_type.setdefault(entity.node_type, []).append(entity)

        for node_type, type_entities in by_type.items():
            nodes = [
                {
                    "id": e.id,
                    "node_type": e.node_type,
                    **e.properties,
                    "_source_document": e.source_document,
                    "_source_chunk_id": e.source_chunk_id,
                    "_extraction_confidence": e.extraction_confidence,
                    "_created_at": e.created_at.isoformat(),
                    "_updated_at": e.updated_at.isoformat(),
                }
                for e in type_entities
            ]

            result = self.tools.batch_create_nodes(nodes, node_type)
            if result.success:
                ids.extend(e.id for e in type_entities)

        return ids

    def bulk_add_relationships(self, relationships: list[Relationship]) -> list[str]:
        """Add multiple relationships efficiently."""
        ids = []

        # Group by edge type
        by_type: dict[str, list[Relationship]] = {}
        for rel in relationships:
            by_type.setdefault(rel.edge_type, []).append(rel)

        for edge_type, type_rels in by_type.items():
            rels = [
                {
                    "source_id": r.source_id,
                    "target_id": r.target_id,
                    "properties": {
                        "id": r.id,
                        "edge_type": r.edge_type,
                        **r.properties,
                        "_source_document": r.source_document,
                        "_source_chunk_id": r.source_chunk_id,
                        "_extraction_confidence": r.extraction_confidence,
                        "_created_at": r.created_at.isoformat(),
                        "_updated_at": r.updated_at.isoformat(),
                    },
                }
                for r in type_rels
            ]

            # Need to determine source/target labels - use generic match
            query = f"""
            UNWIND $rels AS rel
            MATCH (source {{id: rel.source_id}})
            MATCH (target {{id: rel.target_id}})
            CREATE (source)-[r:{edge_type}]->(target)
            SET r = rel.properties
            RETURN count(r) AS created
            """

            result = self.tools.execute_write(query, {"rels": rels})
            if result.success:
                ids.extend(r.id for r in type_rels)

        return ids

    def search_entities(
        self,
        query: str,
        node_types: list[str] | None = None,
        limit: int = 10,
    ) -> list[Entity]:
        """Search entities by text query."""
        # Try fulltext search first
        result = self.tools.fulltext_search("entity_search", query, limit * 2)

        entities = []
        if result.success:
            for record in result.records:
                node_data = dict(record.get("node", {}))
                node_type = node_data.get("node_type", "Unknown")

                if node_types and node_type not in node_types:
                    continue

                entities.append(self._node_to_entity(node_data, node_type))

                if len(entities) >= limit:
                    break

        return entities

    def close(self) -> None:
        """Close the Neo4j connection."""
        self.tools.close()

    def _node_to_entity(self, node_data: dict[str, Any], node_type: str) -> Entity:
        """Convert a Neo4j node to an Entity object."""
        # Extract standard fields
        entity_id = node_data.pop("id", "")
        node_data.pop("node_type", None)
        source_doc = node_data.pop("_source_document", None)
        source_chunk = node_data.pop("_source_chunk_id", None)
        confidence = node_data.pop("_extraction_confidence", 1.0)
        created_at = node_data.pop("_created_at", None)
        updated_at = node_data.pop("_updated_at", None)

        return Entity(
            id=entity_id,
            node_type=node_type,
            properties=node_data,
            source_document=source_doc,
            source_chunk_id=source_chunk,
            extraction_confidence=confidence,
            created_at=datetime.fromisoformat(created_at) if created_at else datetime.utcnow(),
            updated_at=datetime.fromisoformat(updated_at) if updated_at else datetime.utcnow(),
        )

    def _record_to_relationship(self, record: dict[str, Any]) -> Relationship:
        """Convert a Neo4j record to a Relationship object."""
        rel_data = dict(record.get("r", {}))

        rel_id = rel_data.pop("id", "")
        edge_type = record.get("rel_type", rel_data.pop("edge_type", "UNKNOWN"))
        source_id = record.get("source_id", "")
        target_id = record.get("target_id", "")
        source_doc = rel_data.pop("_source_document", None)
        source_chunk = rel_data.pop("_source_chunk_id", None)
        confidence = rel_data.pop("_extraction_confidence", 1.0)
        created_at = rel_data.pop("_created_at", None)
        updated_at = rel_data.pop("_updated_at", None)

        return Relationship(
            id=rel_id,
            edge_type=edge_type,
            source_id=source_id,
            target_id=target_id,
            properties=rel_data,
            source_document=source_doc,
            source_chunk_id=source_chunk,
            extraction_confidence=confidence,
            created_at=datetime.fromisoformat(created_at) if created_at else datetime.utcnow(),
            updated_at=datetime.fromisoformat(updated_at) if updated_at else datetime.utcnow(),
        )
