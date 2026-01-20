"""Neo4j database tools for Cypher execution and connection management.

Provides utilities for interacting with Neo4j, including connection pooling,
query execution, and error handling.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Generator

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class Neo4jSettings(BaseSettings):
    """Configuration for Neo4j connection."""

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    neo4j_database: str = "neo4j"
    max_connection_lifetime: int = 3600
    max_connection_pool_size: int = 50
    connection_acquisition_timeout: int = 60

    class Config:
        """Pydantic settings config."""

        env_file = ".env"
        env_file_encoding = "utf-8"


class CypherQueryResult(BaseModel):
    """Result of a Cypher query execution."""

    success: bool = Field(..., description="Whether the query succeeded")
    records: list[dict[str, Any]] = Field(default_factory=list, description="Query results")
    summary: dict[str, Any] = Field(default_factory=dict, description="Query summary/statistics")
    error: str | None = Field(default=None, description="Error message if failed")


class Neo4jTools:
    """Tools for Neo4j database operations.

    Provides connection management, query execution, and utility functions
    for working with Neo4j.
    """

    def __init__(self, settings: Neo4jSettings | None = None) -> None:
        """Initialize Neo4j tools.

        Args:
            settings: Optional settings. If not provided, loads from environment.
        """
        self.settings = settings or Neo4jSettings()
        self._driver = None
        self._neo4j_available = self._check_neo4j()

    def _check_neo4j(self) -> bool:
        """Check if neo4j package is available."""
        try:
            import neo4j
            return True
        except ImportError:
            return False

    def _get_driver(self):
        """Get or create the Neo4j driver."""
        if not self._neo4j_available:
            raise ImportError("neo4j package is required. Install with: pip install neo4j")

        if self._driver is None:
            from neo4j import GraphDatabase

            self._driver = GraphDatabase.driver(
                self.settings.neo4j_uri,
                auth=(self.settings.neo4j_user, self.settings.neo4j_password),
                max_connection_lifetime=self.settings.max_connection_lifetime,
                max_connection_pool_size=self.settings.max_connection_pool_size,
                connection_acquisition_timeout=self.settings.connection_acquisition_timeout,
            )

        return self._driver

    def close(self) -> None:
        """Close the Neo4j driver connection."""
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    @contextmanager
    def session(self) -> Generator:
        """Context manager for Neo4j session."""
        driver = self._get_driver()
        session = driver.session(database=self.settings.neo4j_database)
        try:
            yield session
        finally:
            session.close()

    def execute_query(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        write: bool = False,
    ) -> CypherQueryResult:
        """Execute a Cypher query.

        Args:
            query: The Cypher query to execute
            params: Optional query parameters
            write: Whether this is a write transaction

        Returns:
            CypherQueryResult with results or error
        """
        params = params or {}

        try:
            with self.session() as session:
                if write:
                    result = session.execute_write(
                        lambda tx: list(tx.run(query, params))
                    )
                else:
                    result = session.execute_read(
                        lambda tx: list(tx.run(query, params))
                    )

                records = [dict(record) for record in result]

                return CypherQueryResult(
                    success=True,
                    records=records,
                    summary={"query": query, "record_count": len(records)},
                )

        except Exception as e:
            return CypherQueryResult(
                success=False,
                error=str(e),
                summary={"query": query},
            )

    def execute_write(
        self,
        query: str,
        params: dict[str, Any] | None = None,
    ) -> CypherQueryResult:
        """Execute a write query."""
        return self.execute_query(query, params, write=True)

    def execute_read(
        self,
        query: str,
        params: dict[str, Any] | None = None,
    ) -> CypherQueryResult:
        """Execute a read query."""
        return self.execute_query(query, params, write=False)

    def verify_connection(self) -> bool:
        """Verify that the Neo4j connection is working."""
        result = self.execute_read("RETURN 1 AS test")
        return result.success and len(result.records) > 0

    def get_schema_info(self) -> dict[str, Any]:
        """Get schema information from the database."""
        labels_result = self.execute_read("CALL db.labels()")
        rel_types_result = self.execute_read("CALL db.relationshipTypes()")
        properties_result = self.execute_read("CALL db.propertyKeys()")

        return {
            "labels": [r.get("label") for r in labels_result.records] if labels_result.success else [],
            "relationship_types": [r.get("relationshipType") for r in rel_types_result.records]
            if rel_types_result.success
            else [],
            "property_keys": [r.get("propertyKey") for r in properties_result.records]
            if properties_result.success
            else [],
        }

    def get_node_count(self, label: str | None = None) -> int:
        """Get count of nodes, optionally filtered by label."""
        if label:
            query = f"MATCH (n:{label}) RETURN count(n) AS count"
        else:
            query = "MATCH (n) RETURN count(n) AS count"

        result = self.execute_read(query)
        if result.success and result.records:
            return result.records[0].get("count", 0)
        return 0

    def get_relationship_count(self, rel_type: str | None = None) -> int:
        """Get count of relationships, optionally filtered by type."""
        if rel_type:
            query = f"MATCH ()-[r:{rel_type}]->() RETURN count(r) AS count"
        else:
            query = "MATCH ()-[r]->() RETURN count(r) AS count"

        result = self.execute_read(query)
        if result.success and result.records:
            return result.records[0].get("count", 0)
        return 0

    def create_constraints(self, node_type: str, property_name: str = "id") -> CypherQueryResult:
        """Create a uniqueness constraint on a node property."""
        query = f"""
        CREATE CONSTRAINT IF NOT EXISTS
        FOR (n:{node_type})
        REQUIRE n.{property_name} IS UNIQUE
        """
        return self.execute_write(query)

    def create_index(
        self,
        node_type: str,
        property_names: list[str],
        index_name: str | None = None,
    ) -> CypherQueryResult:
        """Create an index on node properties."""
        props = ", ".join(f"n.{p}" for p in property_names)
        name = index_name or f"idx_{node_type}_{'_'.join(property_names)}"
        query = f"""
        CREATE INDEX {name} IF NOT EXISTS
        FOR (n:{node_type})
        ON ({props})
        """
        return self.execute_write(query)

    def create_fulltext_index(
        self,
        index_name: str,
        node_types: list[str],
        property_names: list[str],
    ) -> CypherQueryResult:
        """Create a fulltext index for text search."""
        labels = "|".join(node_types)
        props = ", ".join(f"'{p}'" for p in property_names)
        query = f"""
        CREATE FULLTEXT INDEX {index_name} IF NOT EXISTS
        FOR (n:{labels})
        ON EACH [{props}]
        """
        return self.execute_write(query)

    def fulltext_search(
        self,
        index_name: str,
        search_query: str,
        limit: int = 10,
    ) -> CypherQueryResult:
        """Search using a fulltext index."""
        query = f"""
        CALL db.index.fulltext.queryNodes("{index_name}", $search_query)
        YIELD node, score
        RETURN node, score
        ORDER BY score DESC
        LIMIT $limit
        """
        return self.execute_read(query, {"search_query": search_query, "limit": limit})

    def clear_database(self) -> CypherQueryResult:
        """Remove all nodes and relationships from the database."""
        return self.execute_write("MATCH (n) DETACH DELETE n")

    def batch_create_nodes(
        self,
        nodes: list[dict[str, Any]],
        label: str,
        batch_size: int = 1000,
    ) -> CypherQueryResult:
        """Create multiple nodes efficiently using UNWIND."""
        query = f"""
        UNWIND $nodes AS node
        CREATE (n:{label})
        SET n = node
        RETURN count(n) AS created
        """

        total_created = 0
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i : i + batch_size]
            result = self.execute_write(query, {"nodes": batch})
            if result.success and result.records:
                total_created += result.records[0].get("created", 0)
            elif not result.success:
                return result

        return CypherQueryResult(
            success=True,
            records=[{"created": total_created}],
            summary={"total_nodes": len(nodes), "batch_size": batch_size},
        )

    def batch_create_relationships(
        self,
        relationships: list[dict[str, Any]],
        rel_type: str,
        source_label: str,
        target_label: str,
        batch_size: int = 1000,
    ) -> CypherQueryResult:
        """Create multiple relationships efficiently using UNWIND."""
        query = f"""
        UNWIND $rels AS rel
        MATCH (source:{source_label} {{id: rel.source_id}})
        MATCH (target:{target_label} {{id: rel.target_id}})
        CREATE (source)-[r:{rel_type}]->(target)
        SET r = rel.properties
        RETURN count(r) AS created
        """

        total_created = 0
        for i in range(0, len(relationships), batch_size):
            batch = relationships[i : i + batch_size]
            result = self.execute_write(query, {"rels": batch})
            if result.success and result.records:
                total_created += result.records[0].get("created", 0)
            elif not result.success:
                return result

        return CypherQueryResult(
            success=True,
            records=[{"created": total_created}],
            summary={"total_relationships": len(relationships), "batch_size": batch_size},
        )

    def get_node_by_id(self, node_id: str) -> dict[str, Any] | None:
        """Get a node by its ID property."""
        result = self.execute_read(
            "MATCH (n {id: $id}) RETURN n",
            {"id": node_id},
        )
        if result.success and result.records:
            return dict(result.records[0].get("n", {}))
        return None

    def get_neighbors(
        self,
        node_id: str,
        rel_types: list[str] | None = None,
        direction: str = "both",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get neighboring nodes."""
        rel_filter = ""
        if rel_types:
            rel_filter = ":" + "|".join(rel_types)

        if direction == "outgoing":
            pattern = f"(n)-[r{rel_filter}]->(m)"
        elif direction == "incoming":
            pattern = f"(n)<-[r{rel_filter}]-(m)"
        else:
            pattern = f"(n)-[r{rel_filter}]-(m)"

        query = f"""
        MATCH {pattern}
        WHERE n.id = $node_id
        RETURN m, r, type(r) AS rel_type
        LIMIT $limit
        """

        result = self.execute_read(query, {"node_id": node_id, "limit": limit})
        if result.success:
            return [
                {
                    "node": dict(r.get("m", {})),
                    "relationship": dict(r.get("r", {})),
                    "relationship_type": r.get("rel_type"),
                }
                for r in result.records
            ]
        return []

    def find_path(
        self,
        source_id: str,
        target_id: str,
        max_hops: int = 5,
    ) -> list[dict[str, Any]] | None:
        """Find shortest path between two nodes."""
        query = f"""
        MATCH path = shortestPath((source {{id: $source_id}})-[*1..{max_hops}]-(target {{id: $target_id}}))
        RETURN [node IN nodes(path) | node] AS nodes,
               [rel IN relationships(path) | {{type: type(rel), props: properties(rel)}}] AS relationships
        """

        result = self.execute_read(query, {"source_id": source_id, "target_id": target_id})
        if result.success and result.records:
            record = result.records[0]
            return {
                "nodes": [dict(n) for n in record.get("nodes", [])],
                "relationships": record.get("relationships", []),
            }
        return None
