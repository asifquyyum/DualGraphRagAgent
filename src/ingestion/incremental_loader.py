"""Incremental loader for delta updates to the knowledge graph.

Handles efficient updates to the graph by detecting changes and
applying only the necessary modifications.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from src.graph.graph_interface import GraphStore
from src.schema.ontology import Entity, Relationship


class ChangeType(str, Enum):
    """Type of change detected."""

    ADD = "add"
    UPDATE = "update"
    DELETE = "delete"
    UNCHANGED = "unchanged"


class EntityChange(BaseModel):
    """Detected change for an entity."""

    entity_id: str
    change_type: ChangeType
    old_hash: str | None = None
    new_hash: str | None = None
    entity: Entity | None = None


class RelationshipChange(BaseModel):
    """Detected change for a relationship."""

    relationship_id: str
    change_type: ChangeType
    old_hash: str | None = None
    new_hash: str | None = None
    relationship: Relationship | None = None


class LoaderState(BaseModel):
    """State of the incremental loader for tracking changes."""

    entity_hashes: dict[str, str] = Field(default_factory=dict)
    relationship_hashes: dict[str, str] = Field(default_factory=dict)
    last_update: datetime | None = None
    total_entities: int = 0
    total_relationships: int = 0


class IncrementalLoader:
    """Handles incremental updates to the knowledge graph.

    Tracks content hashes to detect changes and applies only
    necessary modifications to the graph.
    """

    def __init__(self, store: GraphStore) -> None:
        """Initialize the incremental loader.

        Args:
            store: The graph store to update
        """
        self.store = store
        self.state = LoaderState()

    def compute_entity_hash(self, entity: Entity) -> str:
        """Compute a content hash for an entity."""
        # Include relevant fields for change detection
        content = json.dumps(
            {
                "node_type": entity.node_type,
                "properties": entity.properties,
            },
            sort_keys=True,
        )
        return hashlib.md5(content.encode()).hexdigest()

    def compute_relationship_hash(self, relationship: Relationship) -> str:
        """Compute a content hash for a relationship."""
        content = json.dumps(
            {
                "edge_type": relationship.edge_type,
                "source_id": relationship.source_id,
                "target_id": relationship.target_id,
                "properties": relationship.properties,
            },
            sort_keys=True,
        )
        return hashlib.md5(content.encode()).hexdigest()

    def detect_entity_changes(
        self,
        entities: list[Entity],
    ) -> list[EntityChange]:
        """Detect changes between new entities and stored state.

        Args:
            entities: New entities to compare

        Returns:
            List of detected changes
        """
        changes = []
        new_entity_ids = set()

        for entity in entities:
            new_hash = self.compute_entity_hash(entity)
            old_hash = self.state.entity_hashes.get(entity.id)
            new_entity_ids.add(entity.id)

            if old_hash is None:
                # New entity
                changes.append(
                    EntityChange(
                        entity_id=entity.id,
                        change_type=ChangeType.ADD,
                        new_hash=new_hash,
                        entity=entity,
                    )
                )
            elif old_hash != new_hash:
                # Updated entity
                changes.append(
                    EntityChange(
                        entity_id=entity.id,
                        change_type=ChangeType.UPDATE,
                        old_hash=old_hash,
                        new_hash=new_hash,
                        entity=entity,
                    )
                )
            else:
                # Unchanged
                changes.append(
                    EntityChange(
                        entity_id=entity.id,
                        change_type=ChangeType.UNCHANGED,
                        old_hash=old_hash,
                        new_hash=new_hash,
                    )
                )

        # Detect deletions
        for entity_id in self.state.entity_hashes:
            if entity_id not in new_entity_ids:
                changes.append(
                    EntityChange(
                        entity_id=entity_id,
                        change_type=ChangeType.DELETE,
                        old_hash=self.state.entity_hashes[entity_id],
                    )
                )

        return changes

    def detect_relationship_changes(
        self,
        relationships: list[Relationship],
    ) -> list[RelationshipChange]:
        """Detect changes between new relationships and stored state.

        Args:
            relationships: New relationships to compare

        Returns:
            List of detected changes
        """
        changes = []
        new_rel_ids = set()

        for rel in relationships:
            new_hash = self.compute_relationship_hash(rel)
            old_hash = self.state.relationship_hashes.get(rel.id)
            new_rel_ids.add(rel.id)

            if old_hash is None:
                changes.append(
                    RelationshipChange(
                        relationship_id=rel.id,
                        change_type=ChangeType.ADD,
                        new_hash=new_hash,
                        relationship=rel,
                    )
                )
            elif old_hash != new_hash:
                changes.append(
                    RelationshipChange(
                        relationship_id=rel.id,
                        change_type=ChangeType.UPDATE,
                        old_hash=old_hash,
                        new_hash=new_hash,
                        relationship=rel,
                    )
                )
            else:
                changes.append(
                    RelationshipChange(
                        relationship_id=rel.id,
                        change_type=ChangeType.UNCHANGED,
                        old_hash=old_hash,
                        new_hash=new_hash,
                    )
                )

        # Detect deletions
        for rel_id in self.state.relationship_hashes:
            if rel_id not in new_rel_ids:
                changes.append(
                    RelationshipChange(
                        relationship_id=rel_id,
                        change_type=ChangeType.DELETE,
                        old_hash=self.state.relationship_hashes[rel_id],
                    )
                )

        return changes

    def apply_entity_changes(
        self,
        changes: list[EntityChange],
        skip_deletions: bool = False,
    ) -> dict[str, int]:
        """Apply entity changes to the graph.

        Args:
            changes: Changes to apply
            skip_deletions: Whether to skip deletion operations

        Returns:
            Dictionary with counts of operations performed
        """
        counts = {"added": 0, "updated": 0, "deleted": 0, "unchanged": 0}

        for change in changes:
            if change.change_type == ChangeType.ADD and change.entity:
                try:
                    self.store.add_entity(change.entity)
                    self.state.entity_hashes[change.entity_id] = change.new_hash or ""
                    counts["added"] += 1
                except Exception:
                    pass

            elif change.change_type == ChangeType.UPDATE and change.entity:
                try:
                    self.store.update_entity(change.entity_id, change.entity.properties)
                    self.state.entity_hashes[change.entity_id] = change.new_hash or ""
                    counts["updated"] += 1
                except Exception:
                    pass

            elif change.change_type == ChangeType.DELETE and not skip_deletions:
                try:
                    self.store.delete_entity(change.entity_id)
                    del self.state.entity_hashes[change.entity_id]
                    counts["deleted"] += 1
                except Exception:
                    pass

            elif change.change_type == ChangeType.UNCHANGED:
                counts["unchanged"] += 1

        self.state.total_entities = len(self.state.entity_hashes)
        self.state.last_update = datetime.utcnow()

        return counts

    def apply_relationship_changes(
        self,
        changes: list[RelationshipChange],
        skip_deletions: bool = False,
    ) -> dict[str, int]:
        """Apply relationship changes to the graph.

        Args:
            changes: Changes to apply
            skip_deletions: Whether to skip deletion operations

        Returns:
            Dictionary with counts of operations performed
        """
        counts = {"added": 0, "updated": 0, "deleted": 0, "unchanged": 0}

        for change in changes:
            if change.change_type == ChangeType.ADD and change.relationship:
                try:
                    self.store.add_relationship(change.relationship)
                    self.state.relationship_hashes[change.relationship_id] = change.new_hash or ""
                    counts["added"] += 1
                except Exception:
                    pass

            elif change.change_type == ChangeType.UPDATE and change.relationship:
                try:
                    self.store.update_relationship(
                        change.relationship_id, change.relationship.properties
                    )
                    self.state.relationship_hashes[change.relationship_id] = change.new_hash or ""
                    counts["updated"] += 1
                except Exception:
                    pass

            elif change.change_type == ChangeType.DELETE and not skip_deletions:
                try:
                    self.store.delete_relationship(change.relationship_id)
                    del self.state.relationship_hashes[change.relationship_id]
                    counts["deleted"] += 1
                except Exception:
                    pass

            elif change.change_type == ChangeType.UNCHANGED:
                counts["unchanged"] += 1

        self.state.total_relationships = len(self.state.relationship_hashes)
        self.state.last_update = datetime.utcnow()

        return counts

    def load_incremental(
        self,
        entities: list[Entity],
        relationships: list[Relationship],
        skip_deletions: bool = False,
    ) -> dict[str, Any]:
        """Perform incremental load of entities and relationships.

        Args:
            entities: Entities to load
            relationships: Relationships to load
            skip_deletions: Whether to skip deletion operations

        Returns:
            Summary of operations performed
        """
        # Detect and apply entity changes
        entity_changes = self.detect_entity_changes(entities)
        entity_counts = self.apply_entity_changes(entity_changes, skip_deletions)

        # Detect and apply relationship changes
        rel_changes = self.detect_relationship_changes(relationships)
        rel_counts = self.apply_relationship_changes(rel_changes, skip_deletions)

        return {
            "entities": entity_counts,
            "relationships": rel_counts,
            "total_entities": self.state.total_entities,
            "total_relationships": self.state.total_relationships,
            "last_update": self.state.last_update.isoformat() if self.state.last_update else None,
        }

    def full_reload(
        self,
        entities: list[Entity],
        relationships: list[Relationship],
    ) -> dict[str, Any]:
        """Perform a full reload, clearing existing data.

        Args:
            entities: Entities to load
            relationships: Relationships to load

        Returns:
            Summary of operations performed
        """
        # Clear existing state
        self.store.clear()
        self.state = LoaderState()

        # Bulk load entities
        entity_ids = self.store.bulk_add_entities(entities)

        # Update state hashes
        for entity in entities:
            if entity.id in entity_ids:
                self.state.entity_hashes[entity.id] = self.compute_entity_hash(entity)

        # Bulk load relationships
        rel_ids = self.store.bulk_add_relationships(relationships)

        # Update state hashes
        for rel in relationships:
            if rel.id in rel_ids:
                self.state.relationship_hashes[rel.id] = self.compute_relationship_hash(rel)

        self.state.total_entities = len(entity_ids)
        self.state.total_relationships = len(rel_ids)
        self.state.last_update = datetime.utcnow()

        return {
            "entities_loaded": len(entity_ids),
            "relationships_loaded": len(rel_ids),
            "total_entities": self.state.total_entities,
            "total_relationships": self.state.total_relationships,
            "last_update": self.state.last_update.isoformat(),
        }

    def get_state_summary(self) -> dict[str, Any]:
        """Get summary of current loader state."""
        return {
            "total_entities": self.state.total_entities,
            "total_relationships": self.state.total_relationships,
            "entity_hashes_tracked": len(self.state.entity_hashes),
            "relationship_hashes_tracked": len(self.state.relationship_hashes),
            "last_update": self.state.last_update.isoformat() if self.state.last_update else None,
        }

    def export_state(self) -> dict[str, Any]:
        """Export loader state for persistence."""
        return self.state.model_dump()

    def import_state(self, state_data: dict[str, Any]) -> None:
        """Import loader state from persisted data."""
        self.state = LoaderState.model_validate(state_data)
