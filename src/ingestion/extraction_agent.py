"""Entity and relationship extraction agent.

Extracts structured entities and relationships from document chunks
using LLM with ontology-guided prompting.
"""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field

from src.ingestion.chunker import DocumentChunk
from src.llm.gemini_provider import GeminiProvider
from src.schema.ontology import DomainOntology, Entity, Relationship


class ExtractedEntity(BaseModel):
    """An entity extracted from text."""

    name: str = Field(..., description="Entity name")
    entity_type: str = Field(..., description="Type of entity")
    properties: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source_text: str = Field(default="", description="Text span this was extracted from")


class ExtractedRelationship(BaseModel):
    """A relationship extracted from text."""

    source_name: str = Field(..., description="Source entity name")
    target_name: str = Field(..., description="Target entity name")
    relationship_type: str = Field(..., description="Type of relationship")
    properties: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source_text: str = Field(default="", description="Text span this was extracted from")


class ExtractionResult(BaseModel):
    """Result of extraction from a single chunk."""

    entities: list[ExtractedEntity] = Field(default_factory=list)
    relationships: list[ExtractedRelationship] = Field(default_factory=list)
    chunk_id: str = Field(default="")


class ExtractionAgent:
    """Agent for extracting entities and relationships from text.

    Uses LLM to identify and extract structured information guided
    by the domain ontology.
    """

    def __init__(
        self,
        llm: GeminiProvider,
        ontology: DomainOntology,
    ) -> None:
        """Initialize the extraction agent.

        Args:
            llm: The LLM provider for extraction
            ontology: Domain ontology defining valid entity and relationship types
        """
        self.llm = llm
        self.ontology = ontology
        self._entity_name_to_id: dict[str, str] = {}

    def _get_ontology_context(self) -> str:
        """Generate ontology context for prompting."""
        node_types = list(self.ontology.node_schemas.keys())
        edge_types = list(self.ontology.edge_schemas.keys())

        node_descriptions = []
        for name, schema in self.ontology.node_schemas.items():
            props = [p.name for p in schema.properties]
            node_descriptions.append(f"- {name}: {schema.description} (properties: {props})")

        edge_descriptions = []
        for name, schema in self.ontology.edge_schemas.items():
            edge_descriptions.append(
                f"- {name}: {schema.description} ({schema.source_types} -> {schema.target_types})"
            )

        return f"""Entity Types:
{chr(10).join(node_descriptions)}

Relationship Types:
{chr(10).join(edge_descriptions)}

Domain Terms:
{self.ontology.domain_terms}"""

    def extract_from_chunk(
        self,
        chunk: DocumentChunk,
        include_low_confidence: bool = False,
    ) -> ExtractionResult:
        """Extract entities and relationships from a document chunk.

        Args:
            chunk: The document chunk to extract from
            include_low_confidence: Include extractions with confidence < 0.5

        Returns:
            ExtractionResult with entities and relationships
        """
        ontology_context = self._get_ontology_context()

        # Get valid types explicitly for the prompt
        valid_entity_types = list(self.ontology.node_schemas.keys())
        valid_rel_types = list(self.ontology.edge_schemas.keys())

        system_instruction = f"""You are an expert at extracting structured information from text
for building knowledge graphs. Use the provided ontology to guide extraction.

{ontology_context}

IMPORTANT: You must ONLY use these entity_type values: {', '.join(valid_entity_types)}
IMPORTANT: You must ONLY use these relationship_type values: {', '.join(valid_rel_types)}

Extract entities that match the defined types. Map domain slang to canonical terms.
For each entity, identify relevant properties based on the schema.
For relationships, ensure source and target types match the schema constraints.

Assign confidence scores:
- 1.0: Explicitly stated
- 0.8: Strongly implied
- 0.5: Inferred with some uncertainty
- 0.3: Speculative"""

        prompt = f"""Extract all entities and relationships from this text:

{chunk.text}

Respond with valid JSON:
{{
    "entities": [
        {{"name": "...", "entity_type": "Concept", "properties": {{}}, "confidence": 0.9, "source_text": "..."}}
    ],
    "relationships": [
        {{"source_name": "...", "target_name": "...", "relationship_type": "RELATES_TO", "properties": {{}}, "confidence": 0.8, "source_text": "..."}}
    ]
}}

CRITICAL: entity_type must be one of: {', '.join(valid_entity_types)}
CRITICAL: relationship_type must be one of: {', '.join(valid_rel_types)}
Ensure relationship source/target names match extracted entity names."""

        try:
            result = self.llm.generate_structured(
                prompt, ExtractionResult, system_instruction=system_instruction, max_retries=5
            )
            result.chunk_id = chunk.id

            # Filter low confidence if requested
            if not include_low_confidence:
                result.entities = [e for e in result.entities if e.confidence >= 0.5]
                result.relationships = [r for r in result.relationships if r.confidence >= 0.5]

            # Validate against ontology
            result = self._validate_extraction(result)

            return result

        except ValueError as e:
            # Return empty result on extraction failure
            return ExtractionResult(chunk_id=chunk.id)

    def _validate_extraction(self, result: ExtractionResult) -> ExtractionResult:
        """Validate extracted entities and relationships against ontology."""
        valid_node_types = set(self.ontology.node_schemas.keys())
        valid_edge_types = set(self.ontology.edge_schemas.keys())

        # Filter entities with valid types
        valid_entities = []
        for entity in result.entities:
            if entity.entity_type in valid_node_types:
                valid_entities.append(entity)
            else:
                # Try to find closest match
                for node_type in valid_node_types:
                    if entity.entity_type.lower() in node_type.lower() or node_type.lower() in entity.entity_type.lower():
                        entity.entity_type = node_type
                        valid_entities.append(entity)
                        break

        result.entities = valid_entities

        # Filter relationships with valid types and entities
        entity_names = {e.name.lower() for e in result.entities}
        valid_relationships = []

        for rel in result.relationships:
            if rel.relationship_type not in valid_edge_types:
                continue

            # Check if source and target entities exist
            if (
                rel.source_name.lower() in entity_names
                and rel.target_name.lower() in entity_names
            ):
                valid_relationships.append(rel)

        result.relationships = valid_relationships

        return result

    def extract_from_chunks(
        self,
        chunks: list[DocumentChunk],
        deduplicate: bool = True,
    ) -> tuple[list[Entity], list[Relationship]]:
        """Extract from multiple chunks and return graph-ready objects.

        Args:
            chunks: List of document chunks to process
            deduplicate: Whether to deduplicate entities by name

        Returns:
            Tuple of (entities, relationships) ready for graph storage
        """
        all_extracted_entities: dict[str, ExtractedEntity] = {}
        all_extracted_relationships: list[ExtractedRelationship] = []

        for chunk in chunks:
            result = self.extract_from_chunk(chunk)

            # Collect entities (deduplicate by name if requested)
            for entity in result.entities:
                key = entity.name.lower() if deduplicate else f"{entity.name}_{chunk.id}"

                if key not in all_extracted_entities:
                    all_extracted_entities[key] = entity
                else:
                    # Merge properties and keep higher confidence
                    existing = all_extracted_entities[key]
                    existing.properties.update(entity.properties)
                    existing.confidence = max(existing.confidence, entity.confidence)

            # Collect relationships
            all_extracted_relationships.extend(result.relationships)

        # Convert to graph objects
        entities = []
        entity_name_to_id: dict[str, str] = {}

        for extracted in all_extracted_entities.values():
            entity_id = str(uuid.uuid4())[:8]
            entity_name_to_id[extracted.name.lower()] = entity_id

            entities.append(
                Entity(
                    id=entity_id,
                    node_type=extracted.entity_type,
                    properties={"name": extracted.name, **extracted.properties},
                    extraction_confidence=extracted.confidence,
                )
            )

        relationships = []
        for extracted in all_extracted_relationships:
            source_id = entity_name_to_id.get(extracted.source_name.lower())
            target_id = entity_name_to_id.get(extracted.target_name.lower())

            if source_id and target_id:
                relationships.append(
                    Relationship(
                        id=str(uuid.uuid4())[:8],
                        edge_type=extracted.relationship_type,
                        source_id=source_id,
                        target_id=target_id,
                        properties=extracted.properties,
                        extraction_confidence=extracted.confidence,
                    )
                )

        self._entity_name_to_id = entity_name_to_id
        return entities, relationships

    def extract_with_coreference(
        self,
        chunks: list[DocumentChunk],
    ) -> tuple[list[Entity], list[Relationship]]:
        """Extract with coreference resolution across chunks.

        Uses LLM to identify when different mentions refer to the same entity.

        Args:
            chunks: List of document chunks to process

        Returns:
            Tuple of (entities, relationships) with coreferences resolved
        """
        # First pass: extract all entities
        entities, relationships = self.extract_from_chunks(chunks, deduplicate=False)

        if len(entities) <= 1:
            return entities, relationships

        # Group potential duplicates by type
        by_type: dict[str, list[Entity]] = {}
        for entity in entities:
            by_type.setdefault(entity.node_type, []).append(entity)

        # Ask LLM to identify coreferences within each type
        merged_entities = []
        id_mapping: dict[str, str] = {}

        for node_type, type_entities in by_type.items():
            if len(type_entities) <= 1:
                merged_entities.extend(type_entities)
                for e in type_entities:
                    id_mapping[e.id] = e.id
                continue

            # Ask LLM to identify same entities
            names = [e.properties.get("name", e.id) for e in type_entities]

            prompt = f"""These are potential {node_type} entities extracted from documents:
{names}

Identify which names refer to the same entity. Group them together.
Respond with JSON array of arrays, where each inner array contains names that refer to the same entity:
[["name1", "name2"], ["name3"], ...]"""

            try:
                response = self.llm.generate(prompt, temperature=0.1)
                import json

                # Clean and parse response
                cleaned = response.strip()
                if cleaned.startswith("```json"):
                    cleaned = cleaned[7:]
                if cleaned.startswith("```"):
                    cleaned = cleaned[3:]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]

                groups = json.loads(cleaned.strip())

                # Merge entities in each group
                name_to_entity = {e.properties.get("name", ""): e for e in type_entities}

                for group in groups:
                    group_entities = [name_to_entity.get(n) for n in group if n in name_to_entity]
                    group_entities = [e for e in group_entities if e]

                    if group_entities:
                        # Merge into first entity
                        merged = group_entities[0]
                        for other in group_entities[1:]:
                            merged.properties.update(other.properties)
                            merged.extraction_confidence = max(
                                merged.extraction_confidence, other.extraction_confidence
                            )
                            id_mapping[other.id] = merged.id

                        merged_entities.append(merged)
                        id_mapping[merged.id] = merged.id

            except (json.JSONDecodeError, ValueError):
                # Fallback: keep all entities
                merged_entities.extend(type_entities)
                for e in type_entities:
                    id_mapping[e.id] = e.id

        # Update relationship IDs
        updated_relationships = []
        for rel in relationships:
            new_source = id_mapping.get(rel.source_id, rel.source_id)
            new_target = id_mapping.get(rel.target_id, rel.target_id)

            # Skip self-loops created by merging
            if new_source != new_target:
                rel.source_id = new_source
                rel.target_id = new_target
                updated_relationships.append(rel)

        return merged_entities, updated_relationships

    def get_entity_id_by_name(self, name: str) -> str | None:
        """Get entity ID by name from last extraction."""
        return self._entity_name_to_id.get(name.lower())
