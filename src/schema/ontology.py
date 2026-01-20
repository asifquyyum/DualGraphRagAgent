"""Domain ontology models for the Graph RAG knowledge graph.

This module defines the schema for nodes (entities) and edges (relationships)
in the knowledge graph, with a focus on quantitative finance domain concepts.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class PropertyType(str, Enum):
    """Supported property types for node and edge attributes."""

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    LIST_STRING = "list[string]"
    LIST_FLOAT = "list[float]"


class PropertySchema(BaseModel):
    """Schema definition for a single property."""

    name: str = Field(..., description="Property name")
    property_type: PropertyType = Field(..., description="Data type of the property")
    required: bool = Field(default=False, description="Whether this property is required")
    description: str = Field(default="", description="Human-readable description")
    default: Any = Field(default=None, description="Default value if not provided")


class NodeSchema(BaseModel):
    """Schema definition for a node type in the knowledge graph.

    Defines what entity types can exist and their allowed properties.
    """

    name: str = Field(..., description="Node type name (e.g., 'Instrument', 'Strategy')")
    description: str = Field(default="", description="Human-readable description of this node type")
    properties: list[PropertySchema] = Field(
        default_factory=list, description="Properties allowed on this node type"
    )
    primary_key: str = Field(
        default="id", description="Property used as unique identifier"
    )
    labels: list[str] = Field(
        default_factory=list,
        description="Additional labels for this node type (Neo4j inheritance)",
    )

    def get_required_properties(self) -> list[str]:
        """Return names of all required properties."""
        return [p.name for p in self.properties if p.required]

    def validate_properties(self, props: dict[str, Any]) -> list[str]:
        """Validate properties against schema, return list of errors."""
        errors = []
        prop_names = {p.name for p in self.properties}

        for required in self.get_required_properties():
            if required not in props:
                errors.append(f"Missing required property: {required}")

        for key in props:
            if key not in prop_names and key != self.primary_key:
                errors.append(f"Unknown property: {key}")

        return errors


class EdgeSchema(BaseModel):
    """Schema definition for an edge type in the knowledge graph.

    Defines what relationship types can exist between node types.
    """

    name: str = Field(..., description="Edge type name (e.g., 'TRADES', 'HEDGES')")
    description: str = Field(
        default="", description="Human-readable description of this relationship"
    )
    source_types: list[str] = Field(
        ..., description="Allowed source node types"
    )
    target_types: list[str] = Field(
        ..., description="Allowed target node types"
    )
    properties: list[PropertySchema] = Field(
        default_factory=list, description="Properties allowed on this edge type"
    )
    directed: bool = Field(
        default=True, description="Whether this relationship is directed"
    )
    cardinality: str = Field(
        default="many-to-many",
        description="Relationship cardinality (one-to-one, one-to-many, many-to-many)",
    )

    def is_valid_connection(self, source_type: str, target_type: str) -> bool:
        """Check if this edge can connect the given node types."""
        return source_type in self.source_types and target_type in self.target_types


class Entity(BaseModel):
    """A concrete entity instance in the knowledge graph."""

    id: str = Field(..., description="Unique identifier")
    node_type: str = Field(..., description="Type of this entity (references NodeSchema)")
    properties: dict[str, Any] = Field(
        default_factory=dict, description="Entity properties"
    )
    source_document: str | None = Field(
        default=None, description="Source document this entity was extracted from"
    )
    source_chunk_id: str | None = Field(
        default=None, description="Chunk ID within source document"
    )
    extraction_confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence score from extraction"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for graph storage."""
        return {
            "id": self.id,
            "node_type": self.node_type,
            **self.properties,
            "_source_document": self.source_document,
            "_source_chunk_id": self.source_chunk_id,
            "_extraction_confidence": self.extraction_confidence,
            "_created_at": self.created_at.isoformat(),
            "_updated_at": self.updated_at.isoformat(),
        }


class Relationship(BaseModel):
    """A concrete relationship instance in the knowledge graph."""

    id: str = Field(..., description="Unique identifier")
    edge_type: str = Field(..., description="Type of this relationship (references EdgeSchema)")
    source_id: str = Field(..., description="ID of source entity")
    target_id: str = Field(..., description="ID of target entity")
    properties: dict[str, Any] = Field(
        default_factory=dict, description="Relationship properties"
    )
    source_document: str | None = Field(
        default=None, description="Source document this relationship was extracted from"
    )
    source_chunk_id: str | None = Field(
        default=None, description="Chunk ID within source document"
    )
    extraction_confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence score from extraction"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for graph storage."""
        return {
            "id": self.id,
            "edge_type": self.edge_type,
            "source_id": self.source_id,
            "target_id": self.target_id,
            **self.properties,
            "_source_document": self.source_document,
            "_source_chunk_id": self.source_chunk_id,
            "_extraction_confidence": self.extraction_confidence,
            "_created_at": self.created_at.isoformat(),
            "_updated_at": self.updated_at.isoformat(),
        }


class DomainOntology(BaseModel):
    """Complete ontology definition for the knowledge graph.

    Contains all node schemas and edge schemas that define what entities
    and relationships can exist in the graph.
    """

    name: str = Field(..., description="Name of this ontology")
    version: str = Field(default="1.0.0", description="Ontology version")
    description: str = Field(default="", description="Description of this ontology")
    node_schemas: dict[str, NodeSchema] = Field(
        default_factory=dict, description="Node type definitions keyed by name"
    )
    edge_schemas: dict[str, EdgeSchema] = Field(
        default_factory=dict, description="Edge type definitions keyed by name"
    )
    domain_terms: dict[str, str] = Field(
        default_factory=dict,
        description="Domain-specific term mappings (slang -> canonical)",
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def add_node_schema(self, schema: NodeSchema) -> None:
        """Add a node schema to the ontology."""
        self.node_schemas[schema.name] = schema
        self.updated_at = datetime.utcnow()

    def add_edge_schema(self, schema: EdgeSchema) -> None:
        """Add an edge schema to the ontology."""
        self.edge_schemas[schema.name] = schema
        self.updated_at = datetime.utcnow()

    def get_node_schema(self, name: str) -> NodeSchema | None:
        """Get a node schema by name."""
        return self.node_schemas.get(name)

    def get_edge_schema(self, name: str) -> EdgeSchema | None:
        """Get an edge schema by name."""
        return self.edge_schemas.get(name)

    def validate_entity(self, entity: Entity) -> list[str]:
        """Validate an entity against the ontology."""
        schema = self.get_node_schema(entity.node_type)
        if schema is None:
            return [f"Unknown node type: {entity.node_type}"]
        return schema.validate_properties(entity.properties)

    def validate_relationship(self, relationship: Relationship) -> list[str]:
        """Validate a relationship against the ontology."""
        schema = self.get_edge_schema(relationship.edge_type)
        if schema is None:
            return [f"Unknown edge type: {relationship.edge_type}"]
        return []

    def get_canonical_term(self, term: str) -> str:
        """Map domain slang to canonical term."""
        return self.domain_terms.get(term.lower(), term)

    def to_cypher_schema(self) -> str:
        """Generate Cypher schema description for LLM context."""
        lines = ["// Graph Schema", "// Node Types:"]

        for name, schema in self.node_schemas.items():
            props = ", ".join(
                f"{p.name}: {p.property_type.value}" for p in schema.properties
            )
            lines.append(f"(:{name} {{{props}}})")

        lines.append("\n// Relationship Types:")
        for name, schema in self.edge_schemas.items():
            sources = "|".join(schema.source_types)
            targets = "|".join(schema.target_types)
            props = ", ".join(
                f"{p.name}: {p.property_type.value}" for p in schema.properties
            )
            prop_str = f" {{{props}}}" if props else ""
            lines.append(f"(:{sources})-[:{name}{prop_str}]->(:{targets})")

        return "\n".join(lines)


def create_quant_finance_ontology() -> DomainOntology:
    """Create a default ontology for quantitative finance domain."""
    ontology = DomainOntology(
        name="QuantFinanceOntology",
        version="1.0.0",
        description="Ontology for quantitative finance and options trading research",
    )

    # Node schemas
    ontology.add_node_schema(
        NodeSchema(
            name="Instrument",
            description="A tradeable financial instrument",
            properties=[
                PropertySchema(name="symbol", property_type=PropertyType.STRING, required=True),
                PropertySchema(name="name", property_type=PropertyType.STRING),
                PropertySchema(name="asset_class", property_type=PropertyType.STRING),
                PropertySchema(name="exchange", property_type=PropertyType.STRING),
            ],
        )
    )

    ontology.add_node_schema(
        NodeSchema(
            name="Strategy",
            description="A trading or hedging strategy",
            properties=[
                PropertySchema(name="name", property_type=PropertyType.STRING, required=True),
                PropertySchema(name="description", property_type=PropertyType.STRING),
                PropertySchema(name="strategy_type", property_type=PropertyType.STRING),
                PropertySchema(name="risk_profile", property_type=PropertyType.STRING),
            ],
        )
    )

    ontology.add_node_schema(
        NodeSchema(
            name="MarketCondition",
            description="A market regime or condition",
            properties=[
                PropertySchema(name="name", property_type=PropertyType.STRING, required=True),
                PropertySchema(name="description", property_type=PropertyType.STRING),
                PropertySchema(name="vix_range", property_type=PropertyType.STRING),
                PropertySchema(name="term_structure", property_type=PropertyType.STRING),
            ],
        )
    )

    ontology.add_node_schema(
        NodeSchema(
            name="RiskFactor",
            description="A risk factor affecting positions",
            properties=[
                PropertySchema(name="name", property_type=PropertyType.STRING, required=True),
                PropertySchema(name="greek", property_type=PropertyType.STRING),
                PropertySchema(name="description", property_type=PropertyType.STRING),
            ],
        )
    )

    ontology.add_node_schema(
        NodeSchema(
            name="Event",
            description="A market event (FOMC, earnings, etc.)",
            properties=[
                PropertySchema(name="name", property_type=PropertyType.STRING, required=True),
                PropertySchema(name="event_type", property_type=PropertyType.STRING),
                PropertySchema(name="frequency", property_type=PropertyType.STRING),
                PropertySchema(name="typical_impact", property_type=PropertyType.STRING),
            ],
        )
    )

    ontology.add_node_schema(
        NodeSchema(
            name="Concept",
            description="A domain concept or principle",
            properties=[
                PropertySchema(name="name", property_type=PropertyType.STRING, required=True),
                PropertySchema(name="definition", property_type=PropertyType.STRING),
                PropertySchema(name="category", property_type=PropertyType.STRING),
            ],
        )
    )

    # Edge schemas
    ontology.add_edge_schema(
        EdgeSchema(
            name="TRADES",
            description="Strategy trades an instrument",
            source_types=["Strategy"],
            target_types=["Instrument"],
            properties=[
                PropertySchema(name="direction", property_type=PropertyType.STRING),
                PropertySchema(name="typical_size", property_type=PropertyType.STRING),
            ],
        )
    )

    ontology.add_edge_schema(
        EdgeSchema(
            name="HEDGES",
            description="Strategy hedges a risk factor",
            source_types=["Strategy"],
            target_types=["RiskFactor", "Instrument"],
            properties=[
                PropertySchema(name="hedge_ratio", property_type=PropertyType.FLOAT),
            ],
        )
    )

    ontology.add_edge_schema(
        EdgeSchema(
            name="PERFORMS_IN",
            description="Strategy performance in market condition",
            source_types=["Strategy"],
            target_types=["MarketCondition"],
            properties=[
                PropertySchema(name="expected_pnl", property_type=PropertyType.STRING),
                PropertySchema(name="win_rate", property_type=PropertyType.FLOAT),
            ],
        )
    )

    ontology.add_edge_schema(
        EdgeSchema(
            name="AFFECTED_BY",
            description="Instrument/Strategy affected by risk factor or event",
            source_types=["Instrument", "Strategy"],
            target_types=["RiskFactor", "Event"],
            properties=[
                PropertySchema(name="sensitivity", property_type=PropertyType.STRING),
                PropertySchema(name="direction", property_type=PropertyType.STRING),
            ],
        )
    )

    ontology.add_edge_schema(
        EdgeSchema(
            name="INDICATES",
            description="Condition indicates another condition or event",
            source_types=["MarketCondition", "RiskFactor"],
            target_types=["MarketCondition", "Event"],
            properties=[
                PropertySchema(name="correlation", property_type=PropertyType.FLOAT),
            ],
        )
    )

    ontology.add_edge_schema(
        EdgeSchema(
            name="RELATED_TO",
            description="General relationship between concepts",
            source_types=["Concept", "Instrument", "Strategy", "RiskFactor"],
            target_types=["Concept", "Instrument", "Strategy", "RiskFactor"],
            properties=[
                PropertySchema(name="relationship_type", property_type=PropertyType.STRING),
            ],
        )
    )

    # Domain term mappings
    ontology.domain_terms = {
        "vol": "volatility",
        "iv": "implied_volatility",
        "rv": "realized_volatility",
        "vrp": "volatility_risk_premium",
        "spx": "S&P 500 Index",
        "vix": "CBOE Volatility Index",
        "contango": "futures_contango",
        "backwardation": "futures_backwardation",
        "gamma": "gamma_exposure",
        "delta": "delta_exposure",
        "theta": "theta_decay",
        "vega": "vega_exposure",
        "straddle": "straddle_strategy",
        "strangle": "strangle_strategy",
        "iron condor": "iron_condor_strategy",
        "butterfly": "butterfly_strategy",
        "fomc": "federal_open_market_committee",
        "opex": "options_expiration",
    }

    return ontology
