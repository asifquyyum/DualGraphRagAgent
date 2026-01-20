"""Meta agent for schema generation from documents.

Analyzes document content to propose and refine the domain ontology,
identifying entity types, relationship types, and domain terminology.
"""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field

from src.llm.gemini_provider import GeminiProvider
from src.schema.ontology import (
    DomainOntology,
    EdgeSchema,
    NodeSchema,
    PropertySchema,
    PropertyType,
)


class ProposedNodeType(BaseModel):
    """A proposed node type from document analysis."""

    name: str = Field(..., description="Name of the node type")
    description: str = Field(..., description="Description of what this node represents")
    example_instances: list[str] = Field(
        default_factory=list, description="Example instances found in documents"
    )
    suggested_properties: list[str] = Field(
        default_factory=list, description="Suggested property names"
    )


class ProposedEdgeType(BaseModel):
    """A proposed edge type from document analysis."""

    name: str = Field(..., description="Name of the relationship type")
    description: str = Field(..., description="Description of the relationship")
    source_types: list[str] = Field(..., description="Types that can be sources")
    target_types: list[str] = Field(..., description="Types that can be targets")
    example_instances: list[str] = Field(
        default_factory=list, description="Example relationships found"
    )


class ProposedDomainTerm(BaseModel):
    """A proposed domain term mapping."""

    slang: str = Field(..., description="Domain slang or abbreviation")
    canonical: str = Field(..., description="Canonical term")
    context: str = Field(default="", description="Context where this term is used")


class SchemaProposal(BaseModel):
    """Complete schema proposal from document analysis."""

    node_types: list[ProposedNodeType] = Field(default_factory=list)
    edge_types: list[ProposedEdgeType] = Field(default_factory=list)
    domain_terms: list[ProposedDomainTerm] = Field(default_factory=list)
    domain_summary: str = Field(default="", description="Summary of the domain")


class MetaAgent:
    """Agent for generating and refining ontology schemas from documents.

    Uses LLM to analyze document content and propose appropriate
    node types, relationship types, and domain terminology.
    """

    def __init__(self, llm: GeminiProvider) -> None:
        """Initialize the meta agent.

        Args:
            llm: The LLM provider for analysis
        """
        self.llm = llm

    def analyze_documents(
        self,
        document_chunks: list[str],
        existing_ontology: DomainOntology | None = None,
    ) -> SchemaProposal:
        """Analyze documents to propose a schema.

        Args:
            document_chunks: List of document text chunks to analyze
            existing_ontology: Optional existing ontology to extend

        Returns:
            Schema proposal with node types, edge types, and domain terms
        """
        # Combine chunks for analysis (limit to avoid token limits)
        sample_text = "\n\n---\n\n".join(document_chunks[:10])

        existing_context = ""
        if existing_ontology:
            existing_context = f"""
Existing ontology to extend:
Node types: {list(existing_ontology.node_schemas.keys())}
Edge types: {list(existing_ontology.edge_schemas.keys())}
Domain terms: {list(existing_ontology.domain_terms.keys())}
"""

        system_instruction = """You are an expert at knowledge graph schema design.
Analyze the provided documents and propose an ontology schema that captures
the key entities, relationships, and domain terminology.

For a quantitative finance domain, look for:
- Financial instruments (stocks, options, futures, ETFs)
- Trading strategies and patterns
- Market conditions and regimes
- Risk factors and Greeks
- Events (economic, corporate)
- Concepts and principles

Propose specific, well-defined types rather than generic ones."""

        prompt = f"""{existing_context}

Analyze these document excerpts and propose a knowledge graph schema:

{sample_text}

Identify:
1. Entity types (nodes) with descriptions and example instances
2. Relationship types (edges) with source/target constraints
3. Domain-specific terminology and abbreviations

Respond with JSON matching this structure:
{{
    "node_types": [
        {{"name": "TypeName", "description": "...", "example_instances": ["..."], "suggested_properties": ["..."]}}
    ],
    "edge_types": [
        {{"name": "RELATIONSHIP_NAME", "description": "...", "source_types": ["..."], "target_types": ["..."], "example_instances": ["..."]}}
    ],
    "domain_terms": [
        {{"slang": "vol", "canonical": "volatility", "context": "..."}}
    ],
    "domain_summary": "Brief summary of the domain covered"
}}"""

        try:
            proposal = self.llm.generate_structured(
                prompt, SchemaProposal, system_instruction=system_instruction
            )
            return proposal
        except ValueError:
            # Fallback to basic proposal
            return SchemaProposal(
                domain_summary="Unable to generate detailed proposal"
            )

    def refine_schema(
        self,
        proposal: SchemaProposal,
        feedback: str,
    ) -> SchemaProposal:
        """Refine a schema proposal based on feedback.

        Args:
            proposal: Current schema proposal
            feedback: User or system feedback for refinement

        Returns:
            Refined schema proposal
        """
        system_instruction = """You are refining a knowledge graph schema based on feedback.
Adjust the proposed types, relationships, and terms according to the feedback
while maintaining consistency and domain relevance."""

        prompt = f"""Current schema proposal:
{proposal.model_dump_json(indent=2)}

Feedback:
{feedback}

Provide a refined schema proposal addressing the feedback.
Respond with JSON matching the same structure."""

        try:
            refined = self.llm.generate_structured(
                prompt, SchemaProposal, system_instruction=system_instruction
            )
            return refined
        except ValueError:
            return proposal

    def proposal_to_ontology(
        self,
        proposal: SchemaProposal,
        ontology_name: str = "GeneratedOntology",
    ) -> DomainOntology:
        """Convert a schema proposal to a DomainOntology.

        Args:
            proposal: The schema proposal to convert
            ontology_name: Name for the new ontology

        Returns:
            A DomainOntology instance
        """
        ontology = DomainOntology(
            name=ontology_name,
            description=proposal.domain_summary,
        )

        # Convert node types
        for node_type in proposal.node_types:
            properties = []
            for prop_name in node_type.suggested_properties:
                properties.append(
                    PropertySchema(
                        name=prop_name,
                        property_type=PropertyType.STRING,
                        required=prop_name == "name",
                    )
                )

            # Ensure name property exists
            if not any(p.name == "name" for p in properties):
                properties.insert(
                    0,
                    PropertySchema(
                        name="name",
                        property_type=PropertyType.STRING,
                        required=True,
                    ),
                )

            ontology.add_node_schema(
                NodeSchema(
                    name=node_type.name,
                    description=node_type.description,
                    properties=properties,
                )
            )

        # Convert edge types
        for edge_type in proposal.edge_types:
            ontology.add_edge_schema(
                EdgeSchema(
                    name=edge_type.name,
                    description=edge_type.description,
                    source_types=edge_type.source_types,
                    target_types=edge_type.target_types,
                )
            )

        # Convert domain terms
        for term in proposal.domain_terms:
            ontology.domain_terms[term.slang.lower()] = term.canonical

        return ontology

    def merge_ontologies(
        self,
        base: DomainOntology,
        extension: DomainOntology,
    ) -> DomainOntology:
        """Merge two ontologies, with extension taking precedence.

        Args:
            base: The base ontology
            extension: The ontology to merge in

        Returns:
            Merged ontology
        """
        merged = DomainOntology(
            name=f"{base.name}_extended",
            description=f"{base.description}\n\nExtended with: {extension.description}",
        )

        # Copy base schemas
        for name, schema in base.node_schemas.items():
            merged.add_node_schema(schema)

        for name, schema in base.edge_schemas.items():
            merged.add_edge_schema(schema)

        merged.domain_terms.update(base.domain_terms)

        # Merge extension (overrides base)
        for name, schema in extension.node_schemas.items():
            merged.add_node_schema(schema)

        for name, schema in extension.edge_schemas.items():
            merged.add_edge_schema(schema)

        merged.domain_terms.update(extension.domain_terms)

        return merged

    def validate_ontology(
        self,
        ontology: DomainOntology,
    ) -> list[str]:
        """Validate an ontology for consistency.

        Args:
            ontology: The ontology to validate

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check edge schemas reference valid node types
        valid_node_types = set(ontology.node_schemas.keys())

        for edge_name, edge_schema in ontology.edge_schemas.items():
            for source_type in edge_schema.source_types:
                if source_type not in valid_node_types:
                    errors.append(
                        f"Edge '{edge_name}' references unknown source type '{source_type}'"
                    )

            for target_type in edge_schema.target_types:
                if target_type not in valid_node_types:
                    errors.append(
                        f"Edge '{edge_name}' references unknown target type '{target_type}'"
                    )

        # Check for duplicate property names within schemas
        for node_name, node_schema in ontology.node_schemas.items():
            prop_names = [p.name for p in node_schema.properties]
            if len(prop_names) != len(set(prop_names)):
                errors.append(f"Node '{node_name}' has duplicate property names")

        return errors

    def suggest_schema_improvements(
        self,
        ontology: DomainOntology,
        sample_entities: list[dict[str, Any]],
    ) -> list[str]:
        """Suggest improvements to an ontology based on sample data.

        Args:
            ontology: Current ontology
            sample_entities: Sample entity data to analyze

        Returns:
            List of improvement suggestions
        """
        system_instruction = """Analyze the ontology and sample data to suggest improvements.
Look for:
- Missing entity types that appear in the data
- Missing relationship types
- Properties that should be added
- Domain terms that could be added"""

        prompt = f"""Ontology:
{ontology.to_cypher_schema()}

Sample entities:
{sample_entities[:20]}

Suggest specific improvements to the ontology."""

        response = self.llm.generate(prompt, system_instruction=system_instruction)

        # Parse suggestions from response
        suggestions = []
        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("-") or line.startswith("*") or line.startswith("•"):
                suggestions.append(line.lstrip("-*• "))
            elif line and not line.endswith(":"):
                suggestions.append(line)

        return suggestions[:10]  # Limit suggestions
