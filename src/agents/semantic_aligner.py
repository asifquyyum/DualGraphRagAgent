"""Semantic alignment agent for mapping user queries to ontology terms.

Maps domain slang, abbreviations, and natural language to canonical
ontology concepts for accurate graph queries.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.llm.gemini_provider import GeminiProvider
from src.schema.ontology import DomainOntology


class AlignedTerm(BaseModel):
    """A term aligned to the ontology."""

    original: str = Field(..., description="Original term from query")
    canonical: str = Field(..., description="Canonical ontology term")
    term_type: str = Field(..., description="Type: entity_type, relationship_type, property, concept")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class AlignedQuery(BaseModel):
    """Query with aligned terms and extracted components."""

    original_query: str = Field(..., description="Original user query")
    aligned_terms: list[AlignedTerm] = Field(default_factory=list)
    entity_references: list[str] = Field(
        default_factory=list, description="Entity types or names referenced"
    )
    relationship_references: list[str] = Field(
        default_factory=list, description="Relationship types referenced"
    )
    property_filters: dict[str, Any] = Field(
        default_factory=dict, description="Property filters extracted"
    )
    query_intent: str = Field(
        default="retrieval", description="Intent: retrieval, comparison, causal, counterfactual"
    )
    sub_questions: list[str] = Field(
        default_factory=list, description="Decomposed sub-questions"
    )


class SemanticAligner:
    """Agent for aligning user queries to the domain ontology.

    Uses LLM to understand user intent and map terminology to
    canonical ontology concepts.
    """

    def __init__(
        self,
        llm: GeminiProvider,
        ontology: DomainOntology,
    ) -> None:
        """Initialize the semantic aligner.

        Args:
            llm: The LLM provider
            ontology: The domain ontology
        """
        self.llm = llm
        self.ontology = ontology

    def _get_ontology_context(self) -> str:
        """Generate ontology context for prompting."""
        node_types = list(self.ontology.node_schemas.keys())
        edge_types = list(self.ontology.edge_schemas.keys())

        # Get all properties across schemas
        all_properties = set()
        for schema in self.ontology.node_schemas.values():
            all_properties.update(p.name for p in schema.properties)

        return f"""Domain Ontology:
Entity Types: {', '.join(node_types)}
Relationship Types: {', '.join(edge_types)}
Properties: {', '.join(all_properties)}

Domain Term Mappings:
{self.ontology.domain_terms}"""

    def align_query(self, query: str) -> AlignedQuery:
        """Align a user query to the ontology.

        Args:
            query: The user's natural language query

        Returns:
            AlignedQuery with mapped terms and extracted components
        """
        ontology_context = self._get_ontology_context()

        system_instruction = f"""You are a semantic alignment expert for a quantitative finance knowledge graph.
Map user queries to canonical ontology terms.

{ontology_context}

For each query:
1. Identify domain slang and map to canonical terms
2. Extract entity types being referenced
3. Extract relationship types being discussed
4. Identify any property filters (e.g., "VIX > 25")
5. Determine query intent (retrieval, comparison, causal, counterfactual)
6. Decompose complex queries into sub-questions if needed"""

        prompt = f"""Analyze this query and align it to the ontology:

Query: "{query}"

Respond with JSON:
{{
    "original_query": "{query}",
    "aligned_terms": [
        {{"original": "vol", "canonical": "volatility", "term_type": "concept", "confidence": 0.95}}
    ],
    "entity_references": ["Instrument", "Strategy"],
    "relationship_references": ["TRADES", "AFFECTED_BY"],
    "property_filters": {{"vix_level": "> 25"}},
    "query_intent": "retrieval",
    "sub_questions": ["What instruments are affected?", "How does the strategy perform?"]
}}"""

        try:
            result = self.llm.generate_structured(
                prompt, AlignedQuery, system_instruction=system_instruction
            )
            return result
        except ValueError:
            # Fallback with basic alignment
            return self._basic_alignment(query)

    def _basic_alignment(self, query: str) -> AlignedQuery:
        """Perform basic alignment without LLM."""
        aligned_terms = []
        query_lower = query.lower()

        # Map known domain terms
        for slang, canonical in self.ontology.domain_terms.items():
            if slang in query_lower:
                aligned_terms.append(
                    AlignedTerm(
                        original=slang,
                        canonical=canonical,
                        term_type="concept",
                        confidence=1.0,
                    )
                )

        # Detect entity types
        entity_refs = []
        for node_type in self.ontology.node_schemas.keys():
            if node_type.lower() in query_lower:
                entity_refs.append(node_type)

        # Detect relationship types
        rel_refs = []
        for edge_type in self.ontology.edge_schemas.keys():
            if edge_type.lower().replace("_", " ") in query_lower:
                rel_refs.append(edge_type)

        # Detect query intent
        intent = "retrieval"
        if "what if" in query_lower or "would happen" in query_lower:
            intent = "counterfactual"
        elif "why" in query_lower or "cause" in query_lower:
            intent = "causal"
        elif "compare" in query_lower or "vs" in query_lower:
            intent = "comparison"

        return AlignedQuery(
            original_query=query,
            aligned_terms=aligned_terms,
            entity_references=entity_refs,
            relationship_references=rel_refs,
            query_intent=intent,
        )

    def expand_query(self, aligned_query: AlignedQuery) -> list[str]:
        """Expand an aligned query into search terms.

        Args:
            aligned_query: The aligned query to expand

        Returns:
            List of search terms to use for retrieval
        """
        terms = [aligned_query.original_query]

        # Add canonical terms
        for term in aligned_query.aligned_terms:
            if term.canonical not in terms:
                terms.append(term.canonical)

        # Add entity references
        terms.extend(aligned_query.entity_references)

        return terms

    def generate_cypher_hints(self, aligned_query: AlignedQuery) -> dict[str, Any]:
        """Generate hints for Cypher query generation.

        Args:
            aligned_query: The aligned query

        Returns:
            Dictionary with hints for Cypher generation
        """
        hints = {
            "suggested_labels": aligned_query.entity_references,
            "suggested_relationships": aligned_query.relationship_references,
            "property_filters": aligned_query.property_filters,
            "pattern_type": "simple",
        }

        # Determine pattern complexity
        if len(aligned_query.entity_references) > 2:
            hints["pattern_type"] = "multi_hop"
        elif aligned_query.query_intent == "causal":
            hints["pattern_type"] = "path"

        return hints

    def get_similar_queries(self, query: str, num_examples: int = 3) -> list[str]:
        """Generate similar example queries for few-shot prompting.

        Args:
            query: The original query
            num_examples: Number of examples to generate

        Returns:
            List of similar example queries
        """
        aligned = self.align_query(query)

        # Generate variations based on aligned terms
        examples = []

        if "volatility" in [t.canonical for t in aligned.aligned_terms]:
            examples.extend(
                [
                    "What happens to a straddle when VIX spikes?",
                    "How does implied volatility affect option prices?",
                    "Which strategies benefit from high volatility?",
                ]
            )

        if any(ref in ["Strategy", "Instrument"] for ref in aligned.entity_references):
            examples.extend(
                [
                    "What instruments does the iron condor strategy trade?",
                    "How does the SPX perform in high volatility regimes?",
                    "Which strategies hedge against delta exposure?",
                ]
            )

        return examples[:num_examples]
