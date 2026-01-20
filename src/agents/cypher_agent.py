"""Self-correcting Cypher query generation agent.

Generates Cypher queries from natural language with automatic
error detection and correction.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.graph.graph_interface import GraphStore
from src.llm.gemini_provider import GeminiProvider
from src.schema.ontology import DomainOntology


class CypherQueryAttempt(BaseModel):
    """Record of a Cypher query attempt."""

    query: str = Field(..., description="The generated Cypher query")
    error: str | None = Field(default=None, description="Error message if failed")
    results: list[dict[str, Any]] = Field(default_factory=list)
    success: bool = Field(default=False)
    reflection: str | None = Field(default=None, description="Reflection on the error")


class CypherGenerationResult(BaseModel):
    """Result of Cypher generation process."""

    final_query: str = Field(..., description="Final working query")
    results: list[dict[str, Any]] = Field(default_factory=list)
    attempts: list[CypherQueryAttempt] = Field(default_factory=list)
    success: bool = Field(default=False)
    fallback_used: bool = Field(default=False)


class CypherAgent:
    """Agent for generating self-correcting Cypher queries.

    Uses LLM to generate Cypher from natural language, with automatic
    error detection and correction through reflection.
    """

    def __init__(
        self,
        llm: GeminiProvider,
        ontology: DomainOntology,
        store: GraphStore,
        max_retries: int = 5,
    ) -> None:
        """Initialize the Cypher agent.

        Args:
            llm: The LLM provider
            ontology: The domain ontology
            store: The graph store for query execution
            max_retries: Maximum retry attempts
        """
        self.llm = llm
        self.ontology = ontology
        self.store = store
        self.max_retries = max_retries

    def _get_schema_context(self) -> str:
        """Generate schema context for prompting."""
        return self.ontology.to_cypher_schema()

    def _get_examples(self) -> list[tuple[str, str]]:
        """Get example question-Cypher pairs."""
        return [
            (
                "What instruments does the straddle strategy trade?",
                """MATCH (s:Strategy {name: 'Straddle'})-[r:TRADES]->(i:Instrument)
RETURN i.name AS instrument, i.symbol AS symbol, r.direction AS direction""",
            ),
            (
                "Find strategies that perform well in high volatility",
                """MATCH (s:Strategy)-[p:PERFORMS_IN]->(m:MarketCondition)
WHERE m.name CONTAINS 'High Volatility' OR m.vix_range CONTAINS '25'
RETURN s.name AS strategy, p.expected_pnl AS expected_pnl, m.name AS condition""",
            ),
            (
                "What affects VIX?",
                """MATCH (i:Instrument {symbol: 'VIX'})<-[r:AFFECTED_BY]-(source)
RETURN source.name AS factor, labels(source) AS type, r.sensitivity AS sensitivity""",
            ),
        ]

    def generate_cypher(
        self,
        question: str,
        hints: dict[str, Any] | None = None,
    ) -> CypherGenerationResult:
        """Generate a Cypher query from natural language.

        Args:
            question: The natural language question
            hints: Optional hints from semantic alignment

        Returns:
            CypherGenerationResult with query and results
        """
        schema_context = self._get_schema_context()
        examples = self._get_examples()

        examples_str = "\n\n".join(
            f"Question: {q}\nCypher:\n{c}" for q, c in examples
        )

        hints_str = ""
        if hints:
            hints_str = f"""
Hints:
- Suggested labels: {hints.get('suggested_labels', [])}
- Suggested relationships: {hints.get('suggested_relationships', [])}
- Property filters: {hints.get('property_filters', {})}
"""

        attempts = []

        for attempt_num in range(self.max_retries):
            # Generate query (with reflection from previous attempts if any)
            reflection_context = ""
            if attempts:
                last_attempt = attempts[-1]
                reflection_context = f"""
Previous attempt failed:
Query: {last_attempt.query}
Error: {last_attempt.error}
Reflection: {last_attempt.reflection}

Generate a corrected query addressing this error.
"""

            system_instruction = f"""You are a Cypher query expert for Neo4j.
Generate valid Cypher queries based on the schema and question.

{schema_context}

Examples:
{examples_str}
{hints_str}
{reflection_context}

Rules:
- Use exact label and relationship names from the schema
- Always include RETURN clause
- Use parameters ($param) for user-provided values where appropriate
- Handle potential null values gracefully
- Return meaningful columns with aliases

Respond with ONLY the Cypher query, no explanations."""

            prompt = f"Generate a Cypher query for: {question}"

            cypher = self.llm.generate(
                prompt,
                system_instruction=system_instruction,
                temperature=0.1 + (0.1 * attempt_num),  # Increase temperature on retries
            )

            # Clean up query
            cypher = self._clean_cypher(cypher)

            # Try to execute
            try:
                results = self.store.execute_query(cypher)

                attempt = CypherQueryAttempt(
                    query=cypher,
                    results=results,
                    success=True,
                )
                attempts.append(attempt)

                return CypherGenerationResult(
                    final_query=cypher,
                    results=results,
                    attempts=attempts,
                    success=True,
                )

            except Exception as e:
                error_msg = str(e)

                # Generate reflection on the error
                reflection = self._reflect_on_error(cypher, error_msg, schema_context)

                attempt = CypherQueryAttempt(
                    query=cypher,
                    error=error_msg,
                    success=False,
                    reflection=reflection,
                )
                attempts.append(attempt)

        # All retries exhausted, use fallback
        return self._fallback_search(question, attempts)

    def _clean_cypher(self, cypher: str) -> str:
        """Clean up generated Cypher query."""
        cypher = cypher.strip()

        # Remove markdown code blocks
        if cypher.startswith("```cypher"):
            cypher = cypher[9:]
        if cypher.startswith("```"):
            cypher = cypher[3:]
        if cypher.endswith("```"):
            cypher = cypher[:-3]

        return cypher.strip()

    def _reflect_on_error(
        self,
        query: str,
        error: str,
        schema: str,
    ) -> str:
        """Generate reflection on query error."""
        prompt = f"""Analyze why this Cypher query failed:

Query: {query}
Error: {error}
Schema: {schema}

Identify the specific issue and suggest how to fix it. Be concise."""

        return self.llm.generate(prompt, temperature=0.3)

    def _fallback_search(
        self,
        question: str,
        attempts: list[CypherQueryAttempt],
    ) -> CypherGenerationResult:
        """Fallback to basic text search when Cypher fails."""
        # Try a simple search query
        fallback_query = """
        MATCH (n)
        WHERE any(prop IN keys(n) WHERE toString(n[prop]) CONTAINS $search_term)
        RETURN n, labels(n) AS types
        LIMIT 10
        """

        # Extract key terms from question
        words = question.lower().split()
        search_terms = [w for w in words if len(w) > 3 and w not in {"what", "which", "does", "that", "have", "with"}]

        results = []
        for term in search_terms[:3]:
            try:
                term_results = self.store.execute_query(
                    fallback_query, {"search_term": term}
                )
                results.extend(term_results)
            except Exception:
                pass

        # Also try entity search
        try:
            entities = self.store.search_entities(question, limit=5)
            for entity in entities:
                results.append({"entity": entity.to_dict(), "types": [entity.node_type]})
        except Exception:
            pass

        return CypherGenerationResult(
            final_query=fallback_query,
            results=results,
            attempts=attempts,
            success=len(results) > 0,
            fallback_used=True,
        )

    def validate_cypher(self, query: str) -> tuple[bool, str | None]:
        """Validate a Cypher query without executing.

        Args:
            query: The Cypher query to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Basic syntax validation
        query_upper = query.upper()

        if "MATCH" not in query_upper and "CREATE" not in query_upper:
            return False, "Query must contain MATCH or CREATE clause"

        if "RETURN" not in query_upper and "DELETE" not in query_upper:
            return False, "Query must contain RETURN or DELETE clause"

        # Check for valid labels against schema
        for label in self.ontology.node_schemas.keys():
            if f":{label}" in query or f":{label.lower()}" in query:
                pass  # Valid label found

        return True, None

    def explain_query(self, query: str) -> str:
        """Generate explanation of a Cypher query.

        Args:
            query: The Cypher query to explain

        Returns:
            Natural language explanation
        """
        prompt = f"""Explain this Cypher query in simple terms:

{query}

Describe:
1. What data it's looking for
2. How it traverses the graph
3. What results it returns"""

        return self.llm.generate(prompt, temperature=0.3)
