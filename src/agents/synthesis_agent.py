"""Synthesis agent for generating final answers.

Synthesizes information from beliefs, evidence, and graph data
into coherent, cited responses.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.llm.gemini_provider import GeminiProvider
from src.schema.world_state import Belief, BeliefStatus, WorldState


class Citation(BaseModel):
    """A citation to supporting evidence."""

    source_type: str = Field(..., description="Type: belief, entity, relationship, document")
    source_id: str = Field(..., description="ID of the source")
    content: str = Field(..., description="Cited content or description")
    confidence: float = Field(default=1.0)


class SynthesizedAnswer(BaseModel):
    """A synthesized answer with citations."""

    answer: str = Field(..., description="The synthesized answer text")
    confidence: float = Field(default=0.5, description="Overall confidence in the answer")
    citations: list[Citation] = Field(default_factory=list)
    uncertainties_noted: list[str] = Field(
        default_factory=list, description="Uncertainties that affect the answer"
    )
    follow_up_questions: list[str] = Field(
        default_factory=list, description="Suggested follow-up questions"
    )


class SynthesisAgent:
    """Agent for synthesizing final answers from world state.

    Combines confirmed beliefs, evidence, and graph data into
    coherent responses with proper citations.
    """

    def __init__(self, llm: GeminiProvider) -> None:
        """Initialize the synthesis agent.

        Args:
            llm: The LLM provider
        """
        self.llm = llm

    def synthesize(
        self,
        world_state: WorldState,
        additional_context: dict[str, Any] | None = None,
    ) -> SynthesizedAnswer:
        """Synthesize a final answer from the world state.

        Args:
            world_state: The current epistemic state
            additional_context: Optional additional context from traversal

        Returns:
            SynthesizedAnswer with citations
        """
        # Gather evidence
        confirmed_beliefs = world_state.get_confirmed_beliefs()
        high_confidence_beliefs = world_state.get_high_confidence_beliefs(0.6)
        unresolved_uncertainties = world_state.get_unresolved_uncertainties()

        # Build context for synthesis
        context = self._build_synthesis_context(
            world_state, confirmed_beliefs, additional_context
        )

        system_instruction = """You are a quantitative finance expert synthesizing information
from a knowledge graph to answer user queries.

Guidelines:
1. Base your answer on the confirmed beliefs and evidence provided
2. Clearly indicate confidence levels and uncertainties
3. Use specific examples and data points when available
4. If the evidence is incomplete, acknowledge this
5. Suggest follow-up questions for unclear areas
6. Be concise but comprehensive"""

        prompt = f"""Synthesize an answer to this query based on the gathered evidence:

Query: {world_state.original_query}

{context}

Provide a comprehensive answer that:
1. Directly addresses the query
2. Cites specific evidence
3. Notes any remaining uncertainties
4. Suggests follow-up questions if relevant"""

        answer_text = self.llm.generate(prompt, system_instruction=system_instruction)

        # Build citations from beliefs
        citations = self._build_citations(confirmed_beliefs, high_confidence_beliefs)

        # Calculate overall confidence
        confidence = self._calculate_confidence(world_state, confirmed_beliefs)

        # Extract uncertainty notes
        uncertainty_notes = [u.description for u in unresolved_uncertainties[:3]]

        # Generate follow-up questions
        follow_ups = self._generate_follow_ups(world_state, answer_text)

        return SynthesizedAnswer(
            answer=answer_text,
            confidence=confidence,
            citations=citations,
            uncertainties_noted=uncertainty_notes,
            follow_up_questions=follow_ups,
        )

    def _build_synthesis_context(
        self,
        world_state: WorldState,
        confirmed_beliefs: list[Belief],
        additional_context: dict[str, Any] | None,
    ) -> str:
        """Build context string for synthesis."""
        sections = []

        # Confirmed beliefs
        if confirmed_beliefs:
            beliefs_text = "\n".join(
                f"- {b.content} (confidence: {b.confidence:.0%})"
                for b in confirmed_beliefs
            )
            sections.append(f"Confirmed Facts:\n{beliefs_text}")

        # High confidence inferences
        inferred = [
            b
            for b in world_state.beliefs.values()
            if b.status == BeliefStatus.INFERRED and b.confidence >= 0.7
        ]
        if inferred:
            inferred_text = "\n".join(
                f"- {b.content} (inferred, confidence: {b.confidence:.0%})"
                for b in inferred[:5]
            )
            sections.append(f"Inferred Information:\n{inferred_text}")

        # Additional context
        if additional_context:
            if "entities" in additional_context:
                entities = additional_context["entities"][:5]
                entities_text = "\n".join(
                    f"- {e.get('type', 'Entity')}: {e.get('properties', {}).get('name', e.get('id', 'unknown'))}"
                    for e in entities
                )
                sections.append(f"Related Entities:\n{entities_text}")

            if "paths" in additional_context:
                paths_text = "\n".join(
                    f"- Path: {p.get('source_id')} -> {p.get('target_id')} ({p.get('length')} hops)"
                    for p in additional_context["paths"][:3]
                )
                sections.append(f"Discovered Connections:\n{paths_text}")

        # Exploration summary
        sections.append(
            f"Exploration: {len(world_state.explored_nodes)} nodes explored, "
            f"{world_state.iteration_count} iterations"
        )

        return "\n\n".join(sections)

    def _build_citations(
        self,
        confirmed_beliefs: list[Belief],
        high_confidence_beliefs: list[Belief],
    ) -> list[Citation]:
        """Build citations from beliefs."""
        citations = []

        for belief in confirmed_beliefs:
            # Add citation from supporting evidence
            for evidence in belief.supporting_evidence[:2]:
                citations.append(
                    Citation(
                        source_type=evidence.source_type,
                        source_id=evidence.source_id,
                        content=belief.content,
                        confidence=belief.confidence,
                    )
                )

        # Add high-confidence beliefs not in confirmed
        confirmed_ids = {b.id for b in confirmed_beliefs}
        for belief in high_confidence_beliefs:
            if belief.id not in confirmed_ids:
                citations.append(
                    Citation(
                        source_type="belief",
                        source_id=belief.id,
                        content=belief.content,
                        confidence=belief.confidence,
                    )
                )

        return citations[:10]  # Limit citations

    def _calculate_confidence(
        self,
        world_state: WorldState,
        confirmed_beliefs: list[Belief],
    ) -> float:
        """Calculate overall answer confidence."""
        if not confirmed_beliefs:
            return 0.3

        # Base confidence on belief confidences
        avg_confidence = sum(b.confidence for b in confirmed_beliefs) / len(
            confirmed_beliefs
        )

        # Adjust for completeness
        completeness_factor = world_state.answer_completeness

        # Penalize for unresolved high-priority uncertainties
        high_priority_uncertainties = [
            u for u in world_state.get_unresolved_uncertainties() if u.priority >= 0.7
        ]
        uncertainty_penalty = min(0.2, 0.05 * len(high_priority_uncertainties))

        confidence = avg_confidence * (0.5 + 0.5 * completeness_factor) - uncertainty_penalty

        return max(0.1, min(1.0, confidence))

    def _generate_follow_ups(
        self,
        world_state: WorldState,
        answer: str,
    ) -> list[str]:
        """Generate follow-up question suggestions."""
        follow_ups = []

        # Add remaining questions from world state
        follow_ups.extend(world_state.remaining_questions[:2])

        # Generate from unresolved uncertainties
        for uncertainty in world_state.get_unresolved_uncertainties()[:2]:
            if "?" not in uncertainty.description:
                follow_ups.append(f"Can you clarify: {uncertainty.description}?")
            else:
                follow_ups.append(uncertainty.description)

        return follow_ups[:4]

    def synthesize_comparison(
        self,
        entities: list[dict[str, Any]],
        comparison_criteria: list[str],
    ) -> str:
        """Synthesize a comparison between entities.

        Args:
            entities: Entities to compare
            comparison_criteria: Criteria for comparison

        Returns:
            Comparison text
        """
        system_instruction = """You are comparing financial entities or strategies.
Provide a structured comparison highlighting key differences and similarities."""

        prompt = f"""Compare these entities:
{entities}

Comparison criteria: {comparison_criteria}

Provide a structured comparison."""

        return self.llm.generate(prompt, system_instruction=system_instruction)

    def synthesize_causal_explanation(
        self,
        effect: str,
        causes: list[dict[str, Any]],
        evidence: list[str],
    ) -> str:
        """Synthesize a causal explanation.

        Args:
            effect: The effect to explain
            causes: Potential causes with evidence
            evidence: Supporting evidence

        Returns:
            Causal explanation text
        """
        system_instruction = """You are explaining causal relationships in quantitative finance.
Be precise about the strength of causal claims based on evidence."""

        prompt = f"""Explain the causes of: {effect}

Potential causes and evidence:
{causes}

Additional evidence:
{evidence}

Provide a causal explanation, being clear about:
1. Direct vs indirect causes
2. Strength of evidence
3. Potential confounding factors"""

        return self.llm.generate(prompt, system_instruction=system_instruction)
