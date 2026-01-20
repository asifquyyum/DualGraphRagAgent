"""Sufficiency checker for determining termination conditions.

Evaluates whether the current world state has sufficient information
to answer the user's query.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.llm.gemini_provider import GeminiProvider
from src.schema.world_state import BeliefStatus, WorldState


class SufficiencyAssessment(BaseModel):
    """Assessment of answer sufficiency."""

    is_sufficient: bool = Field(..., description="Whether we have enough information")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reasoning: str = Field(default="", description="Explanation of the assessment")
    missing_elements: list[str] = Field(
        default_factory=list, description="What's still missing"
    )
    suggested_action: str = Field(
        default="continue", description="synthesize, traverse, simulate, or continue"
    )


class SufficiencyChecker:
    """Checker for evaluating if we have sufficient information to answer.

    Uses multiple criteria to determine if the query loop should terminate:
    - Answer completeness threshold
    - Confirmed belief count
    - Unresolved uncertainty priority
    - Maximum iterations
    """

    def __init__(
        self,
        llm: GeminiProvider,
        completeness_threshold: float = 0.8,
        min_confirmed_beliefs: int = 1,
    ) -> None:
        """Initialize the sufficiency checker.

        Args:
            llm: The LLM provider
            completeness_threshold: Minimum completeness to consider sufficient
            min_confirmed_beliefs: Minimum confirmed beliefs required
        """
        self.llm = llm
        self.completeness_threshold = completeness_threshold
        self.min_confirmed_beliefs = min_confirmed_beliefs

    def check(self, world_state: WorldState) -> SufficiencyAssessment:
        """Check if we have sufficient information.

        Args:
            world_state: Current world state

        Returns:
            SufficiencyAssessment with recommendation
        """
        # Quick checks first
        quick_assessment = self._quick_check(world_state)
        # Return early if we have a definitive action (sufficient, synthesize, traverse, or simulate)
        if quick_assessment.is_sufficient or quick_assessment.suggested_action in ("synthesize", "traverse", "simulate"):
            return quick_assessment

        # Detailed LLM assessment if needed
        return self._detailed_check(world_state)

    def _quick_check(self, world_state: WorldState) -> SufficiencyAssessment:
        """Perform quick rule-based sufficiency check."""
        # Check max iterations
        if world_state.iteration_count >= world_state.max_iterations:
            return SufficiencyAssessment(
                is_sufficient=True,
                confidence=0.6,
                reasoning="Maximum iterations reached",
                suggested_action="synthesize",
            )

        # Check completeness
        if world_state.answer_completeness >= self.completeness_threshold:
            confirmed_count = len(world_state.get_confirmed_beliefs())
            if confirmed_count >= self.min_confirmed_beliefs:
                return SufficiencyAssessment(
                    is_sufficient=True,
                    confidence=world_state.answer_completeness,
                    reasoning=f"Completeness {world_state.answer_completeness:.0%} with {confirmed_count} confirmed beliefs",
                    suggested_action="synthesize",
                )

        # Check for critical unresolved uncertainties
        critical_uncertainties = [
            u for u in world_state.get_unresolved_uncertainties()
            if u.priority >= 0.8
        ]

        if critical_uncertainties:
            uncertainty = critical_uncertainties[0]
            action = "simulate" if uncertainty.uncertainty_type.value == "counterfactual" else "traverse"
            return SufficiencyAssessment(
                is_sufficient=False,
                confidence=world_state.answer_completeness,
                reasoning=f"Critical uncertainty: {uncertainty.description}",
                missing_elements=[uncertainty.description],
                suggested_action=action,
            )

        # Check if no progress is being made
        if world_state.iteration_count > 3 and world_state.answer_completeness < 0.3:
            return SufficiencyAssessment(
                is_sufficient=False,
                confidence=0.3,
                reasoning="Low progress, may need different approach",
                suggested_action="traverse",
            )

        return SufficiencyAssessment(
            is_sufficient=False,
            confidence=world_state.answer_completeness,
            reasoning="More exploration needed",
            suggested_action="continue",
        )

    def _detailed_check(self, world_state: WorldState) -> SufficiencyAssessment:
        """Perform detailed LLM-based sufficiency check."""
        # Build context
        beliefs_summary = []
        for belief in world_state.beliefs.values():
            status_emoji = {
                BeliefStatus.CONFIRMED: "✓",
                BeliefStatus.CONTRADICTED: "✗",
                BeliefStatus.INFERRED: "~",
                BeliefStatus.HYPOTHETICAL: "?",
            }.get(belief.status, "?")
            beliefs_summary.append(
                f"{status_emoji} {belief.content} ({belief.confidence:.0%})"
            )

        uncertainties_summary = [
            f"- {u.description} (priority: {u.priority:.0%})"
            for u in world_state.get_unresolved_uncertainties()
        ]

        system_instruction = """You are evaluating whether there is sufficient information
to answer a user's query about quantitative finance.

Consider:
1. Does the information directly address the query?
2. Are there critical gaps that would make the answer misleading?
3. Is the confidence level acceptable?
4. Would more exploration significantly improve the answer?"""

        prompt = f"""Query: {world_state.original_query}

Current Beliefs:
{chr(10).join(beliefs_summary) if beliefs_summary else "(none)"}

Remaining Uncertainties:
{chr(10).join(uncertainties_summary) if uncertainties_summary else "(none)"}

Completeness: {world_state.answer_completeness:.0%}
Iteration: {world_state.iteration_count}/{world_state.max_iterations}

Assess if we have sufficient information to answer. Respond with JSON:
{{
    "is_sufficient": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "...",
    "missing_elements": ["..."],
    "suggested_action": "synthesize/traverse/simulate"
}}"""

        try:
            assessment = self.llm.generate_structured(
                prompt, SufficiencyAssessment, system_instruction=system_instruction
            )
            return assessment
        except ValueError:
            # Fallback to basic assessment
            return self._quick_check(world_state)

    def estimate_additional_value(
        self,
        world_state: WorldState,
        proposed_action: str,
    ) -> float:
        """Estimate the value of taking an additional action.

        Args:
            world_state: Current world state
            proposed_action: The proposed next action

        Returns:
            Estimated value between 0 and 1
        """
        # Base value depends on current completeness
        base_value = 1.0 - world_state.answer_completeness

        # Adjust based on action type and state
        if proposed_action == "traverse":
            unresolved = len(world_state.get_unresolved_uncertainties())
            if unresolved > 0:
                return min(1.0, base_value * (1 + 0.1 * unresolved))
            return base_value * 0.5  # Diminishing returns if no uncertainties

        if proposed_action == "simulate":
            # High value if we have counterfactual uncertainty
            if world_state.has_counterfactual_uncertainty():
                return min(1.0, base_value * 1.5)
            return 0.0  # No value if no counterfactual need

        return base_value

    def should_terminate(self, world_state: WorldState) -> bool:
        """Simple boolean check for termination.

        Args:
            world_state: Current world state

        Returns:
            True if should terminate
        """
        assessment = self.check(world_state)
        return assessment.is_sufficient or assessment.suggested_action == "synthesize"

    def get_termination_reason(self, world_state: WorldState) -> str:
        """Get human-readable termination reason.

        Args:
            world_state: Current world state

        Returns:
            Reason for termination decision
        """
        if world_state.iteration_count >= world_state.max_iterations:
            return f"Maximum iterations ({world_state.max_iterations}) reached"

        if world_state.has_sufficient_evidence():
            return f"Sufficient evidence gathered (completeness: {world_state.answer_completeness:.0%})"

        confirmed = len(world_state.get_confirmed_beliefs())
        unresolved = len(world_state.get_unresolved_uncertainties())

        return (
            f"Continuing: {confirmed} confirmed beliefs, "
            f"{unresolved} unresolved uncertainties, "
            f"{world_state.answer_completeness:.0%} complete"
        )
