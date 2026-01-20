"""World model agent for epistemic state tracking.

The key agent that updates beliefs and uncertainties based on
new evidence from retrieval, traversal, and simulation.
"""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field

from src.llm.gemini_provider import GeminiProvider
from src.schema.world_state import (
    Belief,
    BeliefStatus,
    EvidenceSource,
    Uncertainty,
    UncertaintyType,
    WorldState,
)


class BeliefUpdate(BaseModel):
    """An update to the belief system."""

    belief_id: str = Field(..., description="ID of belief to update or create")
    content: str = Field(..., description="Belief content")
    status: str = Field(..., description="new, confirmed, contradicted, inferred")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    supporting_evidence: list[str] = Field(default_factory=list)
    contradicting_evidence: list[str] = Field(default_factory=list)


class UncertaintyUpdate(BaseModel):
    """An update to the uncertainty tracking."""

    uncertainty_id: str = Field(..., description="ID of uncertainty")
    description: str = Field(..., description="Uncertainty description")
    uncertainty_type: str = Field(..., description="missing_data, ambiguous, counterfactual, etc.")
    priority: float = Field(default=0.5, ge=0.0, le=1.0)
    resolved: bool = Field(default=False)


class WorldModelUpdate(BaseModel):
    """Complete update to the world model."""

    belief_updates: list[BeliefUpdate] = Field(default_factory=list)
    uncertainty_updates: list[UncertaintyUpdate] = Field(default_factory=list)
    new_questions: list[str] = Field(default_factory=list)
    completeness_estimate: float = Field(default=0.0, ge=0.0, le=1.0)
    reasoning: str = Field(default="")


class WorldModelAgent:
    """Agent for updating the epistemic world model.

    This is the key agent that maintains the system's beliefs about
    the domain, tracks uncertainties, and estimates answer completeness.
    """

    def __init__(self, llm: GeminiProvider) -> None:
        """Initialize the world model agent.

        Args:
            llm: The LLM provider
        """
        self.llm = llm

    def update_world_state(
        self,
        world_state: WorldState,
        new_evidence: list[dict[str, Any]],
    ) -> WorldState:
        """Update the world state with new evidence.

        Args:
            world_state: Current world state
            new_evidence: New evidence to incorporate

        Returns:
            Updated world state
        """
        if not new_evidence:
            return world_state

        # Get LLM analysis of the evidence
        update = self._analyze_evidence(world_state, new_evidence)

        # Apply belief updates
        for belief_update in update.belief_updates:
            self._apply_belief_update(world_state, belief_update, new_evidence)

        # Apply uncertainty updates
        for uncertainty_update in update.uncertainty_updates:
            self._apply_uncertainty_update(world_state, uncertainty_update)

        # Update remaining questions
        if update.new_questions:
            # Add new questions that aren't already present
            existing = set(world_state.remaining_questions)
            for q in update.new_questions:
                if q not in existing:
                    world_state.remaining_questions.append(q)

        # Update completeness estimate
        world_state.answer_completeness = update.completeness_estimate

        return world_state

    def _analyze_evidence(
        self,
        world_state: WorldState,
        new_evidence: list[dict[str, Any]],
    ) -> WorldModelUpdate:
        """Analyze evidence and determine updates."""
        # Build context
        current_beliefs = [
            f"- {b.content} (status: {b.status.value}, confidence: {b.confidence:.0%})"
            for b in world_state.beliefs.values()
        ]

        current_uncertainties = [
            f"- {u.description} (type: {u.uncertainty_type.value}, priority: {u.priority:.0%})"
            for u in world_state.uncertainties.values()
            if u.should_explore()
        ]

        evidence_text = "\n".join(
            f"- [{e.get('source_type', 'unknown')}] {e.get('content', str(e))}"
            for e in new_evidence[:20]  # Limit evidence
        )

        system_instruction = """You are an epistemic reasoning agent that updates beliefs based on evidence.

Your task:
1. Analyze new evidence against existing beliefs
2. Identify which beliefs are confirmed, contradicted, or need updating
3. Identify new beliefs that can be inferred
4. Track remaining uncertainties
5. Estimate how complete the answer is

Belief status types:
- new: A new belief not previously held
- confirmed: Existing belief supported by new evidence
- contradicted: Existing belief contradicted by new evidence
- inferred: Belief derived from other beliefs

Uncertainty types:
- missing_data: Information doesn't exist or wasn't found
- ambiguous: Multiple interpretations possible
- counterfactual: "What if" scenario needs simulation
- conflicting: Sources disagree
- temporal: Time-dependent uncertainty
- causal: Uncertain causal relationship"""

        prompt = f"""Query: {world_state.original_query}

Current Beliefs:
{chr(10).join(current_beliefs) if current_beliefs else "(none)"}

Current Uncertainties:
{chr(10).join(current_uncertainties) if current_uncertainties else "(none)"}

New Evidence:
{evidence_text}

Analyze this evidence and provide updates. Respond with JSON:
{{
    "belief_updates": [
        {{"belief_id": "b1", "content": "...", "status": "confirmed", "confidence": 0.8, "supporting_evidence": ["..."]}}
    ],
    "uncertainty_updates": [
        {{"uncertainty_id": "u1", "description": "...", "uncertainty_type": "missing_data", "priority": 0.7, "resolved": false}}
    ],
    "new_questions": ["What is...?"],
    "completeness_estimate": 0.6,
    "reasoning": "Brief explanation of the update"
}}"""

        try:
            update = self.llm.generate_structured(
                prompt, WorldModelUpdate, system_instruction=system_instruction
            )
            return update
        except ValueError:
            # Fallback: basic evidence processing
            return self._basic_evidence_processing(world_state, new_evidence)

    def _basic_evidence_processing(
        self,
        world_state: WorldState,
        new_evidence: list[dict[str, Any]],
    ) -> WorldModelUpdate:
        """Basic evidence processing without LLM."""
        belief_updates = []
        uncertainty_updates = []

        # Create a belief for each piece of evidence
        for i, evidence in enumerate(new_evidence[:10]):
            content = evidence.get("content", str(evidence))
            if not content or len(content) < 10:
                continue

            belief_updates.append(
                BeliefUpdate(
                    belief_id=f"b_{uuid.uuid4().hex[:8]}",
                    content=content[:200],  # Truncate long content
                    status="new",
                    confidence=0.6,
                    supporting_evidence=[evidence.get("source_type", "unknown")],
                )
            )

        # Calculate completeness based on evidence and existing beliefs
        evidence_count = len(new_evidence)
        belief_count = len(world_state.beliefs)
        completeness = min(0.9, 0.1 + 0.1 * evidence_count + 0.1 * belief_count)

        return WorldModelUpdate(
            belief_updates=belief_updates,
            uncertainty_updates=uncertainty_updates,
            completeness_estimate=completeness,
            reasoning="Basic evidence processing",
        )

    def _apply_belief_update(
        self,
        world_state: WorldState,
        update: BeliefUpdate,
        evidence: list[dict[str, Any]],
    ) -> None:
        """Apply a belief update to the world state."""
        existing = world_state.get_belief(update.belief_id)

        if existing:
            # Update existing belief
            status_map = {
                "confirmed": BeliefStatus.CONFIRMED,
                "contradicted": BeliefStatus.CONTRADICTED,
                "inferred": BeliefStatus.INFERRED,
            }
            existing.status = status_map.get(update.status, existing.status)
            existing.confidence = update.confidence

            # Add evidence
            for ev_desc in update.supporting_evidence:
                existing.add_supporting_evidence(
                    EvidenceSource(
                        source_type="evidence",
                        source_id=str(uuid.uuid4())[:8],
                        content_summary=ev_desc,
                    )
                )

            for ev_desc in update.contradicting_evidence:
                existing.add_contradicting_evidence(
                    EvidenceSource(
                        source_type="evidence",
                        source_id=str(uuid.uuid4())[:8],
                        content_summary=ev_desc,
                    )
                )
        else:
            # Create new belief
            status_map = {
                "new": BeliefStatus.HYPOTHETICAL,
                "confirmed": BeliefStatus.CONFIRMED,
                "contradicted": BeliefStatus.CONTRADICTED,
                "inferred": BeliefStatus.INFERRED,
            }

            belief = Belief(
                id=update.belief_id,
                content=update.content,
                status=status_map.get(update.status, BeliefStatus.HYPOTHETICAL),
                confidence=update.confidence,
            )

            # Add supporting evidence
            for ev_desc in update.supporting_evidence:
                belief.add_supporting_evidence(
                    EvidenceSource(
                        source_type="evidence",
                        source_id=str(uuid.uuid4())[:8],
                        content_summary=ev_desc,
                    )
                )

            world_state.add_belief(belief)

    def _apply_uncertainty_update(
        self,
        world_state: WorldState,
        update: UncertaintyUpdate,
    ) -> None:
        """Apply an uncertainty update to the world state."""
        existing = world_state.get_uncertainty(update.uncertainty_id)

        if existing:
            if update.resolved:
                # Mark as exhausted (max attempts reached)
                existing.exploration_attempts = existing.max_attempts
            else:
                existing.priority = update.priority
        else:
            if not update.resolved:
                # Create new uncertainty
                type_map = {
                    "missing_data": UncertaintyType.MISSING_DATA,
                    "ambiguous": UncertaintyType.AMBIGUOUS,
                    "counterfactual": UncertaintyType.COUNTERFACTUAL,
                    "conflicting": UncertaintyType.CONFLICTING,
                    "temporal": UncertaintyType.TEMPORAL,
                    "causal": UncertaintyType.CAUSAL,
                }

                uncertainty = Uncertainty(
                    id=update.uncertainty_id,
                    description=update.description,
                    uncertainty_type=type_map.get(
                        update.uncertainty_type, UncertaintyType.MISSING_DATA
                    ),
                    priority=update.priority,
                )

                world_state.add_uncertainty(uncertainty)

    def identify_gaps(self, world_state: WorldState) -> list[str]:
        """Identify gaps in knowledge that need filling.

        Args:
            world_state: Current world state

        Returns:
            List of identified knowledge gaps
        """
        gaps = []

        # Check remaining questions
        gaps.extend(world_state.remaining_questions[:3])

        # Check unresolved uncertainties
        for uncertainty in world_state.get_unresolved_uncertainties()[:3]:
            gaps.append(uncertainty.description)

        # Check for beliefs with low confidence
        low_confidence = [
            b for b in world_state.beliefs.values()
            if b.confidence < 0.5 and b.status != BeliefStatus.CONTRADICTED
        ]
        for belief in low_confidence[:2]:
            gaps.append(f"Need more evidence for: {belief.content}")

        return gaps

    def suggest_exploration_strategy(
        self,
        world_state: WorldState,
    ) -> dict[str, Any]:
        """Suggest next exploration strategy.

        Args:
            world_state: Current world state

        Returns:
            Strategy recommendation
        """
        priority_uncertainty = world_state.get_highest_priority_uncertainty()

        if not priority_uncertainty:
            return {
                "strategy": "synthesize",
                "reason": "No unresolved uncertainties",
            }

        if priority_uncertainty.uncertainty_type == UncertaintyType.COUNTERFACTUAL:
            return {
                "strategy": "simulate",
                "target": priority_uncertainty.description,
                "reason": "Counterfactual scenario needs simulation",
            }

        if priority_uncertainty.uncertainty_type == UncertaintyType.MISSING_DATA:
            return {
                "strategy": "traverse",
                "target": priority_uncertainty.description,
                "reason": "Need to explore graph for missing data",
            }

        return {
            "strategy": "traverse",
            "target": priority_uncertainty.description,
            "reason": f"Resolving {priority_uncertainty.uncertainty_type.value} uncertainty",
        }
