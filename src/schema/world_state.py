"""Epistemic state models for tracking beliefs and uncertainties.

This module implements the world state that tracks what the system
believes, what it's uncertain about, and how complete its answer is.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class BeliefStatus(str, Enum):
    """Status of a belief in the epistemic state."""

    HYPOTHETICAL = "hypothetical"  # Proposed but not verified
    INFERRED = "inferred"  # Derived from other beliefs
    CONFIRMED = "confirmed"  # Verified against evidence
    CONTRADICTED = "contradicted"  # Evidence contradicts this belief


class EvidenceSource(BaseModel):
    """Source of evidence supporting or contradicting a belief."""

    source_type: str = Field(..., description="Type: 'document', 'graph', 'simulation'")
    source_id: str = Field(..., description="ID of the source (chunk_id, node_id, etc.)")
    content_summary: str = Field(default="", description="Brief summary of the evidence")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence in source")
    retrieved_at: datetime = Field(default_factory=datetime.utcnow)


class Belief(BaseModel):
    """A belief held by the system about the domain.

    Beliefs are propositions that may be true or false, with varying
    levels of confidence and supporting evidence.
    """

    id: str = Field(..., description="Unique identifier for this belief")
    content: str = Field(..., description="Natural language description of the belief")
    status: BeliefStatus = Field(
        default=BeliefStatus.HYPOTHETICAL, description="Current status of this belief"
    )
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Confidence level (0-1)"
    )
    supporting_evidence: list[EvidenceSource] = Field(
        default_factory=list, description="Evidence supporting this belief"
    )
    contradicting_evidence: list[EvidenceSource] = Field(
        default_factory=list, description="Evidence contradicting this belief"
    )
    related_entity_ids: list[str] = Field(
        default_factory=list, description="IDs of related graph entities"
    )
    derived_from: list[str] = Field(
        default_factory=list, description="IDs of beliefs this was derived from"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def add_supporting_evidence(self, evidence: EvidenceSource) -> None:
        """Add supporting evidence and update status/confidence."""
        self.supporting_evidence.append(evidence)
        self._update_status_and_confidence()
        self.updated_at = datetime.utcnow()

    def add_contradicting_evidence(self, evidence: EvidenceSource) -> None:
        """Add contradicting evidence and update status/confidence."""
        self.contradicting_evidence.append(evidence)
        self._update_status_and_confidence()
        self.updated_at = datetime.utcnow()

    def _update_status_and_confidence(self) -> None:
        """Update belief status and confidence based on evidence."""
        supporting_weight = sum(e.confidence for e in self.supporting_evidence)
        contradicting_weight = sum(e.confidence for e in self.contradicting_evidence)

        total_weight = supporting_weight + contradicting_weight
        if total_weight == 0:
            return

        # Calculate confidence as ratio of supporting to total evidence
        self.confidence = supporting_weight / total_weight

        # Update status based on evidence balance
        if contradicting_weight > supporting_weight:
            self.status = BeliefStatus.CONTRADICTED
        elif supporting_weight > 0 and self.status == BeliefStatus.HYPOTHETICAL:
            self.status = BeliefStatus.CONFIRMED


class UncertaintyType(str, Enum):
    """Types of uncertainty that can be tracked."""

    MISSING_DATA = "missing_data"  # Data doesn't exist in graph
    AMBIGUOUS = "ambiguous"  # Multiple interpretations possible
    COUNTERFACTUAL = "counterfactual"  # "What if" scenario
    CONFLICTING = "conflicting"  # Sources disagree
    TEMPORAL = "temporal"  # Time-dependent uncertainty
    CAUSAL = "causal"  # Uncertain causal relationship


class Uncertainty(BaseModel):
    """An uncertainty or gap in the system's knowledge.

    Uncertainties represent what the system doesn't know or is unsure about.
    They guide further exploration in the query loop.
    """

    id: str = Field(..., description="Unique identifier for this uncertainty")
    description: str = Field(
        ..., description="Natural language description of the uncertainty"
    )
    uncertainty_type: UncertaintyType = Field(
        ..., description="Type of uncertainty"
    )
    priority: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Priority for resolving (0-1)"
    )
    related_belief_ids: list[str] = Field(
        default_factory=list, description="IDs of related beliefs"
    )
    exploration_attempts: int = Field(
        default=0, description="Number of times we've tried to resolve this"
    )
    max_attempts: int = Field(
        default=3, description="Maximum exploration attempts before giving up"
    )
    resolution_strategy: str | None = Field(
        default=None, description="Suggested strategy for resolution"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)

    def should_explore(self) -> bool:
        """Check if this uncertainty should be explored further."""
        return self.exploration_attempts < self.max_attempts

    def record_attempt(self) -> None:
        """Record an exploration attempt."""
        self.exploration_attempts += 1


class WorldState(BaseModel):
    """Complete epistemic state of the system.

    Tracks all beliefs, uncertainties, explored nodes, and progress
    toward answering the user's query.
    """

    beliefs: dict[str, Belief] = Field(
        default_factory=dict, description="All beliefs keyed by ID"
    )
    uncertainties: dict[str, Uncertainty] = Field(
        default_factory=dict, description="All uncertainties keyed by ID"
    )
    explored_nodes: set[str] = Field(
        default_factory=set, description="Graph node IDs that have been explored"
    )
    explored_relationships: set[str] = Field(
        default_factory=set, description="Graph relationship IDs that have been explored"
    )
    answer_completeness: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Estimated completeness of answer (0-1)"
    )
    remaining_questions: list[str] = Field(
        default_factory=list, description="Questions that still need answering"
    )
    original_query: str = Field(default="", description="The original user query")
    iteration_count: int = Field(default=0, description="Number of query loop iterations")
    max_iterations: int = Field(default=10, description="Maximum allowed iterations")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True

    def add_belief(self, belief: Belief) -> None:
        """Add or update a belief."""
        self.beliefs[belief.id] = belief
        self.updated_at = datetime.utcnow()

    def add_uncertainty(self, uncertainty: Uncertainty) -> None:
        """Add an uncertainty."""
        self.uncertainties[uncertainty.id] = uncertainty
        self.updated_at = datetime.utcnow()

    def get_belief(self, belief_id: str) -> Belief | None:
        """Get a belief by ID."""
        return self.beliefs.get(belief_id)

    def get_uncertainty(self, uncertainty_id: str) -> Uncertainty | None:
        """Get an uncertainty by ID."""
        return self.uncertainties.get(uncertainty_id)

    def mark_node_explored(self, node_id: str) -> None:
        """Mark a graph node as explored."""
        self.explored_nodes.add(node_id)
        self.updated_at = datetime.utcnow()

    def mark_relationship_explored(self, rel_id: str) -> None:
        """Mark a relationship as explored."""
        self.explored_relationships.add(rel_id)
        self.updated_at = datetime.utcnow()

    def is_node_explored(self, node_id: str) -> bool:
        """Check if a node has been explored."""
        return node_id in self.explored_nodes

    def increment_iteration(self) -> None:
        """Increment the iteration counter."""
        self.iteration_count += 1
        self.updated_at = datetime.utcnow()

    def get_confirmed_beliefs(self) -> list[Belief]:
        """Get all confirmed beliefs."""
        return [b for b in self.beliefs.values() if b.status == BeliefStatus.CONFIRMED]

    def get_high_confidence_beliefs(self, threshold: float = 0.7) -> list[Belief]:
        """Get beliefs above a confidence threshold."""
        return [b for b in self.beliefs.values() if b.confidence >= threshold]

    def get_unresolved_uncertainties(self) -> list[Uncertainty]:
        """Get uncertainties that should still be explored."""
        return [u for u in self.uncertainties.values() if u.should_explore()]

    def get_highest_priority_uncertainty(self) -> Uncertainty | None:
        """Get the highest priority unresolved uncertainty."""
        unresolved = self.get_unresolved_uncertainties()
        if not unresolved:
            return None
        return max(unresolved, key=lambda u: u.priority)

    def has_counterfactual_uncertainty(self) -> bool:
        """Check if there are any counterfactual uncertainties."""
        return any(
            u.uncertainty_type == UncertaintyType.COUNTERFACTUAL
            for u in self.uncertainties.values()
            if u.should_explore()
        )

    def has_sufficient_evidence(self, threshold: float = 0.8) -> bool:
        """Check if we have sufficient evidence to answer the query.

        Considers:
        - Answer completeness above threshold
        - At least some confirmed beliefs
        - No critical unresolved uncertainties
        """
        if self.answer_completeness < threshold:
            return False

        confirmed = self.get_confirmed_beliefs()
        if not confirmed:
            return False

        # Check for high-priority unresolved uncertainties
        critical_uncertainties = [
            u for u in self.get_unresolved_uncertainties() if u.priority >= 0.8
        ]
        if critical_uncertainties:
            return False

        return True

    def should_continue(self) -> bool:
        """Determine if the query loop should continue."""
        if self.iteration_count >= self.max_iterations:
            return False
        if self.has_sufficient_evidence():
            return False
        if not self.get_unresolved_uncertainties() and not self.remaining_questions:
            return False
        return True

    def get_routing_decision(self) -> str:
        """Determine the next step in the query workflow.

        Returns one of:
        - 'synthesize': Ready to generate final answer
        - 'simulate_scenario': Need counterfactual simulation
        - 'graph_traversal': Need more graph exploration
        """
        if self.has_sufficient_evidence():
            return "synthesize"

        if self.iteration_count >= self.max_iterations:
            return "synthesize"

        if self.has_counterfactual_uncertainty():
            return "simulate_scenario"

        if self.get_unresolved_uncertainties() or self.remaining_questions:
            return "graph_traversal"

        return "synthesize"

    def to_context_string(self) -> str:
        """Generate a context string for LLM prompts."""
        lines = [
            f"Query: {self.original_query}",
            f"Iteration: {self.iteration_count}/{self.max_iterations}",
            f"Completeness: {self.answer_completeness:.0%}",
            "",
            "Confirmed Beliefs:",
        ]

        for belief in self.get_confirmed_beliefs():
            lines.append(f"  - {belief.content} (confidence: {belief.confidence:.0%})")

        if not self.get_confirmed_beliefs():
            lines.append("  (none yet)")

        lines.append("")
        lines.append("Unresolved Uncertainties:")

        for uncertainty in self.get_unresolved_uncertainties()[:5]:
            lines.append(f"  - {uncertainty.description} (priority: {uncertainty.priority:.0%})")

        if not self.get_unresolved_uncertainties():
            lines.append("  (none)")

        if self.remaining_questions:
            lines.append("")
            lines.append("Remaining Questions:")
            for q in self.remaining_questions[:3]:
                lines.append(f"  - {q}")

        return "\n".join(lines)

    def to_summary_dict(self) -> dict[str, Any]:
        """Generate a summary dictionary for logging/debugging."""
        return {
            "query": self.original_query,
            "iteration": self.iteration_count,
            "max_iterations": self.max_iterations,
            "completeness": self.answer_completeness,
            "num_beliefs": len(self.beliefs),
            "num_confirmed": len(self.get_confirmed_beliefs()),
            "num_uncertainties": len(self.uncertainties),
            "num_unresolved": len(self.get_unresolved_uncertainties()),
            "explored_nodes": len(self.explored_nodes),
            "routing_decision": self.get_routing_decision(),
        }
