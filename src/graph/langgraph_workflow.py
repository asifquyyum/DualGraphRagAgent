"""LangGraph workflow for the agentic query loop.

Implements the state machine that orchestrates semantic alignment,
retrieval, world model updates, sufficiency checking, and synthesis.
"""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from src.agents.cypher_agent import CypherAgent
from src.agents.semantic_aligner import AlignedQuery, SemanticAligner
from src.agents.simulation_agent import SimulationAgent
from src.agents.sufficiency_checker import SufficiencyChecker
from src.agents.synthesis_agent import SynthesisAgent, SynthesizedAnswer
from src.agents.traversal_agent import TraversalAgent
from src.agents.world_model_agent import WorldModelAgent
from src.graph.graph_interface import GraphStore
from src.llm.gemini_provider import GeminiProvider
from src.schema.ontology import DomainOntology
from src.schema.world_state import WorldState


class GraphRAGState(TypedDict):
    """State for the Graph RAG workflow."""

    # Input
    query: str
    config: dict[str, Any]

    # Core state
    world_state: WorldState
    aligned_query: AlignedQuery | None

    # Intermediate results
    retrieval_results: list[dict[str, Any]]
    traversal_results: dict[str, Any]
    simulation_results: dict[str, Any] | None

    # Output
    final_answer: SynthesizedAnswer | None
    error: str | None


class GraphRAGWorkflow:
    """LangGraph workflow for agentic Graph RAG.

    Implements the query loop with epistemic state tracking:
    1. Semantic alignment
    2. Initial retrieval
    3. World model update
    4. Sufficiency check
    5. Graph traversal (if needed)
    6. Simulation (for counterfactuals)
    7. Synthesis
    """

    def __init__(
        self,
        llm: GeminiProvider,
        ontology: DomainOntology,
        store: GraphStore,
        max_iterations: int = 10,
    ) -> None:
        """Initialize the workflow.

        Args:
            llm: The LLM provider
            ontology: The domain ontology
            store: The graph store
            max_iterations: Maximum query loop iterations
        """
        self.llm = llm
        self.ontology = ontology
        self.store = store
        self.max_iterations = max_iterations

        # Initialize agents
        self.semantic_aligner = SemanticAligner(llm, ontology)
        self.cypher_agent = CypherAgent(llm, ontology, store)
        self.traversal_agent = TraversalAgent(llm, ontology, store)
        self.world_model_agent = WorldModelAgent(llm)
        self.sufficiency_checker = SufficiencyChecker(llm)
        self.synthesis_agent = SynthesisAgent(llm)
        self.simulation_agent = SimulationAgent(llm, ontology, store)

        # Build the graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine."""
        workflow = StateGraph(GraphRAGState)

        # Add nodes
        workflow.add_node("semantic_alignment", self._semantic_alignment)
        workflow.add_node("retrieval", self._retrieval)
        workflow.add_node("update_world_model", self._update_world_model)
        workflow.add_node("check_sufficiency", self._check_sufficiency)
        workflow.add_node("graph_traversal", self._graph_traversal)
        workflow.add_node("simulate_scenario", self._simulate_scenario)
        workflow.add_node("synthesize", self._synthesize)

        # Set entry point
        workflow.set_entry_point("semantic_alignment")

        # Add edges
        workflow.add_edge("semantic_alignment", "retrieval")
        workflow.add_edge("retrieval", "update_world_model")
        workflow.add_edge("update_world_model", "check_sufficiency")

        # Conditional routing from check_sufficiency
        workflow.add_conditional_edges(
            "check_sufficiency",
            self._route_from_sufficiency,
            {
                "synthesize": "synthesize",
                "graph_traversal": "graph_traversal",
                "simulate_scenario": "simulate_scenario",
            },
        )

        # Return to update_world_model after traversal/simulation
        workflow.add_edge("graph_traversal", "update_world_model")
        workflow.add_edge("simulate_scenario", "update_world_model")

        # End after synthesis
        workflow.add_edge("synthesize", END)

        return workflow.compile()

    def _semantic_alignment(self, state: GraphRAGState) -> dict[str, Any]:
        """Perform semantic alignment on the query."""
        query = state["query"]

        try:
            aligned_query = self.semantic_aligner.align_query(query)

            # Initialize world state
            world_state = WorldState(
                original_query=query,
                max_iterations=self.max_iterations,
            )

            # Add initial questions based on alignment
            if aligned_query.sub_questions:
                world_state.remaining_questions = aligned_query.sub_questions

            return {
                "aligned_query": aligned_query,
                "world_state": world_state,
            }

        except Exception as e:
            return {"error": f"Semantic alignment failed: {str(e)}"}

    def _retrieval(self, state: GraphRAGState) -> dict[str, Any]:
        """Perform initial retrieval using Cypher."""
        aligned_query = state.get("aligned_query")
        if not aligned_query:
            return {"retrieval_results": []}

        try:
            # Generate hints from alignment
            hints = self.semantic_aligner.generate_cypher_hints(aligned_query)

            # Generate and execute Cypher query
            result = self.cypher_agent.generate_cypher(
                aligned_query.original_query, hints
            )

            return {"retrieval_results": result.results}

        except Exception as e:
            return {"retrieval_results": [], "error": f"Retrieval failed: {str(e)}"}

    def _update_world_model(self, state: GraphRAGState) -> dict[str, Any]:
        """Update the epistemic world model with new evidence."""
        world_state = state["world_state"]
        retrieval_results = state.get("retrieval_results", [])
        traversal_results = state.get("traversal_results", {})
        simulation_results = state.get("simulation_results")

        # Collect all evidence
        evidence = []

        for result in retrieval_results:
            evidence.append(
                {
                    "source_type": "graph",
                    "content": str(result),
                }
            )

        if traversal_results:
            for item in traversal_results.get("new_evidence", []):
                evidence.append(
                    {
                        "source_type": "traversal",
                        "content": item,
                    }
                )

        if simulation_results:
            evidence.append(
                {
                    "source_type": "simulation",
                    "content": simulation_results.get("summary", ""),
                }
            )

        # Update world model
        updated_state = self.world_model_agent.update_world_state(
            world_state, evidence
        )

        # Increment iteration
        updated_state.increment_iteration()

        return {"world_state": updated_state}

    def _check_sufficiency(self, state: GraphRAGState) -> dict[str, Any]:
        """Check if we have sufficient information."""
        world_state = state["world_state"]

        # Use sufficiency checker
        decision = self.sufficiency_checker.check(world_state)

        return {"world_state": world_state}

    def _route_from_sufficiency(self, state: GraphRAGState) -> str:
        """Route based on sufficiency check."""
        world_state = state["world_state"]
        return world_state.get_routing_decision()

    def _graph_traversal(self, state: GraphRAGState) -> dict[str, Any]:
        """Perform graph traversal for more context."""
        world_state = state["world_state"]
        retrieval_results = state.get("retrieval_results", [])

        # Get entity IDs from retrieval results
        entity_ids = []
        for result in retrieval_results:
            if isinstance(result, dict):
                for key, value in result.items():
                    if isinstance(value, dict) and "id" in value:
                        entity_ids.append(value["id"])

        # If no entities, try exploring for uncertainties
        if not entity_ids:
            uncertainty = world_state.get_highest_priority_uncertainty()
            if uncertainty:
                traversal_result = self.traversal_agent.explore_for_uncertainty(
                    uncertainty, world_state
                )
            else:
                return {"traversal_results": {}}
        else:
            traversal_result = self.traversal_agent.explore_from_entities(
                entity_ids[:5], world_state
            )

        # Mark explored nodes
        for step in traversal_result.steps:
            world_state.mark_node_explored(step.source_entity_id)
            for entity in step.entities_found:
                if "id" in entity:
                    world_state.mark_node_explored(entity["id"])

        return {
            "traversal_results": {
                "entities": traversal_result.entities_discovered,
                "relationships": traversal_result.relationships_discovered,
                "new_evidence": traversal_result.new_evidence,
            }
        }

    def _simulate_scenario(self, state: GraphRAGState) -> dict[str, Any]:
        """Run counterfactual simulation."""
        world_state = state["world_state"]
        aligned_query = state.get("aligned_query")

        # Find counterfactual uncertainty
        counterfactual_uncertainty = None
        for u in world_state.uncertainties.values():
            if u.uncertainty_type.value == "counterfactual":
                counterfactual_uncertainty = u
                break

        if not counterfactual_uncertainty:
            return {"simulation_results": None}

        # Run simulation
        simulation_result = self.simulation_agent.simulate(
            counterfactual_uncertainty.description,
            world_state,
        )

        # Mark uncertainty as explored
        counterfactual_uncertainty.record_attempt()

        return {
            "simulation_results": {
                "scenario": simulation_result.scenario,
                "impacts": simulation_result.impacts,
                "confidence": simulation_result.confidence,
                "summary": simulation_result.summary,
            }
        }

    def _synthesize(self, state: GraphRAGState) -> dict[str, Any]:
        """Synthesize the final answer."""
        world_state = state["world_state"]
        traversal_results = state.get("traversal_results", {})
        simulation_results = state.get("simulation_results")

        # Build additional context
        additional_context = {}

        if traversal_results:
            additional_context["entities"] = traversal_results.get("entities", [])
            additional_context["paths"] = traversal_results.get("relationships", [])

        if simulation_results:
            additional_context["simulation"] = simulation_results

        # Synthesize answer
        answer = self.synthesis_agent.synthesize(world_state, additional_context)

        return {"final_answer": answer}

    def run(
        self,
        query: str,
        config: dict[str, Any] | None = None,
    ) -> SynthesizedAnswer:
        """Run the workflow on a query.

        Args:
            query: The user's query
            config: Optional configuration

        Returns:
            Synthesized answer
        """
        initial_state: GraphRAGState = {
            "query": query,
            "config": config or {},
            "world_state": WorldState(),
            "aligned_query": None,
            "retrieval_results": [],
            "traversal_results": {},
            "simulation_results": None,
            "final_answer": None,
            "error": None,
        }

        # Run the graph
        result = self.graph.invoke(initial_state)

        if result.get("error"):
            return SynthesizedAnswer(
                answer=f"Error processing query: {result['error']}",
                confidence=0.0,
            )

        return result.get("final_answer") or SynthesizedAnswer(
            answer="Unable to generate answer",
            confidence=0.0,
        )

    def run_verbose(
        self,
        query: str,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run the workflow with verbose output.

        Args:
            query: The user's query
            config: Optional configuration

        Returns:
            Full state including intermediate results
        """
        initial_state: GraphRAGState = {
            "query": query,
            "config": config or {},
            "world_state": WorldState(),
            "aligned_query": None,
            "retrieval_results": [],
            "traversal_results": {},
            "simulation_results": None,
            "final_answer": None,
            "error": None,
        }

        # Run and return full state
        return self.graph.invoke(initial_state)
