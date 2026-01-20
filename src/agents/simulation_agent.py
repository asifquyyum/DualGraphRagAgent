"""Simulation agent for counterfactual scenario analysis.

Generates and evaluates "what if" scenarios for quantitative
finance questions.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.graph.graph_interface import GraphStore
from src.llm.gemini_provider import GeminiProvider
from src.schema.ontology import DomainOntology
from src.schema.world_state import WorldState
from src.tools.simulation_tools import SimulationTools, StrategyImpact


class ScenarioDefinition(BaseModel):
    """Definition of a counterfactual scenario."""

    name: str = Field(..., description="Scenario name")
    description: str = Field(..., description="Scenario description")
    market_changes: dict[str, float] = Field(
        default_factory=dict, description="Market variable changes"
    )
    affected_instruments: list[str] = Field(default_factory=list)
    affected_strategies: list[str] = Field(default_factory=list)


class SimulationResult(BaseModel):
    """Result of a counterfactual simulation."""

    scenario: str = Field(..., description="Scenario description")
    impacts: list[StrategyImpact] = Field(default_factory=list)
    market_effects: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    summary: str = Field(default="")
    caveats: list[str] = Field(default_factory=list)


class SimulationAgent:
    """Agent for running counterfactual simulations.

    Uses LLM to interpret scenarios and simulation tools to
    calculate impacts.
    """

    def __init__(
        self,
        llm: GeminiProvider,
        ontology: DomainOntology,
        store: GraphStore,
    ) -> None:
        """Initialize the simulation agent.

        Args:
            llm: The LLM provider
            ontology: The domain ontology
            store: The graph store
        """
        self.llm = llm
        self.ontology = ontology
        self.store = store
        self.tools = SimulationTools()

    def simulate(
        self,
        scenario_description: str,
        world_state: WorldState,
    ) -> SimulationResult:
        """Run a counterfactual simulation.

        Args:
            scenario_description: Natural language scenario description
            world_state: Current world state

        Returns:
            SimulationResult with impacts
        """
        # Parse the scenario
        scenario = self._parse_scenario(scenario_description, world_state)

        # Get affected entities from graph
        affected_entities = self._get_affected_entities(scenario)

        # Run simulations
        impacts = self._calculate_impacts(scenario, affected_entities)

        # Calculate market effects
        market_effects = self._calculate_market_effects(scenario)

        # Generate summary
        summary = self._generate_summary(scenario, impacts, market_effects)

        return SimulationResult(
            scenario=scenario.description,
            impacts=impacts,
            market_effects=market_effects,
            confidence=self._estimate_confidence(scenario, impacts),
            summary=summary,
            caveats=self._generate_caveats(scenario),
        )

    def _parse_scenario(
        self,
        description: str,
        world_state: WorldState,
    ) -> ScenarioDefinition:
        """Parse a scenario description into structured form."""
        system_instruction = """You are parsing a financial scenario for simulation.
Extract:
1. The name of the scenario
2. Specific market variable changes (e.g., VIX +20%, SPX -5%)
3. Instruments that would be affected
4. Strategies that would be affected

Common market variables:
- vix_change: VIX percentage change
- spx_change: S&P 500 percentage change
- iv_change: Implied volatility change
- rates_change: Interest rate change (basis points)
- time_decay: Days of time decay"""

        prompt = f"""Parse this scenario:
"{description}"

Context from current beliefs:
{world_state.to_context_string() if world_state else "No context"}

Respond with JSON:
{{
    "name": "VIX Spike Scenario",
    "description": "...",
    "market_changes": {{"vix_change": 20, "spx_change": -3}},
    "affected_instruments": ["SPX", "VIX"],
    "affected_strategies": ["straddle", "iron_condor"]
}}"""

        try:
            scenario = self.llm.generate_structured(
                prompt, ScenarioDefinition, system_instruction=system_instruction
            )
            return scenario
        except ValueError:
            # Fallback parsing
            return self._basic_scenario_parse(description)

    def _basic_scenario_parse(self, description: str) -> ScenarioDefinition:
        """Basic scenario parsing without LLM."""
        description_lower = description.lower()

        market_changes = {}
        affected_instruments = []
        affected_strategies = []

        # Detect VIX changes
        if "vix" in description_lower:
            affected_instruments.append("VIX")
            if "spike" in description_lower or "jump" in description_lower:
                market_changes["vix_change"] = 20
            elif "double" in description_lower:
                market_changes["vix_change"] = 100
            elif "20%" in description or "20 %" in description:
                market_changes["vix_change"] = 20

        # Detect SPX changes
        if "spx" in description_lower or "s&p" in description_lower:
            affected_instruments.append("SPX")
            if "crash" in description_lower or "drop" in description_lower:
                market_changes["spx_change"] = -10

        # Detect strategy mentions
        strategy_keywords = {
            "straddle": "long_straddle",
            "strangle": "long_strangle",
            "iron condor": "iron_condor",
            "butterfly": "butterfly",
        }
        for keyword, strategy in strategy_keywords.items():
            if keyword in description_lower:
                affected_strategies.append(strategy)

        return ScenarioDefinition(
            name="Parsed Scenario",
            description=description,
            market_changes=market_changes,
            affected_instruments=affected_instruments,
            affected_strategies=affected_strategies,
        )

    def _get_affected_entities(
        self,
        scenario: ScenarioDefinition,
    ) -> list[dict[str, Any]]:
        """Get entities affected by the scenario from the graph."""
        affected = []

        # Search for instruments
        for instrument in scenario.affected_instruments:
            entities = self.store.search_entities(instrument, node_types=["Instrument"], limit=3)
            for entity in entities:
                affected.append(
                    {
                        "type": "instrument",
                        "entity": entity.to_dict(),
                    }
                )

        # Search for strategies
        for strategy in scenario.affected_strategies:
            entities = self.store.search_entities(strategy, node_types=["Strategy"], limit=3)
            for entity in entities:
                affected.append(
                    {
                        "type": "strategy",
                        "entity": entity.to_dict(),
                    }
                )

        return affected

    def _calculate_impacts(
        self,
        scenario: ScenarioDefinition,
        affected_entities: list[dict[str, Any]],
    ) -> list[StrategyImpact]:
        """Calculate impacts on strategies."""
        impacts = []
        vix_change = scenario.market_changes.get("vix_change", 0)

        # Calculate impact for each affected strategy
        strategy_types = set()
        for entity in affected_entities:
            if entity["type"] == "strategy":
                props = entity["entity"].get("properties", {})
                strategy_type = props.get("strategy_type", props.get("name", "unknown"))
                strategy_types.add(strategy_type)

        # Also add strategies from scenario definition
        strategy_types.update(scenario.affected_strategies)

        for strategy_type in strategy_types:
            if vix_change != 0:
                impact = self.tools.estimate_vix_spike_impact(
                    vix_change_percent=vix_change,
                    strategy_type=strategy_type,
                )
                impacts.append(impact)

        return impacts

    def _calculate_market_effects(
        self,
        scenario: ScenarioDefinition,
    ) -> dict[str, Any]:
        """Calculate broader market effects."""
        effects = {}

        vix_change = scenario.market_changes.get("vix_change", 0)
        if vix_change != 0:
            # Simulate term structure change
            # Using typical VIX levels
            effects["term_structure"] = self.tools.simulate_term_structure_change(
                vix_spot=20,  # Typical VIX level
                vix_futures=[21, 22, 23],
                shock_percent=vix_change,
            )

        return effects

    def _generate_summary(
        self,
        scenario: ScenarioDefinition,
        impacts: list[StrategyImpact],
        market_effects: dict[str, Any],
    ) -> str:
        """Generate a narrative summary of the simulation."""
        system_instruction = """Summarize the simulation results in 2-3 sentences.
Focus on:
1. The key market changes
2. Which strategies benefit or suffer
3. Any important risk factors"""

        impacts_text = "\n".join(
            f"- {i.strategy_name}: P&L estimate ${i.pnl_estimate:,.0f} ({i.recommendation})"
            for i in impacts
        )

        prompt = f"""Scenario: {scenario.description}

Market Changes: {scenario.market_changes}

Strategy Impacts:
{impacts_text if impacts else "No specific impacts calculated"}

Market Effects:
{market_effects}

Summarize these simulation results."""

        return self.llm.generate(prompt, system_instruction=system_instruction, temperature=0.3)

    def _estimate_confidence(
        self,
        scenario: ScenarioDefinition,
        impacts: list[StrategyImpact],
    ) -> float:
        """Estimate confidence in the simulation results."""
        confidence = 0.7  # Base confidence

        # Reduce confidence for extreme scenarios
        for change in scenario.market_changes.values():
            if abs(change) > 50:
                confidence -= 0.1
            if abs(change) > 100:
                confidence -= 0.1

        # Increase confidence if we have calculated impacts
        if impacts:
            confidence += 0.1

        return max(0.3, min(0.95, confidence))

    def _generate_caveats(self, scenario: ScenarioDefinition) -> list[str]:
        """Generate caveats for the simulation."""
        caveats = [
            "Simulations are based on simplified models and historical relationships",
            "Actual market behavior may differ significantly during stress events",
        ]

        for var, change in scenario.market_changes.items():
            if abs(change) > 30:
                caveats.append(
                    f"Large {var} change ({change}%) may trigger non-linear effects not captured"
                )

        return caveats
