"""Tests for simulation tools and agent."""

import pytest
from unittest.mock import MagicMock

from src.agents.simulation_agent import (
    SimulationAgent,
    ScenarioDefinition,
    SimulationResult,
)
from src.graph.networkx_store import NetworkXStore
from src.schema.ontology import Entity, create_quant_finance_ontology
from src.schema.world_state import WorldState
from src.tools.simulation_tools import SimulationTools, GreeksResult, StrategyImpact


class TestSimulationTools:
    """Tests for SimulationTools."""

    def test_calculate_greeks_call(self):
        greeks = SimulationTools.calculate_black_scholes_greeks(
            spot=100,
            strike=100,
            time_to_expiry=0.25,
            volatility=0.20,
            risk_free_rate=0.05,
            is_call=True,
        )

        assert isinstance(greeks, GreeksResult)
        assert 0 < greeks.delta < 1  # Call delta between 0 and 1
        assert greeks.gamma > 0  # Gamma always positive
        assert greeks.theta < 0  # Theta typically negative for long options
        assert greeks.vega > 0  # Vega positive for long options

    def test_calculate_greeks_put(self):
        greeks = SimulationTools.calculate_black_scholes_greeks(
            spot=100,
            strike=100,
            time_to_expiry=0.25,
            volatility=0.20,
            risk_free_rate=0.05,
            is_call=False,
        )

        assert -1 < greeks.delta < 0  # Put delta between -1 and 0

    def test_calculate_greeks_edge_cases(self):
        # Zero time to expiry
        greeks = SimulationTools.calculate_black_scholes_greeks(
            spot=100, strike=100, time_to_expiry=0, volatility=0.20, risk_free_rate=0.05
        )
        assert greeks.delta == 0  # Returns empty result for invalid inputs

    def test_simulate_straddle_pnl(self):
        result = SimulationTools.simulate_straddle_pnl(
            spot=100,
            strike=100,
            call_premium=5,
            put_premium=5,
            new_spot=110,  # Underlying moved up
            new_iv=0.25,  # IV increased
            days_passed=0,
            original_iv=0.20,
            time_to_expiry=0.1,
        )

        assert "pnl" in result
        assert "original_premium" in result
        assert result["original_premium"] == 10

    def test_estimate_vix_spike_impact_long_straddle(self):
        impact = SimulationTools.estimate_vix_spike_impact(
            vix_change_percent=20,
            strategy_type="long_straddle",
            position_size=100000,
        )

        assert isinstance(impact, StrategyImpact)
        assert impact.pnl_estimate > 0  # Long vol profits from VIX spike
        assert "vega" in impact.risk_factors[0].lower()

    def test_estimate_vix_spike_impact_short_straddle(self):
        impact = SimulationTools.estimate_vix_spike_impact(
            vix_change_percent=20,
            strategy_type="short_straddle",
            position_size=100000,
        )

        assert impact.pnl_estimate < 0  # Short vol loses from VIX spike
        assert "volatility risk" in impact.risk_factors[1].lower()

    def test_estimate_vix_spike_impact_iron_condor(self):
        impact = SimulationTools.estimate_vix_spike_impact(
            vix_change_percent=30,
            strategy_type="iron_condor",
            position_size=50000,
        )

        assert impact.pnl_estimate < 0  # Iron condor is short vol

    def test_simulate_term_structure_change(self):
        result = SimulationTools.simulate_term_structure_change(
            vix_spot=15,
            vix_futures=[16, 17, 18],
            shock_percent=20,
        )

        assert "original" in result
        assert "new" in result
        assert result["new"]["spot"] > result["original"]["spot"]
        # Front month should move more than back months
        assert result["new"]["futures"][0] - result["original"]["futures"][0] > \
               result["new"]["futures"][-1] - result["original"]["futures"][-1]

    def test_simulate_term_structure_empty_futures(self):
        result = SimulationTools.simulate_term_structure_change(
            vix_spot=15,
            vix_futures=[],
            shock_percent=20,
        )
        assert "error" in result

    def test_calculate_var(self):
        result = SimulationTools.calculate_var(
            position_value=100000,
            volatility=0.20,
            confidence_level=0.95,
            time_horizon_days=1,
        )

        assert "var" in result
        assert "var_percent" in result
        assert result["var"] > 0
        assert result["confidence_level"] == 0.95


class TestSimulationAgent:
    """Tests for SimulationAgent."""

    @pytest.fixture
    def mock_llm(self):
        llm = MagicMock()
        llm.generate.return_value = "Simulation summary"
        return llm

    @pytest.fixture
    def ontology(self):
        return create_quant_finance_ontology()

    @pytest.fixture
    def store(self, ontology):
        store = NetworkXStore()
        store.initialize(ontology)

        # Add sample entities
        store.add_entity(
            Entity(
                id="strat_1",
                node_type="Strategy",
                properties={"name": "Long Straddle", "strategy_type": "long_straddle"},
            )
        )
        store.add_entity(
            Entity(
                id="inst_vix",
                node_type="Instrument",
                properties={"symbol": "VIX", "name": "CBOE Volatility Index"},
            )
        )

        return store

    def test_basic_scenario_parse(self, mock_llm, ontology, store):
        agent = SimulationAgent(mock_llm, ontology, store)

        scenario = agent._basic_scenario_parse("What if VIX spikes 20%?")

        assert "VIX" in scenario.affected_instruments
        assert scenario.market_changes.get("vix_change") == 20

    def test_basic_scenario_parse_straddle(self, mock_llm, ontology, store):
        agent = SimulationAgent(mock_llm, ontology, store)

        scenario = agent._basic_scenario_parse(
            "What happens to my straddle if VIX doubles?"
        )

        assert "long_straddle" in scenario.affected_strategies
        assert scenario.market_changes.get("vix_change") == 100

    def test_simulate(self, mock_llm, ontology, store):
        # Make LLM fail to use fallback
        mock_llm.generate_structured.side_effect = ValueError("Mock error")

        agent = SimulationAgent(mock_llm, ontology, store)
        world_state = WorldState(original_query="What if VIX spikes?")

        result = agent.simulate("What if VIX spikes 20%?", world_state)

        assert isinstance(result, SimulationResult)
        assert result.scenario != ""
        assert len(result.caveats) > 0  # Should have caveats

    def test_calculate_impacts(self, mock_llm, ontology, store):
        agent = SimulationAgent(mock_llm, ontology, store)

        scenario = ScenarioDefinition(
            name="VIX Spike",
            description="VIX increases 20%",
            market_changes={"vix_change": 20},
            affected_strategies=["long_straddle", "iron_condor"],
        )

        impacts = agent._calculate_impacts(scenario, [])

        assert len(impacts) == 2
        # Long straddle should profit
        long_straddle = next(i for i in impacts if "straddle" in i.strategy_name.lower())
        assert long_straddle.pnl_estimate > 0

    def test_estimate_confidence(self, mock_llm, ontology, store):
        agent = SimulationAgent(mock_llm, ontology, store)

        # Normal scenario
        scenario = ScenarioDefinition(
            name="Normal",
            description="Test",
            market_changes={"vix_change": 10},
        )
        confidence = agent._estimate_confidence(scenario, [])
        assert confidence >= 0.6

        # Extreme scenario
        extreme_scenario = ScenarioDefinition(
            name="Extreme",
            description="Test",
            market_changes={"vix_change": 150},
        )
        extreme_confidence = agent._estimate_confidence(extreme_scenario, [])
        assert extreme_confidence < confidence

    def test_generate_caveats(self, mock_llm, ontology, store):
        agent = SimulationAgent(mock_llm, ontology, store)

        # Normal scenario
        scenario = ScenarioDefinition(
            name="Normal",
            description="Test",
            market_changes={"vix_change": 10},
        )
        caveats = agent._generate_caveats(scenario)
        assert len(caveats) >= 2  # Standard caveats

        # Large change scenario
        extreme_scenario = ScenarioDefinition(
            name="Extreme",
            description="Test",
            market_changes={"vix_change": 50},
        )
        extreme_caveats = agent._generate_caveats(extreme_scenario)
        assert len(extreme_caveats) > len(caveats)  # Extra caveat for large change
