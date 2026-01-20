"""Simulation tools for quantitative finance scenarios.

Provides utilities for simulating market scenarios, calculating
option Greeks, and evaluating strategy performance.
"""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, Field


class ScenarioInput(BaseModel):
    """Input parameters for a scenario simulation."""

    base_price: float = Field(default=100.0, description="Base price")
    volatility_change: float = Field(default=0.0, description="Change in volatility (percentage points)")
    price_change: float = Field(default=0.0, description="Change in underlying price (percentage)")
    time_decay_days: int = Field(default=0, description="Days of time decay to simulate")
    interest_rate: float = Field(default=0.05, description="Risk-free interest rate")


class GreeksResult(BaseModel):
    """Result of Greeks calculation."""

    delta: float = Field(default=0.0, description="Price sensitivity")
    gamma: float = Field(default=0.0, description="Delta sensitivity")
    theta: float = Field(default=0.0, description="Time decay per day")
    vega: float = Field(default=0.0, description="Volatility sensitivity")
    rho: float = Field(default=0.0, description="Interest rate sensitivity")


class StrategyImpact(BaseModel):
    """Impact of a scenario on a strategy."""

    strategy_name: str = Field(..., description="Name of the strategy")
    pnl_estimate: float = Field(default=0.0, description="Estimated P&L")
    pnl_range: tuple[float, float] = Field(default=(0.0, 0.0), description="P&L range (low, high)")
    risk_factors: list[str] = Field(default_factory=list, description="Dominant risk factors")
    recommendation: str = Field(default="", description="Action recommendation")


class SimulationTools:
    """Tools for financial scenario simulation."""

    @staticmethod
    def calculate_black_scholes_greeks(
        spot: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        risk_free_rate: float,
        is_call: bool = True,
    ) -> GreeksResult:
        """Calculate Black-Scholes Greeks for an option.

        Args:
            spot: Current spot price
            strike: Strike price
            time_to_expiry: Time to expiry in years
            volatility: Implied volatility (decimal, e.g., 0.20 for 20%)
            risk_free_rate: Risk-free rate (decimal)
            is_call: True for call, False for put

        Returns:
            GreeksResult with all Greeks
        """
        if time_to_expiry <= 0 or volatility <= 0:
            return GreeksResult()

        sqrt_t = math.sqrt(time_to_expiry)
        d1 = (math.log(spot / strike) + (risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / (
            volatility * sqrt_t
        )
        d2 = d1 - volatility * sqrt_t

        # Standard normal PDF and CDF
        def norm_pdf(x: float) -> float:
            return math.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)

        def norm_cdf(x: float) -> float:
            return (1 + math.erf(x / math.sqrt(2))) / 2

        # Greeks
        if is_call:
            delta = norm_cdf(d1)
        else:
            delta = norm_cdf(d1) - 1

        gamma = norm_pdf(d1) / (spot * volatility * sqrt_t)
        vega = spot * norm_pdf(d1) * sqrt_t / 100  # Per 1% vol change
        theta = (
            -spot * norm_pdf(d1) * volatility / (2 * sqrt_t)
            - risk_free_rate * strike * math.exp(-risk_free_rate * time_to_expiry) * norm_cdf(d2 if is_call else -d2)
        ) / 365  # Per day

        rho = (
            strike * time_to_expiry * math.exp(-risk_free_rate * time_to_expiry) * norm_cdf(d2 if is_call else -d2)
        ) / 100  # Per 1% rate change

        return GreeksResult(
            delta=round(delta, 4),
            gamma=round(gamma, 4),
            theta=round(theta, 4),
            vega=round(vega, 4),
            rho=round(rho, 4),
        )

    @staticmethod
    def simulate_straddle_pnl(
        spot: float,
        strike: float,
        call_premium: float,
        put_premium: float,
        new_spot: float,
        new_iv: float,
        days_passed: int = 0,
        original_iv: float = 0.20,
        time_to_expiry: float = 0.1,
    ) -> dict[str, Any]:
        """Simulate P&L for a long straddle position.

        Args:
            spot: Original spot price
            strike: Strike price
            call_premium: Premium paid for call
            put_premium: Premium paid for put
            new_spot: New spot price
            new_iv: New implied volatility
            days_passed: Days that have passed
            original_iv: Original implied volatility
            time_to_expiry: Original time to expiry in years

        Returns:
            Dictionary with P&L breakdown
        """
        total_premium = call_premium + put_premium
        new_tte = max(0.001, time_to_expiry - days_passed / 365)

        # Intrinsic values at new spot
        call_intrinsic = max(0, new_spot - strike)
        put_intrinsic = max(0, strike - new_spot)

        # Estimate new premiums using simplified pricing
        # This is a rough approximation for simulation purposes
        iv_impact_call = (new_iv - original_iv) * spot * 0.4 * math.sqrt(new_tte)
        iv_impact_put = (new_iv - original_iv) * spot * 0.4 * math.sqrt(new_tte)

        time_decay = total_premium * (1 - math.sqrt(new_tte / time_to_expiry)) * 0.5

        estimated_call_value = call_intrinsic + max(0, call_premium - time_decay / 2 + iv_impact_call)
        estimated_put_value = put_intrinsic + max(0, put_premium - time_decay / 2 + iv_impact_put)

        new_total = estimated_call_value + estimated_put_value
        pnl = new_total - total_premium

        return {
            "original_premium": total_premium,
            "new_value": new_total,
            "pnl": pnl,
            "pnl_percent": pnl / total_premium * 100 if total_premium > 0 else 0,
            "components": {
                "intrinsic_call": call_intrinsic,
                "intrinsic_put": put_intrinsic,
                "iv_impact": iv_impact_call + iv_impact_put,
                "time_decay": -time_decay,
            },
        }

    @staticmethod
    def estimate_vix_spike_impact(
        vix_change_percent: float,
        strategy_type: str,
        position_size: float = 100000,
    ) -> StrategyImpact:
        """Estimate impact of VIX spike on different strategy types.

        Args:
            vix_change_percent: VIX change (e.g., 20 for 20% increase)
            strategy_type: Type of strategy (straddle, strangle, iron_condor, etc.)
            position_size: Notional position size

        Returns:
            StrategyImpact with estimated effects
        """
        strategy_type = strategy_type.lower()

        # Strategy sensitivity to VIX (rough multipliers)
        sensitivities = {
            "long_straddle": 1.5,  # Profits from vol
            "long_strangle": 1.2,
            "short_straddle": -1.5,  # Loses from vol
            "short_strangle": -1.2,
            "iron_condor": -0.8,  # Short vol
            "butterfly": -0.3,
            "calendar_spread": 0.5,
            "vix_call": 2.0,  # Direct VIX exposure
            "vix_put": -2.0,
        }

        sensitivity = sensitivities.get(strategy_type, 0.0)
        pnl_percent = sensitivity * vix_change_percent / 100
        pnl_estimate = position_size * pnl_percent

        # Determine risk factors
        risk_factors = []
        if abs(sensitivity) > 1:
            risk_factors.append("High vega exposure")
        if sensitivity < 0:
            risk_factors.append("Short volatility risk")
        if vix_change_percent > 30:
            risk_factors.append("Extreme volatility scenario")

        # Generate recommendation
        if pnl_estimate > position_size * 0.1:
            recommendation = "Position benefits significantly from this scenario"
        elif pnl_estimate < -position_size * 0.1:
            recommendation = "Consider hedging or reducing position"
        else:
            recommendation = "Moderate impact, monitor position"

        return StrategyImpact(
            strategy_name=strategy_type,
            pnl_estimate=round(pnl_estimate, 2),
            pnl_range=(round(pnl_estimate * 0.7, 2), round(pnl_estimate * 1.3, 2)),
            risk_factors=risk_factors,
            recommendation=recommendation,
        )

    @staticmethod
    def simulate_term_structure_change(
        vix_spot: float,
        vix_futures: list[float],
        shock_percent: float,
    ) -> dict[str, Any]:
        """Simulate change in VIX term structure.

        Args:
            vix_spot: Current VIX spot level
            vix_futures: List of VIX futures prices (front to back)
            shock_percent: Percentage shock to apply

        Returns:
            Dictionary with term structure analysis
        """
        if not vix_futures:
            return {"error": "No futures data provided"}

        # Calculate current term structure
        front_month = vix_futures[0]
        is_contango = front_month > vix_spot

        # Apply shock (spot typically moves more than futures)
        new_spot = vix_spot * (1 + shock_percent / 100)
        shock_dampening = [1.0, 0.8, 0.6, 0.5, 0.4]  # Decreasing impact
        new_futures = []

        for i, fut in enumerate(vix_futures):
            dampening = shock_dampening[min(i, len(shock_dampening) - 1)]
            new_futures.append(fut * (1 + shock_percent / 100 * dampening))

        new_is_contango = new_futures[0] > new_spot if new_futures else is_contango

        return {
            "original": {
                "spot": vix_spot,
                "futures": vix_futures,
                "structure": "contango" if is_contango else "backwardation",
            },
            "new": {
                "spot": round(new_spot, 2),
                "futures": [round(f, 2) for f in new_futures],
                "structure": "contango" if new_is_contango else "backwardation",
            },
            "structure_changed": is_contango != new_is_contango,
            "spot_change": round(new_spot - vix_spot, 2),
        }

    @staticmethod
    def calculate_var(
        position_value: float,
        volatility: float,
        confidence_level: float = 0.95,
        time_horizon_days: int = 1,
    ) -> dict[str, float]:
        """Calculate Value at Risk (VaR) for a position.

        Args:
            position_value: Current position value
            volatility: Position volatility (annualized, decimal)
            confidence_level: VaR confidence level
            time_horizon_days: Time horizon in days

        Returns:
            Dictionary with VaR metrics
        """
        # Z-scores for common confidence levels
        z_scores = {
            0.90: 1.28,
            0.95: 1.645,
            0.99: 2.326,
        }
        z = z_scores.get(confidence_level, 1.645)

        # Convert to daily volatility
        daily_vol = volatility / math.sqrt(252)

        # Scale to time horizon
        period_vol = daily_vol * math.sqrt(time_horizon_days)

        var = position_value * z * period_vol
        expected_shortfall = var * 1.25  # Rough approximation

        return {
            "var": round(var, 2),
            "var_percent": round(var / position_value * 100, 2),
            "expected_shortfall": round(expected_shortfall, 2),
            "confidence_level": confidence_level,
            "time_horizon_days": time_horizon_days,
        }
