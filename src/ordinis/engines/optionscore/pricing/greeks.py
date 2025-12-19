"""
Greeks Calculator for Options Risk Management

Provides comprehensive Greeks calculation (Delta, Gamma, Theta, Vega, Rho) for
individual options and multi-leg strategies.

Mathematical foundations from Hull, J.C. (2018). Options, Futures, and Other Derivatives, 10th ed.

Author: Ordinis Project
License: MIT
"""

import numpy as np
from scipy.stats import norm

from .black_scholes import BlackScholesEngine, OptionType, PricingParameters


class GreeksCalculator:
    """
    Calculate Greeks for options and strategies.

    Integrates with BlackScholesEngine for pricing and provides risk metrics
    used by all strategy modules.

    Usage:
        >>> engine = BlackScholesEngine()
        >>> calc = GreeksCalculator(engine)
        >>> params = PricingParameters(S=100, K=100, T=0.25, r=0.05, sigma=0.20)
        >>> greeks = calc.all_greeks(params, OptionType.CALL)
    """

    def __init__(self, pricing_engine: BlackScholesEngine):
        """
        Initialize Greeks calculator.

        Args:
            pricing_engine: Black-Scholes pricing engine instance
        """
        self.engine = pricing_engine

    def delta_call(self, params: PricingParameters) -> float:
        """
        Calculate delta for call option.

        Delta = ∂C/∂S = e^(-qT)·N(d1)

        Interpretation: Change in option price for $1 change in underlying.
        Range: [0, 1] for calls
        """
        if params.T == 0:
            return 1.0 if params.S > params.K else 0.0

        d1 = self.engine._d1(params)
        return np.exp(-params.q * params.T) * norm.cdf(d1)

    def delta_put(self, params: PricingParameters) -> float:
        """
        Calculate delta for put option.

        Delta = ∂P/∂S = -e^(-qT)·N(-d1)

        Range: [-1, 0] for puts
        """
        if params.T == 0:
            return -1.0 if params.S < params.K else 0.0

        d1 = self.engine._d1(params)
        return -np.exp(-params.q * params.T) * norm.cdf(-d1)

    def delta(self, params: PricingParameters, option_type: OptionType) -> float:
        """Calculate delta for specified option type."""
        if option_type == OptionType.CALL:
            return self.delta_call(params)
        return self.delta_put(params)

    def gamma(self, params: PricingParameters) -> float:
        """
        Calculate gamma (same for calls and puts).

        Gamma = ∂²V/∂S² = e^(-qT)·n(d1) / (S·σ·√T)

        Interpretation: Rate of change in delta per $1 change in underlying.
        Always positive for long options.
        """
        if params.T == 0:
            return 0.0

        d1 = self.engine._d1(params)
        gamma = (np.exp(-params.q * params.T) * norm.pdf(d1)) / (
            params.S * params.sigma * np.sqrt(params.T)
        )

        return gamma

    def theta_call(self, params: PricingParameters) -> float:
        """
        Calculate theta for call option (daily).

        Theta = ∂C/∂t (converted to daily)

        Interpretation: Change in option price per day of time decay.
        Typically negative for long options.
        """
        if params.T == 0:
            return 0.0

        d1 = self.engine._d1(params)
        d2 = self.engine._d2(params, d1)

        term1 = -(params.S * np.exp(-params.q * params.T) * norm.pdf(d1) * params.sigma) / (
            2 * np.sqrt(params.T)
        )
        term2 = params.q * params.S * np.exp(-params.q * params.T) * norm.cdf(d1)
        term3 = params.r * params.K * np.exp(-params.r * params.T) * norm.cdf(d2)

        theta_annual = term1 + term2 - term3

        # Convert to daily theta
        return theta_annual / 365

    def theta_put(self, params: PricingParameters) -> float:
        """Calculate theta for put option (daily)."""
        if params.T == 0:
            return 0.0

        d1 = self.engine._d1(params)
        d2 = self.engine._d2(params, d1)

        term1 = -(params.S * np.exp(-params.q * params.T) * norm.pdf(d1) * params.sigma) / (
            2 * np.sqrt(params.T)
        )
        term2 = params.q * params.S * np.exp(-params.q * params.T) * norm.cdf(-d1)
        term3 = params.r * params.K * np.exp(-params.r * params.T) * norm.cdf(-d2)

        theta_annual = term1 - term2 + term3

        return theta_annual / 365

    def theta(self, params: PricingParameters, option_type: OptionType) -> float:
        """Calculate theta for specified option type."""
        if option_type == OptionType.CALL:
            return self.theta_call(params)
        return self.theta_put(params)

    def vega(self, params: PricingParameters) -> float:
        """
        Calculate vega (same for calls and puts).

        Vega = ∂V/∂σ (per 1% volatility change)

        Interpretation: Change in option price per 1% change in IV.
        Always positive for long options.
        """
        if params.T == 0:
            return 0.0

        d1 = self.engine._d1(params)
        vega = params.S * np.exp(-params.q * params.T) * norm.pdf(d1) * np.sqrt(params.T)

        # Per 1% volatility change
        return vega / 100

    def rho_call(self, params: PricingParameters) -> float:
        """
        Calculate rho for call option.

        Rho = ∂C/∂r (per 1% rate change)

        Interpretation: Change in option price per 1% change in interest rates.
        """
        if params.T == 0:
            return 0.0

        d2 = self.engine._d2(params)
        rho = params.K * params.T * np.exp(-params.r * params.T) * norm.cdf(d2)

        # Per 1% rate change
        return rho / 100

    def rho_put(self, params: PricingParameters) -> float:
        """Calculate rho for put option."""
        if params.T == 0:
            return 0.0

        d2 = self.engine._d2(params)
        rho = -params.K * params.T * np.exp(-params.r * params.T) * norm.cdf(-d2)

        return rho / 100

    def rho(self, params: PricingParameters, option_type: OptionType) -> float:
        """Calculate rho for specified option type."""
        if option_type == OptionType.CALL:
            return self.rho_call(params)
        return self.rho_put(params)

    def all_greeks(self, params: PricingParameters, option_type: OptionType) -> dict[str, float]:
        """
        Calculate all Greeks for an option.

        Args:
            params: Pricing parameters
            option_type: CALL or PUT

        Returns:
            Dictionary with price and all Greeks

        Example:
            >>> engine = BlackScholesEngine()
            >>> calc = GreeksCalculator(engine)
            >>> params = PricingParameters(S=100, K=100, T=0.25, r=0.05, sigma=0.20)
            >>> greeks = calc.all_greeks(params, OptionType.CALL)
            >>> for key, value in greeks.items():
            ...     print(f"{key}: {value:.4f}")
        """
        price = self.engine.price(params, option_type)

        return {
            "price": price,
            "delta": self.delta(params, option_type),
            "gamma": self.gamma(params),
            "theta": self.theta(params, option_type),
            "vega": self.vega(params),
            "rho": self.rho(params, option_type),
        }

    def strategy_greeks(self, legs: list[dict], aggregate: bool = True) -> dict[str, float]:
        """
        Calculate aggregate Greeks for a multi-leg strategy.

        Args:
            legs: List of leg dictionaries with keys:
                  - params: PricingParameters
                  - option_type: OptionType
                  - position_side: 'long' or 'short'
                  - quantity: int
            aggregate: If True, return total Greeks; if False, return per-leg

        Returns:
            Dictionary with strategy-level Greeks

        Example (Covered Call):
            >>> legs = [
            ...     {
            ...         'params': PricingParameters(S=100, K=105, T=0.25, r=0.05, sigma=0.20),
            ...         'option_type': OptionType.CALL,
            ...         'position_side': 'short',
            ...         'quantity': 1
            ...     }
            ... ]
            >>> greeks = calc.strategy_greeks(legs)
        """
        if aggregate:
            total_greeks = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}

            for leg in legs:
                leg_greeks = self.all_greeks(leg["params"], leg["option_type"])
                multiplier = leg["quantity"] * 100  # Options are per 100 shares
                sign = -1 if leg["position_side"] == "short" else 1

                total_greeks["delta"] += leg_greeks["delta"] * multiplier * sign
                total_greeks["gamma"] += leg_greeks["gamma"] * multiplier * sign
                total_greeks["theta"] += leg_greeks["theta"] * multiplier * sign
                total_greeks["vega"] += leg_greeks["vega"] * multiplier * sign
                total_greeks["rho"] += leg_greeks["rho"] * multiplier * sign

            return total_greeks
        # Return per-leg breakdown
        return [self.all_greeks(leg["params"], leg["option_type"]) for leg in legs]


if __name__ == "__main__":
    print("=== Options Greeks Calculator ===\n")

    engine = BlackScholesEngine()
    calc = GreeksCalculator(engine)

    params = PricingParameters(S=100.0, K=100.0, T=0.25, r=0.05, sigma=0.20)

    print("ATM Call Option Greeks:")
    call_greeks = calc.all_greeks(params, OptionType.CALL)
    for greek, value in call_greeks.items():
        print(f"  {greek.capitalize()}: {value:.4f}")

    print("\nATM Put Option Greeks:")
    put_greeks = calc.all_greeks(params, OptionType.PUT)
    for greek, value in put_greeks.items():
        print(f"  {greek.capitalize()}: {value:.4f}")
