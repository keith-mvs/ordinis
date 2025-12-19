"""
Black-Scholes Options Pricing Engine

Reference implementation of the Black-Scholes-Merton model for European options pricing.
Used as the canonical pricing engine for all options strategies in the Ordinis platform.

Mathematical Foundation:
    Black, F., & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities.
    Journal of Political Economy, 81(3), 637-654.

Author: Ordinis Project
License: MIT
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.stats import norm


class OptionType(Enum):
    """Option contract type."""

    CALL = "call"
    PUT = "put"


@dataclass
class PricingParameters:
    """
    Parameters for Black-Scholes pricing.

    Attributes:
        S: Current underlying asset price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free interest rate (annual, as decimal)
        sigma: Implied volatility (annual, as decimal)
        q: Dividend yield (annual, as decimal), default 0
    """

    S: float
    K: float
    T: float
    r: float
    sigma: float
    q: float = 0.0

    def __post_init__(self):
        """Validate parameters."""
        if self.S <= 0:
            raise ValueError("Underlying price must be positive")
        if self.K <= 0:
            raise ValueError("Strike price must be positive")
        if self.T < 0:
            raise ValueError("Time to expiration cannot be negative")
        if self.sigma < 0:
            raise ValueError("Volatility cannot be negative")


class BlackScholesEngine:
    """
    Black-Scholes-Merton pricing engine for European options.

    This engine provides the foundational pricing mechanism used by all
    options strategy skills in the Ordinis platform.

    Usage:
        >>> engine = BlackScholesEngine()
        >>> params = PricingParameters(S=100, K=100, T=0.25, r=0.05, sigma=0.20)
        >>> call_price = engine.price_call(params)
        >>> put_price = engine.price_put(params)

    Integration:
        - Called by GreeksCalculator for risk metrics
        - Used by strategy modules (covered_call.py, etc.) for leg pricing
        - Referenced by skill packages for educational pricing examples
    """

    def __init__(self):
        """Initialize Black-Scholes pricing engine."""

    @staticmethod
    def _d1(params: PricingParameters) -> float:
        """
        Calculate d1 component of Black-Scholes formula.

        d1 = [ln(S/K) + (r - q + σ²/2)T] / (σ√T)
        """
        if params.T == 0:
            return float("inf") if params.S > params.K else float("-inf")

        numerator = (
            np.log(params.S / params.K) + (params.r - params.q + params.sigma**2 / 2) * params.T
        )
        denominator = params.sigma * np.sqrt(params.T)

        return numerator / denominator

    @staticmethod
    def _d2(params: PricingParameters, d1: float | None = None) -> float:
        """
        Calculate d2 component of Black-Scholes formula.

        d2 = d1 - σ√T
        """
        if params.T == 0:
            return float("inf") if params.S > params.K else float("-inf")

        if d1 is None:
            d1 = BlackScholesEngine._d1(params)

        return d1 - params.sigma * np.sqrt(params.T)

    def price_call(self, params: PricingParameters) -> float:
        """
        Calculate European call option price.

        Formula:
            C = S·e^(-qT)·N(d1) - K·e^(-rT)·N(d2)

        Args:
            params: Pricing parameters

        Returns:
            Call option price

        Example:
            >>> engine = BlackScholesEngine()
            >>> params = PricingParameters(S=100, K=100, T=0.25, r=0.05, sigma=0.20)
            >>> price = engine.price_call(params)
            >>> print(f"Call price: ${price:.2f}")
        """
        if params.T == 0:
            return max(params.S - params.K, 0)

        d1 = self._d1(params)
        d2 = self._d2(params, d1)

        call_price = params.S * np.exp(-params.q * params.T) * norm.cdf(d1) - params.K * np.exp(
            -params.r * params.T
        ) * norm.cdf(d2)

        return call_price

    def price_put(self, params: PricingParameters) -> float:
        """
        Calculate European put option price.

        Formula:
            P = K·e^(-rT)·N(-d2) - S·e^(-qT)·N(-d1)

        Args:
            params: Pricing parameters

        Returns:
            Put option price

        Example:
            >>> engine = BlackScholesEngine()
            >>> params = PricingParameters(S=100, K=100, T=0.25, r=0.05, sigma=0.20)
            >>> price = engine.price_put(params)
            >>> print(f"Put price: ${price:.2f}")
        """
        if params.T == 0:
            return max(params.K - params.S, 0)

        d1 = self._d1(params)
        d2 = self._d2(params, d1)

        put_price = params.K * np.exp(-params.r * params.T) * norm.cdf(-d2) - params.S * np.exp(
            -params.q * params.T
        ) * norm.cdf(-d1)

        return put_price

    def price(self, params: PricingParameters, option_type: OptionType) -> float:
        """
        Price an option of specified type.

        Args:
            params: Pricing parameters
            option_type: CALL or PUT

        Returns:
            Option price
        """
        if option_type == OptionType.CALL:
            return self.price_call(params)
        if option_type == OptionType.PUT:
            return self.price_put(params)
        raise ValueError(f"Unknown option type: {option_type}")

    def implied_volatility(
        self,
        market_price: float,
        params: PricingParameters,
        option_type: OptionType,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ) -> float:
        """
        Calculate implied volatility using Newton-Raphson method.

        Args:
            market_price: Observed market price
            params: Pricing parameters (sigma will be solved for)
            option_type: CALL or PUT
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance

        Returns:
            Implied volatility (annual, as decimal)

        Raises:
            ValueError: If convergence fails
        """
        from .greeks import GreeksCalculator

        greeks_calc = GreeksCalculator(self)

        # Initial guess
        sigma = 0.25

        for i in range(max_iterations):
            # Create temporary params with current sigma
            temp_params = PricingParameters(
                S=params.S, K=params.K, T=params.T, r=params.r, sigma=sigma, q=params.q
            )

            # Calculate price and vega
            price = self.price(temp_params, option_type)
            vega = greeks_calc.vega(temp_params)

            # Check convergence
            diff = price - market_price
            if abs(diff) < tolerance:
                return sigma

            # Newton-Raphson update
            if vega == 0:
                raise ValueError("Vega is zero, cannot calculate IV")

            sigma = sigma - diff / (vega * 100)  # vega is per 1% change

            # Ensure sigma stays positive
            if sigma <= 0:
                sigma = 0.01

        raise ValueError(f"IV calculation did not converge after {max_iterations} iterations")


if __name__ == "__main__":
    # Example usage
    print("=== Black-Scholes Pricing Engine ===\n")

    engine = BlackScholesEngine()

    # Example 1: ATM options
    print("Example 1: ATM Call and Put")
    params = PricingParameters(S=100.0, K=100.0, T=0.25, r=0.05, sigma=0.20)

    call_price = engine.price_call(params)
    put_price = engine.price_put(params)

    print(f"Underlying: ${params.S:.2f}")
    print(f"Strike: ${params.K:.2f}")
    print(f"Time: {params.T*365:.0f} days")
    print(f"IV: {params.sigma*100:.1f}%")
    print(f"\nCall Price: ${call_price:.2f}")
    print(f"Put Price: ${put_price:.2f}")

    # Verify put-call parity
    parity_lhs = call_price - put_price
    parity_rhs = params.S - params.K * np.exp(-params.r * params.T)
    print("\nPut-Call Parity Check:")
    print(f"C - P = ${parity_lhs:.4f}")
    print(f"S - K·e^(-rT) = ${parity_rhs:.4f}")
    print(f"Difference: ${abs(parity_lhs - parity_rhs):.6f}")
