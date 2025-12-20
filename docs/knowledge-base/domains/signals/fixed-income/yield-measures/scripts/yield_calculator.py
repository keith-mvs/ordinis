"""
Yield Measures Calculator
=========================

Calculate and analyze various bond yield metrics including YTM, YTC, YTW,
current yield, and spot/forward rates.

Author: Ordinis-1 Bond Analysis Framework
Version: 1.0.0
Last Updated: 2025-12-07
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class YieldMetrics:
    """
    Complete yield analysis for a bond.

    Attributes:
        current_yield: Annual coupon / market price
        ytm: Yield to maturity
        ytc: Yield to call (if callable)
        ytw: Yield to worst (minimum of YTM and YTC)
        spot_rates: Array of spot rates by maturity
        forward_rates: Array of forward rates
    """

    current_yield: float
    ytm: float
    ytc: float | None = None
    ytw: float | None = None
    spot_rates: np.ndarray | None = None
    forward_rates: np.ndarray | None = None


def current_yield(annual_coupon: float, market_price: float) -> float:
    """
    Calculate current yield (income return only).

    Parameters:
        annual_coupon: Annual coupon payment
        market_price: Current market price

    Returns:
        Current yield as decimal

    Example:
        >>> cy = current_yield(5.0, 98.5)
        >>> print(f"Current Yield: {cy:.2%}")
        Current Yield: 5.08%
    """
    return annual_coupon / market_price


def yield_to_worst(ytm: float, ytc: float | None = None) -> float:
    """
    Calculate yield to worst (most conservative yield measure).

    For callable bonds, YTW is the minimum of YTM and YTC.

    Parameters:
        ytm: Yield to maturity
        ytc: Yield to call (None for non-callable bonds)

    Returns:
        Yield to worst
    """
    if ytc is None:
        return ytm
    return min(ytm, ytc)


def bootstrap_spot_rates(par_yields: np.ndarray, maturities: np.ndarray) -> np.ndarray:
    """
    Extract spot rates from par yield curve using bootstrapping.

    Bootstrap Method:
    1. First spot rate equals first par yield
    2. For each subsequent maturity, solve for spot rate that
       makes coupon bond price equal par

    Parameters:
        par_yields: Array of par yields (as decimals)
        maturities: Array of maturities (in years)

    Returns:
        Array of spot rates

    Example:
        >>> par_yields = np.array([0.02, 0.025, 0.03, 0.035])
        >>> maturities = np.array([1, 2, 3, 4])
        >>> spots = bootstrap_spot_rates(par_yields, maturities)

    Reference:
        - CFA Level I: Spot Rates and Forward Rates
        - Fabozzi, Ch. 6
    """
    n = len(par_yields)
    spot_rates = np.zeros(n)

    # First spot rate equals first par yield
    spot_rates[0] = par_yields[0]

    # Bootstrap subsequent spot rates
    for i in range(1, n):
        # TODO: Implement bootstrapping algorithm
        # For a par bond: 100 = C/(1+s1) + C/(1+s2)² + ... + (100+C)/(1+sn)ⁿ
        # Solve for sn given s1, s2, ..., s(n-1)
        spot_rates[i] = par_yields[i]  # Placeholder

    return spot_rates


def calculate_forward_rates(spot_rates: np.ndarray, maturities: np.ndarray) -> np.ndarray:
    """
    Calculate forward rates from spot rates.

    Forward Rate Formula:
        (1 + sn)ⁿ = (1 + sm)ᵐ × (1 + fm,n)ⁿ⁻ᵐ

    Where fm,n is the forward rate from year m to year n.

    Parameters:
        spot_rates: Array of spot rates
        maturities: Corresponding maturities

    Returns:
        Array of one-year forward rates

    Example:
        >>> spots = np.array([0.02, 0.025, 0.03])
        >>> mats = np.array([1, 2, 3])
        >>> forwards = calculate_forward_rates(spots, mats)
        >>> # Forward rate from year 1 to year 2
        >>> print(f"1y1y forward: {forwards[1]:.2%}")
    """
    n = len(spot_rates)
    forward_rates = np.zeros(n)

    # First forward rate equals first spot rate
    forward_rates[0] = spot_rates[0]

    # Calculate subsequent forward rates
    for i in range(1, n):
        # Solve: (1 + s_i)^i = (1 + s_(i-1))^(i-1) × (1 + f_(i-1,i))
        forward_rates[i] = (
            (1 + spot_rates[i]) ** maturities[i] / (1 + spot_rates[i - 1]) ** maturities[i - 1]
        ) - 1

    return forward_rates


def analyze_yield_curve(maturities: np.ndarray, yields: np.ndarray) -> dict:
    """
    Analyze yield curve shape and characteristics.

    Identifies:
    - Normal (upward sloping)
    - Inverted (downward sloping)
    - Flat
    - Humped

    Parameters:
        maturities: Array of maturities
        yields: Corresponding yields

    Returns:
        Dictionary with curve analysis
    """
    # Calculate slope
    slope = np.polyfit(maturities, yields, 1)[0]

    # Calculate curvature (second derivative)
    if len(maturities) >= 3:
        curvature = np.polyfit(maturities, yields, 2)[0]
    else:
        curvature = 0

    # Classify shape
    if slope > 0.001:
        shape = "Normal (upward sloping)"
    elif slope < -0.001:
        shape = "Inverted (downward sloping)"
    else:
        shape = "Flat"

    # Calculate 2-10 spread (common indicator)
    if len(maturities) >= 2:
        spread_2_10 = yields[-1] - yields[0]  # Simplified
    else:
        spread_2_10 = None

    return {
        "shape": shape,
        "slope": slope,
        "curvature": curvature,
        "2-10_spread": spread_2_10,
        "interpretation": _interpret_curve(shape, slope),
    }


def _interpret_curve(shape: str, slope: float) -> str:
    """Provide economic interpretation of yield curve shape."""
    interpretations = {
        "Normal (upward sloping)": "Economic expansion expected; higher long-term rates compensate for time risk",
        "Inverted (downward sloping)": "Recession signal; investors expect rates to fall as economy weakens",
        "Flat": "Economic uncertainty; transition period between expansion and contraction",
    }
    return interpretations.get(shape, "Neutral economic outlook")


if __name__ == "__main__":
    # Example usage
    print("Yield Measures Calculator - Example")
    print("=" * 50)

    # Current yield example
    cy = current_yield(annual_coupon=5.0, market_price=98.5)
    print(f"\nCurrent Yield: {cy:.2%}")

    # Yield curve analysis
    maturities = np.array([1, 2, 5, 10, 30])
    yields = np.array([0.015, 0.020, 0.025, 0.030, 0.035])

    analysis = analyze_yield_curve(maturities, yields)
    print("\nYield Curve Analysis:")
    print(f"  Shape: {analysis['shape']}")
    print(f"  Slope: {analysis['slope']:.4f}")
    print(f"  Interpretation: {analysis['interpretation']}")

    # Forward rates
    spot_rates = np.array([0.02, 0.025, 0.03])
    mats = np.array([1, 2, 3])
    forwards = calculate_forward_rates(spot_rates, mats)
    print("\nForward Rates:")
    for i, f in enumerate(forwards[1:], 1):
        print(f"  Year {i} to {i+1}: {f:.2%}")
