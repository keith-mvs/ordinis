"""
Duration and Convexity Calculator
==================================

Calculate price sensitivity measures for interest rate risk management.

Implements:
- Macaulay Duration
- Modified Duration
- Effective Duration (for embedded options)
- Convexity
- Portfolio duration aggregation

Author: Ordinis-1 Bond Analysis Framework
Version: 1.0.0
Last Updated: 2025-12-07
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class DurationMetrics:
    """
    Complete duration/convexity analysis.

    Attributes:
        macaulay_duration: Weighted average time to cash flows (years)
        modified_duration: Price sensitivity to yield changes
        effective_duration: Price sensitivity including embedded options
        convexity: Second-order price sensitivity
        dollar_duration: Price change for 1bp yield change
        dv01: Dollar value of 01 basis point
    """

    macaulay_duration: float
    modified_duration: float
    effective_duration: Optional[float] = None
    convexity: float = 0.0
    dollar_duration: Optional[float] = None
    dv01: Optional[float] = None


def macaulay_duration(
    cash_flows: np.ndarray, times: np.ndarray, yield_rate: float, frequency: int = 2
) -> float:
    """
    Calculate Macaulay Duration.

    Definition: Weighted average time to receive cash flows,
                where weights are PV of each cash flow.

    Formula:
        D_Mac = Σ [t × PV(CFt)] / Bond_Price

    Parameters:
        cash_flows: Array of cash flows (coupons + principal)
        times: Time to each cash flow (in years)
        yield_rate: Yield to maturity (annual)
        frequency: Payments per year (2 for semi-annual)

    Returns:
        Macaulay duration in years

    Example:
        >>> cfs = np.array([2.5, 2.5, 2.5, 2.5, 102.5])  # 5% semi-annual
        >>> times = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        >>> ytm = 0.04
        >>> mac_dur = macaulay_duration(cfs, times, ytm, frequency=2)
        >>> print(f"Macaulay Duration: {mac_dur:.2f} years")

    Reference:
        - CFA Level I: Understanding Fixed-Income Risk and Return
        - Macaulay, F. (1938). Some Theoretical Problems...
    """
    periodic_yield = yield_rate / frequency

    # Calculate PV of each cash flow
    pv_cash_flows = cash_flows / (1 + periodic_yield) ** (times * frequency)

    # Bond price (sum of PVs)
    bond_price = np.sum(pv_cash_flows)

    # Weighted average time
    weighted_times = times * pv_cash_flows
    mac_duration = np.sum(weighted_times) / bond_price

    return mac_duration


def modified_duration(macaulay_dur: float, yield_rate: float, frequency: int = 2) -> float:
    """
    Calculate Modified Duration from Macaulay Duration.

    Formula:
        D_Mod = D_Mac / (1 + y/k)

    Where:
        y = yield to maturity
        k = compounding frequency

    Interpretation: Modified duration measures the approximate percentage
                    price change for a 1% change in yield.

    Parameters:
        macaulay_dur: Macaulay duration (years)
        yield_rate: Annual yield to maturity
        frequency: Compounding frequency per year

    Returns:
        Modified duration

    Example:
        >>> mac_dur = 4.5
        >>> mod_dur = modified_duration(mac_dur, 0.05, frequency=2)
        >>> print(f"Modified Duration: {mod_dur:.4f}")
        >>> print(f"1% yield increase → {-mod_dur:.2%} price change")
    """
    return macaulay_dur / (1 + yield_rate / frequency)


def effective_duration(
    price_down: float, price_up: float, initial_price: float, yield_change: float
) -> float:
    """
    Calculate Effective Duration (for bonds with embedded options).

    Formula:
        D_Eff = (P- - P+) / (2 × P0 × Δy)

    Where:
        P- = Price if yields fall by Δy
        P+ = Price if yields rise by Δy
        P0 = Initial price
        Δy = Yield change

    Use effective duration when:
    - Bond has embedded call/put options
    - Cash flows are path-dependent (MBS)
    - Standard duration is inaccurate

    Parameters:
        price_down: Price if yields decrease
        price_up: Price if yields increase
        initial_price: Current price
        yield_change: Size of yield shift (e.g., 0.01 for 100bp)

    Returns:
        Effective duration

    Example:
        >>> # Callable bond sensitivity
        >>> p_down = 103.5  # Price if yields -100bp
        >>> p_up = 98.2     # Price if yields +100bp
        >>> p0 = 100.5      # Current price
        >>> eff_dur = effective_duration(p_down, p_up, p0, 0.01)
        >>> print(f"Effective Duration: {eff_dur:.2f}")
    """
    return (price_down - price_up) / (2 * initial_price * yield_change)


def convexity(
    cash_flows: np.ndarray,
    times: np.ndarray,
    yield_rate: float,
    frequency: int = 2,
    bond_price: Optional[float] = None,
) -> float:
    """
    Calculate Convexity.

    Formula:
        C = [1 / (P × (1+y)²)] × Σ [t(t+1) × CFt / (1+y)^t]

    Convexity measures the curvature of the price-yield relationship,
    capturing the non-linear price changes for larger yield shifts.

    Parameters:
        cash_flows: Array of cash flows
        times: Time to each cash flow (in years)
        yield_rate: Annual yield to maturity
        frequency: Payments per year
        bond_price: Current bond price (calculated if not provided)

    Returns:
        Convexity

    Example:
        >>> cfs = np.array([2.5, 2.5, 2.5, 2.5, 102.5])
        >>> times = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        >>> conv = convexity(cfs, times, 0.04, frequency=2)
        >>> print(f"Convexity: {conv:.4f}")

    Reference:
        - CFA Level I: Understanding Fixed-Income Risk and Return
    """
    periodic_yield = yield_rate / frequency

    # Calculate bond price if not provided
    if bond_price is None:
        pv_cash_flows = cash_flows / (1 + periodic_yield) ** (times * frequency)
        bond_price = np.sum(pv_cash_flows)

    # Calculate convexity components
    periods = times * frequency
    convexity_sum = 0

    for cf, t in zip(cash_flows, periods, strict=False):
        pv_cf = cf / (1 + periodic_yield) ** t
        convexity_sum += t * (t + 1) * pv_cf

    # Convexity formula
    conv = convexity_sum / (bond_price * (1 + periodic_yield) ** 2)

    # Adjust for frequency (semi-annual → annual)
    conv = conv / (frequency**2)

    return conv


def price_change_estimate(
    modified_duration: float, convexity: float, yield_change: float
) -> tuple[float, float, float]:
    """
    Estimate bond price change using duration and convexity.

    Formulas:
        Duration Effect:  ΔP/P ≈ -D_Mod × Δy
        Convexity Effect: ΔP/P ≈ -D_Mod × Δy + (1/2) × C × (Δy)²

    Parameters:
        modified_duration: Modified duration
        convexity: Convexity
        yield_change: Change in yield (as decimal, e.g., 0.01 for 100bp)

    Returns:
        Tuple of (duration_effect, convexity_adjustment, total_change)
        All as percentages

    Example:
        >>> mod_dur = 7.5
        >>> conv = 65.0
        >>> yield_chg = 0.01  # 100bp increase
        >>> dur_effect, conv_adj, total = price_change_estimate(
        ...     mod_dur, conv, yield_chg
        ... )
        >>> print(f"Duration effect: {dur_effect:.2%}")
        >>> print(f"Convexity adjustment: {conv_adj:.2%}")
        >>> print(f"Total price change: {total:.2%}")
    """
    # Duration effect (first-order)
    duration_effect = -modified_duration * yield_change

    # Convexity adjustment (second-order)
    convexity_adjustment = 0.5 * convexity * (yield_change**2)

    # Total estimated price change
    total_change = duration_effect + convexity_adjustment

    return duration_effect, convexity_adjustment, total_change


def portfolio_duration(bond_durations: list[float], bond_weights: list[float]) -> float:
    """
    Calculate portfolio duration as weighted average.

    Formula:
        D_Portfolio = Σ (wi × Di)

    Where:
        wi = Market value weight of bond i
        Di = Duration of bond i

    Parameters:
        bond_durations: List of individual bond durations
        bond_weights: Market value weights (must sum to 1.0)

    Returns:
        Portfolio duration

    Example:
        >>> durations = [5.2, 7.8, 3.1, 9.5]
        >>> weights = [0.25, 0.35, 0.20, 0.20]
        >>> port_dur = portfolio_duration(durations, weights)
        >>> print(f"Portfolio Duration: {port_dur:.2f} years")
    """
    if abs(sum(bond_weights) - 1.0) > 1e-6:
        raise ValueError("Bond weights must sum to 1.0")

    return sum(d * w for d, w in zip(bond_durations, bond_weights, strict=False))


if __name__ == "__main__":
    # Example usage
    print("Duration and Convexity Calculator - Example")
    print("=" * 50)

    # 5-year bond, 5% coupon, 4% YTM, semi-annual
    cash_flows = np.array([2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 102.5])
    times = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    ytm = 0.04

    # Calculate metrics
    mac_dur = macaulay_duration(cash_flows, times, ytm, frequency=2)
    mod_dur = modified_duration(mac_dur, ytm, frequency=2)
    conv = convexity(cash_flows, times, ytm, frequency=2)

    print("\nBond Metrics:")
    print(f"  Macaulay Duration: {mac_dur:.2f} years")
    print(f"  Modified Duration: {mod_dur:.4f}")
    print(f"  Convexity: {conv:.2f}")

    # Price change estimate for 100bp yield increase
    yield_change = 0.01
    dur_eff, conv_adj, total = price_change_estimate(mod_dur, conv, yield_change)

    print("\nPrice Change for +100bp yield shift:")
    print(f"  Duration effect: {dur_eff:.2%}")
    print(f"  Convexity adjustment: {conv_adj:.2%}")
    print(f"  Total estimated change: {total:.2%}")
