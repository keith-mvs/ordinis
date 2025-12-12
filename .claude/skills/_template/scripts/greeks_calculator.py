#!/usr/bin/env python3
"""
Options Greeks Calculator Template

Calculates option Greeks (Delta, Gamma, Theta, Vega, Rho) for strategies.
Uses Black-Scholes model for European options.

Usage:
    from greeks_calculator import calculate_greeks, BlackScholes

    # Single option Greeks
    greeks = calculate_greeks(
        spot=450, strike=445, time=0.123,
        rate=0.05, volatility=0.20, option_type='call'
    )

    # Strategy Greeks (combine multiple options)
    position_greeks = calculate_position_greeks([...])

Customize for your strategy by:
1. Use calculate_greeks() for single options
2. Implement calculate_position_greeks() to combine Greeks for multi-leg strategies
3. Add strategy-specific Greek calculations if needed

Author: Ordinis-1 Project
Version: 1.0.0
Python: 3.11+
"""

import numpy as np
from scipy.stats import norm
from typing import Dict, List, Optional


class BlackScholes:
    """
    Black-Scholes option pricing model for European options.

    Calculates theoretical option prices and Greeks.
    """

    @staticmethod
    def d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d1 term in Black-Scholes formula."""
        return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    @staticmethod
    def d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d2 term in Black-Scholes formula."""
        return BlackScholes.d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

    @staticmethod
    def call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate European call option price."""
        if T <= 0:
            return max(0, S - K)

        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)

        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    @staticmethod
    def put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate European put option price."""
        if T <= 0:
            return max(0, K - S)

        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)

        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def calculate_greeks(spot: float,
                    strike: float,
                    time: float,
                    rate: float,
                    volatility: float,
                    option_type: str = 'call') -> Dict[str, float]:
    """
    Calculate all Greeks for a single option.

    Args:
        spot: Current underlying price
        strike: Option strike price
        time: Time to expiration in years
        rate: Risk-free interest rate (decimal)
        volatility: Implied volatility (decimal)
        option_type: 'call' or 'put'

    Returns:
        Dictionary with delta, gamma, theta, vega, rho

    Example:
        >>> greeks = calculate_greeks(
        ...     spot=450, strike=445, time=45/365,
        ...     rate=0.05, volatility=0.20, option_type='call'
        ... )
        >>> print(f"Delta: {greeks['delta']:.4f}")
    """
    if time <= 0:
        # At expiration
        if option_type == 'call':
            delta = 1.0 if spot > strike else 0.0
        else:
            delta = -1.0 if spot < strike else 0.0

        return {
            'delta': delta,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'rho': 0.0
        }

    # Calculate d1, d2
    d1 = BlackScholes.d1(spot, strike, time, rate, volatility)
    d2 = BlackScholes.d2(spot, strike, time, rate, volatility)

    # Delta
    if option_type == 'call':
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1

    # Gamma (same for calls and puts)
    gamma = norm.pdf(d1) / (spot * volatility * np.sqrt(time))

    # Vega (same for calls and puts, divided by 100 for 1% change)
    vega = spot * norm.pdf(d1) * np.sqrt(time) / 100

    # Theta (per day, so divide by 365)
    if option_type == 'call':
        theta = (-(spot * norm.pdf(d1) * volatility) / (2 * np.sqrt(time))
                 - rate * strike * np.exp(-rate * time) * norm.cdf(d2)) / 365
    else:
        theta = (-(spot * norm.pdf(d1) * volatility) / (2 * np.sqrt(time))
                 + rate * strike * np.exp(-rate * time) * norm.cdf(-d2)) / 365

    # Rho (divided by 100 for 1% change)
    if option_type == 'call':
        rho = strike * time * np.exp(-rate * time) * norm.cdf(d2) / 100
    else:
        rho = -strike * time * np.exp(-rate * time) * norm.cdf(-d2) / 100

    return {
        'delta': float(delta),
        'gamma': float(gamma),
        'theta': float(theta),
        'vega': float(vega),
        'rho': float(rho)
    }


def calculate_position_greeks(options: List[Dict]) -> Dict[str, float]:
    """
    Calculate combined Greeks for a multi-leg options position.

    Args:
        options: List of option dictionaries, each containing:
            - spot, strike, time, rate, volatility, option_type
            - quantity: number of contracts (positive = long, negative = short)

    Returns:
        Dictionary with aggregated Greeks

    Example (Bull Call Spread):
        >>> options = [
        ...     {  # Long call
        ...         'spot': 450, 'strike': 445, 'time': 45/365,
        ...         'rate': 0.05, 'volatility': 0.20,
        ...         'option_type': 'call', 'quantity': 1
        ...     },
        ...     {  # Short call
        ...         'spot': 450, 'strike': 455, 'time': 45/365,
        ...         'rate': 0.05, 'volatility': 0.20,
        ...         'option_type': 'call', 'quantity': -1
        ...     }
        ... ]
        >>> position_greeks = calculate_position_greeks(options)
    """
    total_greeks = {
        'delta': 0.0,
        'gamma': 0.0,
        'theta': 0.0,
        'vega': 0.0,
        'rho': 0.0
    }

    for option in options:
        quantity = option.get('quantity', 1)

        greeks = calculate_greeks(
            spot=option['spot'],
            strike=option['strike'],
            time=option['time'],
            rate=option['rate'],
            volatility=option['volatility'],
            option_type=option['option_type']
        )

        # Aggregate Greeks (multiply by quantity and scale by 100 for contract)
        for greek in total_greeks:
            total_greeks[greek] += greeks[greek] * quantity * 100

    return total_greeks


if __name__ == "__main__":
    # Example usage
    print("Greeks Calculator Template - Example Usage\n")
    print("=" * 50)

    # Example 1: Single option
    print("\n1. Single Call Option Greeks:")
    greeks = calculate_greeks(
        spot=450.0,
        strike=445.0,
        time=45/365,
        rate=0.05,
        volatility=0.20,
        option_type='call'
    )

    print(f"   Delta: {greeks['delta']:.4f}")
    print(f"   Gamma: {greeks['gamma']:.4f}")
    print(f"   Theta: {greeks['theta']:.4f} (per day)")
    print(f"   Vega:  {greeks['vega']:.4f} (per 1% IV change)")
    print(f"   Rho:   {greeks['rho']:.4f} (per 1% rate change)")

    # Example 2: Bull Call Spread
    print("\n2. Bull Call Spread Position Greeks:")
    options = [
        {  # Long 445 call
            'spot': 450, 'strike': 445, 'time': 45/365,
            'rate': 0.05, 'volatility': 0.20,
            'option_type': 'call', 'quantity': 1
        },
        {  # Short 455 call
            'spot': 450, 'strike': 455, 'time': 45/365,
            'rate': 0.05, 'volatility': 0.20,
            'option_type': 'call', 'quantity': -1
        }
    ]

    position_greeks = calculate_position_greeks(options)
    print(f"   Position Delta: {position_greeks['delta']:.2f}")
    print(f"   Position Gamma: {position_greeks['gamma']:.2f}")
    print(f"   Position Theta: {position_greeks['theta']:.2f}")
    print(f"   Position Vega:  {position_greeks['vega']:.2f}")

    print("\n" + "=" * 50)
