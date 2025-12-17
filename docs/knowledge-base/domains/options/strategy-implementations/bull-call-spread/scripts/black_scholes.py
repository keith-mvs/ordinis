#!/usr/bin/env python3
"""Black-Scholes Options Pricing Module

Pure implementation of Black-Scholes formulas for European call and put options.
Includes pricing and all Greeks calculations.
"""

import numpy as np
from scipy.stats import norm


class BlackScholes:
    """Black-Scholes options pricing and Greeks calculator."""

    @staticmethod
    def calculate_d1_d2(
        S: float, K: float, T: float, r: float, sigma: float
    ) -> tuple[float, float]:
        """Calculate d1 and d2 terms in Black-Scholes formula."""
        if S <= 0 or K <= 0 or T < 0 or sigma <= 0:
            raise ValueError("Invalid parameters")
        if T == 0:
            return (
                float("inf") if S > K else float("-inf"),
                float("inf") if S > K else float("-inf"),
            )
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2

    @staticmethod
    def call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate European call option price."""
        if T == 0:
            return max(S - K, 0)
        d1, d2 = BlackScholes.calculate_d1_d2(S, K, T, r, sigma)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    @staticmethod
    def put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate European put option price."""
        if T == 0:
            return max(K - S, 0)
        d1, d2 = BlackScholes.calculate_d1_d2(S, K, T, r, sigma)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    @staticmethod
    def call_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate call option delta."""
        if T == 0:
            return 1.0 if S > K else 0.0
        d1, _ = BlackScholes.calculate_d1_d2(S, K, T, r, sigma)
        return norm.cdf(d1)

    @staticmethod
    def put_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate put option delta."""
        if T == 0:
            return -1.0 if S < K else 0.0
        d1, _ = BlackScholes.calculate_d1_d2(S, K, T, r, sigma)
        return norm.cdf(d1) - 1

    @staticmethod
    def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate option gamma."""
        if T == 0:
            return 0.0
        d1, _ = BlackScholes.calculate_d1_d2(S, K, T, r, sigma)
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))

    @staticmethod
    def call_theta(
        S: float, K: float, T: float, r: float, sigma: float, annual: bool = False
    ) -> float:
        """Calculate call option theta."""
        if T == 0:
            return 0.0
        d1, d2 = BlackScholes.calculate_d1_d2(S, K, T, r, sigma)
        theta_annual = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(
            -r * T
        ) * norm.cdf(d2)
        return theta_annual if annual else theta_annual / 365

    @staticmethod
    def put_theta(
        S: float, K: float, T: float, r: float, sigma: float, annual: bool = False
    ) -> float:
        """Calculate put option theta."""
        if T == 0:
            return 0.0
        d1, d2 = BlackScholes.calculate_d1_d2(S, K, T, r, sigma)
        theta_annual = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(
            -r * T
        ) * norm.cdf(-d2)
        return theta_annual if annual else theta_annual / 365

    @staticmethod
    def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate option vega."""
        if T == 0:
            return 0.0
        d1, _ = BlackScholes.calculate_d1_d2(S, K, T, r, sigma)
        return S * norm.pdf(d1) * np.sqrt(T) / 100

    @staticmethod
    def call_rho(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate call option rho."""
        if T == 0:
            return 0.0
        _, d2 = BlackScholes.calculate_d1_d2(S, K, T, r, sigma)
        return K * T * np.exp(-r * T) * norm.cdf(d2) / 100

    @staticmethod
    def put_rho(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate put option rho."""
        if T == 0:
            return 0.0
        _, d2 = BlackScholes.calculate_d1_d2(S, K, T, r, sigma)
        return -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    @staticmethod
    def calculate_all_greeks(
        S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call"
    ) -> dict[str, float]:
        """Calculate all Greeks for an option."""
        if option_type.lower() not in ["call", "put"]:
            raise ValueError("option_type must be 'call' or 'put'")
        is_call = option_type.lower() == "call"
        return {
            "price": BlackScholes.call_price(S, K, T, r, sigma)
            if is_call
            else BlackScholes.put_price(S, K, T, r, sigma),
            "delta": BlackScholes.call_delta(S, K, T, r, sigma)
            if is_call
            else BlackScholes.put_delta(S, K, T, r, sigma),
            "gamma": BlackScholes.gamma(S, K, T, r, sigma),
            "theta": BlackScholes.call_theta(S, K, T, r, sigma)
            if is_call
            else BlackScholes.put_theta(S, K, T, r, sigma),
            "vega": BlackScholes.vega(S, K, T, r, sigma),
            "rho": BlackScholes.call_rho(S, K, T, r, sigma)
            if is_call
            else BlackScholes.put_rho(S, K, T, r, sigma),
        }


if __name__ == "__main__":
    print("Black-Scholes Calculator Example\n")
    S, K, T, r, sigma = 100, 100, 0.25, 0.05, 0.20
    print(f"Parameters: S=${S}, K=${K}, T={T}yr, r={r}, sigma={sigma}\n")
    call = BlackScholes.calculate_all_greeks(S, K, T, r, sigma, "call")
    print("Call Option:")
    for k, v in call.items():
        print(f"  {k.capitalize():8s}: {v:10.4f}")
    put = BlackScholes.calculate_all_greeks(S, K, T, r, sigma, "put")
    print("\nPut Option:")
    for k, v in put.items():
        print(f"  {k.capitalize():8s}: {v:10.4f}")
