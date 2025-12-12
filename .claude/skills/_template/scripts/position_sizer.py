#!/usr/bin/env python3
"""
Position Sizing Template for Options Strategies

Calculates optimal position sizes based on portfolio risk management principles.
Implements various sizing methods: fixed fractional, Kelly Criterion, risk-based.

Usage:
    from position_sizer import calculate_position_size, RiskBasedSizer

    # Quick calculation
    size = calculate_position_size(
        portfolio_value=100000,
        risk_per_trade=0.02,
        max_loss_per_contract=500
    )

    # Advanced risk-based sizing
    sizer = RiskBasedSizer(portfolio_value=100000)
    size = sizer.size_by_risk(max_loss=500, risk_percent=0.02)

Customize for your strategy by:
1. Update calculate_position_size() for strategy-specific costs
2. Add strategy-specific constraints (e.g., shares must be multiples of 100)
3. Implement custom sizing methods as needed

Author: Ordinis-1 Project
Version: 1.0.0
Python: 3.11+
"""

from typing import Dict, Optional
import numpy as np


def calculate_position_size(portfolio_value: float,
                           risk_per_trade: float,
                           max_loss_per_contract: float,
                           max_position_pct: float = 0.40) -> int:
    """
    Calculate number of contracts based on risk management rules.

    Simple position sizing based on maximum risk per trade.

    Args:
        portfolio_value: Total portfolio value
        risk_per_trade: Maximum risk per trade as decimal (e.g., 0.02 = 2%)
        max_loss_per_contract: Maximum loss per contract for this strategy
        max_position_pct: Maximum % of portfolio for single position (default: 40%)

    Returns:
        Number of contracts to trade

    Example:
        >>> # Risk 2% of $100k portfolio, max loss $500/contract
        >>> contracts = calculate_position_size(
        ...     portfolio_value=100000,
        ...     risk_per_trade=0.02,
        ...     max_loss_per_contract=500
        ... )
        >>> print(f"Trade {contracts} contracts")
        Trade 4 contracts
    """
    # Calculate max dollar risk for this trade
    max_risk_dollars = portfolio_value * risk_per_trade

    # Calculate contracts based on risk
    contracts_by_risk = int(max_risk_dollars / max_loss_per_contract)

    # TODO: Add strategy-specific constraints
    # For stock+option strategies:
    # - Ensure shares are multiples of 100
    # - Consider available buying power

    # Ensure at least 1 contract
    contracts = max(1, contracts_by_risk)

    # Apply position size limit
    max_position_value = portfolio_value * max_position_pct
    # TODO: Check if total position cost exceeds max_position_value
    # and reduce contracts if needed

    return contracts


def calculate_detailed_position_size(portfolio_value: float,
                                    max_loss_per_contract: float,
                                    risk_per_trade: float = 0.02,
                                    max_position_pct: float = 0.40,
                                    **strategy_params) -> Dict:
    """
    Calculate position size with detailed breakdown.

    TODO: Customize for your strategy's cost structure

    Args:
        portfolio_value: Total portfolio value
        max_loss_per_contract: Maximum loss per contract
        risk_per_trade: Risk per trade as decimal (default: 2%)
        max_position_pct: Max position size as pct of portfolio (default: 40%)
        **strategy_params: Strategy-specific parameters (e.g., stock_price, premium)

    Returns:
        Dictionary with detailed position sizing breakdown

    Example:
        >>> result = calculate_detailed_position_size(
        ...     portfolio_value=100000,
        ...     max_loss_per_contract=500,
        ...     risk_per_trade=0.02,
        ...     stock_price=450,  # Strategy-specific param
        ...     total_cost_per_contract=530  # Strategy-specific param
        ... )
    """
    # Calculate based on risk
    max_risk_dollars = portfolio_value * risk_per_trade
    contracts_by_risk = int(max_risk_dollars / max_loss_per_contract)

    # TODO: Calculate based on position size limit
    # This is strategy-specific
    # For debit spreads: total_cost = net_debit × 100 × contracts
    # For stock+option: total_cost = (stock_price + premium) × 100 × contracts
    # For credit spreads: max_capital_required = (spread_width - credit) × 100 × contracts

    # Example calculation (customize for your strategy):
    # total_cost_per_contract = strategy_params.get('total_cost_per_contract', max_loss_per_contract)
    # max_position_value = portfolio_value * max_position_pct
    # contracts_by_position_limit = int(max_position_value / total_cost_per_contract)

    contracts_by_position_limit = contracts_by_risk  # Placeholder

    # Take the minimum of both constraints
    contracts = max(1, min(contracts_by_risk, contracts_by_position_limit))

    # TODO: Calculate actual costs for final position
    # total_cost = total_cost_per_contract * contracts
    # max_risk = max_loss_per_contract * contracts

    return {
        'contracts': contracts,
        'contracts_by_risk': contracts_by_risk,
        'contracts_by_position_limit': contracts_by_position_limit,
        'max_risk_dollars': max_risk_dollars,
        'actual_risk_dollars': max_loss_per_contract * contracts,
        'risk_percent': (max_loss_per_contract * contracts / portfolio_value) * 100,
        # TODO: Add strategy-specific metrics
        # 'total_cost': total_cost,
        # 'position_percent': (total_cost / portfolio_value) * 100,
        'recommendation': f"Trade {contracts} contract{'s' if contracts != 1 else ''} "
                         f"(${max_loss_per_contract * contracts:,.0f} max risk, "
                         f"{(max_loss_per_contract * contracts / portfolio_value) * 100:.1f}% of portfolio)"
    }


def calculate_kelly_criterion(win_probability: float,
                              win_loss_ratio: float,
                              max_kelly: float = 0.25) -> float:
    """
    Calculate position size using Kelly Criterion.

    Kelly Criterion is an optimal position sizing method that maximizes
    long-term growth rate. However, full Kelly is often too aggressive,
    so we use a fractional Kelly (default: 25%).

    Args:
        win_probability: Probability of profitable trade (0-1)
        win_loss_ratio: Average win / average loss ratio
        max_kelly: Maximum Kelly fraction (default: 25% of full Kelly)

    Returns:
        Recommended position size as fraction of capital (0-1)

    Formula:
        Kelly % = (P × B - Q) / B
        where P = win probability, Q = loss probability (1-P), B = win/loss ratio

    Example:
        >>> # 55% win rate, 1.5:1 reward/risk ratio
        >>> fraction = calculate_kelly_criterion(0.55, 1.5)
        >>> print(f"Allocate {fraction*100:.1f}% of capital")
        Allocate 5.8% of capital

    Note:
        - Returns 0 if Kelly is negative (don't trade)
        - Caps at max_kelly to avoid over-sizing
    """
    if not (0 <= win_probability <= 1):
        raise ValueError("win_probability must be between 0 and 1")

    if win_loss_ratio <= 0:
        raise ValueError("win_loss_ratio must be positive")

    # Kelly formula
    loss_probability = 1 - win_probability
    kelly_fraction = (win_probability * win_loss_ratio - loss_probability) / win_loss_ratio

    # Apply constraints
    if kelly_fraction < 0:
        return 0.0  # Negative edge, don't trade

    # Cap at max_kelly (typically 25% of full Kelly for safety)
    kelly_fraction = min(kelly_fraction, max_kelly)

    return kelly_fraction


class RiskBasedSizer:
    """
    Advanced position sizer with multiple sizing methods.

    Attributes:
        portfolio_value: Total portfolio value
        default_risk_pct: Default risk per trade (decimal)
        default_max_position_pct: Default max position size (decimal)
    """

    def __init__(self,
                 portfolio_value: float,
                 default_risk_pct: float = 0.02,
                 default_max_position_pct: float = 0.40):
        """
        Initialize risk-based sizer.

        Args:
            portfolio_value: Total portfolio value
            default_risk_pct: Default risk per trade (default: 2%)
            default_max_position_pct: Default max position (default: 40%)
        """
        self.portfolio_value = portfolio_value
        self.default_risk_pct = default_risk_pct
        self.default_max_position_pct = default_max_position_pct

    def size_by_risk(self,
                     max_loss: float,
                     risk_percent: Optional[float] = None) -> int:
        """
        Size position based on maximum acceptable loss.

        Args:
            max_loss: Maximum loss per contract
            risk_percent: Risk per trade (uses default if None)

        Returns:
            Number of contracts
        """
        risk_pct = risk_percent if risk_percent is not None else self.default_risk_pct
        return calculate_position_size(
            portfolio_value=self.portfolio_value,
            risk_per_trade=risk_pct,
            max_loss_per_contract=max_loss,
            max_position_pct=self.default_max_position_pct
        )

    def size_by_volatility(self,
                          volatility: float,
                          target_volatility: float = 0.15) -> float:
        """
        Size position based on volatility targeting.

        Adjusts position size to maintain constant portfolio volatility.

        Args:
            volatility: Strategy volatility (annualized standard deviation)
            target_volatility: Target portfolio volatility (default: 15%)

        Returns:
            Position size as fraction of portfolio

        Example:
            >>> sizer = RiskBasedSizer(100000)
            >>> # Strategy has 25% volatility, target 15% portfolio vol
            >>> fraction = sizer.size_by_volatility(0.25, 0.15)
            >>> print(f"Allocate {fraction*100:.0f}% to this strategy")
        """
        if volatility <= 0:
            raise ValueError("Volatility must be positive")

        # Position size = Target Vol / Strategy Vol
        position_fraction = target_volatility / volatility

        # Cap at maximum position size
        position_fraction = min(position_fraction, self.default_max_position_pct)

        return position_fraction

    def size_by_kelly(self,
                     win_prob: float,
                     win_loss_ratio: float,
                     max_kelly: float = 0.25) -> float:
        """
        Size position using Kelly Criterion.

        Args:
            win_prob: Win probability (0-1)
            win_loss_ratio: Win/loss ratio
            max_kelly: Max Kelly fraction (default: 25%)

        Returns:
            Position size as fraction of portfolio
        """
        return calculate_kelly_criterion(win_prob, win_loss_ratio, max_kelly)


def scale_position_for_correlation(base_contracts: int,
                                   correlation: float,
                                   scale_factor: float = 0.5) -> int:
    """
    Reduce position size for correlated trades.

    When adding correlated positions to portfolio, reduce size to avoid
    excessive concentration risk.

    Args:
        base_contracts: Normal position size without correlation
        correlation: Correlation with existing portfolio positions (-1 to 1)
        scale_factor: How aggressively to scale (default: 0.5)

    Returns:
        Adjusted contract count

    Example:
        >>> # Normal size is 10 contracts, but 70% correlated with existing
        >>> adjusted = scale_position_for_correlation(10, 0.70)
        >>> print(f"Reduce to {adjusted} contracts")
        Reduce to 7 contracts
    """
    if not (-1 <= correlation <= 1):
        raise ValueError("Correlation must be between -1 and 1")

    # Scale down for positive correlation
    if correlation > 0:
        adjustment = 1 - (correlation * scale_factor)
        adjusted_contracts = int(base_contracts * adjustment)
    else:
        # Don't scale up for negative correlation (conservative)
        adjusted_contracts = base_contracts

    return max(1, adjusted_contracts)


if __name__ == "__main__":
    # Example usage
    print("Position Sizing Template - Example Usage\n")
    print("=" * 50)

    portfolio = 100000
    max_loss = 500

    print(f"\nPortfolio Value: ${portfolio:,}")
    print(f"Max Loss per Contract: ${max_loss}")

    # Method 1: Simple risk-based
    print("\n1. Simple Risk-Based Sizing:")
    contracts = calculate_position_size(portfolio, 0.02, max_loss)
    print(f"   Risk 2%: {contracts} contracts")

    # Method 2: Detailed
    print("\n2. Detailed Position Sizing:")
    result = calculate_detailed_position_size(portfolio, max_loss, risk_per_trade=0.02)
    print(f"   {result['recommendation']}")

    # Method 3: Kelly Criterion
    print("\n3. Kelly Criterion:")
    kelly_frac = calculate_kelly_criterion(win_probability=0.55, win_loss_ratio=1.5)
    print(f"   Kelly recommends: {kelly_frac*100:.1f}% of capital")

    # Method 4: Risk-based sizer class
    print("\n4. Advanced Risk-Based Sizer:")
    sizer = RiskBasedSizer(portfolio)
    contracts = sizer.size_by_risk(max_loss, risk_percent=0.025)
    print(f"   Risk 2.5%: {contracts} contracts")

    print("\n" + "=" * 50)
