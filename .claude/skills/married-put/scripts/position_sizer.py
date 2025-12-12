"""
Position Sizing for Married-Put Strategies

Calculates optimal position sizes based on available capital, risk tolerance,
and portfolio allocation constraints.

Author: Ordinis-1 Project
Version: 1.0.0
Python: 3.8+
"""

from typing import Dict, Optional
import numpy as np


def calculate_position_size(capital_available: float,
                           stock_price: float,
                           put_premium: float,
                           max_position_pct: float = 0.40,
                           target_contracts: Optional[int] = None,
                           transaction_cost: float = 0.65) -> Dict:
    """
    Calculate optimal position size with married-put protection.
    
    Args:
        capital_available: Total capital available for position
        stock_price: Current stock price
        put_premium: Put option premium per share
        max_position_pct: Maximum % of capital for single position (default: 40%)
        target_contracts: Force specific number of contracts (optional)
        transaction_cost: Transaction cost per contract
        
    Returns:
        Dictionary with position sizing recommendations
        
    Example:
        >>> params = calculate_position_size(
        ...     capital_available=25000,
        ...     stock_price=48.25,
        ...     put_premium=2.15,
        ...     max_position_pct=0.40
        ... )
        >>> print(f"Buy {params['shares']} shares with {params['contracts']} puts")
    """
    # Maximum capital to deploy
    max_capital = capital_available * max_position_pct
    
    # Cost per protected share (stock + put premium + transaction cost per share)
    cost_per_share = stock_price + put_premium + (transaction_cost / 100)
    
    if target_contracts is not None:
        # User specified exact contracts
        shares = target_contracts * 100
        contracts = target_contracts
    else:
        # Calculate maximum shares affordable
        max_shares = int(max_capital / cost_per_share)
        
        # Round down to nearest 100 (full contracts)
        contracts = max_shares // 100
        shares = contracts * 100
    
    if contracts == 0:
        return {
            'error': 'Insufficient capital for position',
            'minimum_required': cost_per_share * 100,
            'available': capital_available,
            'shares': 0,
            'contracts': 0
        }
    
    # Calculate actual costs
    stock_cost = shares * stock_price
    put_cost = shares * put_premium
    txn_cost = contracts * transaction_cost
    total_cost = stock_cost + put_cost + txn_cost
    
    # Remaining capital
    remaining_capital = capital_available - total_cost
    
    # Position metrics
    position_pct = (total_cost / capital_available) * 100
    
    return {
        'shares': shares,
        'contracts': contracts,
        'stock_cost': stock_cost,
        'put_cost': put_cost,
        'transaction_cost': txn_cost,
        'total_cost': total_cost,
        'cost_per_share': cost_per_share,
        'position_pct': position_pct,
        'remaining_capital': remaining_capital,
        'capital_available': capital_available,
        'recommendation': f"Buy {shares} shares ({contracts} contract{'s' if contracts != 1 else ''}) "
                         f"for ${total_cost:,.2f} ({position_pct:.1f}% of capital)"
    }


def calculate_kelly_criterion(win_probability: float,
                              win_loss_ratio: float,
                              max_kelly: float = 0.25) -> float:
    """
    Calculate position size using Kelly Criterion.
    
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
        >>> size = calculate_kelly_criterion(0.55, 1.5)
        >>> print(f"Position size: {size*100:.1f}% of capital")
    """
    if not 0 <= win_probability <= 1:
        raise ValueError("win_probability must be between 0 and 1")
    if win_loss_ratio <= 0:
        raise ValueError("win_loss_ratio must be positive")
    if not 0 < max_kelly <= 1:
        raise ValueError("max_kelly must be between 0 and 1")
    
    loss_probability = 1 - win_probability
    
    # Kelly formula
    kelly_pct = (win_probability * win_loss_ratio - loss_probability) / win_loss_ratio
    
    # Apply fractional Kelly (usually 1/4 or 1/2 Kelly for safety)
    adjusted_kelly = max(0, kelly_pct * max_kelly)
    
    return adjusted_kelly


def portfolio_heat_check(positions: list,
                        total_portfolio_value: float,
                        max_portfolio_risk: float = 0.02) -> Dict:
    """
    Calculate total portfolio risk ("heat") across multiple positions.
    
    Args:
        positions: List of dicts with 'max_loss' for each position
        total_portfolio_value: Total portfolio value
        max_portfolio_risk: Maximum acceptable portfolio risk (default: 2%)
        
    Returns:
        Dictionary with portfolio heat analysis
        
    Example:
        >>> positions = [
        ...     {'name': 'AAPL', 'max_loss': 450},
        ...     {'name': 'MSFT', 'max_loss': 380},
        ...     {'name': 'GOOGL', 'max_loss': 520}
        ... ]
        >>> heat = portfolio_heat_check(positions, 100000, 0.02)
        >>> print(f"Portfolio heat: {heat['total_heat_pct']:.2f}%")
    """
    total_risk = sum(pos.get('max_loss', 0) for pos in positions)
    total_heat_pct = (total_risk / total_portfolio_value) * 100
    max_acceptable_risk = total_portfolio_value * max_portfolio_risk
    
    risk_status = "SAFE" if total_risk <= max_acceptable_risk else "EXCESSIVE"
    
    # Calculate available risk budget
    available_risk = max(0, max_acceptable_risk - total_risk)
    
    return {
        'total_positions': len(positions),
        'total_risk_dollars': total_risk,
        'total_heat_pct': total_heat_pct,
        'max_acceptable_risk': max_acceptable_risk,
        'max_acceptable_pct': max_portfolio_risk * 100,
        'risk_status': risk_status,
        'available_risk_budget': available_risk,
        'positions': positions
    }


def diversification_calculator(capital_available: float,
                              num_positions: int,
                              position_overlap: float = 0.10) -> Dict:
    """
    Calculate capital allocation for diversified portfolio.
    
    Args:
        capital_available: Total capital to allocate
        num_positions: Target number of positions
        position_overlap: % overlap between positions (0-1, default: 10%)
        
    Returns:
        Dictionary with diversification analysis
        
    Example:
        >>> allocation = diversification_calculator(50000, 5, 0.10)
        >>> print(f"Capital per position: ${allocation['capital_per_position']:.2f}")
    """
    if num_positions <= 0:
        raise ValueError("num_positions must be positive")
    if not 0 <= position_overlap <= 1:
        raise ValueError("position_overlap must be between 0 and 1")
    
    # Adjust for correlation (overlap)
    # With perfect correlation (1.0), diversification = 1 position
    # With zero correlation (0.0), full diversification
    effective_positions = num_positions * (1 - position_overlap)
    
    # Capital per position
    capital_per_position = capital_available / effective_positions
    
    # Risk reduction from diversification
    # Std dev reduces by sqrt(N) with uncorrelated positions
    risk_reduction_pct = (1 - 1/np.sqrt(effective_positions)) * 100
    
    return {
        'capital_available': capital_available,
        'target_positions': num_positions,
        'effective_positions': effective_positions,
        'position_overlap': position_overlap * 100,
        'capital_per_position': capital_per_position,
        'risk_reduction_pct': risk_reduction_pct,
        'recommendation': f"Allocate ${capital_per_position:,.2f} per position "
                         f"across {num_positions} holdings for "
                         f"{risk_reduction_pct:.1f}% risk reduction"
    }


if __name__ == "__main__":
    print("Position Sizing for Married-Put Strategies\n")
    print("=" * 70)
    
    # Example 1: Basic position sizing
    print("\nExample 1: Small Account ($7,500)")
    print("-" * 70)
    
    pos1 = calculate_position_size(
        capital_available=7500,
        stock_price=38.50,
        put_premium=1.45,
        max_position_pct=0.80  # Aggressive for small account
    )
    
    print(f"Available Capital: ${pos1['capital_available']:,.2f}")
    print(f"Stock Price: ${pos1['stock_cost']/pos1['shares']:.2f}")
    print(f"\nRecommendation: {pos1['recommendation']}")
    print(f"  Stock Cost: ${pos1['stock_cost']:,.2f}")
    print(f"  Put Cost: ${pos1['put_cost']:,.2f}")
    print(f"  Transaction Cost: ${pos1['transaction_cost']:.2f}")
    print(f"  Total Investment: ${pos1['total_cost']:,.2f}")
    print(f"  Remaining Capital: ${pos1['remaining_capital']:,.2f}")
    
    # Example 2: Mid-size account
    print("\n" + "=" * 70)
    print("\nExample 2: Mid-Size Account ($25,000)")
    print("-" * 70)
    
    pos2 = calculate_position_size(
        capital_available=25000,
        stock_price=52.75,
        put_premium=2.35,
        max_position_pct=0.40
    )
    
    print(f"Available Capital: ${pos2['capital_available']:,.2f}")
    print(f"\nRecommendation: {pos2['recommendation']}")
    print(f"  Position Size: {pos2['position_pct']:.1f}% of capital")
    print(f"  Cost per Share (with protection): ${pos2['cost_per_share']:.2f}")
    print(f"  Remaining for Other Positions: ${pos2['remaining_capital']:,.2f}")
    
    # Example 3: Larger account
    print("\n" + "=" * 70)
    print("\nExample 3: Larger Account ($50,000)")
    print("-" * 70)
    
    pos3 = calculate_position_size(
        capital_available=50000,
        stock_price=175.50,
        put_premium=6.25,
        max_position_pct=0.30
    )
    
    print(f"Available Capital: ${pos3['capital_available']:,.2f}")
    print(f"\nRecommendation: {pos3['recommendation']}")
    print(f"  Contracts: {pos3['contracts']}")
    print(f"  Total Cost: ${pos3['total_cost']:,.2f}")
    
    # Example 4: Kelly Criterion
    print("\n" + "=" * 70)
    print("\nExample 4: Kelly Criterion Position Sizing")
    print("-" * 70)
    
    scenarios = [
        (0.55, 1.5, "Moderate edge: 55% win, 1.5:1 ratio"),
        (0.60, 2.0, "Strong edge: 60% win, 2:1 ratio"),
        (0.50, 1.0, "Break-even: 50% win, 1:1 ratio"),
    ]
    
    print("\nKelly Criterion (using 25% fractional Kelly):")
    for win_prob, ratio, description in scenarios:
        kelly = calculate_kelly_criterion(win_prob, ratio, max_kelly=0.25)
        print(f"\n{description}")
        print(f"  Full Kelly: {kelly/0.25*100:.1f}%")
        print(f"  25% Fractional Kelly: {kelly*100:.1f}% of capital")
        
        if kelly > 0:
            example_capital = 25000
            position_size = example_capital * kelly
            print(f"  Example with $25,000: ${position_size:,.2f} position")
    
    # Example 5: Portfolio heat check
    print("\n" + "=" * 70)
    print("\nExample 5: Portfolio Risk Management")
    print("-" * 70)
    
    current_positions = [
        {'name': 'Position 1', 'max_loss': 450},
        {'name': 'Position 2', 'max_loss': 380},
        {'name': 'Position 3', 'max_loss': 520},
    ]
    
    heat = portfolio_heat_check(
        positions=current_positions,
        total_portfolio_value=100000,
        max_portfolio_risk=0.02  # 2% max risk
    )
    
    print(f"Portfolio Value: ${heat['total_heat_pct']:.2f}")
    print(f"Current Positions: {heat['total_positions']}")
    print(f"Total Risk: ${heat['total_risk_dollars']:,.2f} "
          f"({heat['total_heat_pct']:.2f}% of portfolio)")
    print(f"Max Acceptable Risk: ${heat['max_acceptable_risk']:,.2f} "
          f"({heat['max_acceptable_pct']:.1f}%)")
    print(f"Risk Status: {heat['risk_status']}")
    print(f"Available Risk Budget: ${heat['available_risk_budget']:,.2f}")
    
    # Example 6: Diversification
    print("\n" + "=" * 70)
    print("\nExample 6: Portfolio Diversification")
    print("-" * 70)
    
    div_analysis = diversification_calculator(
        capital_available=50000,
        num_positions=5,
        position_overlap=0.15  # 15% correlation between positions
    )
    
    print(f"Total Capital: ${div_analysis['capital_available']:,.2f}")
    print(f"Target Positions: {div_analysis['target_positions']}")
    print(f"Position Correlation: {div_analysis['position_overlap']:.1f}%")
    print(f"Effective Positions: {div_analysis['effective_positions']:.1f}")
    print(f"\n{div_analysis['recommendation']}")
    
    print("\n" + "=" * 70)
    print("Position sizing analysis complete!")
