"""
Married-Put Strategy Calculator

Core calculation engine for analyzing married-put options positions.
Calculates breakeven points, profit/loss scenarios, and key metrics.

Author: Ordinis-1 Project
Version: 1.0.0
Python: 3.8+
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class MarriedPut:
    """
    Represents a married-put options position.
    
    A married put combines stock ownership with protective put options,
    providing downside insurance while maintaining upside potential.
    
    Attributes:
        stock_price (float): Current price per share of stock
        shares (int): Number of shares owned (typically 100)
        put_strike (float): Strike price of protective put option
        put_premium (float): Premium paid per share for put option
        transaction_cost (float): Brokerage cost per contract (default: $0.65)
        days_to_expiration (int): Days until put option expires
    
    Example:
        >>> position = MarriedPut(
        ...     stock_price=45.00,
        ...     shares=100,
        ...     put_strike=43.00,
        ...     put_premium=2.10,
        ...     transaction_cost=0.65
        ... )
        >>> print(f"Breakeven: ${position.breakeven_price:.2f}")
        Breakeven: $47.11
    """
    
    stock_price: float
    shares: int
    put_strike: float
    put_premium: float
    transaction_cost: float = 0.65
    days_to_expiration: Optional[int] = None
    
    def __post_init__(self):
        """Validate inputs after initialization."""
        if self.stock_price <= 0:
            raise ValueError("Stock price must be positive")
        if self.shares <= 0:
            raise ValueError("Shares must be positive")
        if self.put_strike <= 0:
            raise ValueError("Put strike must be positive")
        if self.put_premium < 0:
            raise ValueError("Put premium cannot be negative")
        if self.shares % 100 != 0:
            raise ValueError("Shares must be multiple of 100 (standard contracts)")
    
    @property
    def contracts(self) -> int:
        """Number of option contracts (1 contract = 100 shares)."""
        return self.shares // 100
    
    @property
    def stock_cost(self) -> float:
        """Total cost of stock purchase."""
        return self.stock_price * self.shares
    
    @property
    def put_cost(self) -> float:
        """Total cost of put options including transaction fees."""
        return (self.put_premium * self.shares) + self.transaction_cost
    
    @property
    def total_cost(self) -> float:
        """Total investment: stock + put premium + transaction cost."""
        return self.stock_cost + self.put_cost
    
    @property
    def breakeven_price(self) -> float:
        """
        Stock price needed to break even at expiration.
        
        Formula: Stock Price + Put Premium + (Transaction Cost / Shares)
        """
        return self.stock_price + self.put_premium + (self.transaction_cost / self.shares)
    
    @property
    def max_loss(self) -> float:
        """
        Maximum possible loss on the position.
        
        Formula: (Stock Price - Put Strike) + Put Premium + (Transaction Cost / Shares)
        
        This represents the worst-case scenario if stock falls to zero,
        but put provides floor at strike price.
        """
        loss_per_share = (self.stock_price - self.put_strike) + self.put_premium + (self.transaction_cost / self.shares)
        return loss_per_share * self.shares
    
    @property
    def max_loss_percentage(self) -> float:
        """Maximum loss as percentage of total investment."""
        return (self.max_loss / self.total_cost) * 100
    
    @property
    def protection_percentage(self) -> float:
        """
        Percentage decline protected against before max loss reached.
        
        Example: Stock at $50, put strike $48 = 4% protection
        """
        return ((self.stock_price - self.put_strike) / self.stock_price) * 100
    
    @property
    def protection_cost_percentage(self) -> float:
        """Put premium as percentage of stock price."""
        return (self.put_premium / self.stock_price) * 100
    
    @property
    def annualized_protection_cost(self) -> Optional[float]:
        """
        Annualized cost of protection as percentage.
        
        Returns None if days_to_expiration not provided.
        """
        if self.days_to_expiration is None:
            return None
        return (self.put_premium / self.stock_price) * (365 / self.days_to_expiration) * 100
    
    def calculate_pl_at_price(self, final_stock_price: float) -> float:
        """
        Calculate profit/loss at a specific stock price at expiration.
        
        Args:
            final_stock_price: Stock price to evaluate
            
        Returns:
            Total profit (positive) or loss (negative) in dollars
            
        Example:
            >>> position = MarriedPut(stock_price=45, shares=100, 
            ...                       put_strike=43, put_premium=2.10)
            >>> position.calculate_pl_at_price(50)  # Stock rises to $50
            293.50
            >>> position.calculate_pl_at_price(40)  # Stock falls to $40
            -411.00
        """
        # Stock P/L
        stock_pl = (final_stock_price - self.stock_price) * self.shares
        
        # Put P/L at expiration
        if final_stock_price <= self.put_strike:
            # Put is in-the-money, has intrinsic value
            put_intrinsic = (self.put_strike - final_stock_price) * self.shares
            put_pl = put_intrinsic - self.put_cost
        else:
            # Put expires worthless
            put_pl = -self.put_cost
        
        return stock_pl + put_pl
    
    def calculate_pl_table(self, 
                          price_range: Optional[Tuple[float, float]] = None,
                          num_points: int = 20) -> List[Dict[str, float]]:
        """
        Generate profit/loss table across range of stock prices.
        
        Args:
            price_range: (min_price, max_price) tuple, defaults to ±30% of stock price
            num_points: Number of price points to calculate
            
        Returns:
            List of dicts with 'stock_price', 'pl_dollars', 'pl_percent', 'pl_per_share'
            
        Example:
            >>> position = MarriedPut(stock_price=45, shares=100, 
            ...                       put_strike=43, put_premium=2.10)
            >>> table = position.calculate_pl_table()
            >>> for row in table[:3]:
            ...     print(f"${row['stock_price']:.2f}: {row['pl_percent']:.1f}%")
        """
        if price_range is None:
            min_price = self.stock_price * 0.70
            max_price = self.stock_price * 1.30
        else:
            min_price, max_price = price_range
        
        prices = np.linspace(min_price, max_price, num_points)
        
        table = []
        for price in prices:
            pl_dollars = self.calculate_pl_at_price(price)
            pl_percent = (pl_dollars / self.total_cost) * 100
            pl_per_share = pl_dollars / self.shares
            
            table.append({
                'stock_price': float(price),
                'pl_dollars': float(pl_dollars),
                'pl_percent': float(pl_percent),
                'pl_per_share': float(pl_per_share)
            })
        
        return table
    
    def get_metrics_summary(self) -> Dict[str, float]:
        """
        Get comprehensive summary of position metrics.
        
        Returns:
            Dictionary containing all key metrics
        """
        metrics = {
            'stock_price': self.stock_price,
            'shares': self.shares,
            'contracts': self.contracts,
            'put_strike': self.put_strike,
            'put_premium': self.put_premium,
            'stock_cost': self.stock_cost,
            'put_cost': self.put_cost,
            'total_cost': self.total_cost,
            'breakeven_price': self.breakeven_price,
            'max_loss': self.max_loss,
            'max_loss_percentage': self.max_loss_percentage,
            'protection_percentage': self.protection_percentage,
            'protection_cost_percentage': self.protection_cost_percentage,
        }
        
        if self.days_to_expiration is not None:
            metrics['days_to_expiration'] = self.days_to_expiration
            metrics['annualized_protection_cost'] = self.annualized_protection_cost
        
        return metrics
    
    def __str__(self) -> str:
        """String representation of position."""
        return (f"MarriedPut(stock=${self.stock_price:.2f}, "
                f"strike=${self.put_strike:.2f}, "
                f"premium=${self.put_premium:.2f}, "
                f"shares={self.shares})")
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return self.__str__()


def compare_positions(positions: List[MarriedPut]) -> Dict:
    """
    Compare multiple married-put positions side by side.
    
    Args:
        positions: List of MarriedPut instances to compare
        
    Returns:
        Dictionary with comparison metrics
        
    Example:
        >>> pos1 = MarriedPut(stock_price=45, shares=100, 
        ...                   put_strike=43, put_premium=2.10)
        >>> pos2 = MarriedPut(stock_price=45, shares=100, 
        ...                   put_strike=45, put_premium=3.50)
        >>> comparison = compare_positions([pos1, pos2])
    """
    if not positions:
        raise ValueError("Must provide at least one position")
    
    comparison = {
        'positions': len(positions),
        'metrics': []
    }
    
    for i, pos in enumerate(positions, 1):
        metrics = pos.get_metrics_summary()
        metrics['position_number'] = i
        comparison['metrics'].append(metrics)
    
    return comparison


def calculate_optimal_strike(stock_price: float,
                            available_strikes: List[float],
                            available_premiums: List[float],
                            risk_tolerance: str = 'moderate') -> Dict:
    """
    Recommend optimal strike price based on risk tolerance.
    
    Args:
        stock_price: Current stock price
        available_strikes: List of available strike prices
        available_premiums: Corresponding premiums for each strike
        risk_tolerance: 'conservative', 'moderate', or 'aggressive'
        
    Returns:
        Dictionary with recommended strike and analysis
        
    Risk Tolerance Guidelines:
        - Conservative: Favor higher strikes (ITM), more protection
        - Moderate: Balance cost and protection (ATM)
        - Aggressive: Lower strikes (OTM), minimize cost
    """
    if len(available_strikes) != len(available_premiums):
        raise ValueError("Strikes and premiums must have same length")
    
    analysis = []
    for strike, premium in zip(available_strikes, available_premiums):
        protection_pct = ((stock_price - strike) / stock_price) * 100
        cost_pct = (premium / stock_price) * 100
        
        # Simple scoring based on risk tolerance
        if risk_tolerance == 'conservative':
            score = protection_pct - (cost_pct * 0.5)  # Weight protection heavily
        elif risk_tolerance == 'moderate':
            score = protection_pct - cost_pct  # Balance both
        else:  # aggressive
            score = protection_pct - (cost_pct * 2.0)  # Minimize cost
        
        analysis.append({
            'strike': strike,
            'premium': premium,
            'protection_pct': protection_pct,
            'cost_pct': cost_pct,
            'score': score
        })
    
    # Find highest scoring option
    best = max(analysis, key=lambda x: x['score'])
    
    return {
        'recommended_strike': best['strike'],
        'recommended_premium': best['premium'],
        'protection_percentage': best['protection_pct'],
        'cost_percentage': best['cost_pct'],
        'all_options': analysis,
        'risk_tolerance': risk_tolerance
    }


if __name__ == "__main__":
    # Example usage and testing
    print("Married-Put Calculator - Example Usage\n")
    print("=" * 50)
    
    # Example 1: Basic position
    print("\n1. Basic Position Analysis:")
    position = MarriedPut(
        stock_price=45.00,
        shares=100,
        put_strike=43.00,
        put_premium=2.10,
        transaction_cost=0.65,
        days_to_expiration=45
    )
    
    print(f"Position: {position}")
    print(f"Total Cost: ${position.total_cost:,.2f}")
    print(f"Breakeven: ${position.breakeven_price:.2f}")
    print(f"Max Loss: ${position.max_loss:.2f} ({position.max_loss_percentage:.1f}%)")
    print(f"Protection: {position.protection_percentage:.1f}% decline")
    print(f"Protection Cost: {position.protection_cost_percentage:.1f}% of stock price")
    
    # Example 2: P/L at various prices
    print("\n2. Profit/Loss at Various Prices:")
    test_prices = [35, 40, 43, 45, 50, 55, 60]
    for price in test_prices:
        pl = position.calculate_pl_at_price(price)
        print(f"Stock at ${price:5.2f}: P/L = ${pl:7.2f}")
    
    # Example 3: Strike comparison
    print("\n3. Strike Price Comparison:")
    strikes = [40, 43, 45, 48]
    premiums = [0.85, 2.10, 3.50, 5.40]
    
    optimal = calculate_optimal_strike(
        stock_price=45.00,
        available_strikes=strikes,
        available_premiums=premiums,
        risk_tolerance='moderate'
    )
    
    print(f"Recommended Strike: ${optimal['recommended_strike']:.2f}")
    print(f"Premium: ${optimal['recommended_premium']:.2f}")
    print(f"Protection: {optimal['protection_percentage']:.1f}%")
    print(f"Cost: {optimal['cost_percentage']:.1f}%")
    
    print("\n" + "=" * 50)
    print("Calculations complete!")
