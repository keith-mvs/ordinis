"""
Visualization Tools for Married-Put Strategies

Creates payoff diagrams, comparison charts, and analysis visualizations
for married-put options positions.

Author: Ordinis-1 Project
Version: 1.0.0
Python: 3.8+
"""

from typing import Optional, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from married_put_calculator import MarriedPut


# Set professional style
plt.style.use('seaborn-v0_8-darkgrid')
COLORS = {
    'profit': '#2ecc71',
    'loss': '#e74c3c',
    'breakeven': '#3498db',
    'stock': '#9b59b6',
    'put': '#f39c12',
    'combined': '#34495e',
}


def plot_married_put_payoff(stock_price: float,
                            put_strike: float,
                            put_premium: float,
                            shares: int = 100,
                            price_range: Optional[Tuple[float, float]] = None,
                            save_path: Optional[str] = None,
                            show_components: bool = True,
                            figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Create payoff diagram for married-put strategy.
    
    Args:
        stock_price: Initial stock price
        put_strike: Put option strike price
        put_premium: Put option premium
        shares: Number of shares (default: 100)
        price_range: (min, max) price range for x-axis
        save_path: Path to save figure (optional)
        show_components: Show individual stock and put P/L (default: True)
        figsize: Figure size in inches
        
    Example:
        >>> plot_married_put_payoff(
        ...     stock_price=45.00,
        ...     put_strike=43.00,
        ...     put_premium=2.10,
        ...     save_path='married_put_diagram.png'
        ... )
    """
    position = MarriedPut(stock_price, shares, put_strike, put_premium)
    
    # Generate price range
    if price_range is None:
        min_price = stock_price * 0.65
        max_price = stock_price * 1.35
    else:
        min_price, max_price = price_range
    
    prices = np.linspace(min_price, max_price, 200)
    
    # Calculate P/L for each component
    stock_pl = (prices - stock_price) * shares
    
    put_pl = np.where(
        prices <= put_strike,
        (put_strike - prices) * shares - position.put_cost,
        -position.put_cost
    )
    
    combined_pl = stock_pl + put_pl
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot components if requested
    if show_components:
        ax.plot(prices, stock_pl, '--', color=COLORS['stock'], 
                linewidth=2, label='Stock Only', alpha=0.7)
        ax.plot(prices, put_pl, '--', color=COLORS['put'], 
                linewidth=2, label='Put Only', alpha=0.7)
    
    # Plot combined P/L
    ax.plot(prices, combined_pl, '-', color=COLORS['combined'], 
            linewidth=3, label='Married Put (Combined)')
    
    # Color profit/loss areas
    ax.fill_between(prices, 0, combined_pl, 
                     where=(combined_pl >= 0), 
                     color=COLORS['profit'], alpha=0.2, label='Profit Zone')
    ax.fill_between(prices, 0, combined_pl, 
                     where=(combined_pl < 0), 
                     color=COLORS['loss'], alpha=0.2, label='Loss Zone')
    
    # Add zero line
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    
    # Add vertical lines for key prices
    ax.axvline(x=stock_price, color='gray', linestyle=':', 
               linewidth=1.5, alpha=0.7, label=f'Initial Stock: ${stock_price:.2f}')
    ax.axvline(x=put_strike, color=COLORS['put'], linestyle=':', 
               linewidth=1.5, alpha=0.7, label=f'Put Strike: ${put_strike:.2f}')
    ax.axvline(x=position.breakeven_price, color=COLORS['breakeven'], 
               linestyle='--', linewidth=2, label=f'Breakeven: ${position.breakeven_price:.2f}')
    
    # Format axes
    ax.set_xlabel('Stock Price at Expiration ($)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Profit / Loss ($)', fontsize=12, fontweight='bold')
    ax.set_title(f'Married-Put Payoff Diagram\n'
                 f'{shares} shares @ ${stock_price:.2f} + ${put_strike:.2f} Put @ ${put_premium:.2f}',
                 fontsize=14, fontweight='bold', pad=20)
    
    # Format y-axis as currency
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${x:.0f}'))
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Legend
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    
    # Add metrics text box
    metrics_text = (
        f'Total Cost: ${position.total_cost:,.2f}\n'
        f'Max Loss: ${position.max_loss:,.2f} ({position.max_loss_percentage:.1f}%)\n'
        f'Protection: {position.protection_percentage:.1f}% decline\n'
        f'Premium Cost: {position.protection_cost_percentage:.1f}% of stock price'
    )
    
    ax.text(0.98, 0.02, metrics_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved payoff diagram to: {save_path}")
    
    plt.show()


def plot_strike_comparison(stock_price: float,
                          strikes: List[float],
                          premiums: List[float],
                          labels: Optional[List[str]] = None,
                          shares: int = 100,
                          save_path: Optional[str] = None,
                          figsize: Tuple[int, int] = (14, 8)) -> None:
    """
    Compare payoff diagrams for different strike prices.
    
    Args:
        stock_price: Current stock price
        strikes: List of strike prices to compare
        premiums: Corresponding premiums for each strike
        labels: Custom labels for each strike (optional)
        shares: Number of shares (default: 100)
        save_path: Path to save figure (optional)
        figsize: Figure size in inches
    """
    if len(strikes) != len(premiums):
        raise ValueError("strikes and premiums must have same length")
    
    if labels is None:
        labels = [f"${strike:.2f} Strike" for strike in strikes]
    
    # Generate price range
    min_price = stock_price * 0.70
    max_price = stock_price * 1.30
    prices = np.linspace(min_price, max_price, 200)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Colors for different strikes
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(strikes)))
    
    # Plot 1: Payoff Diagrams
    for strike, premium, label, color in zip(strikes, premiums, labels, colors):
        position = MarriedPut(stock_price, shares, strike, premium)
        pl = [position.calculate_pl_at_price(p) for p in prices]
        ax1.plot(prices, pl, linewidth=2.5, label=label, color=color)
    
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax1.axvline(x=stock_price, color='gray', linestyle=':', 
                linewidth=1.5, alpha=0.7, label=f'Stock: ${stock_price:.2f}')
    
    ax1.set_xlabel('Stock Price at Expiration ($)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Profit / Loss ($)', fontsize=11, fontweight='bold')
    ax1.set_title('Payoff Comparison by Strike Price', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Plot 2: Cost vs Protection
    protections = [((stock_price - strike) / stock_price) * 100 for strike in strikes]
    costs = [(premium / stock_price) * 100 for premium in premiums]
    
    ax2.scatter(protections, costs, s=200, c=colors, alpha=0.7, edgecolors='black', linewidth=2)
    
    for i, (prot, cost, label) in enumerate(zip(protections, costs, labels)):
        ax2.annotate(label, (prot, cost), 
                    textcoords="offset points", 
                    xytext=(0, 10), 
                    ha='center', 
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    ax2.set_xlabel('Protection Level (% below stock price)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Premium Cost (% of stock price)', fontsize=11, fontweight='bold')
    ax2.set_title('Cost vs Protection Trade-off', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(ticker.PercentFormatter())
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter())
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison chart to: {save_path}")
    
    plt.show()


def plot_expiration_comparison(stock_price: float,
                              put_strike: float,
                              premiums_dict: dict,
                              shares: int = 100,
                              save_path: Optional[str] = None,
                              figsize: Tuple[int, int] = (12, 6)) -> None:
    """
    Compare costs across different expiration cycles.
    
    Args:
        stock_price: Current stock price
        put_strike: Put strike price
        premiums_dict: Dict with keys '30d', '60d', '90d' and premium values
        shares: Number of shares (default: 100)
        save_path: Path to save figure (optional)
        figsize: Figure size in inches
        
    Example:
        >>> plot_expiration_comparison(
        ...     stock_price=50.00,
        ...     put_strike=48.00,
        ...     premiums_dict={'30d': 1.85, '60d': 2.95, '90d': 3.85}
        ... )
    """
    expirations = ['30d', '60d', '90d']
    days = [30, 60, 90]
    
    # Calculate costs
    total_costs = []
    monthly_costs = []
    
    for exp, d in zip(expirations, days):
        premium = premiums_dict[exp]
        position = MarriedPut(stock_price, shares, put_strike, premium, days_to_expiration=d)
        total_cost = position.put_cost
        monthly_cost = (total_cost * 365 / d) / 12
        
        total_costs.append(total_cost)
        monthly_costs.append(monthly_cost)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Total Premium Cost
    bars1 = ax1.bar(expirations, total_costs, 
                    color=['#e74c3c', '#f39c12', '#2ecc71'],
                    edgecolor='black', linewidth=1.5, alpha=0.8)
    
    ax1.set_ylabel('Total Premium Cost ($)', fontsize=11, fontweight='bold')
    ax1.set_title('Total Premium by Expiration', fontsize=13, fontweight='bold')
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Add value labels on bars
    for bar, cost in zip(bars1, total_costs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'${cost:.2f}',
                ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Monthly Cost (Annualized)
    bars2 = ax2.bar(expirations, monthly_costs,
                    color=['#e74c3c', '#f39c12', '#2ecc71'],
                    edgecolor='black', linewidth=1.5, alpha=0.8)
    
    ax2.set_ylabel('Monthly Cost ($)', fontsize=11, fontweight='bold')
    ax2.set_title('Annualized Monthly Cost', fontsize=13, fontweight='bold')
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Add value labels
    for bar, cost in zip(bars2, monthly_costs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'${cost:.2f}',
                ha='center', va='bottom', fontweight='bold')
    
    # Highlight best value
    best_idx = monthly_costs.index(min(monthly_costs))
    bars2[best_idx].set_edgecolor('gold')
    bars2[best_idx].set_linewidth(3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved expiration comparison to: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    print("Married-Put Visualization Examples\n")
    print("=" * 70)
    
    # Example 1: Basic payoff diagram
    print("\nExample 1: Basic Payoff Diagram")
    print("-" * 70)
    
    plot_married_put_payoff(
        stock_price=45.00,
        put_strike=43.00,
        put_premium=2.10,
        shares=100,
        show_components=True,
        save_path='/home/claude/married-put-strategy/married_put_basic.png'
    )
    
    # Example 2: Strike comparison
    print("\nExample 2: Strike Price Comparison")
    print("-" * 70)
    
    plot_strike_comparison(
        stock_price=50.00,
        strikes=[45, 48, 50, 52],
        premiums=[0.85, 1.75, 2.90, 4.30],
        labels=['Deep OTM', 'OTM', 'ATM', 'ITM'],
        save_path='/home/claude/married-put-strategy/strike_comparison.png'
    )
    
    # Example 3: Expiration comparison
    print("\nExample 3: Expiration Cycle Comparison")
    print("-" * 70)
    
    plot_expiration_comparison(
        stock_price=52.75,
        put_strike=50.00,
        premiums_dict={'30d': 1.85, '60d': 2.95, '90d': 3.85},
        save_path='/home/claude/married-put-strategy/expiration_comparison.png'
    )
    
    print("\n" + "=" * 70)
    print("Visualizations complete!")
    print("Check the married-put-strategy directory for saved images.")
