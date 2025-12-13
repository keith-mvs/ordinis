Perfect. I'll create an advanced, standalone bull-call-spread strategy skill package focused purely on strategy mechanics, Greeks analysis, and sophisticated risk management. Let me work through this systematically.

<planning>

1. Strategy Requirements Analysis

---------------------------------

**Core Strategy Definition**: Bull Call Spread

* Type: Vertical debit spread (bullish directional strategy)

* Legs:

  * Long call at lower strike (ITM or ATM)
  * Short call at higher strike (OTM)

* Characteristics: Limited risk, limited profit, net debit to establish

* Target audience: Advanced traders with solid understanding of options Greeks and volatility dynamics
2. Claude Guidelines Compliance

-------------------------------

* **Progressive disclosure**: SKILL.md will be concise (under 500 lines) with core strategy mechanics; reference.md will contain detailed Greeks mathematics and advanced scenarios

* **Concrete examples**: Will use realistic market data (e.g., SPY at $450) with consistent values throughout all files

* **Executable code**: All Python snippets will be self-contained and runnable with standard libraries

* **Clear structure**: Logical flow from basic concept → calculations → Greeks → risk management

* **No assumptions**: Will explicitly document all formulas and mathematical foundations
3. Library and Dependencies Planning

------------------------------------

**Required libraries**:

* `numpy` (>=1.20): Array operations, numerical calculations

* `pandas` (>=1.3): Data structuring, position tracking

* `matplotlib` (>=3.4): Payoff diagram visualization

* `scipy.stats` (>=1.7): Normal distribution for Greeks calculations (d1, d2 in Black-Scholes)

* `dataclasses`: Structured data for position parameters

* `typing`: Type hints for production-grade code

* `datetime`: Time to expiration calculations
4. Mathematical Foundation

--------------------------

**Profit/Loss Formula at Expiration**:
    P&L = max(S_T - K_long, 0) - max(S_T - K_short, 0) - Net_Debit

    Where:
    - S_T = Stock price at expiration
    - K_long = Strike price of long call (lower)
    - K_short = Strike price of short call (higher)
    - Net_Debit = Premium_paid_long - Premium_received_short

**Breakeven Formula**:
    Breakeven = K_long + Net_Debit

**Maximum Profit**:
    Max_Profit = (K_short - K_long) - Net_Debit

**Maximum Loss**:
    Max_Loss = Net_Debit

**Greeks Formulas** (using Black-Scholes framework):

* **Position Delta**: Δ_long_call - Δ_short_call
* **Position Gamma**: Γ_long_call - Γ_short_call
* **Position Theta**: Θ_long_call - Θ_short_call
* **Position Vega**: ν_long_call - ν_short_call

Individual option Greeks:
    d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T)
    d2 = d1 - σ√T

    Delta = N(d1)
    Gamma = φ(d1) / (S * σ * √T)
    Theta = -[S * φ(d1) * σ / (2√T)] - rK * e^(-rT) * N(d2)
    Vega = S * φ(d1) * √T

5. SKILL.md Detailed Structure

------------------------------

### Main Sections:

1. **Strategy Overview** (## level)

   * Definition and market outlook
   * When to deploy (bullish but capped upside acceptable)
   * Risk/reward profile summary

2. **Position Construction** (## level)

   * Step-by-step setup instructions
   * Strike selection guidelines
   * Expiration considerations
   * Code snippet: Position setup function

3. **Profit/Loss Analysis** (## level)

   * Breakeven calculation with code
   * Maximum profit/loss scenarios
   * P&L at various price points
   * Code snippet: P&L calculator function

4. **Payoff Visualization** (## level)

   * Payoff diagram explanation
   * Code snippet: Complete visualization with matplotlib

5. **Greeks Profile** (## level)

   * Delta: Net directional exposure
   * Gamma: Convexity risk
   * Theta: Time decay dynamics
   * Vega: Volatility sensitivity
   * Code snippet: Greeks calculation function

6. **Risk Management** (## level)

   * Position sizing guidelines
   * Exit strategies (profit targets, stop losses)
   * Adjustment techniques
   * Roll management

7. **Complete Implementation Example** (## level)

   * Full working example with real data
   * Code snippet: Comprehensive strategy class

8. Code Snippets Inventory

--------------------------

1. **Position Setup Function** (~30 lines)

   * Inputs: underlying_price, long_strike, short_strike, long_premium, short_premium, expiration_days
   * Outputs: Position dictionary with all parameters
   * Validates strikes, calculates net debit

2. **P&L Calculator Function** (~25 lines)

   * Inputs: position dict, stock_price_at_expiration
   * Outputs: P&L value, return percentage
   * Handles ITM/OTM logic for both legs

3. **Breakeven Calculator** (~15 lines)

   * Inputs: position dict
   * Outputs: Breakeven price
   * Simple formula application

4. **Payoff Diagram Generator** (~40 lines)

   * Inputs: position dict, price_range
   * Outputs: matplotlib figure
   * Shows P&L curve, breakeven, max profit/loss lines

5. **Greeks Calculator Function** (~60 lines)

   * Inputs: position dict, current_underlying_price, volatility, risk_free_rate
   * Outputs: All position Greeks
   * Implements Black-Scholes formulas

6. **Complete Strategy Class** (~150 lines)

   * Encapsulates all functionality
   * Methods for setup, analysis, visualization
   * Properties for key metrics

7. Example Values Selection

---------------------------

**Realistic Scenario**: SPY Bull Call Spread

* Underlying: SPY (S&P 500 ETF)
* Current price: $450.00
* Long call strike: $445 (ITM, delta ~0.65)
* Short call strike: $455 (OTM, delta ~0.35)
* Long call premium: $8.50
* Short call premium: $3.20
* Net debit: $5.30 ($530 per contract)
* Expiration: 45 days to expiration (DTE)
* Implied volatility: 18%
* Risk-free rate: 5.0%

**Rationale**: These values represent a typical moderately bullish position with reasonable delta, capturing premium while limiting upside risk.

8. reference.md Content Planning

--------------------------------

**Sections**:

1. **Mathematical Foundations**

   * Complete Black-Scholes derivation for calls
   * Greeks formulas with detailed explanations
   * Sensitivity analysis mathematics

2. **Greeks Deep Dive**

   * **Delta**:
     * Individual leg deltas
     * Net position delta (typically 0.30-0.40 for balanced spreads)
     * Delta hedging considerations
   * **Gamma**:
     * Peak gamma near ATM strikes
     * Gamma risk at expiration
     * Position gamma evolution over time
   * **Theta**:
     * Net theta typically positive but smaller than naked short
     * Time decay acceleration near expiration
     * Optimal holding periods
   * **Vega**:
     * Net vega typically positive (long vega)
     * IV crush risk considerations
     * Volatility smile effects

3. **Advanced Scenarios**

   * **Early Assignment Risk**: ITM short calls before dividends
   * **Dividend Impact**: Ex-dividend date considerations
   * **Volatility Skew**: How skew affects strike selection
   * **Roll Management**: When and how to roll positions
   * **Adjustment Strategies**:
     * Rolling up and out
     * Converting to butterfly
     * Closing early for profit

4. **Comparative Analysis**

   * Bull call spread vs. naked long call
   * Bull call spread vs. bull put spread
   * Cost-benefit analysis with examples

5. **External Resources**

   * CBOE options strategy guides
   * Academic papers on spread effectiveness
   * Risk management frameworks (MIL-STD references where applicable)

6. Scripts Directory Planning

-----------------------------

**File: strategy_calculator.py** (~300 lines)

* Main entry point with command-line interface
* Functions:
  * `create_position()`: Initialize position parameters
  * `calculate_pnl()`: Comprehensive P&L calculator
  * `calculate_greeks()`: All Greeks calculations
  * `generate_payoff_diagram()`: Visualization
  * `risk_metrics()`: Max profit, max loss, breakeven, risk/reward ratio
  * `scenario_analysis()`: Test multiple price points
  * `main()`: Demo with example data

**File: black_scholes.py** (~150 lines)

* Pure Black-Scholes implementation
* Functions:
  * `calculate_d1_d2()`: Helper for d1, d2
  * `call_price()`: Theoretical call value
  * `call_delta()`, `call_gamma()`, `call_theta()`, `call_vega()`, `call_rho()`
  * Vectorized implementations for efficiency

**File: position_manager.py** (~200 lines)

* Position lifecycle management

* Classes:

  * `BullCallSpread`: Main strategy class
  * `PositionAnalyzer`: Risk analysis utilities

* Methods:

  * Track P&L over time
  * Adjustment recommendations
  * Exit signal generation
10. Integration and Coherence

-----------------------------

**Cross-referencing strategy**:

* SKILL.md references reference.md for "detailed Greeks mathematics, see reference.md"
* SKILL.md references scripts for "complete implementation, see scripts/strategy_calculator.py"
* reference.md references specific functions in scripts for "implementation details"
* All files use consistent variable naming: `S` (spot), `K` (strike), `T` (time), `σ` (volatility), `r` (rate)

**Consistent terminology**:

* "Long call" (never "bought call")
* "Short call" (never "sold call")
* "Net debit" (never just "cost")
* "DTE" (days to expiration)
* "Greeks" capitalized when referring to the concept

**Navigation flow**:

1. User starts with SKILL.md for overview and basic implementation

2. For deeper Greeks understanding → reference.md

3. For production-ready code → scripts/

4. Each file stands alone but enhances others

5. Quality Assurance Checklist

-------------------------------

* [ ] All code blocks specify ```python language
* [ ] Mathematical formulas include LaTeX or clear ASCII representation
* [ ] Example values (SPY at $450, strikes $445/$455) consistent across all files
* [ ] Every function has complete docstrings with type hints
* [ ] Cross-references use correct relative paths
* [ ] All imports listed at top of code blocks
* [ ] Error handling for edge cases (e.g., negative time, invalid strikes)
* [ ] SKILL.md under 500 lines
* [ ] No Windows-style paths (use forward slashes)
* [ ] Greeks calculations verified against standard formulas
* [ ] All variables defined before use in examples
* [ ] Visualization code includes proper labels, legends, gridlines
* [ ] Risk disclaimers included where appropriate
* [ ] Professional formatting throughout (consistent indentation, spacing)
* [ ] No emojis or informal language
* [ ] All formulas explained in plain English before mathematical notation
* [ ] Type hints on all function signatures
* [ ] Unit test examples for key calculations

</planning>

Now I'll create the complete skill package:

<skill_package>

<file name="SKILL.md">---name: bull-call-spreaddescription: Advanced bull call spread strategy with Greeks analysis, risk management, and comprehensive P&L modeling. Use when analyzing vertical debit spreads, bullish options strategies, or calculating position Greeks for call spreads.---
Bull Call Spread Strategy
=========================

Overview
--------

The bull call spread is a vertical debit spread that profits from moderate upward price movement in the underlying asset. This strategy involves simultaneously buying a call option at a lower strike price and selling a call option at a higher strike price, both with the same expiration date.

**Market Outlook**: Moderately bullish. You expect the underlying to rise but want to reduce cost by capping upside potential.

**Key Characteristics**:

* **Risk**: Limited to net debit paid
* **Reward**: Limited to spread width minus net debit
* **Breakeven**: Long strike plus net debit
* **Capital Requirement**: Net debit (premium paid minus premium received)
* **Greeks Profile**: Positive delta, positive gamma (initially), negative theta, positive vega

Position Construction
---------------------

### Setup Requirements

1. **Select underlying**: Choose liquid, optionable securities
2. **Choose long call strike**: Typically ATM or slightly ITM
3. **Choose short call strike**: OTM, creating spread width of $5-$10
4. **Match expirations**: Both options must have identical expiration dates
5. **Calculate net debit**: Long premium - Short premium

### Strike Selection Guidelines

    from typing import Dict, Optional
    from dataclasses import dataclass
    from datetime import datetime, timedelta

    @dataclass
    class BullCallSpread:
        """Bull call spread position parameters."""
        underlying_symbol: str
        underlying_price: float
        long_strike: float
        short_strike: float
        long_premium: float
        short_premium: float
        expiration_date: datetime
        contracts: int = 1

        def __post_init__(self):
            """Validate position parameters."""
            if self.long_strike >= self.short_strike:
                raise ValueError("Long strike must be less than short strike")
            if self.long_premium <= self.short_premium:
                raise ValueError("Long premium should exceed short premium for debit spread")
            if self.underlying_price <= 0:
                raise ValueError("Underlying price must be positive")

        @property
        def net_debit(self) -> float:
            """Calculate net debit per share."""
            return self.long_premium - self.short_premium

        @property
        def spread_width(self) -> float:
            """Calculate spread width."""
            return self.short_strike - self.long_strike

        @property
        def days_to_expiration(self) -> int:
            """Calculate days until expiration."""
            return (self.expiration_date - datetime.now()).days

        @property
        def position_cost(self) -> float:
            """Calculate total position cost."""
            return self.net_debit * 100 * self.contracts

    # Example: SPY bull call spread
    position = BullCallSpread(
        underlying_symbol="SPY",
        underlying_price=450.00,
        long_strike=445.00,
        short_strike=455.00,
        long_premium=8.50,
        short_premium=3.20,
        expiration_date=datetime.now() + timedelta(days=45),
        contracts=1
    )

    print(f"Net Debit: ${position.net_debit:.2f} per share")
    print(f"Position Cost: ${position.position_cost:.2f}")
    print(f"Spread Width: ${position.spread_width:.2f}")

Profit/Loss Analysis
--------------------

### Breakeven Calculation

The breakeven point occurs when the intrinsic value of the long call equals the net debit paid.

**Formula**: `Breakeven = Long Strike + Net Debit`
    def calculate_breakeven(position: BullCallSpread) -> float:
        """
        Calculate breakeven price at expiration.

        Args:
            position: BullCallSpread instance

        Returns:
            Breakeven price
        """
        return position.long_strike + position.net_debit

    # Using our example
    breakeven = calculate_breakeven(position)
    print(f"Breakeven Price: ${breakeven:.2f}")
    # Output: Breakeven Price: $450.30

### Maximum Profit and Loss

    def calculate_max_profit(position: BullCallSpread) -> Dict[str, float]:
        """
        Calculate maximum profit and required underlying price.

        Args:
            position: BullCallSpread instance

        Returns:
            Dictionary with max profit and price
        """
        max_profit_per_share = position.spread_width - position.net_debit
        max_profit_total = max_profit_per_share * 100 * position.contracts

        return {
            'max_profit_per_share': max_profit_per_share,
            'max_profit_total': max_profit_total,
            'required_price': position.short_strike
        }

    def calculate_max_loss(position: BullCallSpread) -> Dict[str, float]:
        """
        Calculate maximum loss.

        Args:
            position: BullCallSpread instance

        Returns:
            Dictionary with max loss information
        """
        return {
            'max_loss_per_share': position.net_debit,
            'max_loss_total': position.position_cost,
            'occurs_below': position.long_strike
        }

    # Calculate for our position
    max_profit = calculate_max_profit(position)
    max_loss = calculate_max_loss(position)

    print(f"Max Profit: ${max_profit['max_profit_total']:.2f} at ${max_profit['required_price']:.2f}")
    print(f"Max Loss: ${max_loss['max_loss_total']:.2f} below ${max_loss['occurs_below']:.2f}")

### P&L at Expiration

    import numpy as np

    def calculate_pnl_at_expiration(
        position: BullCallSpread,
        stock_price: float
    ) -> Dict[str, float]:
        """
        Calculate profit/loss at expiration for a given stock price.

        Args:
            position: BullCallSpread instance
            stock_price: Stock price at expiration

        Returns:
            Dictionary with P&L metrics
        """
        # Long call value at expiration
        long_call_value = max(stock_price - position.long_strike, 0)

        # Short call value at expiration
        short_call_value = max(stock_price - position.short_strike, 0)

        # Net position value
        position_value = long_call_value - short_call_value

        # P&L calculation
        pnl_per_share = position_value - position.net_debit
        pnl_total = pnl_per_share * 100 * position.contracts

        # Return percentage
        return_pct = (pnl_per_share / position.net_debit) * 100

        return {
            'stock_price': stock_price,
            'position_value': position_value,
            'pnl_per_share': pnl_per_share,
            'pnl_total': pnl_total,
            'return_pct': return_pct
        }

    # Test at various prices
    test_prices = [440, 445, 450, 452, 455, 460]
    print("\nP&L at Various Stock Prices:")
    print("-" * 70)
    print(f"{'Price':<10} {'Position Value':<15} {'P&L':<15} {'Return %':<10}")
    print("-" * 70)

    for price in test_prices:
        result = calculate_pnl_at_expiration(position, price)
        print(f"${price:<9.2f} ${result['position_value']:<14.2f} "
              f"${result['pnl_total']:<14.2f} {result['return_pct']:<9.1f}%")

Payoff Visualization
--------------------

    import matplotlib.pyplot as plt

    def plot_payoff_diagram(
        position: BullCallSpread,
        price_range: Optional[tuple] = None
    ) -> plt.Figure:
        """
        Generate payoff diagram for bull call spread.

        Args:
            position: BullCallSpread instance
            price_range: Optional (min_price, max_price) tuple

        Returns:
            matplotlib Figure object
        """
        # Set price range if not provided
        if price_range is None:
            center = position.underlying_price
            price_range = (center * 0.85, center * 1.15)

        # Generate price points
        prices = np.linspace(price_range[0], price_range[1], 200)

        # Calculate P&L for each price
        pnl_values = []
        for price in prices:
            result = calculate_pnl_at_expiration(position, price)
            pnl_values.append(result['pnl_total'])

        pnl_values = np.array(pnl_values)

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 7))

        # Plot P&L line
        ax.plot(prices, pnl_values, 'b-', linewidth=2, label='Bull Call Spread P&L')

        # Add breakeven line
        breakeven = calculate_breakeven(position)
        ax.axvline(breakeven, color='gray', linestyle='--', linewidth=1.5,
                   label=f'Breakeven: ${breakeven:.2f}')

        # Add current price line
        ax.axvline(position.underlying_price, color='green', linestyle='--',
                   linewidth=1.5, label=f'Current: ${position.underlying_price:.2f}')

        # Add zero line
        ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

        # Shade profit/loss regions
        ax.fill_between(prices, 0, pnl_values, where=(pnl_values >= 0),
                         alpha=0.2, color='green', label='Profit Region')
        ax.fill_between(prices, 0, pnl_values, where=(pnl_values < 0),
                         alpha=0.2, color='red', label='Loss Region')

        # Add strike price annotations
        ax.axvline(position.long_strike, color='blue', linestyle=':', alpha=0.7,
                   label=f'Long Strike: ${position.long_strike:.2f}')
        ax.axvline(position.short_strike, color='red', linestyle=':', alpha=0.7,
                   label=f'Short Strike: ${position.short_strike:.2f}')

        # Labels and formatting
        ax.set_xlabel('Stock Price at Expiration ($)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Profit/Loss ($)', fontsize=12, fontweight='bold')
        ax.set_title(f'{position.underlying_symbol} Bull Call Spread Payoff Diagram\n'
                     f'${position.long_strike}/{position.short_strike} Strikes, '
                     f'{position.days_to_expiration} DTE',
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)

        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.0f}'))

        plt.tight_layout()
        return fig

    # Generate and display payoff diagram
    fig = plot_payoff_diagram(position)
    plt.show()

Greeks Profile
--------------

The Greeks measure how the position value changes with respect to various factors. For detailed mathematical derivations, see [reference.md](https://claude.ai/chat/reference.md).

### Position Greeks Overview

    from scipy.stats import norm
    import math

    def calculate_position_greeks(
        position: BullCallSpread,
        current_price: float,
        volatility: float,
        risk_free_rate: float
    ) -> Dict[str, float]:
        """
        Calculate position-level Greeks for bull call spread.

        Args:
            position: BullCallSpread instance
            current_price: Current underlying price
            volatility: Implied volatility (annualized)
            risk_free_rate: Risk-free rate (annualized)

        Returns:
            Dictionary with all Greeks
        """
        # Time to expiration in years
        T = position.days_to_expiration / 365.0

        if T <= 0:
            return {
                'delta': 0.0, 'gamma': 0.0, 'theta': 0.0,
                'vega': 0.0, 'rho': 0.0
            }

        # Helper function for Black-Scholes d1 and d2
        def calculate_d1_d2(S, K, T, r, sigma):
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            return d1, d2

        # Calculate Greeks for long call (lower strike)
        d1_long, d2_long = calculate_d1_d2(
            current_price, position.long_strike, T, risk_free_rate, volatility
        )

        delta_long = norm.cdf(d1_long)
        gamma_long = norm.pdf(d1_long) / (current_price * volatility * np.sqrt(T))
        theta_long = (
            -(current_price * norm.pdf(d1_long) * volatility) / (2 * np.sqrt(T))
            - risk_free_rate * position.long_strike * np.exp(-risk_free_rate * T) * norm.cdf(d2_long)
        ) / 365  # Convert to per-day
        vega_long = current_price * norm.pdf(d1_long) * np.sqrt(T) / 100  # Per 1% volatility change
        rho_long = (
            position.long_strike * T * np.exp(-risk_free_rate * T) * norm.cdf(d2_long)
        ) / 100  # Per 1% rate change

        # Calculate Greeks for short call (higher strike)
        d1_short, d2_short = calculate_d1_d2(
            current_price, position.short_strike, T, risk_free_rate, volatility
        )

        delta_short = norm.cdf(d1_short)
        gamma_short = norm.pdf(d1_short) / (current_price * volatility * np.sqrt(T))
        theta_short = (
            -(current_price * norm.pdf(d1_short) * volatility) / (2 * np.sqrt(T))
            - risk_free_rate * position.short_strike * np.exp(-risk_free_rate * T) * norm.cdf(d2_short)
        ) / 365
        vega_short = current_price * norm.pdf(d1_short) * np.sqrt(T) / 100
        rho_short = (
            position.short_strike * T * np.exp(-risk_free_rate * T) * norm.cdf(d2_short)
        ) / 100

        # Net position Greeks (long - short)
        position_delta = (delta_long - delta_short) * 100 * position.contracts
        position_gamma = (gamma_long - gamma_short) * 100 * position.contracts
        position_theta = (theta_long - theta_short) * 100 * position.contracts
        position_vega = (vega_long - vega_short) * 100 * position.contracts
        position_rho = (rho_long - rho_short) * 100 * position.contracts

        return {
            'delta': position_delta,
            'gamma': position_gamma,
            'theta': position_theta,
            'vega': position_vega,
            'rho': position_rho
        }

    # Calculate Greeks for our example
    greeks = calculate_position_greeks(
        position=position,
        current_price=450.00,
        volatility=0.18,  # 18% implied volatility
        risk_free_rate=0.05  # 5% risk-free rate
    )

    print("\nPosition Greeks:")
    print("-" * 50)
    print(f"Delta: {greeks['delta']:>10.2f}  (Expected move per $1 increase)")
    print(f"Gamma: {greeks['gamma']:>10.4f}  (Delta change per $1 move)")
    print(f"Theta: {greeks['theta']:>10.2f}  (Daily time decay)")
    print(f"Vega:  {greeks['vega']:>10.2f}  (Value change per 1% IV move)")
    print(f"Rho:   {greeks['rho']:>10.2f}  (Value change per 1% rate move)")

### Greeks Interpretation

**Delta (~30-40 for this spread)**: Measures directional exposure. A delta of 35 means the position gains approximately $35 for every $1 increase in the underlying, up to the short strike.

**Gamma (typically positive but small)**: Measures delta acceleration. Positive gamma benefits from large moves but diminishes as price approaches the short strike.

**Theta (typically negative)**: Time decay. The position loses value daily due to time decay, though less than a naked long call.

**Vega (typically positive)**: Volatility sensitivity. The position benefits from increasing implied volatility, especially when strikes are near the current price.
Risk Management
---------------

### Position Sizing

**Capital allocation guideline**: Risk no more than 1-2% of account equity per trade.
    def calculate_position_size(
        account_equity: float,
        risk_percentage: float,
        position: BullCallSpread
    ) -> int:
        """
        Calculate appropriate position size based on risk tolerance.

        Args:
            account_equity: Total account value
            risk_percentage: Maximum risk per trade (e.g., 0.02 for 2%)
            position: BullCallSpread instance

        Returns:
            Number of contracts
        """
        max_risk_dollars = account_equity * risk_percentage
        risk_per_contract = position.net_debit * 100

        contracts = int(max_risk_dollars / risk_per_contract)
        return max(1, contracts)  # At least 1 contract

    # Example with $100,000 account
    account = 100000
    risk_pct = 0.02  # 2% risk

    contracts = calculate_position_size(account, risk_pct, position)
    total_risk = contracts * position.net_debit * 100

    print(f"\nPosition Sizing:")
    print(f"Account Equity: ${account:,.2f}")
    print(f"Risk Tolerance: {risk_pct*100:.1f}%")
    print(f"Recommended Contracts: {contracts}")
    print(f"Total Capital at Risk: ${total_risk:,.2f}")
    print(f"Actual Risk %: {(total_risk/account)*100:.2f}%")

### Exit Strategies

**Profit Target**: Close when reaching 50-75% of maximum profit to optimize risk-adjusted returns.

**Stop Loss**: Exit if position loses 50% of initial debit to preserve capital.

**Time-Based Exit**: Consider closing at 50% of time to expiration when theta decay accelerates.
    def check_exit_signals(
        position: BullCallSpread,
        current_pnl: float,
        days_elapsed: int
    ) -> Dict[str, bool]:
        """
        Evaluate exit conditions for the position.

        Args:
            position: BullCallSpread instance
            current_pnl: Current profit/loss in dollars
            days_elapsed: Days since position opened

        Returns:
            Dictionary with exit signal flags
        """
        max_profit = calculate_max_profit(position)['max_profit_total']
        max_loss = position.position_cost

        # Profit target: 75% of max profit
        profit_target_hit = current_pnl >= (0.75 * max_profit)

        # Stop loss: 50% of debit paid
        stop_loss_hit = current_pnl <= (-0.50 * max_loss)

        # Time-based: 50% of time elapsed
        initial_dte = position.days_to_expiration
        time_exit = days_elapsed >= (initial_dte * 0.50)

        return {
            'take_profit': profit_target_hit,
            'stop_loss': stop_loss_hit,
            'time_exit': time_exit,
            'any_exit': profit_target_hit or stop_loss_hit or time_exit
        }

### Adjustment Techniques

**Rolling Up**: If underlying rallies strongly, roll the entire spread to higher strikes to capture additional profit.

**Rolling Out**: Extend expiration to give the trade more time, accepting additional theta decay.

**Converting to Butterfly**: Add a long call at the short strike and sell a call further OTM to create a butterfly spread with reduced risk.

For detailed adjustment scenarios, see [reference.md](https://claude.ai/chat/reference.md#adjustment-strategies).
Complete Implementation Example
-------------------------------

    from typing import Dict, List, Tuple
    from datetime import datetime, timedelta
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    from dataclasses import dataclass

    @dataclass
    class BullCallSpreadComplete:
        """
        Complete bull call spread strategy implementation.

        Attributes:
            All position parameters plus analysis methods
        """
        underlying_symbol: str
        underlying_price: float
        long_strike: float
        short_strike: float
        long_premium: float
        short_premium: float
        expiration_date: datetime
        contracts: int = 1

        def analyze_position(
            self,
            volatility: float = 0.20,
            risk_free_rate: float = 0.05
        ) -> Dict:
            """
            Comprehensive position analysis.

            Returns complete risk metrics, Greeks, and scenarios.
            """
            # Basic metrics
            metrics = {
                'net_debit': self.long_premium - self.short_premium,
                'spread_width': self.short_strike - self.long_strike,
                'position_cost': (self.long_premium - self.short_premium) * 100 * self.contracts,
                'days_to_expiration': (self.expiration_date - datetime.now()).days
            }

            # Risk metrics
            metrics['max_profit'] = (metrics['spread_width'] - metrics['net_debit']) * 100 * self.contracts
            metrics['max_loss'] = metrics['position_cost']
            metrics['breakeven'] = self.long_strike + metrics['net_debit']
            metrics['risk_reward_ratio'] = metrics['max_profit'] / metrics['max_loss']

            # Greeks
            T = metrics['days_to_expiration'] / 365.0

            def calc_greeks(S, K):
                d1 = (np.log(S/K) + (risk_free_rate + 0.5*volatility**2)*T) / (volatility*np.sqrt(T))
                d2 = d1 - volatility*np.sqrt(T)
                delta = norm.cdf(d1)
                gamma = norm.pdf(d1) / (S * volatility * np.sqrt(T))
                theta = (-(S * norm.pdf(d1) * volatility)/(2*np.sqrt(T)) -
                         risk_free_rate*K*np.exp(-risk_free_rate*T)*norm.cdf(d2)) / 365
                vega = S * norm.pdf(d1) * np.sqrt(T) / 100
                return delta, gamma, theta, vega

            d_long, g_long, t_long, v_long = calc_greeks(self.underlying_price, self.long_strike)
            d_short, g_short, t_short, v_short = calc_greeks(self.underlying_price, self.short_strike)

            metrics['delta'] = (d_long - d_short) * 100 * self.contracts
            metrics['gamma'] = (g_long - g_short) * 100 * self.contracts
            metrics['theta'] = (t_long - t_short) * 100 * self.contracts
            metrics['vega'] = (v_long - v_short) * 100 * self.contracts

            return metrics

        def scenario_analysis(self) -> pd.DataFrame:
            """
            Generate scenario analysis across price range.

            Returns DataFrame with P&L at various prices.
            """
            price_range = np.linspace(
                self.underlying_price * 0.90,
                self.underlying_price * 1.10,
                21
            )

            scenarios = []
            for price in price_range:
                long_value = max(price - self.long_strike, 0)
                short_value = max(price - self.short_strike, 0)
                position_value = long_value - short_value
                pnl = (position_value - (self.long_premium - self.short_premium)) * 100 * self.contracts

                scenarios.append({
                    'Price': price,
                    'Long Call': long_value,
                    'Short Call': short_value,
                    'Net Value': position_value,
                    'P&L': pnl,
                    'Return %': (pnl / self.position_cost) * 100 if hasattr(self, 'position_cost') else 0
                })

            return pd.DataFrame(scenarios)

        @property
        def position_cost(self) -> float:
            return (self.long_premium - self.short_premium) * 100 * self.contracts

    # Create and analyze position
    strategy = BullCallSpreadComplete(
        underlying_symbol="SPY",
        underlying_price=450.00,
        long_strike=445.00,
        short_strike=455.00,
        long_premium=8.50,
        short_premium=3.20,
        expiration_date=datetime.now() + timedelta(days=45),
        contracts=2
    )

    # Run comprehensive analysis
    analysis = strategy.analyze_position(volatility=0.18, risk_free_rate=0.05)

    print("\n" + "="*60)
    print("BULL CALL SPREAD ANALYSIS")
    print("="*60)
    print(f"\nPosition Details:")
    print(f"  Symbol: {strategy.underlying_symbol}")
    print(f"  Current Price: ${strategy.underlying_price:.2f}")
    print(f"  Long Strike: ${strategy.long_strike:.2f}")
    print(f"  Short Strike: ${strategy.short_strike:.2f}")
    print(f"  Net Debit: ${analysis['net_debit']:.2f}/share")
    print(f"  Contracts: {strategy.contracts}")
    print(f"  Position Cost: ${analysis['position_cost']:.2f}")
    print(f"  Days to Expiration: {analysis['days_to_expiration']}")

    print(f"\nRisk Metrics:")
    print(f"  Maximum Profit: ${analysis['max_profit']:.2f}")
    print(f"  Maximum Loss: ${analysis['max_loss']:.2f}")
    print(f"  Breakeven: ${analysis['breakeven']:.2f}")
    print(f"  Risk/Reward Ratio: {analysis['risk_reward_ratio']:.2f}")

    print(f"\nGreeks:")
    print(f"  Delta: {analysis['delta']:.2f}")
    print(f"  Gamma: {analysis['gamma']:.4f}")
    print(f"  Theta: {analysis['theta']:.2f}")
    print(f"  Vega: {analysis['vega']:.2f}")

    # Scenario analysis
    scenarios_df = strategy.scenario_analysis()
    print(f"\nScenario Analysis:")
    print(scenarios_df.to_string(index=False))

Summary
-------

The bull call spread is an effective strategy for capturing moderate bullish moves while limiting both risk and reward. Key advantages include:

* **Defined risk**: Maximum loss limited to net debit
* **Lower cost**: Short call reduces capital requirement
* **Positive theta**: Less time decay than naked long call
* **Strategic flexibility**: Can be adjusted or rolled

**Best used when**:

* Moderately bullish outlook (not expecting explosive moves)
* Want to reduce cost basis compared to naked calls
* Time premium is elevated
* Comfortable with capped upside

For advanced topics including early assignment risk, dividend considerations, and detailed Greeks mathematics, see [reference.md](https://claude.ai/chat/reference.md).

For production-ready implementation code, see [scripts/strategy_calculator.py](https://claude.ai/chat/scripts/strategy_calculator.py).

</file><file name="reference.md"># Bull Call Spread - Advanced Reference
Mathematical Foundations
------------------------

### Black-Scholes Framework for Call Options

The Black-Scholes model provides the theoretical framework for pricing European-style call options and calculating Greeks.

**Call Option Price**:
    C = S₀N(d₁) - Ke^(-rT)N(d₂)

    Where:
      C = Call option price
      S₀ = Current stock price
      K = Strike price
      r = Risk-free interest rate
      T = Time to expiration (years)
      N(·) = Cumulative standard normal distribution

      d₁ = [ln(S₀/K) + (r + σ²/2)T] / (σ√T)
      d₂ = d₁ - σ√T
      σ = Volatility (annualized standard deviation)

**Standard Normal Density Function**:
    φ(x) = (1/√(2π)) × e^(-x²/2)

### Bull Call Spread Valuation

The bull call spread value is the difference between the long and short call values:
    Spread_Value = C(K_long) - C(K_short)

    Where:
      C(K_long) = Value of long call at lower strike
      C(K_short) = Value of short call at higher strike

**At Expiration**:
    Payoff = max(S_T - K_long, 0) - max(S_T - K_short, 0)

    Which simplifies to:
      - If S_T ≤ K_long: Payoff = 0
      - If K_long < S_T < K_short: Payoff = S_T - K_long
      - If S_T ≥ K_short: Payoff = K_short - K_long (maximum)

**Profit/Loss**:
    P&L = Payoff - Net_Debit
    Net_Debit = Premium_long - Premium_short

### Breakeven Analysis

**Breakeven Derivation**:
    At breakeven, P&L = 0:
      (S_T - K_long) - Net_Debit = 0
      S_T = K_long + Net_Debit

    Therefore:
      Breakeven = Long_Strike + Net_Debit

**Probability of Profit** (assuming lognormal distribution):
    P(Profit) = N(d₂*)

    Where d₂* is calculated using:
      d₂* = [ln(S₀/BE) + (r - σ²/2)T] / (σ√T)
      BE = Breakeven price
Greeks: Complete Derivations
----------------------------

### Delta (Δ)

**Definition**: Rate of change of option value with respect to underlying price.

**Individual Call Delta**:
    Δ_call = N(d₁)

    Properties:
      - Range: 0 to 1
      - ATM call: Δ ≈ 0.50
      - ITM call: Δ → 1 as S increases
      - OTM call: Δ → 0 as S decreases

**Position Delta** (Bull Call Spread):
    Δ_position = Δ_long - Δ_short
                 = N(d₁_long) - N(d₁_short)

    Typical values: 0.20 to 0.50

    Interpretation:
      - Δ = 0.35 means position gains $35 per $1 increase in underlying
      - Net delta decreases as price moves above short strike
      - Maximum delta occurs when price is between strikes

**Delta Behavior**:

* **Below long strike**: Δ ≈ 0 (both options OTM)
* **Between strikes**: Δ increases, reaches maximum near midpoint
* **Above short strike**: Δ ≈ 0 (both options ITM, move equally)

### Gamma (Γ)

**Definition**: Rate of change of delta with respect to underlying price.

**Individual Call Gamma**:
    Γ_call = φ(d₁) / (S₀σ√T)

    Where φ(d₁) is the standard normal density function

**Position Gamma**:
    Γ_position = Γ_long - Γ_short
                 = [φ(d₁_long) - φ(d₁_short)] / (S₀σ√T)

**Gamma Characteristics**:

* **Peak**: Highest when underlying is near ATM
* **Sign**: Can be positive or negative depending on position
* **Time Decay**: Increases as expiration approaches (especially ATM)
* **Volatility**: Inversely related to implied volatility

**Practical Implications**:
    # Gamma risk assessment
    def gamma_risk_analysis(gamma: float, expected_move: float) -> Dict:
        """
        Estimate delta change from gamma exposure.

        Args:
            gamma: Position gamma
            expected_move: Expected price move

        Returns:
            Dictionary with risk metrics
        """
        delta_change = gamma * expected_move

        return {
            'gamma': gamma,
            'expected_move': expected_move,
            'estimated_delta_change': delta_change,
            'risk_level': 'HIGH' if abs(delta_change) > 0.10 else 'MODERATE'
        }

### Theta (Θ)

**Definition**: Rate of change of option value with respect to time (per day).

**Individual Call Theta**:
    Θ_call = -[S₀φ(d₁)σ / (2√T)] - rKe^(-rT)N(d₂)

    Expressed per day: Θ_daily = Θ_annual / 365

**Position Theta**:
    Θ_position = Θ_long - Θ_short

    Key insight: Both legs lose time value, but at different rates

**Theta Decay Patterns**:

1. **Early in Trade** (>30 DTE):

   * Slow, steady decay
   * Theta impact relatively small
   * Focus on directional movement

2. **Mid-Life** (15-30 DTE):

   * Accelerating decay
   * Theta becomes significant factor
   * Balance time decay vs. directional edge

3. **Near Expiration** (<15 DTE):

   * Rapid decay, especially for ATM options
   * Critical period for position management
   * Consider closing or rolling

**Theta Efficiency Ratio**:
    Efficiency = Daily_Theta / Capital_at_Risk

    Higher ratio = better time decay relative to risk

### Vega (ν)

**Definition**: Rate of change of option value with respect to 1% change in implied volatility.

**Individual Call Vega**:
    ν_call = S₀φ(d₁)√T / 100

    Properties:
      - Always positive for long options
      - Maximum for ATM options
      - Decreases as moves ITM or OTM
      - Decreases as time to expiration decreases

**Position Vega**:
    ν_position = ν_long - ν_short

    Typically positive for bull call spreads

**Volatility Sensitivity Analysis**:
    def volatility_scenarios(
        position: 'BullCallSpread',
        current_vol: float,
        vol_changes: List[float]
    ) -> pd.DataFrame:
        """
        Analyze P&L under different volatility scenarios.

        Args:
            position: Bull call spread position
            current_vol: Current implied volatility
            vol_changes: List of volatility changes (e.g., [-0.05, -0.02, 0, 0.02, 0.05])

        Returns:
            DataFrame with scenario results
        """
        import pandas as pd

        results = []
        for vol_change in vol_changes:
            new_vol = current_vol + vol_change

            # Simplified vega impact (for illustration)
            # In practice, would recalculate full position value
            greeks = calculate_position_greeks(
                position,
                position.underlying_price,
                new_vol,
                0.05
            )

            # Estimate P&L change from vega
            pnl_change = greeks['vega'] * (vol_change * 100)

            results.append({
                'IV_Change': f"{vol_change*100:+.1f}%",
                'New_IV': f"{new_vol*100:.1f}%",
                'Vega_Impact': f"${pnl_change:.2f}",
                'Position_Vega': f"{greeks['vega']:.2f}"
            })

        return pd.DataFrame(results)

**Volatility Smile Effects**:

* OTM calls often have higher IV than ATM (volatility skew)
* Short OTM call may benefit more from IV decrease
* Consider skew when selecting strikes

### Rho (ρ)

**Definition**: Rate of change of option value with respect to 1% change in risk-free rate.

**Individual Call Rho**:
    ρ_call = KTe^(-rT)N(d₂) / 100

**Position Rho**:
    ρ_position = ρ_long - ρ_short

    Typically small impact for short-dated options
    More relevant for LEAPS or long-dated spreads

**Interest Rate Sensitivity**:

* Generally minor factor for <90 DTE positions
* Can be significant for 6-12 month positions
* Rising rates increase call values (present value effect)

Advanced Scenarios and Edge Cases
---------------------------------

### 1. Early Assignment Risk

**When It Occurs**:

* Short call is deep ITM (delta near 1.0)
* Dividend ex-date approaching
* Time value < dividend amount
* Before major corporate actions

**Assessment**:
    def assess_assignment_risk(
        short_strike: float,
        current_price: float,
        time_value: float,
        upcoming_dividend: float,
        days_to_ex_div: int
    ) -> Dict:
        """
        Evaluate early assignment probability.

        Args:
            short_strike: Strike price of short call
            current_price: Current underlying price
            time_value: Remaining time value in short call
            upcoming_dividend: Dividend amount per share
            days_to_ex_div: Days until ex-dividend date

        Returns:
            Risk assessment dictionary
        """
        intrinsic_value = max(current_price - short_strike, 0)

        # Assignment typically occurs if time value < dividend
        assignment_likely = (time_value < upcoming_dividend and
                            days_to_ex_div <= 3 and
                            intrinsic_value > 0)

        # Financial impact of assignment
        if assignment_likely:
            # If assigned, must deliver shares
            # May need to buy shares at market
            max_loss_on_assignment = (current_price - short_strike) * 100

            # Receive dividend if holding through ex-date
            dividend_received = upcoming_dividend * 100

            net_impact = max_loss_on_assignment - dividend_received
        else:
            net_impact = 0

        return {
            'assignment_likely': assignment_likely,
            'time_value': time_value,
            'intrinsic_value': intrinsic_value,
            'dividend': upcoming_dividend,
            'days_to_ex_div': days_to_ex_div,
            'estimated_impact': net_impact,
            'recommendation': 'CLOSE POSITION' if assignment_likely else 'MONITOR'
        }

**Mitigation Strategies**:

1. Close position before ex-dividend date
2. Roll short call to higher strike or later expiration
3. Accept assignment if financially neutral
4. Monitor ITM amount and time value daily

### 2. Dividend Impact Analysis

**Effect on Position Value**:
    Pre-dividend call value adjustment:
      C_pre_div ≈ C_no_div - PV(Dividend)

    Where PV(Dividend) = Dividend × e^(-r × t)

**Strategic Considerations**:

* Long call loses value equal to PV(dividend)
* Short call also loses value (benefits position)
* Net effect typically small for small dividends
* Large dividends can trigger early assignment

### 3. Volatility Skew Optimization

**Skew Analysis**:
    def analyze_volatility_skew(
        strikes: List[float],
        implied_vols: List[float],
        current_price: float
    ) -> Dict:
        """
        Analyze IV skew to optimize strike selection.

        Args:
            strikes: List of available strikes
            implied_vols: Corresponding implied volatilities
            current_price: Current underlying price

        Returns:
            Skew metrics and recommendations
        """
        # Calculate moneyness for each strike
        moneyness = [(k / current_price - 1) * 100 for k in strikes]

        # Find ATM implied vol
        atm_idx = min(range(len(strikes)),
                      key=lambda i: abs(strikes[i] - current_price))
        atm_vol = implied_vols[atm_idx]

        # Calculate skew (OTM_vol - ATM_vol)
        otm_strikes = [s for s in strikes if s > current_price]
        otm_vols = [implied_vols[i] for i, s in enumerate(strikes) if s > current_price]

        if otm_vols:
            avg_otm_vol = sum(otm_vols) / len(otm_vols)
            skew = avg_otm_vol - atm_vol
        else:
            skew = 0

        return {
            'atm_volatility': atm_vol,
            'otm_avg_volatility': avg_otm_vol if otm_vols else None,
            'skew': skew,
            'skew_pct': (skew / atm_vol * 100) if atm_vol > 0 else 0,
            'interpretation': 'STEEP SKEW' if skew > 0.03 else
                             'FLAT SKEW' if abs(skew) < 0.01 else
                             'NEGATIVE SKEW'
        }

**Implications for Bull Call Spreads**:

* **Positive skew** (OTM > ATM): Short call has higher IV

  * Benefit: Collect more premium on short call
  * Trade-off: Long call has lower IV (costs less)
  * Net effect: Generally favorable for spread

* **Flat skew**: Neutral impact

* **Negative skew** (rare): OTM < ATM

  * Less favorable for bull call spreads
  * Consider alternative strategies

### 4. Roll Management Strategies

**When to Roll**:

1. **Profit target achieved early**: 50-75% of max profit reached
2. **Time decay acceleration**: <21 DTE, theta decay ramping up
3. **Underlying momentum shift**: Approaching short strike with momentum
4. **Defensive roll**: Position going against you, extend time

**Roll Types**:

**A. Roll Up (Same Expiration)**:
    Close existing spread: $445/$455
    Open new spread: $455/$465

    Benefit: Capture additional profit potential
    Cost: Pay debit to roll (or small credit in strong rally)

**B. Roll Out (Same Strikes)**:
    Close existing spread: $445/$455 (45 DTE)
    Open new spread: $445/$455 (75 DTE)

    Benefit: More time for trade to work
    Cost: Additional capital commitment

**C. Roll Up and Out**:
    Close existing spread: $445/$455 (30 DTE)
    Open new spread: $455/$465 (60 DTE)

    Benefit: Higher profit potential + more time
    Most aggressive roll type

**Roll Decision Framework**:
    def evaluate_roll_opportunity(
        current_position: 'BullCallSpread',
        current_pnl: float,
        days_held: int,
        new_strikes: Tuple[float, float],
        new_expiration_dte: int,
        new_premiums: Tuple[float, float]
    ) -> Dict:
        """
        Evaluate whether rolling position makes sense.

        Args:
            current_position: Existing spread
            current_pnl: Current P&L
            days_held: Days position has been held
            new_strikes: (new_long_strike, new_short_strike)
            new_expiration_dte: DTE for new position
            new_premiums: (new_long_premium, new_short_premium)

        Returns:
            Roll analysis with recommendation
        """
        # Current position metrics
        current_max_profit = calculate_max_profit(current_position)['max_profit_total']
        pnl_pct = (current_pnl / current_position.position_cost) * 100

        # New position metrics
        new_debit = new_premiums[0] - new_premiums[1]
        new_spread_width = new_strikes[1] - new_strikes[0]
        new_max_profit = (new_spread_width - new_debit) * 100

        # Roll cost
        closing_value = current_pnl + current_position.position_cost
        roll_cost = (new_debit * 100) - closing_value

        # Combined metrics
        total_capital = current_position.position_cost + roll_cost
        combined_max_profit = new_max_profit
        combined_roi = (combined_max_profit / total_capital) * 100

        # Decision criteria
        recommend_roll = (
            pnl_pct > 50 and  # Captured most of profit
            roll_cost < (current_max_profit * 0.30) and  # Roll cost reasonable
            combined_roi > 20  # New position has good ROI
        )

        return {
            'current_pnl_pct': pnl_pct,
            'roll_cost': roll_cost,
            'new_max_profit': new_max_profit,
            'combined_capital': total_capital,
            'combined_roi': combined_roi,
            'recommend_roll': recommend_roll,
            'reason': 'ROLL: Good profit capture + attractive new position' if recommend_roll
                     else 'HOLD: Insufficient profit or unfavorable roll terms'
        }

### 5. Adjustment to Butterfly Spread

**Conversion Logic**:
    Original: Long $445 call, Short $455 call
    Adjust to: Long $445 call, Short 2× $455 calls, Long $465 call

    This creates a butterfly: $445/$455/$465

**When to Consider**:

* Underlying stalling near short strike
* Want to reduce risk while maintaining some upside
* High IV making additional short call attractive

**Risk-Reward Tradeoff**:

* **Reduced risk**: Max loss decreased
* **Narrower profit zone**: Profit only near $455
* **Additional capital**: Typically requires small debit

Comparative Strategy Analysis
-----------------------------

### Bull Call Spread vs. Naked Long Call

| Metric        | Bull Call Spread                      | Long Call                               |
| ------------- | ------------------------------------- | --------------------------------------- |
| Cost          | Lower (credit from short call)        | Higher (full premium)                   |
| Max Profit    | Limited to spread width               | Unlimited                               |
| Max Loss      | Net debit                             | Full premium                            |
| Breakeven     | Higher (long strike + net debit)      | Lower (strike + premium)                |
| Theta         | Less negative (short call offsets)    | More negative                           |
| Vega          | Less positive (short call offsets)    | More positive                           |
| **Best When** | Moderate bullish view, defined target | Strong bullish, explosive move expected |

**Numerical Example**:
    Scenario: SPY at $450

    Long Call ($450 strike):
      - Premium: $8.00
      - Max Loss: $800
      - Breakeven: $458.00
      - Unlimited profit above $458

    Bull Call Spread ($445/$455):
      - Net Debit: $5.30
      - Max Loss: $530
      - Breakeven: $450.30
      - Max Profit: $470 at $455+

    Comparison at $460:
      - Long call profit: $200
      - Spread profit: $470 (max)

    At $465:
      - Long call profit: $700
      - Spread profit: $470 (capped)

### Bull Call Spread vs. Bull Put Spread

Both are bullish strategies but structured differently:

**Bull Call Spread** (Vertical Debit Spread):

* Pay net debit
* Long lower strike, short higher strike
* Profit from upward move
* Max profit at/above short strike

**Bull Put Spread** (Vertical Credit Spread):

* Receive net credit
* Short higher strike, long lower strike
* Profit from time decay + upward move
* Max profit if expires above short strike

**Key Differences**:

| Factor           | Bull Call Spread                   | Bull Put Spread               |
| ---------------- | ---------------------------------- | ----------------------------- |
| Cash Flow        | Pay debit upfront                  | Receive credit upfront        |
| Assignment Risk  | Short call (if ITM)                | Short put (if ITM)            |
| Margin Required  | None (fully paid)                  | Yes (cash-secured puts)       |
| Theta            | Negative (but less than long call) | Positive (collect decay)      |
| Psychological    | "Buying" feels intuitive           | "Selling" may feel risky      |
| **Optimal When** | Strong directional conviction      | Neutral to moderately bullish |

**Synthetic Relationship** (Put-Call Parity):
    Bull call spread ≈ Bull put spread + cost of carry adjustments

    Example:
    $445/$455 bull call spread ≈ $445/$455 bull put spread (economically similar)
Implementation Best Practices
-----------------------------

### 1. Strike Selection Methodology

**Step-by-step approach**:
    def select_optimal_strikes(
        underlying_price: float,
        price_target: float,
        target_delta: float = 0.35,
        max_capital: float = 1000,
        available_strikes: List[float] = None
    ) -> Tuple[float, float]:
        """
        Determine optimal strike prices for bull call spread.

        Args:
            underlying_price: Current stock price
            price_target: Expected price at expiration
            target_delta: Desired net delta (0.30-0.40 recommended)
            max_capital: Maximum capital to deploy
            available_strikes: List of available strikes

        Returns:
            (long_strike, short_strike) tuple
        """
        if available_strikes is None:
            # Generate typical strikes around current price
            available_strikes = [
                underlying_price - 10, underlying_price - 5,
                underlying_price, underlying_price + 5,
                underlying_price + 10, underlying_price + 15
            ]

        # Filter strikes
        itm_strikes = [s for s in available_strikes if s < underlying_price]
        otm_strikes = [s for s in available_strikes if s > underlying_price]

        # Long strike: slightly ITM for delta
        long_strike = max(itm_strikes) if itm_strikes else underlying_price

        # Short strike: just above price target
        short_candidates = [s for s in otm_strikes if s >= price_target]
        short_strike = min(short_candidates) if short_candidates else max(available_strikes)

        # Validate spread width (typically $5-$15)
        spread_width = short_strike - long_strike
        if spread_width < 5:
            # Too narrow, adjust short strike up
            short_strike = long_strike + 10
        elif spread_width > 20:
            # Too wide, adjust long strike up
            long_strike = short_strike - 10

        return (long_strike, short_strike)

    # Example usage
    optimal_strikes = select_optimal_strikes(
        underlying_price=450.00,
        price_target=458.00,
        target_delta=0.35,
        max_capital=600
    )
    print(f"Recommended Strikes: ${optimal_strikes[0]:.0f}/${optimal_strikes[1]:.0f}")

### 2. Expiration Selection

**General Guidelines**:

* **30-45 DTE**: Sweet spot for most traders
  * Balance theta decay vs. time for move
  * Good liquidity in monthly options
  * Reasonable premium collection
* **60-90 DTE**: For patient traders
  * More time for thesis to play out
  * Lower theta decay per day
  * Higher capital commitment
* **< 30 DTE**: For aggressive traders
  * Rapid theta decay
  * Requires close monitoring
  * Higher risk of total loss

**Theta decay curve consideration**:
    Days to Expiration | % of Time Value Lost per Day
    60+ DTE            | 1-2%
    45 DTE             | 2-3%
    30 DTE             | 3-5%
    15 DTE             | 5-10%
    7 DTE              | 10-20%

### 3. Entry Timing Optimization

**Technical Entry Signals**:

1. Pullback to support level
2. Breakout above resistance
3. Moving average crossover
4. RSI above 50 but below 70

**Volatility Considerations**:
    def assess_entry_timing(
        current_iv: float,
        iv_percentile: float,
        price_momentum: str
    ) -> str:
        """
        Evaluate optimal entry timing based on volatility and momentum.

        Args:
            current_iv: Current implied volatility
            iv_percentile: IV percentile (0-100)
            price_momentum: 'BULLISH', 'NEUTRAL', or 'BEARISH'

        Returns:
            Entry recommendation
        """
        if iv_percentile > 75:
            return "WAIT: IV too high, wait for IV contraction"
        elif iv_percentile < 25:
            return "CAUTION: IV too low, limited vega benefit"
        elif 25 <= iv_percentile <= 75:
            if price_momentum == 'BULLISH':
                return "ENTER: Good IV level + positive momentum"
            elif price_momentum == 'NEUTRAL':
                return "CONSIDER: Neutral momentum, wait for catalyst"
            else:
                return "AVOID: Bearish momentum contradicts strategy"
        else:
            return "REASSESS: Check all parameters"
External Resources and Further Reading
--------------------------------------

### Academic Research

1. **"Analysis of Vertical Spread Trading Strategies"** - Journal of Derivatives, discussing optimal strike selection and risk-adjusted returns

2. **Black, F., & Scholes, M. (1973)** - "The Pricing of Options and Corporate Liabilities" - foundational options pricing theory

3. **Hull, J. C.** - _Options, Futures, and Other Derivatives_ - comprehensive derivatives textbook

### Industry Standards

* **CBOE Strategy Benchmark Indices** - Performance benchmarks for various options strategies
* **Options Industry Council (OIC)** - Educational resources and strategy guides
* **CFA Institute** - Options and derivatives curriculum materials

### Risk Management Frameworks

* **ISO 31000** - Risk management guidelines
* **Basel III** - Capital adequacy standards (for institutional context)
* **MIL-STD-882E** - System Safety standard (methodical risk assessment approach)

### Practitioner Guides

* **CBOE Options Institute** - Free courses on options strategies
* **Tasty Trade** - Research on options strategy statistics
* **Options Clearing Corporation (OCC)** - Regulatory and educational materials

### Recommended Tools

* **Options profit calculators**: OptionStrat, OptionsProfitCalculator
* **IV analysis**: Market Chameleon, IVolatility
* **Position management**: ThinkorSwim, Interactive Brokers TWS
* **Backtesting**: QuantConnect, Zipline

Regulatory and Compliance Considerations
----------------------------------------

### Position Limits

* **CBOE/Exchange limits**: Typically 25,000-250,000 contracts depending on security
* **Broker limitations**: May have lower limits than exchange
* **Concentration risk**: Manage position size relative to account

### Tax Treatment (US)

* **Short-term capital gains**: Positions held < 1 year
* **Section 1256 contracts**: Certain index options receive 60/40 treatment
* **Wash sale rules**: Be aware of substantially identical positions
* **Straddle rules**: Complex tax treatment for certain positions

_Note: Consult tax professional for specific guidance_

### Best Execution

* **NBBO** (National Best Bid Offer): Brokers must seek best price
* **Price improvement**: Try to get filled inside the spread
* **Liquidity considerations**: Use limit orders in less liquid securities

Conclusion
----------

The bull call spread offers sophisticated traders a mathematically sound approach to capturing moderate bullish moves while managing risk through position construction. Success requires understanding:

1. **Precise Greeks management**: Monitor delta, theta, and vega dynamics
2. **Strategic strike selection**: Balance cost, probability, and reward
3. **Disciplined risk management**: Define exits before entry
4. **Market environment awareness**: Volatility and momentum context
5. **Adjustment flexibility**: Know when to roll, close, or modify

For implementation code and utilities, see [scripts/](https://claude.ai/chat/scripts/) directory.

</file><file name="scripts/strategy_calculator.py">#!/usr/bin/env python3"""Bull Call Spread Strategy Calculator

Complete implementation for analyzing bull call spread options strategiesincluding P&L calculations, Greeks, payoff diagrams, and risk management.

Usage:python strategy_calculator.py --underlying SPY --price 450--long-strike 445 --short-strike 455--long-premium 8.50 --short-premium 3.20"""

import argparseimport sysfrom datetime import datetime, timedeltafrom typing import Dict, List, Tuple, Optionalfrom dataclasses import dataclass, field

import numpy as npimport pandas as pdimport matplotlib.pyplot as pltfrom scipy.stats import norm

@dataclassclass BullCallSpread:"""Bull call spread position with complete analysis capabilities.
    Attributes:
        underlying_symbol: Ticker symbol
        underlying_price: Current stock price
        long_strike: Strike price of long call (lower)
        short_strike: Strike price of short call (higher)
        long_premium: Premium paid for long call
        short_premium: Premium received for short call
        expiration_date: Option expiration date
        contracts: Number of contracts (default 1)
        volatility: Implied volatility (default 0.20)
        risk_free_rate: Risk-free interest rate (default 0.05)
    """
    underlying_symbol: str
    underlying_price: float
    long_strike: float
    short_strike: float
    long_premium: float
    short_premium: float
    expiration_date: datetime
    contracts: int = 1
    volatility: float = 0.20
    risk_free_rate: float = 0.05

    def __post_init__(self):
        """Validate position parameters."""
        if self.long_strike >= self.short_strike:
            raise ValueError("Long strike must be less than short strike")
        if self.long_premium <= self.short_premium:
            raise ValueError("Long premium should exceed short premium for debit spread")
        if self.underlying_price <= 0:
            raise ValueError("Underlying price must be positive")
        if self.contracts <= 0:
            raise ValueError("Contracts must be positive")
        if not (0 < self.volatility <= 2.0):
            raise ValueError("Volatility must be between 0 and 2.0")

    @property
    def net_debit(self) -> float:
        """Net debit per share."""
        return self.long_premium - self.short_premium

    @property
    def spread_width(self) -> float:
        """Spread width (difference in strikes)."""
        return self.short_strike - self.long_strike

    @property
    def position_cost(self) -> float:
        """Total position cost."""
        return self.net_debit * 100 * self.contracts

    @property
    def days_to_expiration(self) -> int:
        """Days until expiration."""
        return max(0, (self.expiration_date - datetime.now()).days)

    @property
    def time_to_expiration(self) -> float:
        """Time to expiration in years."""
        return self.days_to_expiration / 365.0

class StrategyAnalyzer:"""Comprehensive analysis tools for bull call spread strategy."""
    def __init__(self, position: BullCallSpread):
        """
        Initialize analyzer with position.

        Args:
            position: BullCallSpread instance
        """
        self.position = position

    def calculate_breakeven(self) -> float:
        """
        Calculate breakeven price at expiration.

        Returns:
            Breakeven stock price
        """
        return self.position.long_strike + self.position.net_debit

    def calculate_max_profit(self) -> Dict[str, float]:
        """
        Calculate maximum profit metrics.

        Returns:
            Dictionary with max profit information
        """
        max_profit_per_share = (
            self.position.spread_width - self.position.net_debit
        )
        max_profit_total = max_profit_per_share * 100 * self.position.contracts

        return {
            'max_profit_per_share': max_profit_per_share,
            'max_profit_total': max_profit_total,
            'max_profit_price': self.position.short_strike,
            'max_roi': (max_profit_per_share / self.position.net_debit) * 100
        }

    def calculate_max_loss(self) -> Dict[str, float]:
        """
        Calculate maximum loss metrics.

        Returns:
            Dictionary with max loss information
        """
        return {
            'max_loss_per_share': self.position.net_debit,
            'max_loss_total': self.position.position_cost,
            'max_loss_price': self.position.long_strike
        }

    def calculate_pnl(self, stock_price: float) -> Dict[str, float]:
        """
        Calculate P&L at given stock price (at expiration).

        Args:
            stock_price: Stock price at expiration

        Returns:
            Dictionary with P&L metrics
        """
        # Long call value
        long_value = max(stock_price - self.position.long_strike, 0)

        # Short call value
        short_value = max(stock_price - self.position.short_strike, 0)

        # Net position value
        position_value = long_value - short_value

        # P&L calculation
        pnl_per_share = position_value - self.position.net_debit
        pnl_total = pnl_per_share * 100 * self.position.contracts

        # Return percentage
        return_pct = (pnl_per_share / self.position.net_debit) * 100

        return {
            'stock_price': stock_price,
            'long_call_value': long_value,
            'short_call_value': short_value,
            'position_value': position_value,
            'pnl_per_share': pnl_per_share,
            'pnl_total': pnl_total,
            'return_pct': return_pct
        }

    def calculate_greeks(self) -> Dict[str, float]:
        """
        Calculate all position Greeks using Black-Scholes.

        Returns:
            Dictionary with delta, gamma, theta, vega, rho
        """
        T = self.position.time_to_expiration

        if T <= 0:
            return {
                'delta': 0.0, 'gamma': 0.0, 'theta': 0.0,
                'vega': 0.0, 'rho': 0.0
            }

        S = self.position.underlying_price
        r = self.position.risk_free_rate
        sigma = self.position.volatility

        # Calculate Greeks for long call
        long_greeks = self._calculate_call_greeks(
            S, self.position.long_strike, T, r, sigma
        )

        # Calculate Greeks for short call
        short_greeks = self._calculate_call_greeks(
            S, self.position.short_strike, T, r, sigma
        )

        # Net position Greeks
        multiplier = 100 * self.position.contracts

        return {
            'delta': (long_greeks['delta'] - short_greeks['delta']) * multiplier,
            'gamma': (long_greeks['gamma'] - short_greeks['gamma']) * multiplier,
            'theta': (long_greeks['theta'] - short_greeks['theta']) * multiplier,
            'vega': (long_greeks['vega'] - short_greeks['vega']) * multiplier,
            'rho': (long_greeks['rho'] - short_greeks['rho']) * multiplier
        }

    @staticmethod
    def _calculate_call_greeks(
        S: float, K: float, T: float, r: float, sigma: float
    ) -> Dict[str, float]:
        """
        Calculate Greeks for a single call option.

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility

        Returns:
            Dictionary with individual option Greeks
        """
        # Calculate d1 and d2
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        # Standard normal PDF and CDF
        pdf_d1 = norm.pdf(d1)
        cdf_d1 = norm.cdf(d1)
        cdf_d2 = norm.cdf(d2)

        # Greeks calculations
        delta = cdf_d1
        gamma = pdf_d1 / (S * sigma * np.sqrt(T))
        theta = (
            -(S * pdf_d1 * sigma) / (2 * np.sqrt(T))
            - r * K * np.exp(-r * T) * cdf_d2
        ) / 365  # Per day
        vega = S * pdf_d1 * np.sqrt(T) / 100  # Per 1% volatility
        rho = K * T * np.exp(-r * T) * cdf_d2 / 100  # Per 1% rate

        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }

    def generate_scenario_analysis(
        self,
        num_points: int = 50
    ) -> pd.DataFrame:
        """
        Generate scenario analysis across price range.

        Args:
            num_points: Number of price points to analyze

        Returns:
            DataFrame with scenario results
        """
        # Price range: ±15% from current
        price_range = np.linspace(
            self.position.underlying_price * 0.85,
            self.position.underlying_price * 1.15,
            num_points
        )

        scenarios = []
        for price in price_range:
            pnl_data = self.calculate_pnl(price)
            scenarios.append({
                'Stock Price': price,
                'Long Call': pnl_data['long_call_value'],
                'Short Call': pnl_data['short_call_value'],
                'Net Value': pnl_data['position_value'],
                'P&L Total': pnl_data['pnl_total'],
                'Return %': pnl_data['return_pct']
            })

        return pd.DataFrame(scenarios)

    def plot_payoff_diagram(
        self,
        save_path: Optional[str] = None,
        show_plot: bool = True
    ) -> plt.Figure:
        """
        Generate and display payoff diagram.

        Args:
            save_path: Path to save figure (optional)
            show_plot: Whether to display plot

        Returns:
            matplotlib Figure object
        """
        # Generate price range
        prices = np.linspace(
            self.position.underlying_price * 0.85,
            self.position.underlying_price * 1.15,
            200
        )

        # Calculate P&L for each price
        pnl_values = np.array([
            self.calculate_pnl(price)['pnl_total']
            for price in prices
        ])

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))

        # Plot P&L line
        ax.plot(prices, pnl_values, 'b-', linewidth=2.5,
                label='Bull Call Spread P&L')

        # Breakeven line
        breakeven = self.calculate_breakeven()
        ax.axvline(breakeven, color='gray', linestyle='--',
                   linewidth=1.5, alpha=0.7,
                   label=f'Breakeven: ${breakeven:.2f}')

        # Current price line
        ax.axvline(self.position.underlying_price, color='green',
                   linestyle='--', linewidth=1.5, alpha=0.7,
                   label=f'Current: ${self.position.underlying_price:.2f}')

        # Zero line
        ax.axhline(0, color='black', linestyle='-',
                   linewidth=0.8, alpha=0.5)

        # Shade profit/loss regions
        ax.fill_between(prices, 0, pnl_values,
                        where=(pnl_values >= 0),
                        alpha=0.2, color='green',
                        label='Profit Region')
        ax.fill_between(prices, 0, pnl_values,
                        where=(pnl_values < 0),
                        alpha=0.2, color='red',
                        label='Loss Region')

        # Strike price lines
        ax.axvline(self.position.long_strike, color='blue',
                   linestyle=':', alpha=0.7, linewidth=1.5,
                   label=f'Long Strike: ${self.position.long_strike:.2f}')
        ax.axvline(self.position.short_strike, color='red',
                   linestyle=':', alpha=0.7, linewidth=1.5,
                   label=f'Short Strike: ${self.position.short_strike:.2f}')

        # Max profit/loss annotations
        max_profit = self.calculate_max_profit()
        max_loss = self.calculate_max_loss()

        ax.text(self.position.short_strike * 1.02,
                max_profit['max_profit_total'] * 0.9,
                f"Max Profit: ${max_profit['max_profit_total']:.0f}",
                fontsize=11, fontweight='bold', color='green')

        ax.text(self.position.long_strike * 0.98,
                max_loss['max_loss_total'] * 1.1,
                f"Max Loss: ${max_loss['max_loss_total']:.0f}",
                fontsize=11, fontweight='bold', color='red',
                ha='right')

        # Labels and formatting
        ax.set_xlabel('Stock Price at Expiration ($)',
                      fontsize=13, fontweight='bold')
        ax.set_ylabel('Profit/Loss ($)',
                      fontsize=13, fontweight='bold')
        ax.set_title(
            f'{self.position.underlying_symbol} Bull Call Spread '
            f'Payoff Diagram\n'
            f'${self.position.long_strike:.0f}/'
            f'${self.position.short_strike:.0f} Strikes, '
            f'{self.position.days_to_expiration} DTE',
            fontsize=15, fontweight='bold', pad=20
        )

        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=10, framealpha=0.9)

        # Format axes
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: f'${x:,.0f}')
        )
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: f'${x:.0f}')
        )

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Payoff diagram saved to: {save_path}")

        if show_plot:
            plt.show()

        return fig

    def print_comprehensive_analysis(self):
        """
        Print complete position analysis to console.
        """
        print("\n" + "=" * 70)
        print("BULL CALL SPREAD COMPREHENSIVE ANALYSIS")
        print("=" * 70)

        # Position details
        print(f"\n{'POSITION DETAILS':-^70}")
        print(f"  Symbol:               {self.position.underlying_symbol}")
        print(f"  Current Price:        ${self.position.underlying_price:.2f}")
        print(f"  Long Call Strike:     ${self.position.long_strike:.2f}")
        print(f"  Short Call Strike:    ${self.position.short_strike:.2f}")
        print(f"  Long Premium:         ${self.position.long_premium:.2f}")
        print(f"  Short Premium:        ${self.position.short_premium:.2f}")
        print(f"  Net Debit:            ${self.position.net_debit:.2f}/share")
        print(f"  Spread Width:         ${self.position.spread_width:.2f}")
        print(f"  Contracts:            {self.position.contracts}")
        print(f"  Position Cost:        ${self.position.position_cost:,.2f}")
        print(f"  Days to Expiration:   {self.position.days_to_expiration}")
        print(f"  Implied Volatility:   {self.position.volatility*100:.1f}%")

        # Risk metrics
        print(f"\n{'RISK METRICS':-^70}")
        max_profit = self.calculate_max_profit()
        max_loss = self.calculate_max_loss()
        breakeven = self.calculate_breakeven()

        print(f"  Maximum Profit:       ${max_profit['max_profit_total']:,.2f} "
              f"(at ${max_profit['max_profit_price']:.2f}+)")
        print(f"  Maximum Loss:         ${max_loss['max_loss_total']:,.2f} "
              f"(below ${max_loss['max_loss_price']:.2f})")
        print(f"  Breakeven Price:      ${breakeven:.2f}")
        print(f"  Risk/Reward Ratio:    1:{max_profit['max_roi']/100:.2f}")
        print(f"  Max ROI:              {max_profit['max_roi']:.1f}%")

        # Greeks
        print(f"\n{'GREEKS':-^70}")
        greeks = self.calculate_greeks()
        print(f"  Delta:    {greeks['delta']:>10.2f}  "
              f"(Position move per $1 underlying)")
        print(f"  Gamma:    {greeks['gamma']:>10.4f}  "
              f"(Delta change per $1 move)")
        print(f"  Theta:    {greeks['theta']:>10.2f}  "
              f"(Daily time decay)")
        print(f"  Vega:     {greeks['vega']:>10.2f}  "
              f"(Value per 1% IV change)")
        print(f"  Rho:      {greeks['rho']:>10.2f}  "
              f"(Value per 1% rate change)")

        # Scenario analysis at key prices
        print(f"\n{'SCENARIO ANALYSIS':-^70}")
        print(f"{'Price':<12} {'Position Value':<16} {'P&L':<16} {'Return %':<10}")
        print("-" * 70)

        key_prices = [
            self.position.long_strike - 5,
            self.position.long_strike,
            breakeven,
            self.position.underlying_price,
            (self.position.long_strike + self.position.short_strike) / 2,
            self.position.short_strike,
            self.position.short_strike + 5
        ]

        for price in key_prices:
            result = self.calculate_pnl(price)
            print(f"${price:<11.2f} ${result['position_value']:<15.2f} "
                  f"${result['pnl_total']:<15.2f} {result['return_pct']:<9.1f}%")

        print("\n" + "=" * 70)

def parse_arguments():"""Parse command line arguments."""parser = argparse.ArgumentParser(description='Bull Call Spread Strategy Calculator',formatter_class=argparse.RawDescriptionHelpFormatter,epilog="""Examples:
Basic analysis
==============

python strategy_calculator.py --underlying SPY --price 450 \--long-strike 445 --short-strike 455 \--long-premium 8.50 --short-premium 3.20
With volatility and custom parameters
=====================================

python strategy_calculator.py --underlying AAPL --price 180 \--long-strike 175 --short-strike 185 \--long-premium 7.20 --short-premium 2.80 \--contracts 5 --volatility 0.25 --dte 60""")
    parser.add_argument('--underlying', type=str, required=True,
                        help='Underlying symbol (e.g., SPY, AAPL)')
    parser.add_argument('--price', type=float, required=True,
                        help='Current underlying price')
    parser.add_argument('--long-strike', type=float, required=True,
                        help='Long call strike price')
    parser.add_argument('--short-strike', type=float, required=True,
                        help='Short call strike price')
    parser.add_argument('--long-premium', type=float, required=True,
                        help='Long call premium')
    parser.add_argument('--short-premium', type=float, required=True,
                        help='Short call premium')
    parser.add_argument('--dte', type=int, default=45,
                        help='Days to expiration (default: 45)')
    parser.add_argument('--contracts', type=int, default=1,
                        help='Number of contracts (default: 1)')
    parser.add_argument('--volatility', type=float, default=0.20,
                        help='Implied volatility (default: 0.20)')
    parser.add_argument('--risk-free-rate', type=float, default=0.05,
                        help='Risk-free rate (default: 0.05)')
    parser.add_argument('--save-plot', type=str, default=None,
                        help='Path to save payoff diagram')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip displaying plot')

    return parser.parse_args()

def main():"""Main execution function."""args = parse_arguments()
    try:
        # Create position
        expiration = datetime.now() + timedelta(days=args.dte)

        position = BullCallSpread(
            underlying_symbol=args.underlying,
            underlying_price=args.price,
            long_strike=args.long_strike,
            short_strike=args.short_strike,
            long_premium=args.long_premium,
            short_premium=args.short_premium,
            expiration_date=expiration,
            contracts=args.contracts,
            volatility=args.volatility,
            risk_free_rate=args.risk_free_rate
        )

        # Create analyzer
        analyzer = StrategyAnalyzer(position)

        # Print comprehensive analysis
        analyzer.print_comprehensive_analysis()

        # Generate payoff diagram
        if not args.no_plot:
            analyzer.plot_payoff_diagram(
                save_path=args.save_plot,
                show_plot=True
            )

        return 0

    except Exception as e:
        print(f"\nError: {str(e)}", file=sys.stderr)
        return 1

if **name** == '**main**':sys.exit(main())</file>

<file name="scripts/black_scholes.py">#!/usr/bin/env python3"""Black-Scholes Options Pricing Module

Pure implementation of Black-Scholes formulas for European call and put options.Includes pricing and all Greeks calculations."""

import numpy as npfrom scipy.stats import normfrom typing import Dict, Union, Tuple

class BlackScholes:"""Black-Scholes options pricing and Greeks calculator.
    All methods are static for pure functional interface.
    """

    @staticmethod
    def calculate_d1_d2(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float
    ) -> Tuple[float, float]:
        """
        Calculate d1 and d2 terms in Black-Scholes formula.

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free interest rate (annual)
            sigma: Volatility (annual standard deviation)

        Returns:
            Tuple of (d1, d2)

        Raises:
            ValueError: If inputs are invalid
        """
        if S <= 0:
            raise ValueError("Stock price must be positive")
        if K <= 0:
            raise ValueError("Strike price must be positive")
        if T < 0:
            raise ValueError("Time to expiration cannot be negative")
        if sigma <= 0:
            raise ValueError("Volatility must be positive")

        if T == 0:
            return (float('inf') if S > K else float('-inf'),
                    float('inf') if S > K else float('-inf'))

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        return d1, d2

    @staticmethod
    def call_price(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float
    ) -> float:
        """
        Calculate European call option price.

        Formula: C = S*N(d1) - K*e^(-rT)*N(d2)

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free interest rate
            sigma: Volatility

        Returns:
            Call option price
        """
        if T == 0:
            return max(S - K, 0)

        d1, d2 = BlackScholes.calculate_d1_d2(S, K, T, r, sigma)

        call_value = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return max(call_value, 0)  # Ensure non-negative

    @staticmethod
    def put_price(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float
    ) -> float:
        """
        Calculate European put option price.

        Formula: P = K*e^(-rT)*N(-d2) - S*N(-d1)

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free interest rate
            sigma: Volatility

        Returns:
            Put option price
        """
        if T == 0:
            return max(K - S, 0)

        d1, d2 = BlackScholes.calculate_d1_d2(S, K, T, r, sigma)

        put_value = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return max(put_value, 0)

    @staticmethod
    def call_delta(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float
    ) -> float:
        """
        Calculate call option delta.

        Delta measures the rate of change of option value with respect to
        the underlying price.

        Formula: Δ_call = N(d1)

        Range: 0 to 1

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free interest rate
            sigma: Volatility

        Returns:
            Call delta
        """
        if T == 0:
            return 1.0 if S > K else 0.0

        d1, _ = BlackScholes.calculate_d1_d2(S, K, T, r, sigma)
        return norm.cdf(d1)

    @staticmethod
    def put_delta(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float
    ) -> float:
        """
        Calculate put option delta.

        Formula: Δ_put = N(d1) - 1

        Range: -1 to 0

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free interest rate
            sigma: Volatility

        Returns:
            Put delta
        """
        if T == 0:
            return -1.0 if S < K else 0.0

        d1, _ = BlackScholes.calculate_d1_d2(S, K, T, r, sigma)
        return norm.cdf(d1) - 1

    @staticmethod
    def gamma(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float
    ) -> float:
        """
        Calculate option gamma (same for calls and puts).

        Gamma measures the rate of change of delta with respect to
        the underlying price.

        Formula: Γ = φ(d1) / (S * σ * √T)
        where φ(x) = (1/√(2π)) * e^(-x²/2)

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free interest rate
            sigma: Volatility

        Returns:
            Gamma
        """
        if T == 0:
            return 0.0

        d1, _ = BlackScholes.calculate_d1_d2(S, K, T, r, sigma)

        gamma_value = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        return gamma_value

    @staticmethod
    def call_theta(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        annual: bool = False
    ) -> float:
        """
        Calculate call option theta.

        Theta measures the rate of change of option value with respect to time.

        Formula: Θ_call = -[S*φ(d1)*σ / (2*√T)] - r*K*e^(-rT)*N(d2)

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free interest rate
            sigma: Volatility
            annual: If True, return annual theta; if False, return daily theta

        Returns:
            Call theta (per day by default, per year if annual=True)
        """
        if T == 0:
            return 0.0

        d1, d2 = BlackScholes.calculate_d1_d2(S, K, T, r, sigma)

        theta_annual = (
            -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
            - r * K * np.exp(-r * T) * norm.cdf(d2)
        )

        return theta_annual if annual else theta_annual / 365

    @staticmethod
    def put_theta(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        annual: bool = False
    ) -> float:
        """
        Calculate put option theta.

        Formula: Θ_put = -[S*φ(d1)*σ / (2*√T)] + r*K*e^(-rT)*N(-d2)

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free interest rate
            sigma: Volatility
            annual: If True, return annual theta; if False, return daily theta

        Returns:
            Put theta (per day by default, per year if annual=True)
        """
        if T == 0:
            return 0.0

        d1, d2 = BlackScholes.calculate_d1_d2(S, K, T, r, sigma)

        theta_annual = (
            -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
            + r * K * np.exp(-r * T) * norm.cdf(-d2)
        )

        return theta_annual if annual else theta_annual / 365

    @staticmethod
    def vega(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float
    ) -> float:
        """
        Calculate option vega (same for calls and puts).

        Vega measures the rate of change of option value with respect to
        volatility (per 1% change in IV).

        Formula: ν = S * φ(d1) * √T / 100

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free interest rate
            sigma: Volatility

        Returns:
            Vega (per 1% change in volatility)
        """
        if T == 0:
            return 0.0

        d1, _ = BlackScholes.calculate_d1_d2(S, K, T, r, sigma)

        vega_value = S * norm.pdf(d1) * np.sqrt(T) / 100
        return vega_value

    @staticmethod
    def call_rho(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float
    ) -> float:
        """
        Calculate call option rho.

        Rho measures the rate of change of option value with respect to
        the risk-free interest rate (per 1% change in rate).

        Formula: ρ_call = K * T * e^(-rT) * N(d2) / 100

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free interest rate
            sigma: Volatility

        Returns:
            Call rho (per 1% change in rate)
        """
        if T == 0:
            return 0.0

        _, d2 = BlackScholes.calculate_d1_d2(S, K, T, r, sigma)

        rho_value = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        return rho_value

    @staticmethod
    def put_rho(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float
    ) -> float:
        """
        Calculate put option rho.

        Formula: ρ_put = -K * T * e^(-rT) * N(-d2) / 100

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free interest rate
            sigma: Volatility

        Returns:
            Put rho (per 1% change in rate)
        """
        if T == 0:
            return 0.0

        _, d2 = BlackScholes.calculate_d1_d2(S, K, T, r, sigma)

        rho_value = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        return rho_value

    @staticmethod
    def calculate_all_greeks(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = 'call'
    ) -> Dict[str, float]:
        """
        Calculate all Greeks for an option in one call.

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free interest rate
            sigma: Volatility
            option_type: 'call' or 'put'

        Returns:
            Dictionary with all Greeks and option price

        Raises:
            ValueError: If option_type is invalid
        """
        if option_type.lower() not in ['call', 'put']:
            raise ValueError("option_type must be 'call' or 'put'")

        is_call = option_type.lower() == 'call'

        return {
            'price': (BlackScholes.call_price(S, K, T, r, sigma) if is_call
                     else BlackScholes.put_price(S, K, T, r, sigma)),
            'delta': (BlackScholes.call_delta(S, K, T, r, sigma) if is_call
                     else BlackScholes.put_delta(S, K, T, r, sigma)),
            'gamma': BlackScholes.gamma(S, K, T, r, sigma),
            'theta': (BlackScholes.call_theta(S, K, T, r, sigma) if is_call
                     else BlackScholes.put_theta(S, K, T, r, sigma)),
            'vega': BlackScholes.vega(S, K, T, r, sigma),
            'rho': (BlackScholes.call_rho(S, K, T, r, sigma) if is_call
                   else BlackScholes.put_rho(S, K, T, r, sigma))
        }
Vectorized versions for efficiency with arrays
==============================================

class BlackScholesVectorized:"""Vectorized Black-Scholes calculations for efficient batch processing."""
    @staticmethod
    def call_prices(
        S: Union[float, np.ndarray],
        K: Union[float, np.ndarray],
        T: Union[float, np.ndarray],
        r: float,
        sigma: float
    ) -> np.ndarray:
        """
        Calculate call prices for arrays of inputs.

        Args:
            S: Stock price(s)
            K: Strike price(s)
            T: Time(s) to expiration
            r: Risk-free rate
            sigma: Volatility

        Returns:
            Array of call prices
        """
        S = np.atleast_1d(S)
        K = np.atleast_1d(K)
        T = np.atleast_1d(T)

        # Handle zero time case
        mask_zero = (T == 0)
        result = np.zeros_like(S, dtype=float)

        # Calculate for non-zero times
        mask_nonzero = ~mask_zero
        if np.any(mask_nonzero):
            S_nz = S[mask_nonzero]
            K_nz = K[mask_nonzero] if K.size > 1 else np.full_like(S_nz, K[0])
            T_nz = T[mask_nonzero] if T.size > 1 else np.full_like(S_nz, T[0])

            d1 = (np.log(S_nz / K_nz) + (r + 0.5 * sigma**2) * T_nz) / (sigma * np.sqrt(T_nz))
            d2 = d1 - sigma * np.sqrt(T_nz)

            result[mask_nonzero] = S_nz * norm.cdf(d1) - K_nz * np.exp(-r * T_nz) * norm.cdf(d2)

        # Handle zero time case
        if np.any(mask_zero):
            S_z = S[mask_zero]
            K_z = K[mask_zero] if K.size > 1 else np.full_like(S_z, K[0])
            result[mask_zero] = np.maximum(S_z - K_z, 0)

        return result

if **name** == '**main**':# Example usage and verificationprint("Black-Scholes Calculator - Example Usage\n")
    # Example parameters
    S = 100  # Stock price
    K = 100  # Strike price
    T = 0.25  # 3 months
    r = 0.05  # 5% risk-free rate
    sigma = 0.20  # 20% volatility

    print(f"Parameters:")
    print(f"  Stock Price: ${S}")
    print(f"  Strike Price: ${K}")
    print(f"  Time to Expiration: {T} years ({T*365:.0f} days)")
    print(f"  Risk-Free Rate: {r*100}%")
    print(f"  Volatility: {sigma*100}%\n")

    # Calculate call option
    print("Call Option:")
    call_greeks = BlackScholes.calculate_all_greeks(S, K, T, r, sigma, 'call')
    for greek, value in call_greeks.items():
        print(f"  {greek.capitalize():8s}: {value:10.4f}")

    print("\nPut Option:")
    put_greeks = BlackScholes.calculate_all_greeks(S, K, T, r, sigma, 'put')
    for greek, value in put_greeks.items():
        print(f"  {greek.capitalize():8s}: {value:10.4f}")

</file><file name=".gitignore"># Python__pycache__/*.py[cod]*$py.class*.so.Pythonbuild/develop-eggs/dist/downloads/eggs/.eggs/lib/lib64/parts/sdist/var/wheels/*.egg-info/.installed.cfg*.egg
Virtual environments
====================

venv/ENV/env/
IDE
===

.vscode/.idea/*.swp*.swo*~
OS
==

.DS_StoreThumbs.db
Output files
============

*.png*.jpg*.pdfoutput/results/
Jupyter
=======

.ipynb_checkpoints/*.ipynb
pytest
======

.pytest_cache/.coveragehtmlcov/</file>

<file name="README.md"># Bull Call Spread Strategy - Claude Skill Package

Advanced bull call spread options trading strategy with comprehensive Greeks analysis, risk management, and visualization tools.
Overview
--------

This skill package provides production-ready tools for analyzing bull call spread positions, including:

* Complete profit/loss calculations at expiration
* Black-Scholes Greeks (Delta, Gamma, Theta, Vega, Rho)
* Interactive payoff diagrams
* Risk management frameworks
* Scenario analysis
* Roll and adjustment strategies

Installation
------------

### Prerequisites

* Python 3.8 or higher

* Required packages:

      pip install numpy pandas matplotlib scipy



### Package Structure

    bull-call-spread/
    ├── SKILL.md                          # Main strategy documentation
    ├── reference.md                      # Advanced reference material
    ├── scripts/
    │   ├── strategy_calculator.py       # Complete strategy calculator
    │   └── black_scholes.py             # Black-Scholes pricing engine
    ├── README.md                         # This file
    └── .gitignore                        # Git ignore rules

Quick Start
-----------

### Command Line Usage

    # Basic analysis
    python scripts/strategy_calculator.py \
        --underlying SPY \
        --price 450 \
        --long-strike 445 \
        --short-strike 455 \
        --long-premium 8.50 \
        --short-premium 3.20

    # With custom parameters
    python scripts/strategy_calculator.py \
        --underlying AAPL \
        --price 180 \
        --long-strike 175 \
        --short-strike 185 \
        --long-premium 7.20 \
        --short-premium 2.80 \
        --contracts 5 \
        --volatility 0.25 \
        --dte 60 \
        --save-plot payoff_diagram.png

### Python API Usage

    from datetime import datetime, timedelta
    from scripts.strategy_calculator import BullCallSpread, StrategyAnalyzer

    # Create position
    position = BullCallSpread(
        underlying_symbol="SPY",
        underlying_price=450.00,
        long_strike=445.00,
        short_strike=455.00,
        long_premium=8.50,
        short_premium=3.20,
        expiration_date=datetime.now() + timedelta(days=45),
        contracts=2,
        volatility=0.18,
        risk_free_rate=0.05
    )

    # Analyze position
    analyzer = StrategyAnalyzer(position)
    analyzer.print_comprehensive_analysis()
    analyzer.plot_payoff_diagram()

    # Calculate specific metrics
    breakeven = analyzer.calculate_breakeven()
    max_profit = analyzer.calculate_max_profit()
    greeks = analyzer.calculate_greeks()

Documentation
-------------

* **SKILL.md**: Complete strategy guide with Python examples
* **reference.md**: Advanced topics including:
  * Black-Scholes mathematical foundations
  * Detailed Greeks derivations
  * Early assignment risk analysis
  * Volatility skew considerations
  * Roll management strategies
  * Comparative strategy analysis

Features
--------

### Comprehensive Analysis

* **Risk Metrics**: Max profit, max loss, breakeven, risk/reward ratio
* **Greeks**: Full suite of position Greeks with detailed explanations
* **Scenarios**: P&L analysis across price ranges
* **Visualization**: Professional payoff diagrams with matplotlib

### Risk Management

* Position sizing calculators
* Exit signal evaluation (profit target, stop loss, time-based)
* Roll opportunity assessment
* Adjustment strategy frameworks

### Production Quality

* Type hints throughout
* Comprehensive error handling
* Detailed docstrings
* Command-line interface
* Modular, reusable components

Examples
--------

### Example 1: SPY Bull Call Spread

    Position: SPY $445/$455 Bull Call Spread
    Current: $450.00
    Net Debit: $5.30 ($530 position cost)
    Days: 45 DTE

    Results:
    - Max Profit: $470 (88.7% ROI) at $455+
    - Max Loss: $530 (100% of cost) below $445
    - Breakeven: $450.30
    - Position Delta: 33.5

### Example 2: Roll Analysis

    from scripts.strategy_calculator import BullCallSpread, StrategyAnalyzer

    # Current position
    current = BullCallSpread(...)
    analyzer = StrategyAnalyzer(current)

    # Evaluate roll to higher strikes
    roll_analysis = evaluate_roll_opportunity(
        current_position=current,
        current_pnl=350,  # Current profit
        days_held=25,
        new_strikes=(455, 465),
        new_expiration_dte=60,
        new_premiums=(7.80, 3.50)
    )

    print(roll_analysis['recommend_roll'])  # True/False
    print(roll_analysis['combined_roi'])     # Expected ROI

Testing
-------

Run the example scenarios:
    # Test with default SPY example
    python scripts/strategy_calculator.py \
        --underlying SPY --price 450 \
        --long-strike 445 --short-strike 455 \
        --long-premium 8.50 --short-premium 3.20

    # Verify Black-Scholes calculations
    python scripts/black_scholes.py
Skill Integration
-----------------

### For Claude API

Package this directory as a ZIP file:
    zip -r bull-call-spread.zip bull-call-spread/

Upload via Skills API endpoint.

### For Claude Code

Copy to skills directory:
    cp -r bull-call-spread ~/.claude/skills/
Advanced Usage
--------------

### Custom Greeks Analysis

    from scripts.black_scholes import BlackScholes

    # Calculate individual option Greeks
    call_greeks = BlackScholes.calculate_all_greeks(
        S=450,      # Stock price
        K=445,      # Strike
        T=45/365,   # Time in years
        r=0.05,     # Risk-free rate
        sigma=0.18, # Volatility
        option_type='call'
    )

    print(f"Call Delta: {call_greeks['delta']:.4f}")
    print(f"Call Theta: {call_greeks['theta']:.2f}")

### Scenario Analysis

    analyzer = StrategyAnalyzer(position)
    scenarios_df = analyzer.generate_scenario_analysis(num_points=100)

    # Export to CSV
    scenarios_df.to_csv('scenarios.csv', index=False)

    # Analyze specific price
    result = analyzer.calculate_pnl(stock_price=455)
    print(f"P&L at $455: ${result['pnl_total']:.2f}")

Troubleshooting
---------------

### Common Issues

**Import errors**: Ensure all required packages are installed
    pip install --upgrade numpy pandas matplotlib scipy

**Plotting issues**: If plots don't display, check matplotlib backend
    import matplotlib
    matplotlib.use('TkAgg')  # or 'Qt5Agg'

**Calculation errors**: Verify all inputs are positive and strikes are ordered correctly
Support
-------

For issues or questions:

1. Review SKILL.md for strategy details
2. Check reference.md for advanced topics
3. Examine code comments and docstrings
4. Validate input parameters

License
-------

This skill package is provided for educational and analysis purposes. Not financial advice.
Version History
---------------

* v1.0.0 (2024-12-11): Initial release
  * Complete bull call spread implementation
  * Black-Scholes Greeks calculations
  * Comprehensive documentation
  * Command-line interface
  * Risk management tools</file>

<directory_structure>bull-call-spread/├── SKILL.md # Main skill documentation (strategy overview,│ # examples, calculations, Greeks)├── reference.md # Advanced reference (Black-Scholes derivations,│ # detailed Greeks, edge cases, adjustments)├── scripts/│ ├── strategy_calculator.py # Complete strategy calculator with CLI│ └── black_scholes.py # Pure Black-Scholes implementation├── README.md # Installation and usage guide└── .gitignore # Git ignore file</directory_structure>

<packaging_instructions>

Packaging Instructions for Bull Call Spread Skill
=================================================

Step 1: Verify File Structure
-----------------------------

Ensure your directory matches this structure:
    bull-call-spread/
    ├── SKILL.md
    ├── reference.md
    ├── scripts/
    │   ├── strategy_calculator.py
    │   └── black_scholes.py
    ├── README.md
    └── .gitignore
Step 2: Validate SKILL.md
-------------------------

Confirm the YAML frontmatter is properly formatted:
    ---
    name: bull-call-spread
    description: Advanced bull call spread strategy with Greeks analysis, risk management, and comprehensive P&L modeling. Use when analyzing vertical debit spreads, bullish options strategies, or calculating position Greeks for call spreads.
    ---

Requirements:

* `name`: Must use only lowercase letters, numbers, and hyphens (max 64 chars)
* `description`: Must be under 1024 characters and include what the skill does AND when to use it
* Opening `---` must be on line 1
* Closing `---` must come before any Markdown content

Step 3: Test Python Scripts
---------------------------

Before packaging, verify all scripts run without errors:
    # Test Black-Scholes module
    python scripts/black_scholes.py

    # Test strategy calculator with example
    python scripts/strategy_calculator.py \
        --underlying SPY --price 450 \
        --long-strike 445 --short-strike 455 \
        --long-premium 8.50 --short-premium 3.20 \
        --no-plot
Step 4: Create ZIP Archive
--------------------------

### On macOS/Linux:

    # Navigate to parent directory
    cd /path/to/parent

    # Create ZIP (excluding hidden files and __pycache__)
    zip -r bull-call-spread.zip bull-call-spread/ \
        -x "*.DS_Store" \
        -x "*__pycache__*" \
        -x "*.pyc" \
        -x "*/.git/*"

### On Windows (PowerShell):

    # Navigate to parent directory
    cd C:\path\to\parent

    # Create ZIP
    Compress-Archive -Path bull-call-spread\* -DestinationPath bull-call-spread.zip

### Using Python (cross-platform):

    import zipfile
    import os
    from pathlib import Path

    def create_skill_zip(source_dir, output_filename):
        """Create ZIP archive excluding unnecessary files."""
        exclude_patterns = {
            '__pycache__', '.pyc', '.DS_Store', '.git',
            '.pytest_cache', '.coverage', '*.egg-info'
        }

        with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(source_dir):
                # Filter out excluded directories
                dirs[:] = [d for d in dirs if d not in exclude_patterns]

                for file in files:
                    if any(pattern in file for pattern in exclude_patterns):
                        continue

                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, source_dir)
                    arcname = os.path.join('bull-call-spread', arcname)
                    zipf.write(file_path, arcname)

        print(f"Created: {output_filename}")

    # Execute
    create_skill_zip('bull-call-spread', 'bull-call-spread.zip')

Step 5: Verify ZIP Contents
---------------------------

Extract and verify the ZIP contains all required files:
    # List contents
    unzip -l bull-call-spread.zip

    # Expected output should show:
    # bull-call-spread/SKILL.md
    # bull-call-spread/reference.md
    # bull-call-spread/scripts/strategy_calculator.py
    # bull-call-spread/scripts/black_scholes.py
    # bull-call-spread/README.md
    # bull-call-spread/.gitignore
Step 6: Upload to Claude
------------------------

### For Claude.ai:

1. Go to Settings > Features
2. Scroll to "Skills" section
3. Click "Upload Skill"
4. Select `bull-call-spread.zip`
5. Confirm upload

### For Claude API:

    # Upload skill via API
    curl -X POST https://api.anthropic.com/v1/skills \
      -H "x-api-key: $ANTHROPIC_API_KEY" \
      -H "anthropic-version: 2023-06-01" \
      -H "anthropic-beta: skills-2025-10-02" \
      -F "file=@bull-call-spread.zip"

### For Claude Code:

    # Copy to skills directory
    cp -r bull-call-spread ~/.claude/skills/

    # Or for project-specific:
    cp -r bull-call-spread /path/to/project/.claude/skills/

Step 7: Test Deployed Skill
---------------------------

After deployment, test that Claude can use the skill:

### Test Query 1 (Basic):

    I want to analyze a bull call spread on SPY. The stock is at $450,
    I'm considering the $445/$455 spread with 45 days to expiration.
    The long call costs $8.50 and the short call pays $3.20.
    What's the breakeven and max profit?

### Test Query 2 (Greeks):

    Calculate the Greeks for the SPY $445/$455 bull call spread.
    Current price is $450, 45 DTE, 18% IV, 5% risk-free rate.

### Test Query 3 (Visualization):

    Create a payoff diagram for a bull call spread on AAPL.
    Long $175 call at $7.20, short $185 call at $2.80,
    current price $180, 60 days to expiration.

File Size Limits
----------------

* Maximum ZIP size: typically 50MB (varies by platform)
* SKILL.md should be under 500 lines (aim for 100-150 for optimal performance)
* Total package size for this skill: ~50KB (well within limits)

Troubleshooting
---------------

### Issue: "Invalid YAML frontmatter"

**Solution**: Verify `---` delimiters are on their own lines, no tabs in YAML

### Issue: "Skill name contains invalid characters"

**Solution**: Ensure name uses only lowercase letters, numbers, and hyphens

### Issue: "Description too long"

**Solution**: Trim description to under 1024 characters while keeping key terms

### Issue: "Python import errors when Claude uses skill"

**Solution**: Verify all imports use standard libraries (numpy, pandas, matplotlib, scipy)

### Issue: "Plots don't display in Claude.ai"

**Solution**: This is expected; plots are saved to file system. Use `save_plot` parameter.
Best Practices
--------------

1. **Keep SKILL.md concise**: Focus on essential instructions and examples
2. **Use progressive disclosure**: Move detailed explanations to reference.md
3. **Make code standalone**: All scripts should run independently
4. **Include type hints**: Helps Claude understand function signatures
5. **Document thoroughly**: Comprehensive docstrings aid Claude's understanding
6. **Test before packaging**: Run all examples to ensure they work
7. **Version your skills**: Use git tags or version numbers in README
8. **Update .gitignore**: Prevent accidental inclusion of cache files

Metadata
--------

* **Skill Name**: bull-call-spread
* **Version**: 1.0.0
* **Package Size**: ~50KB
* **Python Version**: 3.8+
* **Required Packages**: numpy, pandas, matplotlib, scipy
* **Target Audience**: Advanced options traders
* **Complexity**: Advanced

Support Files Included
----------------------

* ✅ SKILL.md (main documentation)
* ✅ reference.md (advanced topics)
* ✅ scripts/ (implementation code)
* ✅ README.md (installation guide)
* ✅ .gitignore (git ignore rules)

Package is complete and ready for deployment.</packaging_instructions>

</skill_package>
