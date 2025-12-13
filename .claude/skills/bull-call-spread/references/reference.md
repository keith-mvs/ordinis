# Bull Call Spread - Advanced Reference
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
