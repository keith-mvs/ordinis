---
name: options-strategies
description: Expert-level options trading framework for multi-leg strategy design, quantitative modeling, and programmatic execution using Alpaca Markets APIs. Covers straddles, butterflies, spreads, iron condors, and calendar strategies with complete payoff analysis, risk management, and automated execution capabilities.
---

# Expert Options Trading Strategies

Systematic framework for designing, modeling, and executing advanced options strategies using Alpaca Markets infrastructure, with institutional-grade quantitative analysis and risk management.

## Authoritative Sources

**Primary Alpaca Documentation**
- [Alpaca API Documentation](https://docs.alpaca.markets/)
- [Getting Started Guide](https://docs.alpaca.markets/docs/getting-started)
- [Trading API Reference](https://docs.alpaca.markets/reference/)
- [Issue Tokens - OAuth 2.0 Authentication](https://docs.alpaca.markets/reference/oauth)

**Alpaca Learn Educational Resources**
- [How to Trade Options with Alpaca](https://alpaca.markets/learn/options/)
- [Long Straddle Strategy](https://alpaca.markets/learn/long-straddle-options-strategy/)
- [Iron Butterfly Strategy](https://alpaca.markets/learn/iron-butterfly-options-strategy/)
- [Calendar Spread Strategy](https://alpaca.markets/learn/calendar-spread-options-strategy/)

**Alpaca Official GitHub**
- [https://github.com/alpacahq](https://github.com/alpacahq) - Official open-source organization
- [alpaca-trade-api-python](https://github.com/alpacahq/alpaca-trade-api-python) - Python SDK
- [alpaca-trade-api-js](https://github.com/alpacahq/alpaca-trade-api-js) - Node.js client
- [alpaca-labs](https://github.com/alpacahq/alpaca-labs) - Reference implementations
- [examples](https://github.com/alpacahq/examples) - Strategy templates and sample scripts

**Industry References**
- [Options Industry Council (OIC)](https://www.optionseducation.org/) - Educational resources
- [CBOE Strategy Guides](https://www.cboe.com/education/tools/) - Payoff diagrams and analysis
- Python 3.11.9 Documentation
- Libraries: numpy, pandas, matplotlib, quantlib, yfinance, alpaca-trade-api

## Core Workflow

### Phase 1: Foundational Concepts and Environment Setup

**Establish technical and conceptual foundation:**

1. **Options Fundamentals**
   - Call/put mechanics, strike selection, premiums, expiration cycles
   - Intrinsic value vs. time value decomposition
   - Option chain structure and contract specifications
   - Implied volatility vs. historical volatility analysis

2. **The Greeks - Risk Derivatives**
   - **Delta (Δ)**: Price sensitivity to underlying movement (directional exposure)
   - **Gamma (Γ)**: Rate of delta change (acceleration risk)
   - **Theta (Θ)**: Time decay impact (time value erosion)
   - **Vega (ν)**: Volatility sensitivity (IV expansion/compression)
   - **Rho (ρ)**: Interest rate sensitivity (typically minor for short-dated options)

3. **Alpaca API Environment Setup**
   - Create Alpaca account (paper trading recommended for testing)
   - Generate API keys: Key ID and Secret Key
   - Configure OAuth 2.0 authentication using Issue Tokens endpoint
   - Install alpaca-trade-api: `pip install alpaca-trade-api`
   - Verify connectivity and account access

4. **Development Environment**
   - Python 3.11+ with virtual environment
   - Install dependencies: `pip install numpy pandas matplotlib alpaca-trade-api yfinance`
   - Configure IDE with Alpaca API credentials (use environment variables)
   - Set up version control for strategy code

**Expected Competency:**
Deploy authenticated Alpaca API environment, interpret options data structures, understand core pricing mechanisms and Greeks.

### Phase 2: Core Volatility Strategies - Straddles and Strangles

**Master neutral volatility strategies:**

1. **Long Straddle - ATM Volatility Expansion Play**
   - **Structure**: Buy ATM call + Buy ATM put (same strike, same expiration)
   - **Market View**: Expect large price movement but uncertain direction
   - **Maximum Profit**: Unlimited (theoretically)
   - **Maximum Loss**: Total premium paid (both legs)
   - **Breakeven Points**: Strike ± Total Premium
   - **Best Conditions**: Pre-earnings, major announcements, event-driven volatility
   
   **Reference**: [Alpaca Learn - Long Straddle](https://alpaca.markets/learn/long-straddle-options-strategy/)
   
   **Payoff Formula**:
   ```
   Profit/Loss = max(S - K, 0) + max(K - S, 0) - (C_premium + P_premium)
   where S = stock price, K = strike price
   ```

2. **Long Strangle - OTM Volatility Strategy**
   - **Structure**: Buy OTM call + Buy OTM put (different strikes, same expiration)
   - **Market View**: Expect significant movement, cheaper entry than straddle
   - **Maximum Profit**: Unlimited (theoretically)
   - **Maximum Loss**: Total premium paid (both legs)
   - **Breakeven Points**: Call Strike + Total Premium, Put Strike - Total Premium
   - **Advantage**: Lower cost, wider profit zone
   - **Disadvantage**: Requires larger underlying movement to profit

3. **Implementation with Alpaca API**
   ```python
   from alpaca_trade_api import REST
   
   # Initialize API
   api = REST(key_id, secret_key, base_url='https://paper-api.alpaca.markets')
   
   # Submit multi-leg straddle order
   api.submit_order(
       symbol='SPY',
       qty=1,
       side='buy',
       type='limit',
       time_in_force='day',
       order_class='oto',  # One-triggers-other
       legs=[
           {'symbol': 'SPY250117C00450000', 'qty': 1, 'side': 'buy'},  # Call
           {'symbol': 'SPY250117P00450000', 'qty': 1, 'side': 'buy'}   # Put
       ]
   )
   ```

4. **Volatility Analysis and Trade Selection**
   - Historical volatility calculation (20-day, 30-day)
   - Implied volatility percentile/rank analysis
   - IV expansion/compression cycles
   - Earnings calendar and event-driven volatility forecasting
   - Expected move calculation: `Expected Move = Stock Price × IV × √(Days to Expiration / 365)`

**Expected Competency:**
Construct and visualize straddles/strangles, automate entry logic based on volatility thresholds, understand when each strategy is optimal.

### Phase 3: Intermediate Multi-Leg Strategies - Butterflies and Spreads

**Develop defined-risk, defined-reward structures:**

1. **Iron Butterfly - Volatility Compression Strategy**
   - **Structure**: 
     - Sell ATM call + Sell ATM put (short straddle at middle strike)
     - Buy OTM call + Buy OTM put (long strangle wings for protection)
   - **Market View**: Expect minimal price movement, volatility contraction
   - **Maximum Profit**: Net premium received (occurs at middle strike at expiration)
   - **Maximum Loss**: (Wing Width - Net Premium) × 100
   - **Breakeven Points**: Middle Strike ± Net Premium
   - **Risk/Reward**: Limited on both sides
   
   **Reference**: [Alpaca Learn - Iron Butterfly](https://alpaca.markets/learn/iron-butterfly-options-strategy/)
   
   **Example Structure**:
   ```
   Stock at $100:
   - Buy 1 Put @ $95 strike
   - Sell 1 Put @ $100 strike
   - Sell 1 Call @ $100 strike
   - Buy 1 Call @ $105 strike
   
   Max Profit: Net credit received (e.g., $2.50 × 100 = $250)
   Max Loss: $5 - $2.50 = $2.50 × 100 = $250
   ```

2. **Calendar Spread - Time Decay Arbitrage**
   - **Structure**: Sell near-term option + Buy longer-term option (same strike)
   - **Market View**: Expect minimal movement short-term, volatility in long-term
   - **Profit Mechanism**: Near-term theta decay faster than long-term
   - **Maximum Profit**: Occurs when near-term expires at strike price
   - **Risk**: Underlying moves significantly away from strike
   
   **Reference**: [Alpaca Learn - Calendar Spread](https://alpaca.markets/learn/calendar-spread-options-strategy/)
   
   **Time Decay Comparison**:
   ```
   30-day option theta: -$0.05/day
   60-day option theta: -$0.03/day
   Net theta capture: $0.02/day
   ```

3. **Vertical Spreads - Directional Defined-Risk Plays**
   - **Bull Call Spread**: Buy lower strike call + Sell higher strike call
   - **Bear Put Spread**: Buy higher strike put + Sell lower strike put
   - **Advantage**: Reduced cost, defined risk, lower margin requirements
   - **Max Profit**: Strike difference - Net debit paid
   - **Max Loss**: Net debit paid

4. **Iron Condor - Range-Bound Strategy**
   - **Structure**: OTM bull put spread + OTM bear call spread
   - **Market View**: Expect price to stay within range
   - **Maximum Profit**: Net premium received
   - **Maximum Loss**: (Wing Width - Net Premium) × 100
   - **Typical Setup**: Wide wings (10-15 delta options), symmetric strikes

**Expected Competency:**
Model, evaluate, and execute 3-4 leg strategies, understand profit zones, calculate breakeven points, and assess sensitivity to time decay and volatility changes.

### Phase 4: Quantitative Modeling and Payoff Simulation

**Build analytical models for strategy evaluation:**

1. **Black-Scholes Option Pricing**
   ```python
   from scipy.stats import norm
   import numpy as np
   
   def black_scholes_call(S, K, T, r, sigma):
       """Calculate theoretical call option price."""
       d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
       d2 = d1 - sigma*np.sqrt(T)
       call_price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
       return call_price
   
   def black_scholes_put(S, K, T, r, sigma):
       """Calculate theoretical put option price."""
       d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
       d2 = d1 - sigma*np.sqrt(T)
       put_price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
       return put_price
   ```

2. **Greeks Calculation**
   ```python
   def calculate_delta_call(S, K, T, r, sigma):
       """Delta for call option."""
       d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
       return norm.cdf(d1)
   
   def calculate_gamma(S, K, T, r, sigma):
       """Gamma (same for calls and puts)."""
       d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
       return norm.pdf(d1) / (S * sigma * np.sqrt(T))
   
   def calculate_theta_call(S, K, T, r, sigma):
       """Theta for call option (daily decay)."""
       d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
       d2 = d1 - sigma*np.sqrt(T)
       theta = (-S*norm.pdf(d1)*sigma/(2*np.sqrt(T)) 
                - r*K*np.exp(-r*T)*norm.cdf(d2)) / 365
       return theta
   
   def calculate_vega(S, K, T, r, sigma):
       """Vega (sensitivity to 1% change in IV)."""
       d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
       return S * norm.pdf(d1) * np.sqrt(T) / 100
   ```

3. **Payoff Diagram Generation**
   ```python
   import matplotlib.pyplot as plt
   
   def plot_straddle_payoff(strike, call_premium, put_premium, 
                           underlying_range=None):
       """Generate straddle payoff diagram."""
       if underlying_range is None:
           underlying_range = np.linspace(strike*0.8, strike*1.2, 100)
       
       total_premium = call_premium + put_premium
       
       # Calculate payoff at each price point
       payoffs = []
       for S in underlying_range:
           call_value = max(S - strike, 0)
           put_value = max(strike - S, 0)
           total_value = call_value + put_value - total_premium
           payoffs.append(total_value)
       
       # Plot
       plt.figure(figsize=(10, 6))
       plt.plot(underlying_range, payoffs, linewidth=2)
       plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
       plt.axvline(x=strike, color='r', linestyle='--', alpha=0.3, 
                   label=f'Strike: ${strike}')
       plt.xlabel('Underlying Price at Expiration')
       plt.ylabel('Profit/Loss')
       plt.title('Long Straddle Payoff Diagram')
       plt.grid(True, alpha=0.3)
       plt.legend()
       plt.show()
   ```

4. **Monte Carlo Simulation for Expected Return**
   ```python
   def monte_carlo_option_simulation(S0, K, T, r, sigma, strategy_func, 
                                    n_simulations=10000):
       """
       Simulate strategy performance across random price paths.
       
       strategy_func: Function that takes final stock price and returns P/L
       """
       np.random.seed(42)
       dt = T / 252  # Daily steps
       price_paths = []
       
       for _ in range(n_simulations):
           prices = [S0]
           for _ in range(int(T * 252)):
               drift = (r - 0.5 * sigma**2) * dt
               diffusion = sigma * np.sqrt(dt) * np.random.normal()
               prices.append(prices[-1] * np.exp(drift + diffusion))
           price_paths.append(prices[-1])
       
       # Calculate strategy P/L for each path
       pnls = [strategy_func(price) for price in price_paths]
       
       return {
           'mean_pnl': np.mean(pnls),
           'median_pnl': np.median(pnls),
           'std_pnl': np.std(pnls),
           'prob_profit': sum(1 for pnl in pnls if pnl > 0) / n_simulations,
           'max_profit': max(pnls),
           'max_loss': min(pnls),
           'percentile_5': np.percentile(pnls, 5),
           'percentile_95': np.percentile(pnls, 95)
       }
   ```

5. **3D Surface Plots - Profit vs. Price and Volatility**
   ```python
   from mpl_toolkits.mplot3d import Axes3D
   
   def plot_strategy_surface(strike, premium, days_to_expiry=30):
       """3D surface showing P/L across price and IV changes."""
       prices = np.linspace(strike*0.7, strike*1.3, 50)
       ivs = np.linspace(0.1, 0.8, 50)
       
       X, Y = np.meshgrid(prices, ivs)
       Z = np.zeros_like(X)
       
       for i in range(len(ivs)):
           for j in range(len(prices)):
               # Calculate strategy value at this price/IV combination
               Z[i, j] = calculate_strategy_value(prices[j], ivs[i], 
                                                   days_to_expiry)
       
       fig = plt.figure(figsize=(12, 8))
       ax = fig.add_subplot(111, projection='3d')
       surf = ax.plot_surface(X, Y, Z, cmap='RdYlGn', alpha=0.8)
       ax.set_xlabel('Underlying Price')
       ax.set_ylabel('Implied Volatility')
       ax.set_zlabel('Profit/Loss')
       plt.colorbar(surf)
       plt.title('Strategy P/L Surface')
       plt.show()
   ```

**Expected Competency:**
Create analytical models for options pricing, simulate performance across scenarios, visualize multi-dimensional payoffs, and quantify risk/reward metrics.

### Phase 5: Execution and Risk Management

**Deploy and manage strategies with institutional controls:**

1. **Multi-Leg Order Submission**
   ```python
   def submit_iron_butterfly(api, symbol, expiry, middle_strike, 
                            wing_width, contracts=1):
       """
       Submit iron butterfly order via Alpaca API.
       
       Args:
           api: Alpaca REST API instance
           symbol: Underlying symbol (e.g., 'SPY')
           expiry: Expiration date 'YYMMDD'
           middle_strike: ATM strike price (e.g., 450.00)
           wing_width: Distance to wings (e.g., 5.00)
           contracts: Number of contracts
       """
       # Construct option symbols (OCC format)
       put_wing = f"{symbol}{expiry}P{int((middle_strike-wing_width)*1000):08d}"
       put_middle = f"{symbol}{expiry}P{int(middle_strike*1000):08d}"
       call_middle = f"{symbol}{expiry}C{int(middle_strike*1000):08d}"
       call_wing = f"{symbol}{expiry}C{int((middle_strike+wing_width)*1000):08d}"
       
       # Submit as bracket order
       order = api.submit_order(
           symbol=symbol,
           qty=contracts,
           side='buy',
           type='market',
           time_in_force='day',
           order_class='bracket',
           legs=[
               {'symbol': put_wing, 'qty': contracts, 'side': 'buy'},
               {'symbol': put_middle, 'qty': contracts, 'side': 'sell'},
               {'symbol': call_middle, 'qty': contracts, 'side': 'sell'},
               {'symbol': call_wing, 'qty': contracts, 'side': 'buy'}
           ]
       )
       
       return order
   ```

2. **Position Monitoring and Greeks Aggregation**
   ```python
   def monitor_portfolio_greeks(api):
       """Calculate aggregate Greeks for all open positions."""
       positions = api.list_positions()
       
       total_delta = 0
       total_gamma = 0
       total_theta = 0
       total_vega = 0
       
       for position in positions:
           if 'option' in position.asset_class.lower():
               # Get option details
               qty = float(position.qty)
               
               # Calculate Greeks (would fetch from market data in production)
               greeks = get_option_greeks(position.symbol)
               
               # Aggregate (multiply by position size)
               total_delta += greeks['delta'] * qty * 100
               total_gamma += greeks['gamma'] * qty * 100
               total_theta += greeks['theta'] * qty * 100
               total_vega += greeks['vega'] * qty * 100
       
       return {
           'portfolio_delta': total_delta,
           'portfolio_gamma': total_gamma,
           'portfolio_theta': total_theta,
           'portfolio_vega': total_vega
       }
   ```

3. **Automated Risk Controls**
   ```python
   class RiskManager:
       """Automated risk management for options strategies."""
       
       def __init__(self, max_portfolio_delta=100, max_position_loss=1000):
           self.max_portfolio_delta = max_portfolio_delta
           self.max_position_loss = max_position_loss
       
       def check_position_limits(self, position, current_pnl):
           """Check if position violates risk limits."""
           violations = []
           
           # Loss limit check
           if current_pnl < -self.max_position_loss:
               violations.append({
                   'type': 'MAX_LOSS',
                   'message': f'Position loss ${current_pnl} exceeds limit',
                   'action': 'CLOSE_POSITION'
               })
           
           return violations
       
       def check_portfolio_greeks(self, greeks):
           """Check if portfolio Greeks within limits."""
           violations = []
           
           # Delta neutrality check
           if abs(greeks['portfolio_delta']) > self.max_portfolio_delta:
               violations.append({
                   'type': 'DELTA_LIMIT',
                   'message': f'Portfolio delta {greeks["portfolio_delta"]} '
                             f'exceeds limit',
                   'action': 'HEDGE_DELTA'
               })
           
           return violations
       
       def adjust_delta_neutral(self, api, current_delta, underlying_symbol):
           """Add hedge to bring portfolio delta to neutral."""
           hedge_shares = -int(current_delta / 100)  # Convert to shares
           
           if abs(hedge_shares) > 0:
               api.submit_order(
                   symbol=underlying_symbol,
                   qty=abs(hedge_shares),
                   side='buy' if hedge_shares > 0 else 'sell',
                   type='market',
                   time_in_force='day'
               )
               
               return hedge_shares
           return 0
   ```

4. **Stop-Loss and Take-Profit Automation**
   ```python
   def set_bracket_orders(api, strategy_positions, stop_loss_pct=0.5, 
                         take_profit_pct=0.8):
       """Set automated exit orders for strategy positions."""
       for position in strategy_positions:
           entry_price = float(position.avg_entry_price)
           
           # Calculate exit prices
           stop_price = entry_price * (1 - stop_loss_pct)
           profit_price = entry_price * (1 + take_profit_pct)
           
           # Submit bracket order
           api.submit_order(
               symbol=position.symbol,
               qty=abs(int(position.qty)),
               side='sell' if int(position.qty) > 0 else 'buy',
               type='bracket',
               time_in_force='gtc',
               stop_loss={'stop_price': stop_price},
               take_profit={'limit_price': profit_price}
           )
   ```

5. **Daily Performance Tracking**
   ```python
   import pandas as pd
   from datetime import datetime
   
   def log_daily_performance(api, log_file='performance_log.csv'):
       """Track daily P/L and Greeks."""
       account = api.get_account()
       positions = api.list_positions()
       greeks = monitor_portfolio_greeks(api)
       
       log_entry = {
           'date': datetime.now().strftime('%Y-%m-%d'),
           'portfolio_value': float(account.portfolio_value),
           'cash': float(account.cash),
           'equity': float(account.equity),
           'total_pnl': float(account.equity) - float(account.last_equity),
           'num_positions': len(positions),
           'delta': greeks['portfolio_delta'],
           'gamma': greeks['portfolio_gamma'],
           'theta': greeks['portfolio_theta'],
           'vega': greeks['portfolio_vega']
       }
       
       # Append to CSV
       df = pd.DataFrame([log_entry])
       df.to_csv(log_file, mode='a', header=not os.path.exists(log_file), 
                index=False)
   ```

**Expected Competency:**
Execute multi-leg orders, monitor positions in real-time, implement automated risk controls, and maintain delta neutrality with programmatic hedging.

### Phase 6: Automation, Backtesting, and Optimization

**Build systematic research and deployment pipelines:**

1. **Historical Data Acquisition**
   ```python
   def fetch_historical_options_data(symbol, start_date, end_date):
       """Fetch historical options chain data for backtesting."""
       import yfinance as yf
       
       ticker = yf.Ticker(symbol)
       
       # Get all available expiration dates
       expirations = ticker.options
       
       historical_chains = {}
       for exp_date in expirations:
           chain = ticker.option_chain(exp_date)
           historical_chains[exp_date] = {
               'calls': chain.calls,
               'puts': chain.puts
           }
       
       return historical_chains
   ```

2. **Backtesting Framework**
   ```python
   class StrategyBacktester:
       """Backtest options strategies on historical data."""
       
       def __init__(self, initial_capital=100000):
           self.initial_capital = initial_capital
           self.trades = []
           self.equity_curve = []
       
       def run_backtest(self, strategy_func, data, parameters):
           """
           Execute backtest.
           
           Args:
               strategy_func: Function that generates trade signals
               data: Historical price/options data
               parameters: Strategy parameters dict
           """
           capital = self.initial_capital
           positions = []
           
           for date, market_data in data.items():
               # Generate signals
               signals = strategy_func(market_data, parameters)
               
               # Execute trades
               for signal in signals:
                   if signal['action'] == 'OPEN':
                       positions.append(self._open_position(signal))
                       capital -= signal['cost']
                   elif signal['action'] == 'CLOSE':
                       pnl = self._close_position(signal, positions)
                       capital += pnl
                       self.trades.append({
                           'date': date,
                           'pnl': pnl,
                           'type': signal['type']
                       })
               
               # Update equity curve
               total_value = capital + sum(p['value'] for p in positions)
               self.equity_curve.append({
                   'date': date,
                   'equity': total_value,
                   'return': (total_value / self.initial_capital - 1) * 100
               })
           
           return self._calculate_metrics()
       
       def _calculate_metrics(self):
           """Calculate performance metrics."""
           df = pd.DataFrame(self.trades)
           
           return {
               'total_trades': len(self.trades),
               'win_rate': len(df[df['pnl'] > 0]) / len(df) if len(df) > 0 else 0,
               'avg_win': df[df['pnl'] > 0]['pnl'].mean() if len(df) > 0 else 0,
               'avg_loss': df[df['pnl'] < 0]['pnl'].mean() if len(df) > 0 else 0,
               'profit_factor': abs(df[df['pnl'] > 0]['pnl'].sum() / 
                                   df[df['pnl'] < 0]['pnl'].sum()) 
                               if len(df[df['pnl'] < 0]) > 0 else 0,
               'max_drawdown': self._calculate_max_drawdown(),
               'sharpe_ratio': self._calculate_sharpe_ratio()
           }
       
       def _calculate_max_drawdown(self):
           """Calculate maximum drawdown."""
           equity = pd.Series([e['equity'] for e in self.equity_curve])
           cummax = equity.cummax()
           drawdown = (equity - cummax) / cummax
           return drawdown.min() * 100
       
       def _calculate_sharpe_ratio(self, risk_free_rate=0.04):
           """Calculate annualized Sharpe ratio."""
           returns = pd.Series([e['return'] for e in self.equity_curve]).pct_change()
           excess_returns = returns - (risk_free_rate / 252)
           return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
   ```

3. **Parameter Optimization**
   ```python
   from itertools import product
   
   def optimize_strategy_parameters(strategy_func, data, param_grid):
       """Grid search for optimal strategy parameters."""
       results = []
       
       # Generate all parameter combinations
       param_combinations = list(product(*param_grid.values()))
       param_names = list(param_grid.keys())
       
       for params in param_combinations:
           param_dict = dict(zip(param_names, params))
           
           # Run backtest
           backtester = StrategyBacktester()
           metrics = backtester.run_backtest(strategy_func, data, param_dict)
           
           results.append({
               'parameters': param_dict,
               'sharpe': metrics['sharpe_ratio'],
               'win_rate': metrics['win_rate'],
               'profit_factor': metrics['profit_factor'],
               'max_drawdown': metrics['max_drawdown']
           })
       
       # Sort by Sharpe ratio
       results_df = pd.DataFrame(results)
       return results_df.sort_values('sharpe', ascending=False)
   ```

4. **Automated Strategy Deployment**
   ```python
   class AutomatedStrategyRunner:
       """Automated execution of options strategies."""
       
       def __init__(self, api, strategy_config):
           self.api = api
           self.config = strategy_config
           self.active_strategies = []
       
       def run_daily_scan(self):
           """Scan for strategy entry opportunities."""
           opportunities = []
           
           for symbol in self.config['watchlist']:
               # Fetch market data
               market_data = self._fetch_market_data(symbol)
               
               # Check entry conditions for each strategy type
               for strategy_type in self.config['strategies']:
                   if self._check_entry_conditions(strategy_type, market_data):
                       opportunities.append({
                           'symbol': symbol,
                           'strategy': strategy_type,
                           'market_data': market_data
                       })
           
           return opportunities
       
       def execute_strategies(self, opportunities):
           """Execute identified strategy opportunities."""
           for opp in opportunities:
               try:
                   # Build strategy order
                   order = self._build_strategy_order(
                       opp['strategy'], 
                       opp['symbol'],
                       opp['market_data']
                   )
                   
                   # Submit to Alpaca
                   submitted = self.api.submit_order(**order)
                   
                   # Track active strategy
                   self.active_strategies.append({
                       'order_id': submitted.id,
                       'strategy': opp['strategy'],
                       'entry_date': datetime.now(),
                       'symbol': opp['symbol']
                   })
                   
               except Exception as e:
                   print(f"Error executing {opp['strategy']} on "
                         f"{opp['symbol']}: {e}")
       
       def manage_positions(self):
           """Monitor and adjust active positions."""
           for strategy in self.active_strategies:
               # Check exit conditions
               if self._check_exit_conditions(strategy):
                   self._close_strategy(strategy)
               
               # Check adjustment conditions (e.g., delta hedging)
               elif self._check_adjustment_conditions(strategy):
                   self._adjust_strategy(strategy)
   ```

**Expected Competency:**
Automate strategy research, backtest performance systematically, optimize parameters, and deploy strategies with monitoring and risk controls.

### Phase 7: Governance, Compliance, and Documentation

**Maintain regulatory and operational excellence:**

1. **Alpaca API Compliance**
   - Adhere to [Alpaca Terms of Service](https://alpaca.markets/legal/terms-of-service)
   - Respect API rate limits (200 requests/minute for trading, 10,000/minute for data)
   - Implement exponential backoff for rate limit handling
   - Use appropriate authentication scopes for different operations
   - Label all paper trading clearly in logs and documentation

2. **Security Best Practices**
   ```python
   # Store credentials in environment variables, never in code
   import os
   from dotenv import load_dotenv
   
   load_dotenv()
   
   API_KEY_ID = os.getenv('ALPACA_API_KEY')
   API_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
   BASE_URL = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
   
   # Validate credentials are present
   if not API_KEY_ID or not API_SECRET_KEY:
       raise ValueError("Alpaca API credentials not found in environment")
   ```

3. **Trade Documentation Standards**
   ```python
   def document_trade(strategy_type, entry_data, exit_data=None):
       """Create complete trade record for audit trail."""
       trade_record = {
           'trade_id': str(uuid.uuid4()),
           'strategy_type': strategy_type,
           'entry_date': entry_data['timestamp'],
           'entry_price': entry_data['price'],
           'entry_greeks': entry_data['greeks'],
           'entry_iv': entry_data['implied_volatility'],
           'underlying_price': entry_data['underlying_price'],
           'position_size': entry_data['contracts'],
           'max_risk': entry_data['max_risk'],
           'expected_profit': entry_data['expected_profit'],
           'rationale': entry_data['rationale']
       }
       
       if exit_data:
           trade_record.update({
               'exit_date': exit_data['timestamp'],
               'exit_price': exit_data['price'],
               'realized_pnl': exit_data['pnl'],
               'hold_period': exit_data['hold_days'],
               'exit_reason': exit_data['reason']
           })
       
       # Save to database or CSV
       save_trade_record(trade_record)
       
       return trade_record
   ```

4. **Version Control and Code Review**
   - Maintain all strategy code in Git repository
   - Use semantic versioning for strategy iterations
   - Document all parameter changes in commit messages
   - Implement code review process before deploying new strategies
   - Tag production releases

5. **Performance Reporting**
   ```python
   def generate_monthly_report(trades, positions):
       """Generate comprehensive monthly performance report."""
       df = pd.DataFrame(trades)
       
       report = {
           'period': datetime.now().strftime('%Y-%m'),
           'summary': {
               'total_trades': len(df),
               'winning_trades': len(df[df['realized_pnl'] > 0]),
               'losing_trades': len(df[df['realized_pnl'] < 0]),
               'win_rate': len(df[df['realized_pnl'] > 0]) / len(df) * 100,
               'total_pnl': df['realized_pnl'].sum(),
               'avg_win': df[df['realized_pnl'] > 0]['realized_pnl'].mean(),
               'avg_loss': df[df['realized_pnl'] < 0]['realized_pnl'].mean()
           },
           'by_strategy': df.groupby('strategy_type').agg({
               'realized_pnl': ['sum', 'mean', 'count'],
               'hold_period': 'mean'
           }).to_dict(),
           'risk_metrics': {
               'max_drawdown': calculate_max_drawdown(df),
               'sharpe_ratio': calculate_sharpe_ratio(df),
               'profit_factor': calculate_profit_factor(df)
           }
       }
       
       return report
   ```

6. **Educational and Simulation Labeling**
   - Clearly mark all paper trading activity in logs
   - Include disclaimers in documentation that strategies are for educational purposes
   - Separate production and testing environments
   - Document assumptions and limitations

**Expected Competency:**
Operate within regulatory boundaries, maintain secure credential management, produce auditable documentation, and follow financial engineering best practices.

## Strategy Decision Matrix

| Strategy | Market View | Volatility View | Max Risk | Max Profit | Best Used When |
|----------|------------|-----------------|----------|------------|----------------|
| Long Straddle | Neutral | Expansion | Premium paid | Unlimited | Pre-earnings, major events |
| Long Strangle | Neutral | Expansion | Premium paid | Unlimited | Lower cost volatility play |
| Iron Butterfly | Neutral | Contraction | Wing width - Premium | Premium received | High IV rank, range-bound |
| Calendar Spread | Neutral | Term structure play | Net debit | Limited | Time decay arbitrage |
| Bull Call Spread | Bullish | Neutral/Low | Net debit | Strike difference - Debit | Directional with limited capital |
| Bear Put Spread | Bearish | Neutral/Low | Net debit | Strike difference - Debit | Directional downside play |
| Iron Condor | Range-bound | Contraction | Wing width - Premium | Premium received | Wide expected range, high IV |

## Risk Management Checklist

**Before Every Trade:**
- [ ] Calculate maximum loss and ensure it's acceptable
- [ ] Verify sufficient buying power/margin
- [ ] Confirm expiration date and time decay profile
- [ ] Check implied volatility rank/percentile
- [ ] Review upcoming earnings or events
- [ ] Calculate breakeven points
- [ ] Set profit target and stop-loss levels

**During Trade Management:**
- [ ] Monitor aggregate portfolio Greeks daily
- [ ] Track P/L against maximum loss
- [ ] Adjust delta if exceeds neutrality threshold
- [ ] Roll positions if approaching expiration
- [ ] Document all adjustments with rationale

**Post-Trade Review:**
- [ ] Record actual P/L vs. expected
- [ ] Analyze what worked and what didn't
- [ ] Update strategy parameters if needed
- [ ] Document lessons learned

## Common Pitfalls and Mitigations

**Pitfall**: Over-leveraging with undefined risk strategies
**Mitigation**: Always use defined-risk strategies or maintain adequate margin buffer

**Pitfall**: Ignoring volatility regime changes
**Mitigation**: Monitor IV rank and adjust strategy selection accordingly

**Pitfall**: Holding through expiration without management plan
**Mitigation**: Set calendar reminders for 7 days, 3 days, and 1 day before expiration

**Pitfall**: Chasing losses with larger position sizes
**Mitigation**: Implement strict position sizing rules (e.g., max 2% of capital per trade)

**Pitfall**: Neglecting transaction costs
**Mitigation**: Include commissions and slippage in backtests and profit calculations

## Integration with Existing Ordinis-1 Components

**Portfolio Management Integration:**
- Import positions from portfolio-management skill
- Track options positions alongside equity holdings
- Aggregate Greeks with overall portfolio delta
- Include options P/L in performance analytics

**Due Diligence Integration:**
- Research underlying companies before opening positions
- Analyze earnings history and volatility patterns
- Assess sector and market conditions
- Document strategy rationale using due-diligence framework

**Benchmarking Integration:**
- Compare strategy returns against VIX or CBOE indices
- Benchmark risk-adjusted returns (Sharpe, Sortino)
- Evaluate performance across market conditions

## Advanced Topics (Future Enhancements)

**Volatility Surface Modeling:**
- Build term structure and skew analysis
- Identify mispriced options
- Construct delta-neutral volatility arbitrage

**Machine Learning Integration:**
- Predict optimal entry/exit timing
- Forecast volatility regime changes
- Automate parameter optimization

**Multi-Asset Strategies:**
- Pairs trading with options
- Sector rotation using options leverage
- Cross-asset volatility strategies

## References and Further Study

**Alpaca Official Resources:**
- [Alpaca API Documentation](https://docs.alpaca.markets/)
- [Alpaca GitHub Organization](https://github.com/alpacahq)
- [Alpaca Learn Center](https://alpaca.markets/learn/)

**Industry Education:**
- [Options Industry Council](https://www.optionseducation.org/)
- [CBOE Education](https://www.cboe.com/education/)

**Quantitative Finance:**
- "Options, Futures, and Other Derivatives" by John Hull
- "Option Volatility and Pricing" by Sheldon Natenberg
- "Dynamic Hedging" by Nassim Taleb

**Python Libraries:**
- [QuantLib Python](https://www.quantlib.org/docs.shtml)
- [NumPy Documentation](https://numpy.org/doc/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Tutorials](https://matplotlib.org/stable/tutorials/)

## Support and Community

**Alpaca Community:**
- [Alpaca Community Forum](https://forum.alpaca.markets/)
- [Alpaca Slack](https://alpaca.markets/slack)
- [GitHub Issues](https://github.com/alpacahq)

**Options Trading Communities:**
- r/options (Reddit)
- r/thetagang (Reddit - options selling strategies)
- Elite Trader (options forum)

---

*This skill is designed for educational and research purposes. All examples use Alpaca's paper trading environment. Live trading involves substantial risk and requires appropriate risk management, capital allocation, and regulatory compliance.*
