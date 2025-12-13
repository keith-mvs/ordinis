# Pairs Trading

## Overview

Pairs trading exploits the mean-reverting spread between two cointegrated securities. When the spread deviates significantly from its mean, trade expecting convergence.

---

## Strategy Logic

```
1. Identify cointegrated pair (A, B)
2. Estimate hedge ratio: β = cov(A,B) / var(B)
3. Construct spread: S = A - β × B
4. Calculate z-score: z = (S - μ_S) / σ_S
5. Entry: |z| > threshold (e.g., 2.0)
6. Exit: z returns to 0 or opposite threshold
```

---

## Step-by-Step Implementation

### 1. Pair Selection

```python
def find_cointegrated_pairs(prices: pd.DataFrame, significance=0.05):
    """
    Screen all pairs for cointegration.
    """
    from statsmodels.tsa.stattools import coint

    pairs = []
    tickers = prices.columns.tolist()

    for i in range(len(tickers)):
        for j in range(i+1, len(tickers)):
            t1, t2 = tickers[i], tickers[j]

            # Test cointegration
            stat, p_value, _ = coint(prices[t1], prices[t2])

            if p_value < significance:
                # Estimate hedge ratio
                hedge_ratio = np.cov(prices[t1], prices[t2])[0,1] / np.var(prices[t2])
                spread = prices[t1] - hedge_ratio * prices[t2]

                # Calculate half-life
                half_life = estimate_half_life(spread)

                pairs.append({
                    'pair': (t1, t2),
                    'p_value': p_value,
                    'hedge_ratio': hedge_ratio,
                    'half_life': half_life
                })

    return sorted(pairs, key=lambda x: x['p_value'])
```

### 2. Hedge Ratio Estimation

```python
def calculate_hedge_ratio(series_y, series_x, method='ols'):
    """
    Estimate hedge ratio (beta) for spread construction.
    """
    if method == 'ols':
        # Simple OLS regression
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(series_x.values.reshape(-1, 1), series_y.values)
        return model.coef_[0]

    elif method == 'tls':
        # Total Least Squares (orthogonal regression)
        from scipy.odr import ODR, Model, Data

        def linear(B, x):
            return B[0] * x + B[1]

        linear_model = Model(linear)
        data = Data(series_x, series_y)
        odr = ODR(data, linear_model, beta0=[1., 0.])
        output = odr.run()
        return output.beta[0]

    elif method == 'rolling':
        # Rolling OLS for time-varying hedge ratio
        window = 60  # 60-day rolling window
        ratios = []
        for i in range(window, len(series_y)):
            y = series_y.iloc[i-window:i]
            x = series_x.iloc[i-window:i]
            model = LinearRegression()
            model.fit(x.values.reshape(-1, 1), y.values)
            ratios.append(model.coef_[0])
        return pd.Series(ratios, index=series_y.index[window:])
```

### 3. Spread Construction

```python
def construct_spread(series_y, series_x, hedge_ratio):
    """
    Build the spread series.

    Spread = Y - β × X
    """
    if isinstance(hedge_ratio, (int, float)):
        # Static hedge ratio
        spread = series_y - hedge_ratio * series_x
    else:
        # Time-varying hedge ratio
        spread = series_y.loc[hedge_ratio.index] - hedge_ratio * series_x.loc[hedge_ratio.index]

    return spread

def calculate_zscore(spread, lookback=60):
    """
    Standardize spread to z-score.
    """
    mean = spread.rolling(lookback).mean()
    std = spread.rolling(lookback).std()
    return (spread - mean) / std
```

### 4. Signal Generation

```python
class PairsTradingSignal:
    def __init__(
        self,
        entry_z: float = 2.0,
        exit_z: float = 0.0,
        stop_z: float = 4.0,
        lookback: int = 60
    ):
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_z = stop_z
        self.lookback = lookback
        self.position = 0  # 1 = long spread, -1 = short spread

    def generate_signal(self, zscore: float) -> int:
        """
        Returns:
            1: Long spread (buy Y, short X)
           -1: Short spread (short Y, buy X)
            0: No position / exit
        """
        # Stop loss
        if abs(zscore) > self.stop_z:
            self.position = 0
            return 0

        # Entry signals
        if self.position == 0:
            if zscore < -self.entry_z:
                self.position = 1  # Long spread
                return 1
            elif zscore > self.entry_z:
                self.position = -1  # Short spread
                return -1

        # Exit signals
        if self.position == 1 and zscore >= self.exit_z:
            self.position = 0
            return 0
        elif self.position == -1 and zscore <= self.exit_z:
            self.position = 0
            return 0

        return self.position  # Hold current position
```

### 5. Position Sizing

```python
def pairs_position_size(
    equity: float,
    risk_pct: float,
    spread_std: float,
    entry_z: float,
    stop_z: float,
    price_y: float,
    price_x: float,
    hedge_ratio: float
) -> dict:
    """
    Calculate position sizes for both legs.
    """
    # Risk amount
    risk_amount = equity * risk_pct

    # Stop distance in spread terms
    stop_distance = (stop_z - entry_z) * spread_std

    # Dollar amount per unit of spread
    # 1 unit spread = 1 share Y - hedge_ratio shares X
    spread_dollar_value = price_y + hedge_ratio * price_x

    # Units of spread we can trade
    spread_units = risk_amount / stop_distance

    # Convert to shares
    shares_y = int(spread_units)
    shares_x = int(spread_units * hedge_ratio)

    return {
        'shares_y': shares_y,
        'shares_x': shares_x,
        'notional_y': shares_y * price_y,
        'notional_x': shares_x * price_x,
        'total_capital': shares_y * price_y + shares_x * price_x
    }
```

---

## Half-Life Estimation

```python
def estimate_half_life(spread: pd.Series) -> float:
    """
    Estimate half-life of mean reversion using OU process.

    dS = θ(μ - S)dt + σdW
    Half-life = ln(2) / θ
    """
    # Lag spread
    spread_lag = spread.shift(1).dropna()
    spread_diff = spread.diff().dropna()

    # Align series
    spread_lag = spread_lag.iloc[1:]
    spread_diff = spread_diff.iloc[1:]

    # OLS: ΔS = α + β × S_lag
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(spread_lag.values.reshape(-1, 1), spread_diff.values)

    # θ = -β (mean reversion speed)
    theta = -model.coef_[0]

    if theta <= 0:
        return float('inf')  # Not mean-reverting

    half_life = np.log(2) / theta
    return half_life
```

---

## Risk Management

### Position Limits
```python
PAIRS_RISK_LIMITS = {
    'max_notional_per_pair': 0.10,      # 10% of equity per pair
    'max_total_pairs_exposure': 0.50,   # 50% total pairs exposure
    'max_zscore_deviation': 4.0,        # Force exit beyond this
    'max_holding_days': 30,             # Time-based exit
    'max_drawdown_per_pair': 0.05       # 5% max loss per pair
}
```

### Hedge Ratio Drift
```python
def monitor_hedge_ratio_stability(
    current_ratio: float,
    historical_ratios: pd.Series,
    threshold: float = 0.20
) -> dict:
    """
    Monitor for hedge ratio drift.
    """
    mean_ratio = historical_ratios.mean()
    drift = abs(current_ratio - mean_ratio) / mean_ratio

    return {
        'current_ratio': current_ratio,
        'historical_mean': mean_ratio,
        'drift_pct': drift,
        'alert': drift > threshold
    }
```

---

## Example Pairs

### Classic Pairs
| Pair | Sector | Rationale |
|------|--------|-----------|
| GLD / GDX | Gold | Mining stocks vs metal |
| XLE / USO | Energy | ETF vs commodity |
| KO / PEP | Consumer | Direct competitors |
| V / MA | Payments | Similar business model |

### Sector ETF Pairs
| Pair | Type |
|------|------|
| XLF / KBE | Financials |
| XLK / SMH | Technology |
| XLV / IBB | Healthcare |

---

## Pitfalls

1. **Spurious Cointegration**: Always test on out-of-sample data
2. **Hedge Ratio Instability**: Use rolling estimation
3. **Execution Slippage**: Wide spreads erode edge
4. **Fundamental Changes**: M&A, spinoffs break relationships
5. **Crowded Trades**: Popular pairs have diminished returns

---

## Academic References

- Gatev, Goetzmann, Rouwenhorst (2006): "Pairs Trading: Performance of a Relative-Value Arbitrage Rule"
- Vidyamurthy (2004): "Pairs Trading: Quantitative Methods and Analysis"
- Engle & Granger (1987): Cointegration theory

---

## Implementation

```python
from src.strategies.quantitative import PairsTrader

# Initialize pairs trader
trader = PairsTrader(
    entry_z=2.0,
    exit_z=0.0,
    stop_z=4.0,
    lookback=60,
    hedge_method='rolling'
)

# Find cointegrated pairs
pairs = trader.find_pairs(price_data, significance=0.05)

# Generate signals for a pair
signals = trader.generate_signals(pair='GLD_GDX')

# Execute trades
orders = trader.execute(signals, position_size_pct=0.05)
```
