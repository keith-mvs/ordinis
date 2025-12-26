# Portfolio Management Strategies

Portfolio construction and allocation strategies that determine **how to weight assets** rather than when to trade them.

## Characteristics
- Output: target weights per asset
- Operate on portfolios/universes, not individual securities
- Rebalance periodically to target allocation
- Measure: portfolio return, volatility, Sharpe, max drawdown, turnover

## Strategies

| Strategy | Type | Description |
|----------|------|-------------|
| NETWORK_PARITY | Risk Parity | Weight inversely to correlation network centrality |
| EVT_RISK_GATE | Risk Management | Extreme value theory tail risk limits |
| HMM_REGIME | Regime Detection | Hidden Markov Model regime classification |
| REGIME_ADAPTIVE_MANAGER | Meta-Strategy | Allocate across strategies based on regime |

## Key Difference from Trading Strategies

```
Trading Strategy:      "Should I buy AAPL now?" → YES/NO signal
Portfolio Management:  "How much AAPL should I hold?" → 5.2% weight
```

Portfolio management strategies can be combined WITH trading strategies:
1. Portfolio strategy sets target weights
2. Trading strategy determines entry/exit timing within those weights
