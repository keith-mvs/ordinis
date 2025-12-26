# Trading Strategies

Signal-generation strategies that produce discrete **entry/exit signals** for individual positions.

## Characteristics
- Generate BUY/SELL/HOLD signals
- Operate on individual securities
- Have defined entry conditions, exit conditions, stop losses
- Measure: win rate, avg P&L per trade, holding period, Sharpe

## Strategies

| Strategy | Type | Description |
|----------|------|-------------|
| ATR-RSI | Mean Reversion | RSI signals with ATR-based position sizing |
| BOLLINGER_BANDS | Mean Reversion | Band breakout/reversion |
| FIBONACCI_ADX | Trend | Fib retracements with ADX trend filter |
| GARCH_BREAKOUT | Volatility | GARCH-modeled vol breakouts |
| KALMAN_HYBRID | Adaptive | Kalman filter trend + mean reversion |
| MACD | Momentum | Classic MACD crossover |
| MI_ENSEMBLE | Ensemble | Mutual information weighted ensemble |
| MOMENTUM_BREAKOUT | Momentum | Price/volume breakout |
| MOVING_AVERAGE_CROSSOVER | Trend | MA crossover signals |
| MTF_MOMENTUM | Multi-timeframe | Multi-timeframe momentum confluence |
| OPTIONS_TRADING | Options | Covered calls, spreads, condors |
| OU_PAIRS | Stat Arb | Ornstein-Uhlenbeck pairs trading |
| PARABOLIC_SAR | Trend | SAR-based trend following |
| RSI_MEAN_REVERSION | Mean Reversion | Oversold/overbought RSI |
| TREND_FOLLOWING_EMA | Trend | EMA trend following |
| VOLATILITY_SQUEEZE | Breakout | Bollinger/Keltner squeeze |
| VOLUME_PRICE_CONFIRM | Confirmation | Volume-confirmed price moves |
