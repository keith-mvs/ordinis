"""
Backtest Engine Script.

Runs a backtest using SignalCore models (including Fundamental and Sentiment)
against synthetic historical data.
"""

import asyncio
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from ordinis.engines.signalcore.core.engine import SignalCoreEngine
from ordinis.engines.signalcore.core.model import ModelConfig
from ordinis.engines.signalcore.core.signal import Direction, SignalType
from ordinis.engines.signalcore.models import (
    BollingerBandsModel,
    FundamentalValueModel,
    RSIMeanReversionModel,
    SentimentMomentumModel,
    StatisticalReversionModel,
    VolumeTrendModel,
)


# Mock Portfolio for backtest
class BacktestPortfolio:
    def __init__(self, initial_capital=100000.0):
        self.cash = initial_capital
        self.positions = {}  # symbol -> quantity
        self.equity_curve = []
        self.trades = []

    def update(self, date, prices):
        equity = self.cash
        for symbol, qty in self.positions.items():
            if symbol in prices:
                equity += qty * prices[symbol]
        self.equity_curve.append({"date": date, "equity": equity})
        return equity

    def execute_signal(self, signal, price, date):
        symbol = signal.symbol

        if signal.signal_type == SignalType.ENTRY:
            # Simple position sizing: 10% of equity per trade
            equity = self.equity_curve[-1]["equity"] if self.equity_curve else self.cash
            target_size = equity * 0.10
            quantity = int(target_size / price)

            if quantity > 0:
                if signal.direction == Direction.LONG:
                    cost = quantity * price
                    if self.cash >= cost:
                        self.cash -= cost
                        self.positions[symbol] = self.positions.get(symbol, 0) + quantity
                        self.trades.append(
                            {
                                "date": date,
                                "symbol": symbol,
                                "side": "BUY",
                                "price": price,
                                "qty": quantity,
                                "reason": signal.model_id,
                            }
                        )
                elif signal.direction == Direction.SHORT:
                    # Simplified short: receive cash, negative position
                    # In reality, margin requirements apply
                    self.cash += quantity * price
                    self.positions[symbol] = self.positions.get(symbol, 0) - quantity
                    self.trades.append(
                        {
                            "date": date,
                            "symbol": symbol,
                            "side": "SELL_SHORT",
                            "price": price,
                            "qty": quantity,
                            "reason": signal.model_id,
                        }
                    )

        elif signal.signal_type == SignalType.EXIT:
            current_qty = self.positions.get(symbol, 0)
            if current_qty != 0:
                # Close position
                if current_qty > 0:  # Long exit
                    self.cash += current_qty * price
                    self.trades.append(
                        {
                            "date": date,
                            "symbol": symbol,
                            "side": "SELL",
                            "price": price,
                            "qty": current_qty,
                            "reason": signal.model_id,
                        }
                    )
                else:  # Short exit
                    cost = abs(current_qty) * price
                    self.cash -= cost
                    self.trades.append(
                        {
                            "date": date,
                            "symbol": symbol,
                            "side": "BUY_TO_COVER",
                            "price": price,
                            "qty": abs(current_qty),
                            "reason": signal.model_id,
                        }
                    )
                self.positions[symbol] = 0


async def run_backtest():
    print("[INFO] Starting Backtest Engine...")

    # 1. Setup Engine
    engine = SignalCoreEngine()

    # Register Models
    models = [
        FundamentalValueModel(
            ModelConfig(model_id="fund_v1", model_type="fundamental", min_data_points=1)
        ),
        SentimentMomentumModel(
            ModelConfig(model_id="sent_v1", model_type="sentiment", min_data_points=1)
        ),
        StatisticalReversionModel(
            ModelConfig(model_id="stat_v1", model_type="technical", min_data_points=30)
        ),
        VolumeTrendModel(
            ModelConfig(model_id="vol_v1", model_type="technical", min_data_points=30)
        ),
        BollingerBandsModel(
            ModelConfig(model_id="bb_v1", model_type="technical", min_data_points=20)
        ),
        RSIMeanReversionModel(
            ModelConfig(model_id="rsi_v1", model_type="technical", min_data_points=15)
        ),
    ]

    for model in models:
        engine._registry.register(model)

    print(f"[INFO] Registered {len(models)} models: {engine.list_models()}")

    # 2. Generate Data
    print("[INFO] Generating Synthetic Data...")
    dates = pd.date_range(start="2024-01-01", end="2024-12-31", freq="D", tz="UTC")
    n = len(dates)

    # Price: Random Walk with Trend
    returns = np.random.normal(0.0005, 0.02, n)  # Slight positive drift
    price_path = 100 * np.cumprod(1 + returns)

    # P/E: Correlated with price but mean reverting
    pe_ratios = 20 + (price_path - 100) / 5 + np.random.normal(0, 2, n)

    # Sentiment: Random regime switching
    sentiment = np.zeros(n)
    current_sent = 0.5
    for i in range(n):
        if np.random.random() < 0.1:  # 10% chance to change regime
            current_sent = np.random.random()
        sentiment[i] = current_sent + np.random.normal(0, 0.05)
        sentiment[i] = max(0, min(1, sentiment[i]))  # Clip 0-1

    df = pd.DataFrame(
        {
            "open": price_path,
            "high": price_path * 1.01,
            "low": price_path * 0.99,
            "close": price_path,
            "volume": np.random.randint(100000, 1000000, n),
            "pe_ratio": pe_ratios,
            "sentiment_score": sentiment,
            "symbol": "AAPL",
        },
        index=dates,
    )

    # 3. Run Simulation
    portfolio = BacktestPortfolio()
    print(f"[INFO] Running simulation over {n} days...")

    for i in range(50, n):  # Start after warmup
        current_date = dates[i]
        # Slice data up to current point
        # Note: In a real engine, we'd be more efficient than slicing every step
        window_data = df.iloc[: i + 1]

        # Prepare data dict for engine
        cycle_data = {"AAPL": window_data}

        # Generate Signals
        batch = await engine.generate_batch(cycle_data)

        # Execute Signals
        current_price = df.iloc[i]["close"]
        prices = {"AAPL": current_price}

        for signal in batch.signals:
            # Simple logic: Execute all valid signals
            # In reality, we'd have conflict resolution
            if signal.score > 0.5 or signal.score < -0.5:  # Strong signal
                portfolio.execute_signal(signal, current_price, current_date)

        # Update Portfolio
        portfolio.update(current_date, prices)

    # 4. Results
    final_equity = portfolio.equity_curve[-1]["equity"]
    returns_pct = (final_equity - 100000) / 100000 * 100

    print("-" * 50)
    print("BACKTEST RESULTS")
    print("-" * 50)
    print("Initial Capital: $100,000.00")
    print(f"Final Equity:    ${final_equity:,.2f}")
    print(f"Total Return:    {returns_pct:.2f}%")
    print(f"Total Trades:    {len(portfolio.trades)}")

    if portfolio.trades:
        print("\nSample Trades:")
        for t in portfolio.trades[:5]:
            print(f"  {t['date'].date()} {t['side']} {t['qty']} @ {t['price']:.2f} ({t['reason']})")

    print("-" * 50)


if __name__ == "__main__":
    asyncio.run(run_backtest())
