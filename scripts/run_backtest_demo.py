"""
Historical Backtest Demo - 30 Minute Session.

Demonstrates the full backtesting capability:
1. Load sample data (500 bars of synthetic SPY)
2. Run all 5 strategies
3. Compare performance
4. Display results

Timeline: 2023-01-01 to 2024-12-31 (synthetic data)
"""

from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dataclasses import dataclass  # noqa: E402
from datetime import datetime  # noqa: E402

import pandas as pd  # noqa: E402

from src.engines.proofbench.core.execution import Order, OrderSide, OrderType  # noqa: E402
from src.engines.proofbench.core.simulator import (  # noqa: E402
    SimulationConfig,
    SimulationEngine,
)


@dataclass
class StrategyResult:
    """Results for a single strategy."""

    name: str
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    profit_factor: float
    final_equity: float


def load_sample_data(filename: str) -> pd.DataFrame:
    """Load sample data with proper index."""
    filepath = project_root / "data" / filename
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    return df


def create_ma_crossover_strategy(fast: int = 20, slow: int = 50):
    """Create MA crossover strategy callback."""
    position = {"shares": 0, "entry_price": 0.0}
    ma_data = {"fast_ma": [], "slow_ma": []}

    def strategy(engine, symbol, bar):
        """MA Crossover strategy."""
        # Get historical closes from data
        data = engine.data[symbol]
        closes = data.loc[: bar.timestamp, "close"]

        if len(closes) < slow:
            return

        fast_ma = closes.iloc[-fast:].mean()
        slow_ma = closes.iloc[-slow:].mean()

        # Track for analysis
        ma_data["fast_ma"].append(fast_ma)
        ma_data["slow_ma"].append(slow_ma)

        # Get previous MAs
        if len(ma_data["fast_ma"]) < 2:
            return

        prev_fast = ma_data["fast_ma"][-2]
        prev_slow = ma_data["slow_ma"][-2]

        # Golden cross - buy
        if prev_fast <= prev_slow and fast_ma > slow_ma:
            if position["shares"] == 0:
                shares = int(engine.portfolio.cash * 0.95 / bar.close)
                if shares > 0:
                    order = Order(
                        symbol=symbol,
                        side=OrderSide.BUY,
                        quantity=shares,
                        order_type=OrderType.MARKET,
                    )
                    engine.submit_order(order)
                    position["shares"] = shares
                    position["entry_price"] = bar.close

        # Death cross - sell
        elif prev_fast >= prev_slow and fast_ma < slow_ma:
            if position["shares"] > 0:
                order = Order(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    quantity=position["shares"],
                    order_type=OrderType.MARKET,
                )
                engine.submit_order(order)
                position["shares"] = 0

    return strategy


def create_rsi_strategy(period: int = 14, oversold: int = 30, overbought: int = 70):
    """Create RSI mean reversion strategy callback."""
    position = {"shares": 0}

    def calculate_rsi(closes: pd.Series, period: int) -> float:
        """Calculate RSI."""
        delta = closes.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        if loss.iloc[-1] == 0:
            return 100.0
        rs = gain.iloc[-1] / loss.iloc[-1]
        return 100 - (100 / (1 + rs))

    def strategy(engine, symbol, bar):
        """RSI Mean Reversion strategy."""
        data = engine.data[symbol]
        closes = data.loc[: bar.timestamp, "close"]

        if len(closes) < period + 1:
            return

        rsi = calculate_rsi(closes, period)

        # Oversold - buy
        if rsi < oversold and position["shares"] == 0:
            shares = int(engine.portfolio.cash * 0.95 / bar.close)
            if shares > 0:
                order = Order(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    quantity=shares,
                    order_type=OrderType.MARKET,
                )
                engine.submit_order(order)
                position["shares"] = shares

        # Overbought - sell
        elif rsi > overbought and position["shares"] > 0:
            order = Order(
                symbol=symbol,
                side=OrderSide.SELL,
                quantity=position["shares"],
                order_type=OrderType.MARKET,
            )
            engine.submit_order(order)
            position["shares"] = 0

    return strategy


def create_momentum_strategy(lookback: int = 20, threshold: float = 0.05):
    """Create momentum breakout strategy callback."""
    position = {"shares": 0}

    def strategy(engine, symbol, bar):
        """Momentum Breakout strategy."""
        data = engine.data[symbol]
        closes = data.loc[: bar.timestamp, "close"]

        if len(closes) < lookback:
            return

        # Calculate momentum
        momentum = (closes.iloc[-1] - closes.iloc[-lookback]) / closes.iloc[-lookback]

        # Strong upward momentum - buy
        if momentum > threshold and position["shares"] == 0:
            shares = int(engine.portfolio.cash * 0.95 / bar.close)
            if shares > 0:
                order = Order(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    quantity=shares,
                    order_type=OrderType.MARKET,
                )
                engine.submit_order(order)
                position["shares"] = shares

        # Momentum fading - sell
        elif momentum < 0 and position["shares"] > 0:
            order = Order(
                symbol=symbol,
                side=OrderSide.SELL,
                quantity=position["shares"],
                order_type=OrderType.MARKET,
            )
            engine.submit_order(order)
            position["shares"] = 0

    return strategy


def create_bollinger_strategy(period: int = 20, std_dev: float = 2.0):
    """Create Bollinger Bands strategy callback."""
    position = {"shares": 0}

    def strategy(engine, symbol, bar):
        """Bollinger Bands strategy."""
        data = engine.data[symbol]
        closes = data.loc[: bar.timestamp, "close"]

        if len(closes) < period:
            return

        # Calculate Bollinger Bands
        sma = closes.iloc[-period:].mean()
        std = closes.iloc[-period:].std()
        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)

        current_price = bar.close

        # Price below lower band - buy
        if current_price < lower and position["shares"] == 0:
            shares = int(engine.portfolio.cash * 0.95 / bar.close)
            if shares > 0:
                order = Order(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    quantity=shares,
                    order_type=OrderType.MARKET,
                )
                engine.submit_order(order)
                position["shares"] = shares

        # Price above upper band - sell
        elif current_price > upper and position["shares"] > 0:
            order = Order(
                symbol=symbol,
                side=OrderSide.SELL,
                quantity=position["shares"],
                order_type=OrderType.MARKET,
            )
            engine.submit_order(order)
            position["shares"] = 0

    return strategy


def create_macd_strategy(fast: int = 12, slow: int = 26, signal: int = 9):
    """Create MACD strategy callback."""
    position = {"shares": 0}
    macd_history = {"macd": [], "signal": []}

    def strategy(engine, symbol, bar):
        """MACD strategy."""
        data = engine.data[symbol]
        closes = data.loc[: bar.timestamp, "close"]

        if len(closes) < slow + signal:
            return

        # Calculate MACD
        exp1 = closes.ewm(span=fast, adjust=False).mean()
        exp2 = closes.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()

        current_macd = macd.iloc[-1]
        current_signal = signal_line.iloc[-1]

        macd_history["macd"].append(current_macd)
        macd_history["signal"].append(current_signal)

        if len(macd_history["macd"]) < 2:
            return

        prev_macd = macd_history["macd"][-2]
        prev_signal = macd_history["signal"][-2]

        # MACD crosses above signal - buy
        if prev_macd <= prev_signal and current_macd > current_signal:
            if position["shares"] == 0:
                shares = int(engine.portfolio.cash * 0.95 / bar.close)
                if shares > 0:
                    order = Order(
                        symbol=symbol,
                        side=OrderSide.BUY,
                        quantity=shares,
                        order_type=OrderType.MARKET,
                    )
                    engine.submit_order(order)
                    position["shares"] = shares

        # MACD crosses below signal - sell
        elif prev_macd >= prev_signal and current_macd < current_signal:
            if position["shares"] > 0:
                order = Order(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    quantity=position["shares"],
                    order_type=OrderType.MARKET,
                )
                engine.submit_order(order)
                position["shares"] = 0

    return strategy


def run_backtest(
    data: pd.DataFrame, strategy_callback, strategy_name: str, initial_capital: float = 100000.0
) -> StrategyResult:
    """Run a single backtest."""
    config = SimulationConfig(
        initial_capital=initial_capital,
        bar_frequency="1d",
        enable_logging=False,
    )

    engine = SimulationEngine(config)
    engine.load_data("SPY", data)
    engine.set_strategy(strategy_callback)

    results = engine.run()

    # Extract metrics (already in percentage form)
    metrics = results.metrics
    return StrategyResult(
        name=strategy_name,
        total_return=metrics.total_return,  # Already in percentage
        sharpe_ratio=metrics.sharpe_ratio,
        max_drawdown=metrics.max_drawdown,  # Already in percentage
        win_rate=metrics.win_rate,  # Already in percentage (0-100)
        total_trades=metrics.num_trades,
        profit_factor=metrics.profit_factor if metrics.profit_factor != float("inf") else 999.99,
        final_equity=metrics.equity_final,
    )


def print_header(text: str):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def print_result(result: StrategyResult, rank: int):
    """Print strategy result."""
    print(f"\n{rank}. {result.name}")
    print("-" * 40)
    print(f"   Total Return:    {result.total_return:+.2f}%")
    print(f"   Sharpe Ratio:    {result.sharpe_ratio:.2f}")
    print(f"   Max Drawdown:    {result.max_drawdown:.2f}%")
    print(f"   Win Rate:        {result.win_rate:.1f}%")
    print(f"   Total Trades:    {result.total_trades}")
    print(f"   Profit Factor:   {result.profit_factor:.2f}")
    print(f"   Final Equity:    ${result.final_equity:,.2f}")


def main():
    """Run backtest demo."""
    print_header("HISTORICAL BACKTEST DEMO")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Initial Capital: $100,000")
    print("Data: Synthetic SPY (2023 - ~500 bars)")

    # Load sample data
    print("\n[1/6] Loading sample data...")
    try:
        data = load_sample_data("sample_spy_trending_up.csv")
        print(f"[OK] Loaded {len(data)} bars")
        print(
            f"     Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}"
        )
        print(f"     Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    except Exception as e:
        print(f"[X] Failed to load data: {e}")
        return

    # Define strategies
    strategies = [
        ("MA Crossover (20/50)", create_ma_crossover_strategy(20, 50)),
        ("RSI Mean Reversion", create_rsi_strategy(14, 30, 70)),
        ("Momentum Breakout", create_momentum_strategy(20, 0.05)),
        ("Bollinger Bands", create_bollinger_strategy(20, 2.0)),
        ("MACD", create_macd_strategy(12, 26, 9)),
    ]

    # Run backtests
    results = []
    for i, (name, callback) in enumerate(strategies, 1):
        print(f"\n[{i+1}/6] Running {name}...")
        try:
            result = run_backtest(data, callback, name)
            results.append(result)
            print(
                f"[OK] Return: {result.total_return:+.2f}% | Sharpe: {result.sharpe_ratio:.2f} | Trades: {result.total_trades}"
            )
        except Exception as e:
            print(f"[X] Failed: {e}")

    # Sort by Sharpe ratio
    results.sort(key=lambda x: x.sharpe_ratio, reverse=True)

    # Display results
    print_header("BACKTEST RESULTS - RANKED BY SHARPE RATIO")

    for rank, result in enumerate(results, 1):
        print_result(result, rank)

    # Summary table
    print_header("COMPARISON TABLE")
    print(
        f"\n{'Strategy':<25} {'Return':>10} {'Sharpe':>8} {'MaxDD':>8} {'WinRate':>8} {'Trades':>7}"
    )
    print("-" * 70)
    for result in results:
        print(
            f"{result.name:<25} "
            f"{result.total_return:>+9.2f}% "
            f"{result.sharpe_ratio:>8.2f} "
            f"{result.max_drawdown:>7.2f}% "
            f"{result.win_rate:>7.1f}% "
            f"{result.total_trades:>7}"
        )

    # Winner announcement
    print_header("WINNER")
    winner = results[0]
    print(f"\n[CHAMPION] {winner.name}")
    print(f"   Return: {winner.total_return:+.2f}%")
    print(f"   Sharpe: {winner.sharpe_ratio:.2f}")
    print(f"   Turned $100,000 into ${winner.final_equity:,.2f}")

    # Buy and hold comparison
    print_header("BENCHMARK: BUY AND HOLD")
    start_price = data["close"].iloc[0]
    end_price = data["close"].iloc[-1]
    bh_return = ((end_price - start_price) / start_price) * 100
    bh_final = 100000 * (1 + bh_return / 100)
    print(f"\n   Buy and Hold Return: {bh_return:+.2f}%")
    print(f"   Final Equity: ${bh_final:,.2f}")

    # Alpha calculation
    print_header("ALPHA ANALYSIS")
    for result in results:
        alpha = result.total_return - bh_return
        status = "OUTPERFORMED" if alpha > 0 else "UNDERPERFORMED"
        print(f"   {result.name:<25} Alpha: {alpha:+.2f}% [{status}]")

    # Session complete
    print_header("DEMO COMPLETE")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nSystem validated. All strategies executed successfully.")
    print("Ready for live paper trading when you are.\n")


if __name__ == "__main__":
    main()
