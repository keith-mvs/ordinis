"""
Risk-managed paper trading demonstration.

Integrates:
- Paper Broker (order execution)
- RiskGuard (risk management)
- MA Crossover Strategy (signal generation)

Shows full workflow with risk checks before each trade.
"""

import asyncio
from datetime import UTC, datetime
import sys
from typing import Any

import pandas as pd

sys.path.insert(0, "src")

from engines.flowroute.adapters.paper import PaperBrokerAdapter
from engines.flowroute.core.orders import Order, OrderType
from engines.riskguard.core.engine import (
    PortfolioState,
    Position,
    ProposedTrade,
    RiskGuardEngine,
)
from engines.riskguard.rules.standard import STANDARD_RISK_RULES
from engines.signalcore.core.signal import Direction, Signal, SignalType


class MockMarketData:
    """Mock market data from sample CSV."""

    def __init__(self, data_path: str = "data/sample_spy_trending_up.csv"):
        self.data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        self.current_idx = 0

    async def get_quote(self, symbol: str) -> dict[str, Any]:
        if self.current_idx >= len(self.data):
            self.current_idx = 0
        row = self.data.iloc[self.current_idx]
        return {
            "symbol": symbol,
            "bid": row["close"] * 0.9999,
            "ask": row["close"] * 1.0001,
            "last": row["close"],
        }

    def get_current_bar(self) -> dict[str, Any] | None:
        if self.current_idx >= len(self.data):
            return None
        row = self.data.iloc[self.current_idx]
        return {"timestamp": self.data.index[self.current_idx], "close": row["close"]}

    def advance(self) -> bool:
        self.current_idx += 1
        return self.current_idx < len(self.data)


class MACrossoverStrategy:
    """MA Crossover strategy with signal generation."""

    def __init__(self, fast: int = 20, slow: int = 50):
        self.fast = fast
        self.slow = slow
        self.prices: list[float] = []
        self.prev_fast: float | None = None
        self.prev_slow: float | None = None

    def update(self, price: float) -> Signal | None:
        self.prices.append(price)

        if len(self.prices) < self.slow:
            return None

        fast_ma = sum(self.prices[-self.fast :]) / self.fast
        slow_ma = sum(self.prices[-self.slow :]) / self.slow

        signal = self._check_crossover(fast_ma, slow_ma)
        self.prev_fast = fast_ma
        self.prev_slow = slow_ma

        return signal

    def _check_crossover(self, fast_ma: float, slow_ma: float) -> Signal | None:
        if not self.prev_fast or not self.prev_slow:
            return None

        if self.prev_fast <= self.prev_slow and fast_ma > slow_ma:
            return self._create_signal(SignalType.ENTRY, Direction.LONG, 0.6)
        if self.prev_fast >= self.prev_slow and fast_ma < slow_ma:
            return self._create_signal(SignalType.EXIT, Direction.SHORT, -0.6)
        return None

    def _create_signal(self, sig_type: SignalType, direction: Direction, score: float) -> Signal:
        return Signal(
            symbol="SPY",
            timestamp=datetime.now(UTC),
            signal_type=sig_type,
            direction=direction,
            probability=0.7,
            expected_return=0.02 if score > 0 else -0.02,
            confidence_interval=(0.01, 0.03) if score > 0 else (-0.03, -0.01),
            score=score,
            model_id="MACrossover",
            model_version="1.0",
        )


def build_portfolio_state(broker: PaperBrokerAdapter, initial: float) -> PortfolioState:
    """Build portfolio state from broker."""
    positions = {
        sym: Position(
            symbol=sym,
            quantity=pos["quantity"],
            entry_price=pos["avg_price"],
            current_price=pos["current_price"],
            market_value=pos["quantity"] * pos["current_price"],
            unrealized_pnl=pos["unrealized_pnl"],
        )
        for sym, pos in broker._positions.items()
    }

    total_value = sum(p.market_value for p in positions.values())
    equity = broker._cash + total_value

    return PortfolioState(
        equity=equity,
        cash=broker._cash,
        peak_equity=max(equity, initial),
        daily_pnl=equity - initial,
        daily_trades=len(broker.get_fills()),
        open_positions=positions,
        total_positions=len(positions),
        total_exposure=total_value / equity if equity > 0 else 0,
    )


def print_config(initial: float, position_pct: float, rule_count: int) -> None:
    """Print session configuration."""
    print("\n" + "=" * 70)
    print("RISK-MANAGED PAPER TRADING SESSION")
    print("=" * 70)
    print("\n[CONFIG]")
    print(f"  Initial Capital: ${initial:,.2f}")
    print(f"  Position Size: {position_pct * 100:.0f}%")
    print("  Strategy: MA Crossover (20/50)")
    print(f"  Risk Rules: {rule_count} active")


def print_rules(risk_engine: RiskGuardEngine) -> None:
    """Print risk rules."""
    print("\n[RISK RULES]")
    for rule in risk_engine._rules.values():
        thresh = f"{rule.threshold:.0%}" if abs(rule.threshold) < 1 else str(rule.threshold)
        print(f"  {rule.name}: {thresh} ({rule.action_on_breach})")


def print_summary(stats: dict[str, Any], account: dict[str, Any], initial: float) -> None:
    """Print session summary."""
    print("-" * 70)
    print("\n[SESSION SUMMARY]")
    print(f"  Signals Generated: {stats['signals_generated']}")
    print(f"  Signals Approved: {stats['signals_approved']}")
    print(f"  Signals Rejected: {stats['signals_rejected']}")
    print(f"  Halt Triggered: {'Yes' if stats['halt_triggered'] else 'No'}")

    print("\n[ACCOUNT]")
    print(f"  Starting: ${initial:,.2f}")
    print(f"  Final: ${account['total_equity']:,.2f}")

    pnl = account["total_equity"] - initial
    print(f"\n[P&L] ${pnl:,.2f} ({(pnl / initial) * 100:.2f}%)")

    print("\n" + "=" * 70)
    print("[SESSION COMPLETE]")
    print("=" * 70 + "\n")


async def process_entry_signal(
    signal: Signal,
    bar: dict[str, Any],
    bar_num: int,
    portfolio: PortfolioState,
    risk_engine: RiskGuardEngine,
    broker: PaperBrokerAdapter,
    position_pct: float,
    stats: dict[str, Any],
) -> int:
    """Process entry signal with risk checks. Returns new position size."""
    qty = int(portfolio.cash * position_pct / bar["close"])
    proposed = ProposedTrade(
        symbol="SPY",
        direction="long",
        quantity=qty,
        entry_price=bar["close"],
        stop_price=bar["close"] * 0.95,
    )

    passed, results, adjusted = risk_engine.evaluate_signal(signal, proposed, portfolio)

    for r in results:
        if not r.passed:
            print(
                f"  Bar {bar_num:3d} | RISK | {r.rule_name}: "
                f"{r.current_value:.2%} vs {r.threshold:.2%} -> {r.action_taken}"
            )

    if passed:
        stats["signals_approved"] += 1
        order = Order(
            order_id=f"RM-{bar_num:04d}",
            symbol="SPY",
            side="buy",
            quantity=qty,
            order_type=OrderType.MARKET,
        )
        result = await broker.submit_order(order)
        if result["status"] == "filled":
            ts = bar["timestamp"].strftime("%Y-%m-%d")
            print(f"  Bar {bar_num:3d} | {ts} | BUY  {qty:4d} SPY @ ${result['fill']['price']:.2f}")
            return qty
    elif adjusted:
        stats["signals_resized"] += 1
    else:
        stats["signals_rejected"] += 1
        ts = bar["timestamp"].strftime("%Y-%m-%d")
        print(f"  Bar {bar_num:3d} | {ts} | REJECTED | Risk rules breached")

    return 0


async def process_exit_signal(
    bar: dict[str, Any],
    bar_num: int,
    current_position: int,
    broker: PaperBrokerAdapter,
    stats: dict[str, Any],
) -> int:
    """Process exit signal. Returns 0 (closed position)."""
    stats["signals_approved"] += 1
    order = Order(
        order_id=f"RM-{bar_num:04d}",
        symbol="SPY",
        side="sell",
        quantity=current_position,
        order_type=OrderType.MARKET,
    )
    result = await broker.submit_order(order)
    if result["status"] == "filled":
        ts = bar["timestamp"].strftime("%Y-%m-%d")
        print(
            f"  Bar {bar_num:3d} | {ts} | SELL {current_position:4d} SPY @ ${result['fill']['price']:.2f}"
        )
        return 0
    return current_position


async def run_risk_managed_session(max_bars: int = 200) -> dict[str, Any]:
    """Run risk-managed trading session."""
    market_data = MockMarketData()
    broker = PaperBrokerAdapter(market_data_plugin=market_data)
    initial_capital = 100000.0
    broker.reset(initial_capital)

    risk_engine = RiskGuardEngine(STANDARD_RISK_RULES)
    strategy = MACrossoverStrategy(fast=20, slow=50)
    position_size_pct = 0.10

    print_config(initial_capital, position_size_pct, len(risk_engine._rules))
    print_rules(risk_engine)

    stats = {
        "signals_generated": 0,
        "signals_approved": 0,
        "signals_rejected": 0,
        "signals_resized": 0,
        "halt_triggered": False,
    }

    print("\n[TRADING LOG]")
    print("-" * 70)

    bar_num = 0
    current_position = 0

    while bar_num < max_bars:
        bar = market_data.get_current_bar()
        if bar is None:
            break

        portfolio = build_portfolio_state(broker, initial_capital)
        triggered, reason = risk_engine.check_kill_switches(portfolio)

        if triggered:
            print(f"\n  [HALT] Kill switch triggered: {reason}")
            stats["halt_triggered"] = True
            break

        signal = strategy.update(bar["close"])

        if signal:
            stats["signals_generated"] += 1

            if signal.signal_type == SignalType.ENTRY and current_position == 0:
                current_position = await process_entry_signal(
                    signal, bar, bar_num, portfolio, risk_engine, broker, position_size_pct, stats
                )
            elif signal.signal_type == SignalType.EXIT and current_position > 0:
                current_position = await process_exit_signal(
                    bar, bar_num, current_position, broker, stats
                )

        market_data.advance()
        bar_num += 1

    account = await broker.get_account()
    print_summary(stats, account, initial_capital)

    return {
        "final_equity": account["total_equity"],
        "pnl": account["total_equity"] - initial_capital,
        **stats,
    }


async def main() -> None:
    """Run demo."""
    await run_risk_managed_session(max_bars=500)


if __name__ == "__main__":
    asyncio.run(main())
