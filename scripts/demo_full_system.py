"""
Full System Demo - End-to-End Trading Pipeline

Demonstrates complete workflow:
1. Live market data from multiple APIs
2. Strategy signal generation
3. RiskGuard approval
4. Paper broker execution
5. Portfolio monitoring
"""

from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import asyncio  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
import os  # noqa: E402

from dotenv import load_dotenv  # noqa: E402

from src.engines.flowroute.adapters.paper import PaperBrokerAdapter  # noqa: E402
from src.engines.flowroute.core.orders import Order, OrderType  # noqa: E402
from src.engines.riskguard.core.engine import (  # noqa: E402
    PortfolioState,
    ProposedTrade,
    RiskGuardEngine,
)
from src.engines.signalcore.core.signal import Direction, Signal, SignalType  # noqa: E402
from src.plugins.base import PluginConfig  # noqa: E402
from src.plugins.market_data import (  # noqa: E402
    AlphaVantageDataPlugin,
    FinnhubDataPlugin,
    TwelveDataPlugin,
)


class MultiSourceDataAggregator:
    """Aggregates data from multiple market data sources."""

    def __init__(self):
        self.sources = {}

    async def add_source(self, name: str, plugin):
        """Add a market data source."""
        if await plugin.initialize():
            self.sources[name] = plugin
            print(f"[OK] {name} initialized")
            return True
        print(f"[X] {name} failed to initialize")
        return False

    async def get_consensus_quote(self, symbol: str) -> dict:
        """Get quote from all sources and compute consensus."""
        quotes = {}

        for name, plugin in self.sources.items():
            try:
                quote = await plugin.get_quote(symbol)
                quotes[name] = quote
            except Exception as e:
                print(f"[WARN] {name} failed for {symbol}: {e}")

        if not quotes:
            raise ValueError(f"No quotes available for {symbol}")

        # Compute consensus price (average of last prices)
        prices = [q["last"] for q in quotes.values() if q["last"] > 0]
        consensus_price = sum(prices) / len(prices) if prices else 0

        return {
            "symbol": symbol,
            "consensus_price": consensus_price,
            "sources": quotes,
            "source_count": len(quotes),
            "timestamp": datetime.now(UTC).isoformat(),
        }

    async def shutdown(self):
        """Shutdown all sources."""
        for plugin in self.sources.values():
            await plugin.shutdown()


class SimpleStrategy:
    """Simple strategy that generates buy signals for demonstration."""

    def __init__(self, buy_threshold: float = 0.5):
        self.buy_threshold = buy_threshold
        self.signal_count = 0

    def generate_signal(self, symbol: str, price: float, momentum: float) -> Signal | None:
        """
        Generate a trading signal based on simple momentum.

        Args:
            symbol: Stock symbol
            price: Current price
            momentum: Momentum indicator (change %)

        Returns:
            Signal if conditions met, None otherwise
        """
        # Simple rule: Buy if positive momentum above threshold
        if momentum > self.buy_threshold:
            self.signal_count += 1
            confidence = min(momentum / 2.0, 0.95)
            return Signal(
                symbol=symbol,
                timestamp=datetime.now(UTC),
                signal_type=SignalType.ENTRY,
                direction=Direction.LONG,
                probability=confidence,
                expected_return=momentum / 100.0,  # Convert % to decimal
                confidence_interval=(0.0, momentum / 50.0),
                score=confidence,
                model_id="SimpleStrategy",
                model_version="1.0.0",
                feature_contributions={"momentum": momentum},
                metadata={
                    "price": price,
                    "momentum": momentum,
                },
            )
        return None


async def run_full_system_demo():  # noqa: PLR0915
    """Run complete system demonstration."""
    load_dotenv()

    print("\n" + "=" * 80)
    print("FULL SYSTEM DEMO - END-TO-END TRADING PIPELINE")
    print("=" * 80)

    # Initialize market data aggregator
    print("\n[1/7] Initializing multi-source market data aggregator...")
    aggregator = MultiSourceDataAggregator()

    # Add all available sources
    av_config = PluginConfig(
        name="alphavantage",
        api_key=os.getenv("ALPHAVANTAGE_API_KEY"),
        enabled=True,
        rate_limit_per_minute=5,
    )
    await aggregator.add_source("Alpha Vantage", AlphaVantageDataPlugin(av_config))

    fh_config = PluginConfig(
        name="finnhub",
        api_key=os.getenv("FINNHUB_API_KEY"),
        enabled=True,
        rate_limit_per_minute=60,
    )
    await aggregator.add_source("Finnhub", FinnhubDataPlugin(fh_config))

    # Skip Polygon for now (API key issues)
    # pg_config = PluginConfig(
    #     name="polygon",
    #     api_key=os.getenv("POLYGON_API_KEY"),
    #     enabled=True,
    #     rate_limit_per_minute=5,
    # )
    # await aggregator.add_source("Polygon", PolygonDataPlugin(pg_config))

    td_config = PluginConfig(
        name="twelvedata",
        api_key=os.getenv("TWELVEDATA_API_KEY"),
        enabled=True,
        rate_limit_per_minute=8,
    )
    await aggregator.add_source("Twelve Data", TwelveDataPlugin(td_config))

    print(f"\n[OK] Initialized {len(aggregator.sources)} data sources")

    # Initialize paper broker
    print("\n[2/7] Initializing paper trading broker...")
    # Use first available source for broker
    first_source = next(iter(aggregator.sources.values()))

    broker = PaperBrokerAdapter(
        slippage_bps=5.0,
        commission_per_share=0.005,
        market_data_plugin=first_source,
    )
    initial_capital = 100000.0
    print(f"[OK] Paper broker ready (${initial_capital:,.2f})")

    # Initialize RiskGuard
    print("\n[3/7] Initializing RiskGuard engine...")
    risk_engine = RiskGuardEngine()

    # RiskGuard is ready with default rules
    rule_count = len(risk_engine.rules) if hasattr(risk_engine, "rules") else 0
    print(f"[OK] RiskGuard configured with {rule_count} rules")

    # Initialize strategy
    print("\n[4/7] Initializing trading strategy...")
    strategy = SimpleStrategy(buy_threshold=0.5)
    print("[OK] SimpleStrategy ready")

    # Test symbols
    symbols = ["AAPL", "MSFT", "GOOGL"]
    print(f"\n[5/7] Fetching live market data for {len(symbols)} symbols...")

    all_quotes = {}
    for symbol in symbols:
        try:
            consensus = await aggregator.get_consensus_quote(symbol)
            all_quotes[symbol] = consensus
            print(
                f"[OK] {symbol}: ${consensus['consensus_price']:.2f} "
                f"({consensus['source_count']} sources)"
            )
        except Exception as e:
            print(f"[X] Failed to get quote for {symbol}: {e}")

    # Generate signals
    print("\n[6/7] Generating trading signals...")
    signals = []

    for symbol, quote_data in all_quotes.items():
        price = quote_data["consensus_price"]
        # Use first source's change percent as momentum
        first_quote = next(iter(quote_data["sources"].values()))
        momentum = abs(first_quote.get("change_percent", 0))

        signal = strategy.generate_signal(symbol, price, momentum)
        if signal:
            signals.append(signal)
            print(
                f"[SIGNAL] {symbol} - {signal.direction.name} @ ${price:.2f} "
                f"(probability: {signal.probability:.2%})"
            )

    print(f"\n[OK] Generated {len(signals)} signals")

    # Process signals through RiskGuard
    print("\n[7/7] Processing signals through RiskGuard...")

    # Build portfolio state
    account = await broker.get_account()
    positions = await broker.get_positions()

    # Convert positions to dict
    position_dict = {}
    for pos in positions:
        from src.engines.riskguard.core.engine import Position

        # Calculate market value if not present
        market_value = pos["quantity"] * pos["current_price"]
        position_dict[pos["symbol"]] = Position(
            symbol=pos["symbol"],
            quantity=pos["quantity"],
            entry_price=pos["avg_price"],
            current_price=pos["current_price"],
            market_value=market_value,
            unrealized_pnl=pos["unrealized_pnl"],
        )

    portfolio_state = PortfolioState(
        equity=account["total_equity"],
        cash=account["cash"],
        peak_equity=initial_capital,
        daily_pnl=0.0,
        daily_trades=0,
        open_positions=position_dict,
        total_positions=len(positions),
        total_exposure=account.get("total_position_value", 0.0),
    )

    approved_count = 0
    rejected_count = 0
    order_count = 0

    for signal in signals:
        # Calculate quantity based on simple position sizing (1% of capital per trade)
        price = all_quotes[signal.symbol]["consensus_price"]
        position_size_dollars = initial_capital * 0.01
        quantity = max(1, int(position_size_dollars / price))

        # Create proposed trade
        proposed_trade = ProposedTrade(
            symbol=signal.symbol,
            direction=signal.direction.value,
            quantity=quantity,
            entry_price=price,
        )

        # Check with RiskGuard
        passed, results, adjusted_signal = risk_engine.evaluate_signal(
            signal, proposed_trade, portfolio_state
        )

        if passed:
            approved_count += 1
            final_quantity = quantity  # Use original quantity since no resize rules
            print(f"[APPROVED] {signal.symbol} - {final_quantity} shares @ ${price:.2f}")

            # Submit order to broker
            order_count += 1
            order = Order(
                order_id=f"ORD-{order_count:04d}",
                symbol=signal.symbol,
                side="buy" if signal.direction == Direction.LONG else "sell",
                quantity=final_quantity,
                order_type=OrderType.MARKET,
            )

            result = await broker.submit_order(order)
            print(f"  [ORDER] {result.get('broker_order_id')} - {result.get('status')}")

        else:
            rejected_count += 1
            # Get rejection reason from results
            rejection_reasons = [r.message for r in results if not r.passed]
            reason = "; ".join(rejection_reasons) if rejection_reasons else "Unknown"
            print(f"[REJECTED] {signal.symbol} - {reason}")

    # Process pending orders to trigger fills
    if approved_count > 0:
        print(f"\nProcessing {approved_count} pending orders...")
        await asyncio.sleep(0.2)  # Small delay for realistic fill timing
        fills = await broker.process_pending_orders()
        print(f"[OK] {len(fills)} orders filled")

    # Final status
    print("\n" + "=" * 80)
    print("EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Signals Generated:  {len(signals)}")
    print(f"Signals Approved:   {approved_count}")
    print(f"Signals Rejected:   {rejected_count}")

    final_account = await broker.get_account()
    final_positions = await broker.get_positions()

    print("\nFinal Account State:")
    print(f"  Cash:             ${final_account['cash']:,.2f}")
    print(f"  Total Equity:     ${final_account['total_equity']:,.2f}")
    print(f"  Open Positions:   {len(final_positions)}")

    if final_positions:
        print("\n  Positions:")
        for pos in final_positions:
            symbol = pos["symbol"]
            qty = pos["quantity"]
            avg_price = pos["avg_price"]
            current = all_quotes.get(symbol, {}).get("consensus_price", avg_price)
            pnl = (current - avg_price) * qty
            print(f"    {symbol}: {qty} @ ${avg_price:.2f} | P&L: ${pnl:+.2f}")

    # Cleanup
    await aggregator.shutdown()

    print("\n" + "=" * 80)
    print("[COMPLETE] Full system demo finished successfully!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(run_full_system_demo())
