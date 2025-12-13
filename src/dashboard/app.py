"""
Cognisys Dashboard - Intelligent Trading & Investment Platform.

An intelligent and semi-automated multi-vehicle trading, investment,
and analysis tool.

Run with: streamlit run src/dashboard/app.py
"""

import asyncio
from datetime import UTC, datetime
import os
import sys

from dotenv import load_dotenv
import pandas as pd
import streamlit as st

sys.path.insert(0, "src")
load_dotenv()

from adapters.market_data.twelvedata import TwelveDataPlugin
from engines.flowroute.adapters.paper import PaperBrokerAdapter
from engines.flowroute.core.orders import Order, OrderType
from engines.riskguard.core.engine import PortfolioState, Position, RiskGuardEngine
from engines.riskguard.rules.standard import STANDARD_RISK_RULES
from plugins.base import PluginConfig

# Default watchlist
DEFAULT_WATCHLIST = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "SPY", "QQQ"]


def init_session_state() -> None:
    """Initialize Streamlit session state."""
    if "broker" not in st.session_state:
        st.session_state.broker = PaperBrokerAdapter(
            slippage_bps=5.0,
            commission_per_share=0.005,
        )
        st.session_state.broker.reset(100000.0)

    if "risk_engine" not in st.session_state:
        st.session_state.risk_engine = RiskGuardEngine(STANDARD_RISK_RULES)

    if "trade_log" not in st.session_state:
        st.session_state.trade_log = []

    if "equity_history" not in st.session_state:
        st.session_state.equity_history = []

    if "market_data" not in st.session_state:
        api_key = os.getenv("TWELVEDATA_API_KEY")
        if api_key:
            config = PluginConfig(
                name="twelvedata",
                api_key=api_key,
                rate_limit_per_minute=8,  # Free tier: 8/min
            )
            st.session_state.market_data = TwelveDataPlugin(config)
            st.session_state.market_data_initialized = False
        else:
            st.session_state.market_data = None

    if "watchlist" not in st.session_state:
        st.session_state.watchlist = DEFAULT_WATCHLIST.copy()

    if "quotes_cache" not in st.session_state:
        st.session_state.quotes_cache = {}
        st.session_state.quotes_updated = None


async def init_market_data() -> bool:
    """Initialize market data plugin if needed."""
    if st.session_state.market_data and not st.session_state.market_data_initialized:
        try:
            result = await st.session_state.market_data.initialize()
            st.session_state.market_data_initialized = result
            return result
        except Exception:
            return False
    return st.session_state.market_data_initialized


async def fetch_quote(symbol: str) -> dict | None:
    """Fetch a single quote from market data provider."""
    if not st.session_state.market_data:
        return None
    try:
        await init_market_data()
        return await st.session_state.market_data.get_quote(symbol)
    except Exception:
        return None


async def fetch_watchlist_quotes() -> dict:
    """Fetch quotes for all watchlist symbols."""
    quotes = {}
    if not st.session_state.market_data:
        return quotes

    await init_market_data()

    for symbol in st.session_state.watchlist[:5]:  # Limit to 5 for rate limiting
        try:
            quote = await st.session_state.market_data.get_quote(symbol)
            if quote:
                quotes[symbol] = quote
        except Exception:
            pass

    return quotes


def get_portfolio_state() -> PortfolioState:
    """Get current portfolio state for risk checks."""
    broker = st.session_state.broker
    positions_dict = {}

    for symbol, pos_data in broker._positions.items():
        positions_dict[symbol] = Position(
            symbol=symbol,
            quantity=pos_data["quantity"],
            entry_price=pos_data["avg_price"],
            current_price=pos_data["current_price"],
            market_value=pos_data["quantity"] * pos_data["current_price"],
            unrealized_pnl=pos_data["unrealized_pnl"],
        )

    total_position_value = sum(p.market_value for p in positions_dict.values())
    equity = broker._cash + total_position_value

    # Get peak equity from history
    if st.session_state.equity_history:
        peak_equity = max(e["equity"] for e in st.session_state.equity_history)
    else:
        peak_equity = equity

    return PortfolioState(
        equity=equity,
        cash=broker._cash,
        peak_equity=peak_equity,
        daily_pnl=equity - 100000.0,
        daily_trades=len(broker.get_fills()),
        open_positions=positions_dict,
        total_positions=len(positions_dict),
        total_exposure=total_position_value / equity if equity > 0 else 0,
    )


def render_header() -> None:
    """Render dashboard header."""
    st.set_page_config(
        page_title="Cognisys",
        page_icon="🧠",
        layout="wide",
    )
    st.title("🧠 Cognisys")
    st.markdown("**Intelligent & Semi-Automated Multi-Vehicle Trading, Investment & Analysis**")


def render_account_summary() -> None:
    """Render account summary section."""
    st.header("Account Summary")

    broker = st.session_state.broker
    account = asyncio.run(broker.get_account())

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Cash", f"${account['cash']:,.2f}")

    with col2:
        st.metric("Position Value", f"${account['total_position_value']:,.2f}")

    with col3:
        st.metric("Total Equity", f"${account['total_equity']:,.2f}")

    with col4:
        pnl = account["total_equity"] - 100000.0
        pnl_pct = (pnl / 100000.0) * 100
        st.metric("P&L", f"${pnl:,.2f}", f"{pnl_pct:+.2f}%")


def render_positions() -> None:
    """Render positions table."""
    st.header("Open Positions")

    broker = st.session_state.broker
    positions = asyncio.run(broker.get_positions())

    if not positions:
        st.info("No open positions")
        return

    df = pd.DataFrame(positions)
    df = df[["symbol", "quantity", "avg_price", "current_price", "unrealized_pnl"]]
    df = df.rename(
        columns={
            "symbol": "Symbol",
            "quantity": "Quantity",
            "avg_price": "Avg Price",
            "current_price": "Current Price",
            "unrealized_pnl": "Unrealized P&L",
        }
    )

    st.dataframe(df, width="stretch")


def render_risk_status() -> None:
    """Render risk management status."""
    st.header("Risk Status")

    risk_engine = st.session_state.risk_engine
    portfolio = get_portfolio_state()

    # Check kill switches
    triggered, reason = risk_engine.check_kill_switches(portfolio)

    col1, col2 = st.columns(2)

    with col1:
        if risk_engine.is_halted():
            st.error(f"⛔ TRADING HALTED: {reason}")
        else:
            st.success("✅ Trading Active")

    with col2:
        drawdown = (
            (portfolio.equity - portfolio.peak_equity) / portfolio.peak_equity * 100
            if portfolio.peak_equity > 0
            else 0
        )
        st.metric("Drawdown", f"{drawdown:.2f}%")

    # Risk rules summary
    st.subheader("Risk Rules")

    rules_data = []
    for rule_id, rule in risk_engine._rules.items():
        rules_data.append(
            {
                "Rule": rule.name,
                "Category": rule.category.value,
                "Threshold": f"{rule.threshold:.2%}" if rule.threshold < 1 else str(rule.threshold),
                "Action": rule.action_on_breach,
                "Enabled": "✅" if rule.enabled else "❌",
            }
        )

    df = pd.DataFrame(rules_data)
    st.dataframe(df, width="stretch")


def render_order_form() -> None:
    """Render order entry form."""
    st.header("Submit Order")

    col1, col2, col3 = st.columns(3)

    with col1:
        symbol = st.text_input("Symbol", value="SPY")

    with col2:
        side = st.selectbox("Side", ["buy", "sell"])

    with col3:
        quantity = st.number_input("Quantity", min_value=1, value=10)

    price = st.number_input("Price (for simulation)", min_value=0.01, value=100.00)

    if st.button("Submit Order"):
        broker = st.session_state.broker
        order = Order(
            order_id=f"DASH-{datetime.now(UTC).strftime('%H%M%S')}",
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType.MARKET,
        )

        # Simulate fill immediately
        fill = broker.simulate_fill(order, price)

        st.session_state.trade_log.append(
            {
                "time": datetime.now(UTC).strftime("%H:%M:%S"),
                "side": side.upper(),
                "symbol": symbol,
                "qty": quantity,
                "price": price,
            }
        )

        # Record equity
        account = asyncio.run(broker.get_account())
        st.session_state.equity_history.append(
            {"time": datetime.now(UTC), "equity": account["total_equity"]}
        )

        st.success(f"Order filled: {side.upper()} {quantity} {symbol} @ ${price:.2f}")
        st.rerun()


def render_trade_log() -> None:
    """Render recent trade log."""
    st.header("Recent Trades")

    broker = st.session_state.broker
    fills = broker.get_fills()

    if not fills:
        st.info("No trades yet")
        return

    fills_data = [
        {
            "Fill ID": f.fill_id,
            "Side": f.side.upper(),
            "Symbol": f.symbol,
            "Qty": f.quantity,
            "Price": f"${f.price:.2f}",
            "Commission": f"${f.commission:.2f}",
        }
        for f in fills[-10:]  # Last 10 fills
    ]

    df = pd.DataFrame(fills_data)
    st.dataframe(df, width="stretch")


def render_equity_chart() -> None:
    """Render equity curve chart."""
    st.header("Equity Curve")

    if not st.session_state.equity_history:
        st.info("No equity history yet")
        return

    df = pd.DataFrame(st.session_state.equity_history)
    st.line_chart(df.set_index("time")["equity"])


def render_market_data() -> None:
    """Render live market data watchlist."""
    st.header("Market Data")

    if not st.session_state.market_data:
        st.warning("Market data not configured. Add TWELVEDATA_API_KEY to .env")
        return

    col1, col2 = st.columns([3, 1])

    with col2:
        if st.button("Refresh Quotes"):
            quotes = asyncio.run(fetch_watchlist_quotes())
            st.session_state.quotes_cache = quotes
            st.session_state.quotes_updated = datetime.now(UTC)
            st.rerun()

    with col1:
        if st.session_state.quotes_updated:
            st.caption(f"Last updated: {st.session_state.quotes_updated.strftime('%H:%M:%S')} UTC")

    # Display quotes
    if st.session_state.quotes_cache:
        quotes_data = []
        for symbol, quote in st.session_state.quotes_cache.items():
            change_pct = quote.get("change_percent", 0)
            quotes_data.append(
                {
                    "Symbol": symbol,
                    "Price": f"${quote.get('last', 0):.2f}",
                    "Change": f"{quote.get('change', 0):+.2f}",
                    "Change %": f"{change_pct:+.2f}%",
                    "Volume": f"{quote.get('volume', 0):,}",
                }
            )

        df = pd.DataFrame(quotes_data)
        st.dataframe(df, width="stretch", hide_index=True)
    else:
        st.info("Click 'Refresh Quotes' to load live market data")

    # Watchlist management
    with st.expander("Manage Watchlist"):
        new_symbol = st.text_input("Add symbol:", key="new_symbol")
        if st.button("Add") and new_symbol:
            if new_symbol.upper() not in st.session_state.watchlist:
                st.session_state.watchlist.append(new_symbol.upper())
                st.rerun()

        st.write("Current watchlist:", ", ".join(st.session_state.watchlist))


def render_controls() -> None:
    """Render control buttons."""
    st.sidebar.header("Controls")

    if st.sidebar.button("Reset Account"):
        st.session_state.broker.reset(100000.0)
        st.session_state.trade_log = []
        st.session_state.equity_history = []
        st.rerun()

    if st.sidebar.button("Reset Halt"):
        st.session_state.risk_engine.reset_halt()
        st.rerun()

    st.sidebar.divider()

    st.sidebar.subheader("Risk Settings")

    daily_loss = st.sidebar.slider("Daily Loss Limit %", 1, 10, 3)
    max_dd = st.sidebar.slider("Max Drawdown %", 5, 25, 15)

    # Update rules
    if "daily_loss_limit" in st.session_state.risk_engine._rules:
        st.session_state.risk_engine._rules["daily_loss_limit"].threshold = -daily_loss / 100
    if "max_drawdown" in st.session_state.risk_engine._rules:
        st.session_state.risk_engine._rules["max_drawdown"].threshold = -max_dd / 100

    # Market data status
    st.sidebar.divider()
    st.sidebar.subheader("Data Sources")
    if st.session_state.market_data:
        st.sidebar.success("Twelve Data: Connected")
    else:
        st.sidebar.warning("Twelve Data: Not configured")


def main() -> None:
    """Main dashboard entry point."""
    init_session_state()
    render_header()
    render_controls()

    # Create tabs for organized layout
    tab1, tab2, tab3 = st.tabs(["Trading", "Market Data", "Analysis"])

    with tab1:
        # Main content
        render_account_summary()

        col1, col2 = st.columns(2)

        with col1:
            render_positions()
            render_order_form()

        with col2:
            render_risk_status()

        render_trade_log()
        render_equity_chart()

    with tab2:
        render_market_data()

    with tab3:
        st.header("Analysis")
        st.info("Coming soon: Technical indicators, strategy backtesting, and portfolio analytics.")


if __name__ == "__main__":
    main()
