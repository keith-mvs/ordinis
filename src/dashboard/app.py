"""
Trading Dashboard - Web-based monitoring interface.

Run with: streamlit run src/dashboard/app.py
"""

import asyncio
from datetime import UTC, datetime
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, "src")

from engines.flowroute.adapters.paper import PaperBrokerAdapter
from engines.flowroute.core.orders import Order, OrderType
from engines.riskguard.core.engine import PortfolioState, Position, RiskGuardEngine
from engines.riskguard.rules.standard import STANDARD_RISK_RULES


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
        page_title="Intelligent Investor Dashboard",
        page_icon="📈",
        layout="wide",
    )
    st.title("📈 Intelligent Investor Dashboard")
    st.markdown("**Paper Trading & Risk Monitoring**")


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

    st.dataframe(df, use_container_width=True)


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
    st.dataframe(df, use_container_width=True)


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
    st.dataframe(df, use_container_width=True)


def render_equity_chart() -> None:
    """Render equity curve chart."""
    st.header("Equity Curve")

    if not st.session_state.equity_history:
        st.info("No equity history yet")
        return

    df = pd.DataFrame(st.session_state.equity_history)
    st.line_chart(df.set_index("time")["equity"])


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


def main() -> None:
    """Main dashboard entry point."""
    init_session_state()
    render_header()
    render_controls()

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


if __name__ == "__main__":
    main()
