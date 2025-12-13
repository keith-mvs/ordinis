import datetime as dt

from ordinis.engines.proofbench.core.execution import (
    Bar,
    ExecutionConfig,
    ExecutionSimulator,
    FillMode,
    Order,
    OrderSide,
)


def make_bar() -> Bar:
    return Bar(
        symbol="TEST",
        timestamp=dt.datetime(2024, 1, 1),
        open=100.0,
        high=110.0,
        low=90.0,
        close=105.0,
        volume=1_000,
    )


def base_config(fill_mode: FillMode) -> ExecutionConfig:
    # Zero slippage/commission to isolate fill price logic
    return ExecutionConfig(
        estimated_spread=0.0,
        impact_coefficient=0.0,
        volatility_factor=0.0,
        max_slippage=0.0,
        commission_pct=0.0,
        commission_per_share=0.0,
        commission_per_trade=0.0,
        fill_mode=fill_mode,
    )


def test_fill_mode_bar_open_buy():
    bar = make_bar()
    sim = ExecutionSimulator(config=base_config(FillMode.BAR_OPEN))
    order = Order(symbol="TEST", side=OrderSide.BUY, quantity=10)
    fill = sim._fill_market_order(order, bar, bar.timestamp)
    assert fill.price == bar.open  # no slippage in this config


def test_fill_mode_intra_bar_buy_sell():
    bar = make_bar()
    sim = ExecutionSimulator(config=base_config(FillMode.INTRA_BAR))

    buy = Order(symbol="TEST", side=OrderSide.BUY, quantity=10)
    buy_fill = sim._fill_market_order(buy, bar, bar.timestamp)
    # Bias toward intrabar high for buys
    assert buy_fill.price == (bar.high + bar.open) / 2

    sell = Order(symbol="TEST", side=OrderSide.SELL, quantity=10)
    sell_fill = sim._fill_market_order(sell, bar, bar.timestamp)
    # Bias toward intrabar low for sells
    assert sell_fill.price == (bar.low + bar.open) / 2


def test_fill_mode_realistic_buy_sell():
    bar = make_bar()
    sim = ExecutionSimulator(config=base_config(FillMode.REALISTIC))

    buy = Order(symbol="TEST", side=OrderSide.BUY, quantity=10)
    buy_fill = sim._fill_market_order(buy, bar, bar.timestamp)
    expected_buy = min(bar.high, (bar.open + bar.close + bar.high) / 3)
    assert buy_fill.price == expected_buy

    sell = Order(symbol="TEST", side=OrderSide.SELL, quantity=10)
    sell_fill = sim._fill_market_order(sell, bar, bar.timestamp)
    expected_sell = max(bar.low, (bar.open + bar.close + bar.low) / 3)
    assert sell_fill.price == expected_sell
