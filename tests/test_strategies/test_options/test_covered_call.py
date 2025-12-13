"""
Covered Call Strategy Tests
"""

from datetime import datetime

import pandas as pd
import pytest

from ordinis.application.strategies.options.covered_call import CoveredCallStrategy


@pytest.fixture
def strategy():
    """Create covered call strategy instance."""
    return CoveredCallStrategy(name="Test Covered Call")


@pytest.fixture
def sample_data():
    """Create sample OHLCV data."""
    dates = pd.date_range(end=datetime.now(), periods=50, freq="D")
    data = pd.DataFrame(
        {
            "open": [100 + i * 0.5 for i in range(50)],
            "high": [102 + i * 0.5 for i in range(50)],
            "low": [99 + i * 0.5 for i in range(50)],
            "close": [101 + i * 0.5 for i in range(50)],
            "volume": [1000000] * 50,
        },
        index=dates,
    )
    return data


def test_strategy_configuration(strategy):
    """Test strategy configuration."""
    assert strategy.name == "Test Covered Call"
    assert strategy.params["min_premium_yield"] == 0.12
    assert strategy.params["min_delta"] == 0.25
    assert strategy.params["max_delta"] == 0.40
    assert strategy.params["days_to_expiration"] == 45


def test_strategy_description(strategy):
    """Test strategy description."""
    desc = strategy.get_description()
    assert "Covered Call" in desc
    assert "12%" in desc
    assert "45 days" in desc


def test_required_bars(strategy):
    """Test minimum bars requirement."""
    assert strategy.get_required_bars() == 20


def test_analyze_opportunity(strategy):
    """Test covered call opportunity analysis."""
    analysis = strategy.analyze_opportunity(
        underlying_price=100.0,
        call_strike=105.0,
        call_premium=2.50,
        call_delta=0.30,
        days_to_expiration=45,
    )

    # Verify calculations
    assert analysis["underlying_price"] == 100.0
    assert analysis["call_strike"] == 105.0
    assert analysis["call_premium"] == 2.50
    assert analysis["call_delta"] == 0.30

    # Max profit = premium + (strike - stock price)
    assert analysis["max_profit"] == 2.50 + (105.0 - 100.0)
    assert analysis["max_profit"] == 7.50

    # Max loss = stock price - premium (if stock goes to zero)
    assert analysis["max_loss"] == 100.0 - 2.50
    assert analysis["max_loss"] == 97.50

    # Breakeven = stock price - premium
    assert analysis["breakeven"] == 100.0 - 2.50
    assert analysis["breakeven"] == 97.50

    # Downside protection
    assert analysis["downside_protection_pct"] == 2.50 / 100.0

    # Premium yield
    assert analysis["premium_yield"] == 2.50 / 100.0

    # Annualized yield
    expected_annual = (2.50 / 100.0) * (365 / 45)
    assert abs(analysis["annualized_yield"] - expected_annual) < 0.001

    # Probability of profit (1 - delta)
    assert analysis["prob_profit"] == 1 - 0.30
    assert analysis["prob_profit"] == 0.70


def test_analyze_opportunity_high_yield(strategy):
    """Test high-yield covered call opportunity."""
    analysis = strategy.analyze_opportunity(
        underlying_price=100.0,
        call_strike=102.0,  # Closer to money
        call_premium=4.00,  # Higher premium
        call_delta=0.45,
        days_to_expiration=30,
    )

    # Higher premium means higher annualized yield
    assert analysis["annualized_yield"] > 0.40  # Over 40% annualized

    # But also higher delta (more likely to be called away)
    assert analysis["call_delta"] == 0.45

    # Check if meets criteria (delta too high)
    assert not analysis["meets_criteria"]  # Delta > max_delta (0.40)


def test_calculate_payoff_stock_below_strike(strategy):
    """Test P&L when stock stays below strike."""
    payoff = strategy.calculate_payoff(
        underlying_price=102.0,  # Below strike
        stock_entry=100.0,
        call_strike=105.0,
        call_premium=2.50,
    )

    # Stock P&L
    assert payoff["stock_pnl"] == 102.0 - 100.0
    assert payoff["stock_pnl"] == 2.0

    # Call P&L (expires worthless, keep premium)
    assert payoff["call_pnl"] == 2.50

    # Total P&L
    assert payoff["total_pnl"] == 2.0 + 2.50
    assert payoff["total_pnl"] == 4.50

    # Not called away
    assert not payoff["stock_called_away"]


def test_calculate_payoff_stock_above_strike(strategy):
    """Test P&L when stock goes above strike (called away)."""
    payoff = strategy.calculate_payoff(
        underlying_price=110.0,  # Above strike
        stock_entry=100.0,
        call_strike=105.0,
        call_premium=2.50,
    )

    # Stock P&L would be 10.0 if not called away, but capped at strike
    # Call P&L = premium - intrinsic value
    # Call intrinsic = 110 - 105 = 5.0
    # Call P&L = 2.50 - 5.0 = -2.50

    # Net effect: stock called away at 105, so profit is 5 + 2.50 = 7.50
    assert payoff["stock_pnl"] == 110.0 - 100.0  # Unrealized stock gain
    assert payoff["call_pnl"] == 2.50 - (110.0 - 105.0)  # Call loss
    assert payoff["call_pnl"] == -2.50

    # Total P&L
    assert payoff["total_pnl"] == 10.0 - 2.50
    assert payoff["total_pnl"] == 7.50

    # Stock called away
    assert payoff["stock_called_away"]


def test_calculate_payoff_at_strike(strategy):
    """Test P&L when stock exactly at strike."""
    payoff = strategy.calculate_payoff(
        underlying_price=105.0,  # Exactly at strike
        stock_entry=100.0,
        call_strike=105.0,
        call_premium=2.50,
    )

    # Stock P&L
    assert payoff["stock_pnl"] == 5.0

    # Call expires at-the-money, treat as worthless (keep premium)
    assert payoff["call_pnl"] == 2.50

    # Total P&L
    assert payoff["total_pnl"] == 7.50

    # Not called away (exactly at strike)
    assert not payoff["stock_called_away"]


def test_calculate_payoff_stock_drops(strategy):
    """Test P&L when stock drops significantly."""
    payoff = strategy.calculate_payoff(
        underlying_price=90.0,  # Stock drops 10%
        stock_entry=100.0,
        call_strike=105.0,
        call_premium=2.50,
    )

    # Stock loss
    assert payoff["stock_pnl"] == -10.0

    # Call profit (expires worthless, keep premium)
    assert payoff["call_pnl"] == 2.50

    # Total P&L (loss partially offset by premium)
    assert payoff["total_pnl"] == -10.0 + 2.50
    assert payoff["total_pnl"] == -7.50

    # Downside protection provided by premium
    # Loss without covered call would be -10.0
    # Loss with covered call is -7.50 (2.50 cushion)


def test_generate_signal_no_engine(strategy, sample_data):
    """Test signal generation without options engine."""
    timestamp = datetime.now()
    signal = strategy.generate_signal(sample_data, timestamp, options_engine=None)

    # Should return None without options engine
    assert signal is None


def test_custom_parameters():
    """Test strategy with custom parameters."""
    strategy = CoveredCallStrategy(
        name="Aggressive CC",
        min_premium_yield=0.20,  # 20% annual yield target
        min_delta=0.35,
        max_delta=0.50,
        days_to_expiration=30,
    )

    assert strategy.params["min_premium_yield"] == 0.20
    assert strategy.params["min_delta"] == 0.35
    assert strategy.params["max_delta"] == 0.50
    assert strategy.params["days_to_expiration"] == 30


def test_roi_calculation(strategy):
    """Test return on investment calculation."""
    payoff = strategy.calculate_payoff(
        underlying_price=107.50,
        stock_entry=100.0,
        call_strike=105.0,
        call_premium=2.50,
    )

    # Total P&L should be 7.50 (capped at strike)
    assert payoff["total_pnl"] == 7.50

    # ROI = profit / capital
    assert payoff["roi"] == 7.50 / 100.0
    assert payoff["roi"] == 0.075  # 7.5% return
