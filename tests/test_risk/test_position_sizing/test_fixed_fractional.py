"""
Unit tests for Fixed Fractional position sizing.
"""

import pytest

from ordinis.risk.position_sizing.fixed_fractional import (
    AntiMartingale,
    FixedFractionalSizing,
    FixedRatioSizing,
    PositionSizeResult,
    PositionSizingEngine,
    SizingMethod,
)


class TestPositionSizeResult:
    """Test PositionSizeResult dataclass."""

    def test_valid_result(self):
        """Test creation with valid result."""
        result = PositionSizeResult(
            shares=100.0,
            notional_value=15000.0,
            risk_amount=500.0,
            risk_percent=0.01,
            method=SizingMethod.PERCENT_RISK,
        )

        assert result.shares == 100.0
        assert result.notional_value == 15000.0
        assert result.risk_amount == 500.0
        assert result.risk_percent == 0.01
        assert result.method == SizingMethod.PERCENT_RISK


class TestFixedFractionalSizing:
    """Test FixedFractionalSizing calculator."""

    def test_initialization(self):
        """Test initialization with default and custom parameters."""
        sizer = FixedFractionalSizing()

        assert sizer.risk_percent == 0.01
        assert sizer.max_position_percent == 0.20
        assert sizer.min_position_value == 100.0
        assert sizer.round_to_lots is True
        assert sizer.lot_size == 100

    def test_percent_risk_basic(self):
        """Test basic percent risk calculation."""
        sizer = FixedFractionalSizing(
            risk_percent=0.01, max_position_percent=0.30, round_to_lots=False
        )

        result = sizer.percent_risk(account_value=100000, entry_price=150.0, stop_price=145.0)

        # Risk per share = 5.0
        # Risk amount = 100000 * 0.01 = 1000
        # Shares = 1000 / 5 = 200
        assert result.shares == pytest.approx(200.0)
        assert result.notional_value == pytest.approx(30000.0)
        assert result.risk_amount == pytest.approx(1000.0)
        assert result.risk_percent == pytest.approx(0.01)
        assert result.method == SizingMethod.PERCENT_RISK

    def test_percent_risk_with_lot_rounding(self):
        """Test percent risk with lot size rounding."""
        sizer = FixedFractionalSizing(
            risk_percent=0.01, max_position_percent=0.25, round_to_lots=True, lot_size=100
        )

        result = sizer.percent_risk(account_value=100000, entry_price=150.0, stop_price=145.0)

        # Raw shares = 200, but max position 25% = 166.67 shares max
        # After rounding: 100 shares (1 lot)
        assert result.shares == 100.0
        assert result.shares % 100 == 0

    def test_percent_risk_max_position_limit(self):
        """Test that max position limit is applied."""
        sizer = FixedFractionalSizing(
            risk_percent=0.05, max_position_percent=0.10, round_to_lots=False
        )

        result = sizer.percent_risk(account_value=100000, entry_price=100.0, stop_price=95.0)

        # Risk would want 1000 shares ($100k), but max position is 10%
        # Max shares = (100000 * 0.10) / 100 = 1000
        assert result.notional_value <= 10000.0

    def test_percent_risk_minimum_position(self):
        """Test minimum position value check."""
        sizer = FixedFractionalSizing(
            risk_percent=0.001, min_position_value=500.0, round_to_lots=False
        )

        result = sizer.percent_risk(account_value=10000, entry_price=100.0, stop_price=95.0)

        # Risk = 10, shares = 2, notional = 200 < 500 minimum
        assert result.shares == 0

    def test_percent_risk_invalid_prices_raises(self):
        """Test invalid prices raise ValueError."""
        sizer = FixedFractionalSizing()

        with pytest.raises(ValueError, match="Prices must be positive"):
            sizer.percent_risk(account_value=100000, entry_price=-150.0, stop_price=145.0)

        with pytest.raises(ValueError, match="Entry and stop cannot be equal"):
            sizer.percent_risk(account_value=100000, entry_price=150.0, stop_price=150.0)

    def test_percent_equity_basic(self):
        """Test basic percent equity calculation."""
        sizer = FixedFractionalSizing(round_to_lots=False)

        result = sizer.percent_equity(
            account_value=100000, entry_price=150.0, allocation_percent=0.10
        )

        # 10% of 100k = 10k notional
        # Shares = 10000 / 150 = 66.67
        assert result.shares == pytest.approx(66.6666, abs=0.01)
        assert result.notional_value == pytest.approx(10000.0)
        assert result.risk_percent == 0.10
        assert result.method == SizingMethod.PERCENT_EQUITY

    def test_percent_equity_respects_max_position(self):
        """Test percent equity respects max position limit."""
        sizer = FixedFractionalSizing(max_position_percent=0.15, round_to_lots=False)

        result = sizer.percent_equity(
            account_value=100000, entry_price=150.0, allocation_percent=0.25
        )

        # Requested 25%, but max is 15%
        assert result.risk_percent == 0.15

    def test_percent_volatility_basic(self):
        """Test basic percent volatility calculation."""
        sizer = FixedFractionalSizing(
            risk_percent=0.01, max_position_percent=0.30, round_to_lots=False
        )

        result = sizer.percent_volatility(
            account_value=100000, entry_price=150.0, atr=5.0, atr_multiplier=2.0
        )

        # Stop distance = 5.0 * 2.0 = 10.0
        # Risk amount = 100000 * 0.01 = 1000
        # Shares = 1000 / 10 = 100
        assert result.shares == pytest.approx(100.0)
        assert result.risk_amount == pytest.approx(1000.0)
        assert result.method == SizingMethod.PERCENT_VOLATILITY

    def test_percent_volatility_invalid_atr_raises(self):
        """Test invalid ATR raises ValueError."""
        sizer = FixedFractionalSizing()

        with pytest.raises(ValueError, match="ATR must be positive"):
            sizer.percent_volatility(account_value=100000, entry_price=150.0, atr=-5.0)


class TestFixedRatioSizing:
    """Test Ryan Jones Fixed Ratio sizing."""

    def test_initialization(self):
        """Test initialization with default parameters."""
        sizer = FixedRatioSizing()

        assert sizer.delta == 5000
        assert sizer.starting_units == 1
        assert sizer.max_units == 100

    def test_calculate_units_basic(self):
        """Test basic units calculation."""
        sizer = FixedRatioSizing(delta=5000, starting_units=1)

        # Formula: N = floor(sqrt(2*E/D + 0.25) - 0.5)
        # E=10000, D=5000: floor(sqrt(4 + 0.25) - 0.5) = floor(1.56) = 1
        units = sizer.calculate_units(account_value=10000)
        assert units == 1

        # E=50000: floor(sqrt(20 + 0.25) - 0.5) = floor(4.0) = 4
        units = sizer.calculate_units(account_value=50000)
        assert units == 4

    def test_calculate_units_respects_min(self):
        """Test that starting_units is minimum."""
        sizer = FixedRatioSizing(delta=5000, starting_units=2)

        units = sizer.calculate_units(account_value=1000)  # Very small account
        assert units >= 2

    def test_calculate_units_respects_max(self):
        """Test that max_units is maximum."""
        sizer = FixedRatioSizing(delta=5000, max_units=10)

        units = sizer.calculate_units(account_value=1000000)  # Very large account
        assert units <= 10

    def test_calculate_units_zero_equity(self):
        """Test zero equity returns zero units."""
        sizer = FixedRatioSizing()

        units = sizer.calculate_units(account_value=0)
        assert units == 0

    def test_calculate_position(self):
        """Test full position calculation."""
        sizer = FixedRatioSizing(delta=5000)

        position = sizer.calculate_position(
            account_value=25000, entry_price=150.0, contract_value=1.0
        )

        assert "units" in position
        assert "shares" in position
        assert "notional" in position
        assert "next_level" in position
        assert "prev_level" in position
        assert position["units"] > 0

    def test_next_level_equity(self):
        """Test calculation of next equity level."""
        sizer = FixedRatioSizing(delta=5000)

        # For 2 units, next level (3 units) requires delta*3*4/2 = 30000
        next_level = sizer._next_level_equity(current_units=2)
        assert next_level == 30000

    def test_prev_level_equity(self):
        """Test calculation of previous equity level."""
        sizer = FixedRatioSizing(delta=5000)

        # For 3 units, prev level (2 units) is delta*2*1/2 = 5000
        prev_level = sizer._prev_level_equity(current_units=3)
        assert prev_level == 15000


class TestAntiMartingale:
    """Test AntiMartingale position sizing."""

    def test_initialization(self):
        """Test initialization with default parameters."""
        am = AntiMartingale()

        assert am.base_risk == 0.01
        assert am.current_risk == 0.01
        assert am.consecutive_wins == 0
        assert am.consecutive_losses == 0

    def test_update_after_win(self):
        """Test risk increase after win."""
        am = AntiMartingale(base_risk=0.01, win_increase=0.005, max_risk=0.03)

        new_risk = am.update(trade_result=100.0)  # Win

        assert new_risk > 0.01  # Increased
        assert am.consecutive_wins == 1
        assert am.consecutive_losses == 0

    def test_update_after_loss(self):
        """Test risk decrease after loss."""
        am = AntiMartingale(base_risk=0.01, loss_decrease=0.003, min_risk=0.005)

        new_risk = am.update(trade_result=-100.0)  # Loss

        assert new_risk < 0.01  # Decreased
        assert am.consecutive_losses == 1
        assert am.consecutive_wins == 0

    def test_update_respects_max_risk(self):
        """Test that max risk limit is respected."""
        am = AntiMartingale(base_risk=0.01, max_risk=0.02)

        # Multiple wins
        for _ in range(10):
            am.update(trade_result=100.0)

        assert am.current_risk <= 0.02

    def test_update_respects_min_risk(self):
        """Test that min risk limit is respected."""
        am = AntiMartingale(base_risk=0.01, min_risk=0.003)

        # Multiple losses
        for _ in range(10):
            am.update(trade_result=-100.0)

        assert am.current_risk >= 0.003

    def test_reset(self):
        """Test reset to base risk."""
        am = AntiMartingale(base_risk=0.01)

        # Cause some changes
        am.update(100.0)
        am.update(100.0)
        assert am.current_risk != 0.01

        # Reset
        am.reset()
        assert am.current_risk == 0.01
        assert am.consecutive_wins == 0
        assert am.consecutive_losses == 0


class TestPositionSizingEngine:
    """Test unified PositionSizingEngine."""

    def test_initialization(self):
        """Test initialization with default parameters."""
        engine = PositionSizingEngine(account_value=100000)

        assert engine.account_value == 100000
        assert engine.default_method == SizingMethod.PERCENT_RISK
        assert engine.default_risk == 0.01

    def test_calculate_percent_risk(self):
        """Test calculation with percent risk method."""
        engine = PositionSizingEngine(account_value=100000, default_risk=0.01)

        result = engine.calculate(
            entry_price=150.0,
            stop_price=145.0,
            method=SizingMethod.PERCENT_RISK,
        )

        assert result.method == SizingMethod.PERCENT_RISK
        assert result.shares > 0

    def test_calculate_percent_risk_no_stop_raises(self):
        """Test that percent risk without stop raises ValueError."""
        engine = PositionSizingEngine(account_value=100000)

        with pytest.raises(ValueError, match="Stop price required"):
            engine.calculate(entry_price=150.0, method=SizingMethod.PERCENT_RISK)

    def test_calculate_percent_equity(self):
        """Test calculation with percent equity method."""
        engine = PositionSizingEngine(account_value=100000)
        # Disable lot rounding to avoid rounding to 0 shares
        engine.fixed_fractional.round_to_lots = False

        result = engine.calculate(entry_price=150.0, method=SizingMethod.PERCENT_EQUITY)

        assert result.method == SizingMethod.PERCENT_EQUITY
        assert result.shares > 0

    def test_calculate_percent_volatility(self):
        """Test calculation with percent volatility method."""
        engine = PositionSizingEngine(account_value=100000)

        result = engine.calculate(
            entry_price=150.0, atr=5.0, method=SizingMethod.PERCENT_VOLATILITY
        )

        assert result.method == SizingMethod.PERCENT_VOLATILITY
        assert result.shares > 0

    def test_calculate_percent_volatility_no_atr_raises(self):
        """Test that percent volatility without ATR raises ValueError."""
        engine = PositionSizingEngine(account_value=100000)

        with pytest.raises(ValueError, match="ATR required"):
            engine.calculate(entry_price=150.0, method=SizingMethod.PERCENT_VOLATILITY)

    def test_calculate_unknown_method_raises(self):
        """Test that unknown method raises ValueError."""
        engine = PositionSizingEngine(account_value=100000)

        with pytest.raises(ValueError, match="Unknown method"):
            engine.calculate(entry_price=150.0, method=SizingMethod.OPTIMAL_F)

    def test_calculate_uses_default_method(self):
        """Test that default method is used when not specified."""
        engine = PositionSizingEngine(
            account_value=100000, default_method=SizingMethod.PERCENT_RISK
        )

        result = engine.calculate(entry_price=150.0, stop_price=145.0)

        assert result.method == SizingMethod.PERCENT_RISK

    def test_update_account(self):
        """Test account value update."""
        engine = PositionSizingEngine(account_value=100000)

        engine.update_account(new_value=120000)
        assert engine.account_value == 120000
