"""Tests for strategies.base module."""

from datetime import datetime

import pandas as pd
import pytest

from ordinis.application.strategies.base import BaseStrategy
from ordinis.engines.signalcore.core.signal import Direction, Signal, SignalType


# Create concrete implementation for testing
class ConcreteTestStrategy(BaseStrategy):
    """Concrete strategy implementation for testing."""

    def configure(self):
        """Configure test strategy."""
        self.params.setdefault("test_param", "default_value")
        self.params.setdefault("min_bars", 50)
        self.configured = True

    def generate_signal(self, data: pd.DataFrame, timestamp: datetime) -> Signal | None:
        """Generate test signal."""
        is_valid, msg = self.validate_data(data)
        if not is_valid:
            return None

        # Simple test signal generation
        if len(data) >= 100:
            return Signal(
                symbol="TEST",
                timestamp=timestamp,
                signal_type=SignalType.ENTRY,
                direction=Direction.LONG,
                probability=0.75,
                expected_return=0.05,
                confidence_interval=(-0.02, 0.12),
                score=0.75,
                model_id="test-model",
                model_version="1.0.0",
                metadata={"test": True},
            )
        return None

    def get_description(self) -> str:
        """Get test strategy description."""
        return "Test Strategy for unit testing BaseStrategy functionality"


class TestBaseStrategy:
    """Tests for BaseStrategy abstract class."""

    def test_strategy_initialization(self):
        """Test strategy initialization."""
        strategy = ConcreteTestStrategy(name="test-strategy", param1="value1", param2=42)

        assert strategy.name == "test-strategy"
        assert strategy.params["param1"] == "value1"
        assert strategy.params["param2"] == 42
        assert strategy.configured is True

    def test_strategy_initialization_calls_configure(self):
        """Test that __init__ calls configure()."""
        strategy = ConcreteTestStrategy(name="test")

        # configure() should have been called and set default params
        assert strategy.params["test_param"] == "default_value"
        assert strategy.configured is True

    def test_get_required_bars_default(self):
        """Test get_required_bars returns default."""
        strategy = ConcreteTestStrategy(name="test")

        required = strategy.get_required_bars()

        # Should use min_bars from params (set to 50 in configure)
        assert required == 50

    def test_get_required_bars_custom(self):
        """Test get_required_bars with custom value."""
        strategy = ConcreteTestStrategy(name="test", min_bars=200)

        required = strategy.get_required_bars()

        assert required == 200

    def test_validate_data_empty_dataframe(self):
        """Test validation fails for empty DataFrame."""
        strategy = ConcreteTestStrategy(name="test")
        empty_data = pd.DataFrame()

        is_valid, msg = strategy.validate_data(empty_data)

        assert is_valid is False
        assert "empty" in msg.lower()

    def test_validate_data_none(self):
        """Test validation fails for None data."""
        strategy = ConcreteTestStrategy(name="test")

        is_valid, msg = strategy.validate_data(None)

        assert is_valid is False
        assert "empty" in msg.lower()

    def test_validate_data_missing_columns(self):
        """Test validation fails for missing required columns."""
        strategy = ConcreteTestStrategy(name="test")

        # Missing 'volume' column
        data = pd.DataFrame(
            {
                "open": [100, 101, 102],
                "high": [105, 106, 107],
                "low": [99, 100, 101],
                "close": [102, 103, 104],
            }
        )

        is_valid, msg = strategy.validate_data(data)

        assert is_valid is False
        assert "Missing columns" in msg
        assert "volume" in msg

    def test_validate_data_insufficient_bars(self):
        """Test validation fails for insufficient data."""
        strategy = ConcreteTestStrategy(name="test", min_bars=100)

        # Only 50 bars, need 100
        data = pd.DataFrame(
            {
                "open": range(50),
                "high": range(50),
                "low": range(50),
                "close": range(50),
                "volume": range(50),
            }
        )

        is_valid, msg = strategy.validate_data(data)

        assert is_valid is False
        assert "Insufficient data" in msg
        assert "50 < 100" in msg

    def test_validate_data_valid(self):
        """Test validation succeeds for valid data."""
        strategy = ConcreteTestStrategy(name="test", min_bars=50)

        data = pd.DataFrame(
            {
                "open": range(100),
                "high": range(100),
                "low": range(100),
                "close": range(100),
                "volume": range(100),
            }
        )

        is_valid, msg = strategy.validate_data(data)

        assert is_valid is True
        assert msg == ""

    def test_generate_signal_with_insufficient_data(self):
        """Test signal generation returns None for insufficient data."""
        strategy = ConcreteTestStrategy(name="test")

        data = pd.DataFrame(
            {
                "open": range(30),
                "high": range(30),
                "low": range(30),
                "close": range(30),
                "volume": range(30),
            }
        )

        signal = strategy.generate_signal(data, datetime.utcnow())

        assert signal is None

    def test_generate_signal_with_valid_data(self):
        """Test signal generation with valid data."""
        strategy = ConcreteTestStrategy(name="test")

        data = pd.DataFrame(
            {
                "open": range(100),
                "high": range(100),
                "low": range(100),
                "close": range(100),
                "volume": range(100),
            }
        )

        timestamp = datetime.utcnow()
        signal = strategy.generate_signal(data, timestamp)

        assert signal is not None
        assert signal.symbol == "TEST"
        assert signal.timestamp == timestamp
        assert signal.signal_type == SignalType.ENTRY
        assert signal.direction == Direction.LONG

    def test_get_description(self):
        """Test get_description returns string."""
        strategy = ConcreteTestStrategy(name="test")

        description = strategy.get_description()

        assert isinstance(description, str)
        assert len(description) > 0
        assert "Test Strategy" in description

    def test_str_representation(self):
        """Test __str__ method."""
        strategy = ConcreteTestStrategy(name="my-strategy")

        result = str(strategy)

        assert result == "my-strategy Strategy"

    def test_repr_representation(self):
        """Test __repr__ method."""
        strategy = ConcreteTestStrategy(name="my-strategy", param1="value1")

        result = repr(strategy)

        assert "ConcreteTestStrategy" in result
        assert "name='my-strategy'" in result
        assert "param1" in result
        assert "value1" in result

    def test_strategy_with_custom_params(self):
        """Test strategy with various custom parameters."""
        strategy = ConcreteTestStrategy(
            name="custom-test",
            window=20,
            threshold=0.5,
            enable_feature=True,
        )

        assert strategy.params["window"] == 20
        assert strategy.params["threshold"] == 0.5
        assert strategy.params["enable_feature"] is True

    def test_strategy_params_preserved(self):
        """Test that custom params are preserved after initialization."""
        strategy = ConcreteTestStrategy(
            name="test",
            custom_param="custom_value",
            min_bars=75,
        )

        # Custom param should be preserved
        assert strategy.params["custom_param"] == "custom_value"
        # Should override default from configure()
        assert strategy.params["min_bars"] == 75
        # Default from configure() should still be there
        assert strategy.params["test_param"] == "default_value"


class TestBaseStrategyAbstract:
    """Test that abstract methods cannot be instantiated without implementation."""

    def test_cannot_instantiate_base_strategy_directly(self):
        """Test that BaseStrategy cannot be instantiated directly."""

        class IncompleteStrategy(BaseStrategy):
            """Strategy missing required method implementations."""

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteStrategy(name="test")  # type: ignore[abstract]

    def test_missing_configure_method(self):
        """Test error when configure() not implemented."""

        class MissingConfigure(BaseStrategy):
            """Missing configure implementation."""

            def generate_signal(self, data, timestamp):
                return None

            def get_description(self):
                return "Test"

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            MissingConfigure(name="test")  # type: ignore[abstract]

    def test_missing_generate_signal_method(self):
        """Test error when generate_signal() not implemented."""

        class MissingGenerateSignal(BaseStrategy):
            """Missing generate_signal implementation."""

            def configure(self):
                pass

            def get_description(self):
                return "Test"

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            MissingGenerateSignal(name="test")  # type: ignore[abstract]

    def test_missing_get_description_method(self):
        """Test error when get_description() not implemented."""

        class MissingGetDescription(BaseStrategy):
            """Missing get_description implementation."""

            def configure(self):
                pass

            def generate_signal(self, data, timestamp):
                return None

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            MissingGetDescription(name="test")  # type: ignore[abstract]


class TestBaseStrategyValidation:
    """Tests for data validation edge cases."""

    def test_validate_data_extra_columns(self):
        """Test validation passes with extra columns."""
        strategy = ConcreteTestStrategy(name="test", min_bars=10)

        data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=20),
                "open": range(20),
                "high": range(20),
                "low": range(20),
                "close": range(20),
                "volume": range(20),
                "extra_column": range(20),
            }
        )

        is_valid, msg = strategy.validate_data(data)

        assert is_valid is True
        assert msg == ""

    def test_validate_data_exact_required_bars(self):
        """Test validation with exact number of required bars."""
        strategy = ConcreteTestStrategy(name="test", min_bars=50)

        data = pd.DataFrame(
            {
                "open": range(50),
                "high": range(50),
                "low": range(50),
                "close": range(50),
                "volume": range(50),
            }
        )

        is_valid, msg = strategy.validate_data(data)

        assert is_valid is True
        assert msg == ""

    def test_validate_data_one_less_than_required(self):
        """Test validation fails with one less than required bars."""
        strategy = ConcreteTestStrategy(name="test", min_bars=50)

        data = pd.DataFrame(
            {
                "open": range(49),
                "high": range(49),
                "low": range(49),
                "close": range(49),
                "volume": range(49),
            }
        )

        is_valid, msg = strategy.validate_data(data)

        assert is_valid is False
        assert "Insufficient data" in msg
