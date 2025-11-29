"""
Tests for Signal and SignalBatch classes.
"""

from datetime import datetime

import pytest

from src.engines.signalcore import Direction, Signal, SignalBatch, SignalType


@pytest.mark.unit
def test_signal_creation():
    """Test creating a valid signal."""
    signal = Signal(
        symbol="AAPL",
        timestamp=datetime.now(),
        signal_type=SignalType.ENTRY,
        direction=Direction.LONG,
        probability=0.7,
        expected_return=0.05,
        confidence_interval=(0.02, 0.08),
        score=0.6,
        model_id="test_model",
        model_version="1.0.0",
    )

    assert signal.symbol == "AAPL"
    assert signal.probability == 0.7
    assert signal.score == 0.6
    assert signal.signal_type == SignalType.ENTRY
    assert signal.direction == Direction.LONG


@pytest.mark.unit
def test_signal_validation():
    """Test signal validation."""
    with pytest.raises(ValueError, match="Probability must be in"):
        Signal(
            symbol="AAPL",
            timestamp=datetime.now(),
            signal_type=SignalType.ENTRY,
            direction=Direction.LONG,
            probability=1.5,  # Invalid
            expected_return=0.05,
            confidence_interval=(0.02, 0.08),
            score=0.6,
            model_id="test",
            model_version="1.0.0",
        )


@pytest.mark.unit
def test_signal_is_actionable():
    """Test signal actionability check."""
    # Actionable signal
    signal = Signal(
        symbol="AAPL",
        timestamp=datetime.now(),
        signal_type=SignalType.ENTRY,
        direction=Direction.LONG,
        probability=0.7,
        expected_return=0.05,
        confidence_interval=(0.02, 0.08),
        score=0.6,
        model_id="test",
        model_version="1.0.0",
    )

    assert signal.is_actionable() is True

    # Not actionable (HOLD)
    hold_signal = Signal(
        symbol="AAPL",
        timestamp=datetime.now(),
        signal_type=SignalType.HOLD,
        direction=Direction.NEUTRAL,
        probability=0.7,
        expected_return=0.0,
        confidence_interval=(-0.01, 0.01),
        score=0.1,
        model_id="test",
        model_version="1.0.0",
    )

    assert hold_signal.is_actionable() is False


@pytest.mark.unit
def test_signal_to_dict():
    """Test converting signal to dictionary."""
    signal = Signal(
        symbol="AAPL",
        timestamp=datetime(2024, 1, 1, 12, 0, 0),
        signal_type=SignalType.ENTRY,
        direction=Direction.LONG,
        probability=0.7,
        expected_return=0.05,
        confidence_interval=(0.02, 0.08),
        score=0.6,
        model_id="test",
        model_version="1.0.0",
    )

    signal_dict = signal.to_dict()

    assert signal_dict["symbol"] == "AAPL"
    assert signal_dict["probability"] == 0.7
    assert signal_dict["signal_type"] == "entry"
    assert signal_dict["direction"] == "long"


@pytest.mark.unit
def test_signal_batch_creation():
    """Test creating a signal batch."""
    signals = [
        Signal(
            symbol="AAPL",
            timestamp=datetime.now(),
            signal_type=SignalType.ENTRY,
            direction=Direction.LONG,
            probability=0.7,
            expected_return=0.05,
            confidence_interval=(0.02, 0.08),
            score=0.6,
            model_id="test",
            model_version="1.0.0",
        ),
        Signal(
            symbol="GOOGL",
            timestamp=datetime.now(),
            signal_type=SignalType.EXIT,
            direction=Direction.NEUTRAL,
            probability=0.6,
            expected_return=0.0,
            confidence_interval=(-0.01, 0.01),
            score=-0.4,
            model_id="test",
            model_version="1.0.0",
        ),
    ]

    batch = SignalBatch(
        timestamp=datetime.now(), signals=signals, universe=["AAPL", "GOOGL", "MSFT"]
    )

    assert len(batch.signals) == 2
    assert len(batch.universe) == 3


@pytest.mark.unit
def test_signal_batch_filter_actionable():
    """Test filtering actionable signals."""
    signals = [
        Signal(
            symbol="AAPL",
            timestamp=datetime.now(),
            signal_type=SignalType.ENTRY,
            direction=Direction.LONG,
            probability=0.7,
            expected_return=0.05,
            confidence_interval=(0.02, 0.08),
            score=0.6,
            model_id="test",
            model_version="1.0.0",
        ),
        Signal(
            symbol="GOOGL",
            timestamp=datetime.now(),
            signal_type=SignalType.HOLD,
            direction=Direction.NEUTRAL,
            probability=0.5,
            expected_return=0.0,
            confidence_interval=(-0.01, 0.01),
            score=0.1,
            model_id="test",
            model_version="1.0.0",
        ),
    ]

    batch = SignalBatch(timestamp=datetime.now(), signals=signals, universe=["AAPL", "GOOGL"])

    actionable = batch.filter_actionable()

    assert len(actionable) == 1
    assert actionable[0].symbol == "AAPL"


@pytest.mark.unit
def test_signal_batch_get_by_symbol():
    """Test getting signal by symbol."""
    signal = Signal(
        symbol="AAPL",
        timestamp=datetime.now(),
        signal_type=SignalType.ENTRY,
        direction=Direction.LONG,
        probability=0.7,
        expected_return=0.05,
        confidence_interval=(0.02, 0.08),
        score=0.6,
        model_id="test",
        model_version="1.0.0",
    )

    batch = SignalBatch(timestamp=datetime.now(), signals=[signal], universe=["AAPL"])

    found = batch.get_by_symbol("AAPL")
    assert found is not None
    assert found.symbol == "AAPL"

    not_found = batch.get_by_symbol("GOOGL")
    assert not_found is None
