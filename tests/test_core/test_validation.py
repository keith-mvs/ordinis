"""
Tests for data validation layer.

Tests cover:
- Market data validation (quotes, OHLC bars)
- Order validation
- ValidationResult functionality
- DataValidator orchestrator
"""

from datetime import datetime, timedelta

import pytest

from src.core.validation import (
    DataValidator,
    MarketDataValidator,
    OrderValidator,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
)

# ==================== MARKET DATA VALIDATION TESTS ====================


@pytest.mark.unit
def test_market_data_validator_valid_quote(valid_quote_data):
    """Test validation of valid quote data."""
    validator = MarketDataValidator()
    result = validator.validate(valid_quote_data)

    assert result.valid is True
    assert len(result.issues) == 0
    assert not result.has_errors
    assert not result.has_warnings


@pytest.mark.unit
def test_market_data_validator_missing_symbol():
    """Test validation fails when symbol is missing."""
    validator = MarketDataValidator()
    data = {"bid": 150.0, "ask": 150.10}

    result = validator.validate(data)

    assert result.valid is False
    assert result.has_errors
    assert any("symbol" in i.field for i in result.issues)


@pytest.mark.unit
def test_market_data_validator_bid_ask_crossed():
    """Test validation fails when bid > ask."""
    validator = MarketDataValidator()
    data = {
        "symbol": "AAPL",
        "bid": 150.10,
        "ask": 150.00,  # Bid > Ask (invalid)
    }

    result = validator.validate(data)

    assert result.valid is False
    assert result.has_errors
    assert any("bid_ask" in i.field for i in result.issues)


@pytest.mark.unit
def test_market_data_validator_negative_price():
    """Test validation fails for negative prices."""
    validator = MarketDataValidator()
    data = {
        "symbol": "AAPL",
        "last": -10.0,  # Negative price
    }

    result = validator.validate(data)

    assert result.valid is False
    assert result.has_errors


@pytest.mark.unit
def test_market_data_validator_price_too_high():
    """Test validation fails for unreasonable prices."""
    validator = MarketDataValidator()
    data = {
        "symbol": "AAPL",
        "last": 2_000_000.0,  # Exceeds default max
    }

    result = validator.validate(data)

    assert result.valid is False
    assert result.has_errors


@pytest.mark.unit
def test_market_data_validator_valid_ohlc(valid_ohlc_bar):
    """Test validation of valid OHLC bar."""
    validator = MarketDataValidator()
    result = validator.validate(valid_ohlc_bar)

    assert result.valid is True
    assert len([i for i in result.issues if i.severity == ValidationSeverity.ERROR]) == 0


@pytest.mark.unit
def test_market_data_validator_invalid_ohlc_high(invalid_ohlc_bar):
    """Test validation fails when high is not highest."""
    validator = MarketDataValidator()
    result = validator.validate(invalid_ohlc_bar)

    assert result.valid is False
    assert result.has_errors
    assert any("high" in i.field or "low" in i.field for i in result.issues)


@pytest.mark.unit
def test_market_data_validator_valid_ohlc_all_equal():
    """Test validation allows all equal OHLC prices."""
    validator = MarketDataValidator()
    data = {
        "symbol": "AAPL",
        "open": 150.0,
        "high": 150.0,
        "low": 150.0,
        "close": 150.0,
        "volume": 100,
        "timestamp": datetime.utcnow().timestamp(),
    }

    result = validator.validate(data)

    assert result.valid is True


@pytest.mark.unit
def test_market_data_validator_negative_volume():
    """Test validation fails for negative volume."""
    validator = MarketDataValidator()
    data = {
        "symbol": "AAPL",
        "volume": -1000,  # Negative volume
    }

    result = validator.validate(data)

    assert result.valid is False
    assert result.has_errors
    assert any("volume" in i.field for i in result.issues)


@pytest.mark.unit
def test_market_data_validator_excessive_volume():
    """Test validation warns for excessive volume."""
    validator = MarketDataValidator()
    data = {
        "symbol": "AAPL",
        "volume": 50_000_000_000,  # Exceeds default max
    }

    result = validator.validate(data)

    # Should be warning, not error
    assert result.valid is True  # Warnings don't invalidate
    assert result.has_warnings
    assert any("volume" in i.field for i in result.issues)


@pytest.mark.unit
def test_market_data_validator_stale_timestamp():
    """Test validation warns for stale data."""
    validator = MarketDataValidator({"max_age_seconds": 60})
    old_time = datetime.utcnow() - timedelta(seconds=120)

    data = {
        "symbol": "AAPL",
        "timestamp": old_time,
    }

    result = validator.validate(data)

    assert result.valid is True  # Warnings don't invalidate
    assert result.has_warnings
    assert any("timestamp" in i.field for i in result.issues)


@pytest.mark.unit
def test_market_data_validator_future_timestamp():
    """Test validation warns for future timestamps."""
    validator = MarketDataValidator()
    future_time = datetime.utcnow() + timedelta(seconds=300)

    data = {
        "symbol": "AAPL",
        "timestamp": future_time,
    }

    result = validator.validate(data)

    assert result.valid is True  # Warning only
    assert result.has_warnings


# ==================== ORDER VALIDATION TESTS ====================


@pytest.mark.unit
def test_order_validator_valid_market_order():
    """Test validation of valid market order."""
    validator = OrderValidator()
    order = {
        "symbol": "AAPL",
        "side": "BUY",
        "quantity": 100,
        "order_type": "MARKET",
    }

    result = validator.validate(order)

    assert result.valid is True
    assert not result.has_errors


@pytest.mark.unit
def test_order_validator_valid_limit_order():
    """Test validation of valid limit order."""
    validator = OrderValidator()
    order = {
        "symbol": "AAPL",
        "side": "SELL",
        "quantity": 100,
        "order_type": "LIMIT",
        "limit_price": 150.00,
    }

    result = validator.validate(order)

    assert result.valid is True
    assert not result.has_errors


@pytest.mark.unit
def test_order_validator_missing_required_fields():
    """Test validation fails for missing required fields."""
    validator = OrderValidator()
    order = {
        "symbol": "AAPL",
        # Missing side, quantity, order_type
    }

    result = validator.validate(order)

    assert result.valid is False
    assert result.has_errors
    assert len(result.issues) >= 3  # At least 3 missing fields


@pytest.mark.unit
def test_order_validator_invalid_side():
    """Test validation fails for invalid side."""
    validator = OrderValidator()
    order = {
        "symbol": "AAPL",
        "side": "INVALID",  # Not BUY or SELL
        "quantity": 100,
        "order_type": "MARKET",
    }

    result = validator.validate(order)

    assert result.valid is False
    assert result.has_errors
    assert any("side" in i.field for i in result.issues)


@pytest.mark.unit
def test_order_validator_invalid_order_type():
    """Test validation fails for invalid order type."""
    validator = OrderValidator()
    order = {
        "symbol": "AAPL",
        "side": "BUY",
        "quantity": 100,
        "order_type": "INVALID",  # Not a valid type
    }

    result = validator.validate(order)

    assert result.valid is False
    assert result.has_errors
    assert any("order_type" in i.field for i in result.issues)


@pytest.mark.unit
def test_order_validator_limit_order_missing_price():
    """Test validation fails for limit order without price."""
    validator = OrderValidator()
    order = {
        "symbol": "AAPL",
        "side": "BUY",
        "quantity": 100,
        "order_type": "LIMIT",
        # Missing limit_price
    }

    result = validator.validate(order)

    assert result.valid is False
    assert result.has_errors
    assert any("limit_price" in i.field for i in result.issues)


@pytest.mark.unit
def test_order_validator_stop_order_missing_price():
    """Test validation fails for stop order without stop price."""
    validator = OrderValidator()
    order = {
        "symbol": "AAPL",
        "side": "SELL",
        "quantity": 100,
        "order_type": "STOP",
        # Missing stop_price
    }

    result = validator.validate(order)

    assert result.valid is False
    assert result.has_errors
    assert any("stop_price" in i.field for i in result.issues)


@pytest.mark.unit
def test_order_validator_zero_quantity():
    """Test validation fails for zero quantity."""
    validator = OrderValidator()
    order = {
        "symbol": "AAPL",
        "side": "BUY",
        "quantity": 0,  # Invalid
        "order_type": "MARKET",
    }

    result = validator.validate(order)

    assert result.valid is False
    assert result.has_errors
    assert any("quantity" in i.field for i in result.issues)


@pytest.mark.unit
def test_order_validator_negative_quantity():
    """Test validation fails for negative quantity."""
    validator = OrderValidator()
    order = {
        "symbol": "AAPL",
        "side": "BUY",
        "quantity": -100,  # Invalid
        "order_type": "MARKET",
    }

    result = validator.validate(order)

    assert result.valid is False
    assert result.has_errors


@pytest.mark.unit
def test_order_validator_quantity_exceeds_max():
    """Test validation fails when quantity exceeds maximum."""
    validator = OrderValidator({"max_shares": 1000})
    order = {
        "symbol": "AAPL",
        "side": "BUY",
        "quantity": 5000,  # Exceeds max
        "order_type": "MARKET",
    }

    result = validator.validate(order)

    assert result.valid is False
    assert result.has_errors


@pytest.mark.unit
def test_order_validator_price_deviation_warning():
    """Test validation warns for large price deviation."""
    validator = OrderValidator({"max_price_deviation": 0.05})  # 5% max
    order = {
        "symbol": "AAPL",
        "side": "BUY",
        "quantity": 100,
        "order_type": "LIMIT",
        "limit_price": 200.0,  # 33% above current
    }

    result = validator.validate(order, current_price=150.0)

    assert result.valid is True  # Warning, not error
    assert result.has_warnings
    assert any("limit_price" in i.field for i in result.issues)


@pytest.mark.unit
def test_order_validator_order_value_exceeds_max():
    """Test validation fails when order value exceeds maximum."""
    validator = OrderValidator({"max_value": 10_000})
    order = {
        "symbol": "AAPL",
        "side": "BUY",
        "quantity": 1000,
        "order_type": "LIMIT",
        "limit_price": 150.0,  # Value = $150,000
    }

    result = validator.validate(order)

    assert result.valid is False
    assert result.has_errors
    assert any("order_value" in i.field for i in result.issues)


@pytest.mark.unit
def test_order_validator_invalid_time_in_force():
    """Test validation fails for invalid time in force."""
    validator = OrderValidator()
    order = {
        "symbol": "AAPL",
        "side": "BUY",
        "quantity": 100,
        "order_type": "MARKET",
        "time_in_force": "INVALID",
    }

    result = validator.validate(order)

    assert result.valid is False
    assert result.has_errors
    assert any("time_in_force" in i.field for i in result.issues)


@pytest.mark.unit
def test_order_validator_case_insensitive():
    """Test order validator handles case-insensitive values."""
    validator = OrderValidator()
    order = {
        "symbol": "AAPL",
        "side": "buy",  # Lowercase
        "quantity": 100,
        "order_type": "market",  # Lowercase
    }

    result = validator.validate(order)

    assert result.valid is True  # Should handle case conversion


# ==================== VALIDATION RESULT TESTS ====================


@pytest.mark.unit
def test_validation_result_add_issue():
    """Test adding validation issues."""
    result = ValidationResult(valid=True)

    # Add warning (should not invalidate)
    result.add_issue(
        ValidationIssue(
            field="test", message="Warning message", severity=ValidationSeverity.WARNING
        )
    )

    assert result.valid is True
    assert result.has_warnings
    assert not result.has_errors

    # Add error (should invalidate)
    result.add_issue(
        ValidationIssue(field="test", message="Error message", severity=ValidationSeverity.ERROR)
    )

    assert result.valid is False
    assert result.has_errors


@pytest.mark.unit
def test_validation_result_has_errors_property():
    """Test has_errors property."""
    result = ValidationResult(valid=True)

    assert not result.has_errors

    result.add_issue(
        ValidationIssue(field="test", message="Error", severity=ValidationSeverity.ERROR)
    )

    assert result.has_errors


@pytest.mark.unit
def test_validation_result_has_warnings_property():
    """Test has_warnings property."""
    result = ValidationResult(valid=True)

    assert not result.has_warnings

    result.add_issue(
        ValidationIssue(field="test", message="Warning", severity=ValidationSeverity.WARNING)
    )

    assert result.has_warnings


# ==================== DATA VALIDATOR ORCHESTRATOR TESTS ====================


@pytest.mark.unit
def test_data_validator_validate_market_data():
    """Test DataValidator orchestrator for market data."""
    validator = DataValidator()
    data = {
        "symbol": "AAPL",
        "last": 150.0,
    }

    result = validator.validate_market_data(data)

    assert isinstance(result, ValidationResult)
    assert result.valid is True


@pytest.mark.unit
def test_data_validator_validate_order():
    """Test DataValidator orchestrator for orders."""
    validator = DataValidator()
    order = {
        "symbol": "AAPL",
        "side": "BUY",
        "quantity": 100,
        "order_type": "MARKET",
    }

    result = validator.validate_order(order)

    assert isinstance(result, ValidationResult)
    assert result.valid is True


@pytest.mark.unit
def test_data_validator_generic_validate():
    """Test DataValidator generic validate method."""
    validator = DataValidator()
    data = {
        "symbol": "AAPL",
        "last": 150.0,
    }

    result = validator.validate("market_data", data)

    assert isinstance(result, ValidationResult)
    assert result.valid is True


@pytest.mark.unit
def test_data_validator_unknown_type():
    """Test DataValidator handles unknown data types."""
    validator = DataValidator()
    data = {"test": "data"}

    result = validator.validate("unknown_type", data)

    assert result.valid is False
    assert result.has_errors


@pytest.mark.unit
def test_data_validator_register_custom_validator():
    """Test registering custom validator."""

    class CustomValidator:
        def validate(self, data):
            return ValidationResult(valid=True)

    validator = DataValidator()
    validator.register_validator("custom", CustomValidator())

    result = validator.validate("custom", {})

    assert result.valid is True


# ==================== EDGE CASES ====================


@pytest.mark.unit
def test_validation_non_numeric_price():
    """Test validation handles non-numeric price."""
    validator = MarketDataValidator()
    data = {
        "symbol": "AAPL",
        "last": "not a number",  # String instead of number
    }

    result = validator.validate(data)

    assert result.valid is False
    assert result.has_errors


@pytest.mark.unit
def test_validation_non_numeric_quantity():
    """Test validation handles non-numeric quantity."""
    validator = OrderValidator()
    order = {
        "symbol": "AAPL",
        "side": "BUY",
        "quantity": "100",  # String instead of number
        "order_type": "MARKET",
    }

    result = validator.validate(order)

    assert result.valid is False
    assert result.has_errors
