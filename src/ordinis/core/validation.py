"""
Data validation layer.

Validates market data, orders, and other inputs before processing.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
from typing import Any

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Validation issue severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """A single validation issue."""

    field: str
    message: str
    severity: ValidationSeverity
    value: Any = None
    expected: Any = None


@dataclass
class ValidationResult:
    """Result of a validation check."""

    valid: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    data: dict[str, Any] | None = None
    validated_at: datetime = field(default_factory=datetime.utcnow)

    def add_issue(self, issue: ValidationIssue) -> None:
        """Add a validation issue."""
        self.issues.append(issue)
        if issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]:
            self.valid = False

    @property
    def has_errors(self) -> bool:
        """Check if there are any error-level issues."""
        return any(
            i.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]
            for i in self.issues
        )

    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return any(i.severity == ValidationSeverity.WARNING for i in self.issues)


class Validator(ABC):
    """Abstract base validator."""

    @abstractmethod
    def validate(self, data: dict[str, Any]) -> ValidationResult:
        """Validate data and return result."""


class MarketDataValidator(Validator):
    """
    Validates market data (quotes, bars).

    Checks:
    - Required fields present
    - Price reasonableness
    - Volume validity
    - Timestamp freshness
    - OHLC consistency
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.max_price = self.config.get("max_price", 1_000_000)
        self.min_price = self.config.get("min_price", 0.0001)
        self.max_volume = self.config.get("max_volume", 10_000_000_000)
        self.max_age_seconds = self.config.get("max_age_seconds", 300)

    def validate(self, data: dict[str, Any]) -> ValidationResult:
        """Validate market data."""
        result = ValidationResult(valid=True, data=data)

        # Check required fields
        self._check_required_fields(data, result)

        if not result.valid:
            return result

        # Check price reasonableness
        self._check_prices(data, result)

        # Check OHLC consistency if applicable
        self._check_ohlc_consistency(data, result)

        # Check volume
        self._check_volume(data, result)

        # Check timestamp
        self._check_timestamp(data, result)

        return result

    def _check_required_fields(self, data: dict[str, Any], result: ValidationResult) -> None:
        """Check for required fields."""
        required = ["symbol"]

        for field_name in required:
            if field_name not in data or data[field_name] is None:
                result.add_issue(
                    ValidationIssue(
                        field=field_name,
                        message=f"Required field '{field_name}' is missing",
                        severity=ValidationSeverity.ERROR,
                    )
                )

    def _check_prices(self, data: dict[str, Any], result: ValidationResult) -> None:
        """Validate price fields."""
        price_fields = ["last", "bid", "ask", "open", "high", "low", "close"]

        for field_name in price_fields:
            if field_name in data and data[field_name] is not None:
                price = data[field_name]

                if not isinstance(price, (int, float)):
                    result.add_issue(
                        ValidationIssue(
                            field=field_name,
                            message=f"Price must be numeric, got {type(price).__name__}",
                            severity=ValidationSeverity.ERROR,
                            value=price,
                        )
                    )
                    continue

                if price < self.min_price:
                    result.add_issue(
                        ValidationIssue(
                            field=field_name,
                            message=f"Price {price} is below minimum {self.min_price}",
                            severity=ValidationSeverity.ERROR,
                            value=price,
                            expected=f">= {self.min_price}",
                        )
                    )

                if price > self.max_price:
                    result.add_issue(
                        ValidationIssue(
                            field=field_name,
                            message=f"Price {price} exceeds maximum {self.max_price}",
                            severity=ValidationSeverity.ERROR,
                            value=price,
                            expected=f"<= {self.max_price}",
                        )
                    )

        # Check bid/ask spread
        if "bid" in data and "ask" in data:
            bid, ask = data.get("bid"), data.get("ask")
            if bid and ask and bid > ask:
                result.add_issue(
                    ValidationIssue(
                        field="bid_ask",
                        message=f"Bid ({bid}) is greater than ask ({ask})",
                        severity=ValidationSeverity.ERROR,
                        value={"bid": bid, "ask": ask},
                    )
                )

    def _check_ohlc_consistency(self, data: dict[str, Any], result: ValidationResult) -> None:
        """Validate OHLC bar consistency."""
        open_p = data.get("open")
        high = data.get("high")
        low = data.get("low")
        close = data.get("close")

        # Only validate if all OHLC fields present
        if not all(x is not None for x in [open_p, high, low, close]):
            return

        # High must be >= open, close, low
        if high < open_p or high < close or high < low:
            result.add_issue(
                ValidationIssue(
                    field="high",
                    message=f"High ({high}) must be >= open, close, and low",
                    severity=ValidationSeverity.ERROR,
                    value={"open": open_p, "high": high, "low": low, "close": close},
                )
            )

        # Low must be <= open, close, high
        if low > open_p or low > close or low > high:
            result.add_issue(
                ValidationIssue(
                    field="low",
                    message=f"Low ({low}) must be <= open, close, and high",
                    severity=ValidationSeverity.ERROR,
                    value={"open": open_p, "high": high, "low": low, "close": close},
                )
            )

    def _check_volume(self, data: dict[str, Any], result: ValidationResult) -> None:
        """Validate volume field."""
        volume = data.get("volume")

        if volume is not None:
            if not isinstance(volume, (int, float)):
                result.add_issue(
                    ValidationIssue(
                        field="volume",
                        message=f"Volume must be numeric, got {type(volume).__name__}",
                        severity=ValidationSeverity.ERROR,
                        value=volume,
                    )
                )
                return

            if volume < 0:
                result.add_issue(
                    ValidationIssue(
                        field="volume",
                        message=f"Volume ({volume}) cannot be negative",
                        severity=ValidationSeverity.ERROR,
                        value=volume,
                    )
                )

            if volume > self.max_volume:
                result.add_issue(
                    ValidationIssue(
                        field="volume",
                        message=f"Volume ({volume}) exceeds maximum",
                        severity=ValidationSeverity.WARNING,
                        value=volume,
                        expected=f"<= {self.max_volume}",
                    )
                )

    def _check_timestamp(self, data: dict[str, Any], result: ValidationResult) -> None:
        """Validate timestamp freshness."""
        timestamp = data.get("timestamp")

        if timestamp is None:
            return

        # Parse timestamp if string
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except ValueError:
                result.add_issue(
                    ValidationIssue(
                        field="timestamp",
                        message="Invalid timestamp format",
                        severity=ValidationSeverity.WARNING,
                        value=timestamp,
                    )
                )
                return

        # Check freshness
        if isinstance(timestamp, datetime):
            age = (datetime.utcnow() - timestamp.replace(tzinfo=None)).total_seconds()

            if age > self.max_age_seconds:
                result.add_issue(
                    ValidationIssue(
                        field="timestamp",
                        message=f"Data is {age:.0f}s old, exceeds {self.max_age_seconds}s limit",
                        severity=ValidationSeverity.WARNING,
                        value=timestamp.isoformat(),
                        expected=f"< {self.max_age_seconds}s old",
                    )
                )

            if age < 0:
                result.add_issue(
                    ValidationIssue(
                        field="timestamp",
                        message="Timestamp is in the future",
                        severity=ValidationSeverity.WARNING,
                        value=timestamp.isoformat(),
                    )
                )


class OrderValidator(Validator):
    """
    Validates order data before submission.

    Checks:
    - Required fields
    - Valid order type
    - Price limits
    - Quantity limits
    - Symbol validity
    """

    VALID_ORDER_TYPES = ["MARKET", "LIMIT", "STOP", "STOP_LIMIT"]
    VALID_SIDES = ["BUY", "SELL"]
    VALID_TIF = ["DAY", "GTC", "IOC", "FOK", "OPG", "CLS"]

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.max_shares = self.config.get("max_shares", 100_000)
        self.max_value = self.config.get("max_value", 1_000_000)
        self.max_price_deviation = self.config.get("max_price_deviation", 0.10)

    def validate(
        self, data: dict[str, Any], current_price: float | None = None
    ) -> ValidationResult:
        """Validate order data."""
        result = ValidationResult(valid=True, data=data)

        # Check required fields
        self._check_required_fields(data, result)

        if not result.valid:
            return result

        # Validate order type
        self._check_order_type(data, result)

        # Validate side
        self._check_side(data, result)

        # Validate quantity
        self._check_quantity(data, result)

        # Validate prices
        self._check_prices(data, result, current_price)

        # Validate time in force
        self._check_tif(data, result)

        return result

    def _check_required_fields(self, data: dict[str, Any], result: ValidationResult) -> None:
        """Check required order fields."""
        required = ["symbol", "side", "quantity", "order_type"]

        for field_name in required:
            if field_name not in data or data[field_name] is None:
                result.add_issue(
                    ValidationIssue(
                        field=field_name,
                        message=f"Required field '{field_name}' is missing",
                        severity=ValidationSeverity.ERROR,
                    )
                )

    def _check_order_type(self, data: dict[str, Any], result: ValidationResult) -> None:
        """Validate order type."""
        order_type = data.get("order_type", "").upper()

        if order_type not in self.VALID_ORDER_TYPES:
            result.add_issue(
                ValidationIssue(
                    field="order_type",
                    message=f"Invalid order type: {order_type}",
                    severity=ValidationSeverity.ERROR,
                    value=order_type,
                    expected=self.VALID_ORDER_TYPES,
                )
            )

        # Check that limit orders have limit price
        if order_type in ["LIMIT", "STOP_LIMIT"]:
            if data.get("limit_price") is None:
                result.add_issue(
                    ValidationIssue(
                        field="limit_price",
                        message=f"Limit price required for {order_type} orders",
                        severity=ValidationSeverity.ERROR,
                    )
                )

        # Check that stop orders have stop price
        if order_type in ["STOP", "STOP_LIMIT"]:
            if data.get("stop_price") is None:
                result.add_issue(
                    ValidationIssue(
                        field="stop_price",
                        message=f"Stop price required for {order_type} orders",
                        severity=ValidationSeverity.ERROR,
                    )
                )

    def _check_side(self, data: dict[str, Any], result: ValidationResult) -> None:
        """Validate order side."""
        side = data.get("side", "").upper()

        if side not in self.VALID_SIDES:
            result.add_issue(
                ValidationIssue(
                    field="side",
                    message=f"Invalid side: {side}",
                    severity=ValidationSeverity.ERROR,
                    value=side,
                    expected=self.VALID_SIDES,
                )
            )

    def _check_quantity(self, data: dict[str, Any], result: ValidationResult) -> None:
        """Validate order quantity."""
        quantity = data.get("quantity")

        if not isinstance(quantity, (int, float)):
            result.add_issue(
                ValidationIssue(
                    field="quantity",
                    message=f"Quantity must be numeric, got {type(quantity).__name__}",
                    severity=ValidationSeverity.ERROR,
                    value=quantity,
                )
            )
            return

        if quantity <= 0:
            result.add_issue(
                ValidationIssue(
                    field="quantity",
                    message="Quantity must be positive",
                    severity=ValidationSeverity.ERROR,
                    value=quantity,
                )
            )

        if quantity > self.max_shares:
            result.add_issue(
                ValidationIssue(
                    field="quantity",
                    message=f"Quantity ({quantity}) exceeds maximum ({self.max_shares})",
                    severity=ValidationSeverity.ERROR,
                    value=quantity,
                    expected=f"<= {self.max_shares}",
                )
            )

    def _check_prices(
        self,
        data: dict[str, Any],
        result: ValidationResult,
        current_price: float | None,
    ) -> None:
        """Validate order prices."""
        limit_price = data.get("limit_price")
        stop_price = data.get("stop_price")
        quantity = data.get("quantity", 0)

        # Check limit price
        if limit_price is not None:
            if not isinstance(limit_price, (int, float)) or limit_price <= 0:
                result.add_issue(
                    ValidationIssue(
                        field="limit_price",
                        message="Limit price must be a positive number",
                        severity=ValidationSeverity.ERROR,
                        value=limit_price,
                    )
                )

            # Check price deviation from current
            if current_price and limit_price > 0:
                deviation = abs(limit_price - current_price) / current_price
                if deviation > self.max_price_deviation:
                    result.add_issue(
                        ValidationIssue(
                            field="limit_price",
                            message=f"Limit price deviates {deviation:.1%} from current price",
                            severity=ValidationSeverity.WARNING,
                            value=limit_price,
                            expected=f"within {self.max_price_deviation:.0%} of {current_price}",
                        )
                    )

            # Check order value
            order_value = limit_price * quantity
            if order_value > self.max_value:
                result.add_issue(
                    ValidationIssue(
                        field="order_value",
                        message=f"Order value (${order_value:,.2f}) exceeds maximum",
                        severity=ValidationSeverity.ERROR,
                        value=order_value,
                        expected=f"<= ${self.max_value:,.2f}",
                    )
                )

        # Check stop price
        if stop_price is not None:
            if not isinstance(stop_price, (int, float)) or stop_price <= 0:
                result.add_issue(
                    ValidationIssue(
                        field="stop_price",
                        message="Stop price must be a positive number",
                        severity=ValidationSeverity.ERROR,
                        value=stop_price,
                    )
                )

    def _check_tif(self, data: dict[str, Any], result: ValidationResult) -> None:
        """Validate time in force."""
        tif = data.get("time_in_force", "DAY").upper()

        if tif not in self.VALID_TIF:
            result.add_issue(
                ValidationIssue(
                    field="time_in_force",
                    message=f"Invalid time in force: {tif}",
                    severity=ValidationSeverity.ERROR,
                    value=tif,
                    expected=self.VALID_TIF,
                )
            )


class DataValidator:
    """
    Main data validation orchestrator.

    Provides validation for different data types using specialized validators.
    """

    def __init__(self):
        self.validators = {
            "market_data": MarketDataValidator(),
            "order": OrderValidator(),
        }

    def register_validator(self, name: str, validator: Validator) -> None:
        """Register a custom validator."""
        self.validators[name] = validator

    def validate_market_data(self, data: dict[str, Any]) -> ValidationResult:
        """Validate market data."""
        return self.validators["market_data"].validate(data)

    def validate_order(
        self, data: dict[str, Any], current_price: float | None = None
    ) -> ValidationResult:
        """Validate order data."""
        return self.validators["order"].validate(data, current_price)

    def validate(self, data_type: str, data: dict[str, Any], **kwargs) -> ValidationResult:
        """Generic validation method."""
        if data_type not in self.validators:
            return ValidationResult(
                valid=False,
                issues=[
                    ValidationIssue(
                        field="data_type",
                        message=f"Unknown data type: {data_type}",
                        severity=ValidationSeverity.ERROR,
                    )
                ],
            )

        validator = self.validators[data_type]
        return validator.validate(data, **kwargs)
