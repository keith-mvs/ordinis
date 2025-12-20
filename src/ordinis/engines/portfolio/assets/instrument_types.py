"""
Multi-Asset Support - Instrument types and handlers.

Provides extensible framework for different asset classes:
- Equities (stocks, ETFs)
- Futures (margin, contract specifications)
- Options (Greeks, expiration)
- Crypto (24/7 trading, fractional quantities)

Gap Addressed: No multi-asset support (futures margin, options Greeks).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
from enum import Enum
import logging
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ============================================================================
# Instrument Type Enums
# ============================================================================


class InstrumentType(str, Enum):
    """Supported instrument types."""

    EQUITY = "equity"
    ETF = "etf"
    FUTURE = "future"
    OPTION = "option"
    CRYPTO = "crypto"
    FOREX = "forex"
    BOND = "bond"
    INDEX = "index"


class OptionType(str, Enum):
    """Option type (call or put)."""

    CALL = "call"
    PUT = "put"


class OptionStyle(str, Enum):
    """Option exercise style."""

    AMERICAN = "american"
    EUROPEAN = "european"


class FuturesMonthCode(str, Enum):
    """Standard futures month codes."""

    F = "F"  # January
    G = "G"  # February
    H = "H"  # March
    J = "J"  # April
    K = "K"  # May
    M = "M"  # June
    N = "N"  # July
    Q = "Q"  # August
    U = "U"  # September
    V = "V"  # October
    X = "X"  # November
    Z = "Z"  # December


# ============================================================================
# Base Instrument Specification
# ============================================================================


@dataclass
class InstrumentSpec:
    """Base specification for any tradeable instrument."""

    symbol: str
    instrument_type: InstrumentType
    currency: str = "USD"
    exchange: str = ""
    name: str = ""
    is_tradeable: bool = True
    is_marginable: bool = True
    is_shortable: bool = True
    min_order_qty: Decimal = Decimal("1")
    qty_increment: Decimal = Decimal("1")
    min_price_increment: Decimal = Decimal("0.01")
    multiplier: Decimal = Decimal("1")  # Contract multiplier

    def get_notional_value(self, qty: Decimal, price: Decimal) -> Decimal:
        """Calculate notional value of a position.

        Args:
            qty: Position quantity
            price: Current price

        Returns:
            Notional value (qty * price * multiplier)
        """
        return qty * price * self.multiplier


@dataclass
class EquitySpec(InstrumentSpec):
    """Equity (stock) specification."""

    sector: str = ""
    industry: str = ""
    market_cap: Decimal = Decimal("0")
    avg_daily_volume: Decimal = Decimal("0")
    dividend_yield: Decimal = Decimal("0")

    def __post_init__(self) -> None:
        if not self.instrument_type:
            self.instrument_type = InstrumentType.EQUITY


@dataclass
class ETFSpec(InstrumentSpec):
    """ETF specification."""

    underlying_index: str = ""
    expense_ratio: Decimal = Decimal("0")
    aum: Decimal = Decimal("0")  # Assets under management
    inception_date: date | None = None

    def __post_init__(self) -> None:
        if not self.instrument_type:
            self.instrument_type = InstrumentType.ETF


# ============================================================================
# Futures Specification
# ============================================================================


@dataclass
class FuturesSpec(InstrumentSpec):
    """Futures contract specification."""

    underlying: str = ""
    contract_size: Decimal = Decimal("1")
    tick_size: Decimal = Decimal("0.01")
    tick_value: Decimal = Decimal("0")  # $ per tick
    expiration_date: date | None = None
    first_notice_date: date | None = None
    last_trading_date: date | None = None
    settlement_type: str = "cash"  # "cash" or "physical"
    margin_initial: Decimal = Decimal("0")
    margin_maintenance: Decimal = Decimal("0")
    product_code: str = ""

    def __post_init__(self) -> None:
        if not self.instrument_type:
            self.instrument_type = InstrumentType.FUTURE
        if self.tick_value == 0 and self.tick_size > 0:
            self.tick_value = self.tick_size * self.contract_size

    @property
    def days_to_expiry(self) -> int | None:
        """Calculate days until expiration."""
        if not self.expiration_date:
            return None
        today = date.today()
        return (self.expiration_date - today).days

    @property
    def is_near_expiry(self, threshold_days: int = 5) -> bool:
        """Check if contract is near expiration."""
        days = self.days_to_expiry
        return days is not None and days <= threshold_days

    def get_margin_requirement(self, qty: Decimal, is_initial: bool = True) -> Decimal:
        """Calculate margin requirement.

        Args:
            qty: Position quantity
            is_initial: True for initial margin, False for maintenance

        Returns:
            Total margin required
        """
        margin = self.margin_initial if is_initial else self.margin_maintenance
        return abs(qty) * margin


# ============================================================================
# Options Specification
# ============================================================================


@dataclass
class OptionsGreeks:
    """Option Greeks for risk measurement."""

    delta: float = 0.0  # Price sensitivity to underlying
    gamma: float = 0.0  # Rate of delta change
    theta: float = 0.0  # Time decay ($/day)
    vega: float = 0.0  # Volatility sensitivity
    rho: float = 0.0  # Interest rate sensitivity
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def portfolio_delta(self, qty: int, multiplier: int = 100) -> float:
        """Calculate portfolio delta."""
        return self.delta * qty * multiplier


@dataclass
class OptionsSpec(InstrumentSpec):
    """Options contract specification."""

    underlying: str = ""
    option_type: OptionType = OptionType.CALL
    option_style: OptionStyle = OptionStyle.AMERICAN
    strike: Decimal = Decimal("0")
    expiration_date: date | None = None
    contract_size: int = 100  # Standard options = 100 shares

    # Greeks (updated dynamically)
    greeks: OptionsGreeks = field(default_factory=OptionsGreeks)

    # Implied volatility
    implied_volatility: float = 0.0

    def __post_init__(self) -> None:
        if not self.instrument_type:
            self.instrument_type = InstrumentType.OPTION
        self.multiplier = Decimal(str(self.contract_size))

    @property
    def days_to_expiry(self) -> int | None:
        """Calculate days until expiration."""
        if not self.expiration_date:
            return None
        today = date.today()
        return (self.expiration_date - today).days

    @property
    def is_itm(self) -> bool | None:
        """Check if option is in-the-money (requires underlying price)."""
        # Would need current underlying price to determine
        return None

    def get_intrinsic_value(self, underlying_price: Decimal) -> Decimal:
        """Calculate intrinsic value.

        Args:
            underlying_price: Current price of underlying

        Returns:
            Intrinsic value (non-negative)
        """
        if self.option_type == OptionType.CALL:
            return max(Decimal("0"), underlying_price - self.strike)
        return max(Decimal("0"), self.strike - underlying_price)

    def get_time_value(self, option_price: Decimal, underlying_price: Decimal) -> Decimal:
        """Calculate time value.

        Args:
            option_price: Current option price
            underlying_price: Current underlying price

        Returns:
            Time value (extrinsic value)
        """
        intrinsic = self.get_intrinsic_value(underlying_price)
        return max(Decimal("0"), option_price - intrinsic)


# ============================================================================
# Crypto Specification
# ============================================================================


@dataclass
class CryptoSpec(InstrumentSpec):
    """Cryptocurrency specification."""

    base_currency: str = ""
    quote_currency: str = "USD"
    is_24_7: bool = True
    is_fractional: bool = True
    min_order_value: Decimal = Decimal("1")  # Alpaca: $1 minimum

    def __post_init__(self) -> None:
        if not self.instrument_type:
            self.instrument_type = InstrumentType.CRYPTO
        self.min_order_qty = Decimal("0.00000001")  # Allow fractional
        self.qty_increment = Decimal("0.00000001")


# ============================================================================
# Instrument Handler Protocol
# ============================================================================


@runtime_checkable
class InstrumentHandler(Protocol):
    """Protocol for handling different instrument types."""

    def get_margin_requirement(
        self,
        spec: InstrumentSpec,
        qty: Decimal,
        price: Decimal,
        is_initial: bool = True,
    ) -> Decimal:
        """Calculate margin requirement for a position.

        Args:
            spec: Instrument specification
            qty: Position quantity
            price: Current price
            is_initial: True for initial margin

        Returns:
            Margin required in account currency
        """
        ...

    def get_position_risk(
        self,
        spec: InstrumentSpec,
        qty: Decimal,
        price: Decimal,
        volatility: float,
    ) -> dict[str, float]:
        """Calculate position risk metrics.

        Args:
            spec: Instrument specification
            qty: Position quantity
            price: Current price
            volatility: Annualized volatility

        Returns:
            Risk metrics dict (var, expected_shortfall, etc.)
        """
        ...

    def validate_order(
        self,
        spec: InstrumentSpec,
        qty: Decimal,
        price: Decimal,
    ) -> tuple[bool, str]:
        """Validate an order against instrument constraints.

        Args:
            spec: Instrument specification
            qty: Order quantity
            price: Order price

        Returns:
            Tuple of (valid, reason)
        """
        ...


# ============================================================================
# Concrete Handlers
# ============================================================================


class EquityHandler:
    """Handler for equity positions."""

    def __init__(
        self,
        margin_rate: float = 0.5,  # Reg T 50%
        maintenance_rate: float = 0.25,  # 25% maintenance
    ) -> None:
        self.margin_rate = margin_rate
        self.maintenance_rate = maintenance_rate

    def get_margin_requirement(
        self,
        spec: InstrumentSpec,
        qty: Decimal,
        price: Decimal,
        is_initial: bool = True,
    ) -> Decimal:
        """Calculate equity margin (Reg T: 50% initial, 25% maintenance)."""
        notional = abs(qty) * price
        rate = Decimal(str(self.margin_rate if is_initial else self.maintenance_rate))
        return notional * rate

    def get_position_risk(
        self,
        spec: InstrumentSpec,
        qty: Decimal,
        price: Decimal,
        volatility: float,
    ) -> dict[str, float]:
        """Calculate equity position risk."""
        import math

        notional = float(abs(qty) * price)
        daily_vol = volatility / math.sqrt(252)

        # 1-day 95% VaR
        var_95 = notional * daily_vol * 1.645

        # Expected shortfall (approximate)
        es_95 = var_95 * 1.25

        return {
            "notional": notional,
            "daily_volatility": daily_vol,
            "var_95_1d": var_95,
            "expected_shortfall_95": es_95,
        }

    def validate_order(
        self,
        spec: InstrumentSpec,
        qty: Decimal,
        price: Decimal,
    ) -> tuple[bool, str]:
        """Validate equity order."""
        if qty == 0:
            return False, "Order quantity cannot be zero"

        if price <= 0:
            return False, "Price must be positive"

        if abs(qty) < spec.min_order_qty:
            return False, f"Quantity below minimum ({spec.min_order_qty})"

        # Penny stock warning
        if price < 5:
            return True, "Warning: Penny stock order"

        return True, "Order valid"


class FuturesHandler:
    """Handler for futures positions."""

    def __init__(self, margin_buffer: float = 1.1) -> None:
        """Initialize with margin buffer (10% default)."""
        self.margin_buffer = margin_buffer

    def get_margin_requirement(
        self,
        spec: InstrumentSpec,
        qty: Decimal,
        price: Decimal,
        is_initial: bool = True,
    ) -> Decimal:
        """Calculate futures margin requirement."""
        if not isinstance(spec, FuturesSpec):
            raise TypeError("Expected FuturesSpec")

        return spec.get_margin_requirement(qty, is_initial)

    def get_position_risk(
        self,
        spec: InstrumentSpec,
        qty: Decimal,
        price: Decimal,
        volatility: float,
    ) -> dict[str, float]:
        """Calculate futures position risk."""
        if not isinstance(spec, FuturesSpec):
            raise TypeError("Expected FuturesSpec")

        import math

        notional = float(abs(qty) * price * spec.contract_size)
        daily_vol = volatility / math.sqrt(252)

        var_95 = notional * daily_vol * 1.645
        es_95 = var_95 * 1.25

        # Days to expiry risk
        expiry_risk = 0.0
        if spec.days_to_expiry is not None and spec.days_to_expiry <= 5:
            expiry_risk = notional * 0.02  # 2% roll risk

        return {
            "notional": notional,
            "daily_volatility": daily_vol,
            "var_95_1d": var_95,
            "expected_shortfall_95": es_95,
            "expiry_risk": expiry_risk,
            "days_to_expiry": spec.days_to_expiry or 0,
        }

    def validate_order(
        self,
        spec: InstrumentSpec,
        qty: Decimal,
        price: Decimal,
    ) -> tuple[bool, str]:
        """Validate futures order."""
        if not isinstance(spec, FuturesSpec):
            return False, "Invalid instrument type"

        if qty == 0:
            return False, "Order quantity cannot be zero"

        # Check expiry
        if spec.days_to_expiry is not None and spec.days_to_expiry <= 0:
            return False, "Contract has expired"

        if spec.days_to_expiry is not None and spec.days_to_expiry <= 3:
            return True, "Warning: Near expiration - consider rolling"

        return True, "Order valid"


class OptionsHandler:
    """Handler for options positions."""

    def get_margin_requirement(
        self,
        spec: InstrumentSpec,
        qty: Decimal,
        price: Decimal,
        is_initial: bool = True,
    ) -> Decimal:
        """Calculate options margin requirement.

        Long options: Full premium (no margin)
        Short options: Complex margin calculation
        """
        if not isinstance(spec, OptionsSpec):
            raise TypeError("Expected OptionsSpec")

        notional = abs(qty) * price * spec.multiplier

        # Long options: just the premium
        if qty > 0:
            return notional

        # Short options: simplified margin
        # Real calculation would use SPAN or similar
        return notional * Decimal("1.5")

    def get_position_risk(
        self,
        spec: InstrumentSpec,
        qty: Decimal,
        price: Decimal,
        volatility: float,
    ) -> dict[str, float]:
        """Calculate options position risk using Greeks."""
        if not isinstance(spec, OptionsSpec):
            raise TypeError("Expected OptionsSpec")

        notional = float(abs(qty) * price * spec.multiplier)
        greeks = spec.greeks

        # Delta-based risk
        delta_risk = abs(greeks.delta * float(qty) * float(spec.multiplier))

        # Theta decay (daily)
        theta_decay = abs(greeks.theta * float(qty))

        # Vega risk (1% vol move)
        vega_risk = abs(greeks.vega * float(qty) * 0.01)

        # Gamma risk (1% underlying move)
        gamma_risk = 0.5 * abs(greeks.gamma * float(qty) * float(spec.multiplier)) * 0.01**2

        return {
            "notional": notional,
            "delta_risk": delta_risk,
            "theta_decay_daily": theta_decay,
            "vega_risk_1pct": vega_risk,
            "gamma_risk_1pct": gamma_risk,
            "days_to_expiry": spec.days_to_expiry or 0,
            "implied_volatility": spec.implied_volatility,
        }

    def validate_order(
        self,
        spec: InstrumentSpec,
        qty: Decimal,
        price: Decimal,
    ) -> tuple[bool, str]:
        """Validate options order."""
        if not isinstance(spec, OptionsSpec):
            return False, "Invalid instrument type"

        if qty == 0:
            return False, "Order quantity cannot be zero"

        # Check expiry
        if spec.days_to_expiry is not None and spec.days_to_expiry <= 0:
            return False, "Option has expired"

        # Penny options warning
        if price < 0.10:
            return True, "Warning: Low-priced option - wide bid/ask likely"

        return True, "Order valid"


# ============================================================================
# Instrument Registry
# ============================================================================


class InstrumentRegistry:
    """Registry for instrument specifications and handlers."""

    def __init__(self) -> None:
        self._specs: dict[str, InstrumentSpec] = {}
        self._handlers: dict[InstrumentType, InstrumentHandler] = {
            InstrumentType.EQUITY: EquityHandler(),  # type: ignore
            InstrumentType.ETF: EquityHandler(),  # type: ignore
            InstrumentType.FUTURE: FuturesHandler(),  # type: ignore
            InstrumentType.OPTION: OptionsHandler(),  # type: ignore
        }

    def register_spec(self, spec: InstrumentSpec) -> None:
        """Register an instrument specification.

        Args:
            spec: Instrument specification to register
        """
        self._specs[spec.symbol] = spec
        logger.debug(f"Registered {spec.instrument_type.value}: {spec.symbol}")

    def get_spec(self, symbol: str) -> InstrumentSpec | None:
        """Get instrument specification by symbol.

        Args:
            symbol: Instrument symbol

        Returns:
            Instrument specification or None
        """
        return self._specs.get(symbol)

    def get_handler(self, instrument_type: InstrumentType) -> InstrumentHandler | None:
        """Get handler for an instrument type.

        Args:
            instrument_type: Type of instrument

        Returns:
            Handler for the instrument type
        """
        return self._handlers.get(instrument_type)

    def register_handler(
        self,
        instrument_type: InstrumentType,
        handler: InstrumentHandler,
    ) -> None:
        """Register a custom handler for an instrument type.

        Args:
            instrument_type: Type of instrument
            handler: Handler implementation
        """
        self._handlers[instrument_type] = handler

    def get_margin_requirement(
        self,
        symbol: str,
        qty: Decimal,
        price: Decimal,
        is_initial: bool = True,
    ) -> Decimal:
        """Get margin requirement for a position.

        Args:
            symbol: Instrument symbol
            qty: Position quantity
            price: Current price
            is_initial: True for initial margin

        Returns:
            Margin required
        """
        spec = self.get_spec(symbol)
        if not spec:
            # Default equity margin
            return abs(qty) * price * Decimal("0.5")

        handler = self.get_handler(spec.instrument_type)
        if not handler:
            return abs(qty) * price * Decimal("0.5")

        return handler.get_margin_requirement(spec, qty, price, is_initial)

    def get_position_risk(
        self,
        symbol: str,
        qty: Decimal,
        price: Decimal,
        volatility: float = 0.2,
    ) -> dict[str, float]:
        """Get risk metrics for a position.

        Args:
            symbol: Instrument symbol
            qty: Position quantity
            price: Current price
            volatility: Annualized volatility

        Returns:
            Risk metrics dictionary
        """
        spec = self.get_spec(symbol)
        if not spec:
            # Default equity risk
            handler = EquityHandler()
            return handler.get_position_risk(
                EquitySpec(symbol=symbol, instrument_type=InstrumentType.EQUITY),
                qty,
                price,
                volatility,
            )

        handler = self.get_handler(spec.instrument_type)
        if not handler:
            return {"notional": float(abs(qty) * price)}

        return handler.get_position_risk(spec, qty, price, volatility)


# ============================================================================
# Factory Functions
# ============================================================================


def create_equity_spec(
    symbol: str,
    sector: str = "",
    market_cap: float = 0,
    avg_volume: float = 0,
) -> EquitySpec:
    """Create an equity specification.

    Args:
        symbol: Stock symbol
        sector: GICS sector
        market_cap: Market capitalization
        avg_volume: Average daily volume

    Returns:
        Configured EquitySpec
    """
    return EquitySpec(
        symbol=symbol,
        instrument_type=InstrumentType.EQUITY,
        sector=sector,
        market_cap=Decimal(str(market_cap)),
        avg_daily_volume=Decimal(str(avg_volume)),
    )


def create_futures_spec(
    symbol: str,
    underlying: str,
    expiration_date: date,
    contract_size: float = 1,
    tick_size: float = 0.01,
    initial_margin: float = 0,
    maintenance_margin: float = 0,
) -> FuturesSpec:
    """Create a futures specification.

    Args:
        symbol: Futures symbol
        underlying: Underlying asset
        expiration_date: Contract expiration
        contract_size: Size per contract
        tick_size: Minimum price increment
        initial_margin: Initial margin per contract
        maintenance_margin: Maintenance margin per contract

    Returns:
        Configured FuturesSpec
    """
    return FuturesSpec(
        symbol=symbol,
        instrument_type=InstrumentType.FUTURE,
        underlying=underlying,
        expiration_date=expiration_date,
        contract_size=Decimal(str(contract_size)),
        tick_size=Decimal(str(tick_size)),
        margin_initial=Decimal(str(initial_margin)),
        margin_maintenance=Decimal(str(maintenance_margin)),
    )


def create_option_spec(
    symbol: str,
    underlying: str,
    option_type: OptionType,
    strike: float,
    expiration_date: date,
    greeks: OptionsGreeks | None = None,
) -> OptionsSpec:
    """Create an options specification.

    Args:
        symbol: Option symbol
        underlying: Underlying asset
        option_type: Call or put
        strike: Strike price
        expiration_date: Expiration date
        greeks: Option Greeks

    Returns:
        Configured OptionsSpec
    """
    return OptionsSpec(
        symbol=symbol,
        instrument_type=InstrumentType.OPTION,
        underlying=underlying,
        option_type=option_type,
        strike=Decimal(str(strike)),
        expiration_date=expiration_date,
        greeks=greeks or OptionsGreeks(),
    )
