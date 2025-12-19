"""Tests for Multi-Asset Support - Instrument types and handlers."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
import pytest

from ordinis.engines.portfolio.assets.instrument_types import (
    CryptoSpec,
    EquityHandler,
    EquitySpec,
    FuturesHandler,
    FuturesSpec,
    InstrumentRegistry,
    InstrumentType,
    OptionsGreeks,
    OptionsHandler,
    OptionsSpec,
    OptionType,
    create_equity_spec,
    create_futures_spec,
    create_option_spec,
)


class TestEquitySpec:
    """Tests for EquitySpec."""

    def test_default_equity(self) -> None:
        """Test default equity specification."""
        spec = EquitySpec(
            symbol="AAPL",
            instrument_type=InstrumentType.EQUITY,
        )
        assert spec.symbol == "AAPL"
        assert spec.instrument_type == InstrumentType.EQUITY
        assert spec.multiplier == Decimal("1")
        assert spec.is_marginable

    def test_notional_value(self) -> None:
        """Test notional value calculation."""
        spec = EquitySpec(
            symbol="AAPL",
            instrument_type=InstrumentType.EQUITY,
        )
        notional = spec.get_notional_value(Decimal("100"), Decimal("150"))
        assert notional == Decimal("15000")

    def test_equity_with_metadata(self) -> None:
        """Test equity with full metadata."""
        spec = EquitySpec(
            symbol="AAPL",
            instrument_type=InstrumentType.EQUITY,
            sector="Technology",
            industry="Consumer Electronics",
            market_cap=Decimal("3000000000000"),
            avg_daily_volume=Decimal("50000000"),
        )
        assert spec.sector == "Technology"
        assert spec.market_cap == Decimal("3000000000000")


class TestFuturesSpec:
    """Tests for FuturesSpec."""

    @pytest.fixture
    def es_future(self) -> FuturesSpec:
        """Create E-mini S&P 500 futures spec."""
        return FuturesSpec(
            symbol="ESH24",
            instrument_type=InstrumentType.FUTURE,
            underlying="ES",
            contract_size=Decimal("50"),
            tick_size=Decimal("0.25"),
            expiration_date=date(2024, 3, 15),
            margin_initial=Decimal("12000"),
            margin_maintenance=Decimal("10000"),
        )

    def test_futures_notional(self, es_future: FuturesSpec) -> None:
        """Test futures notional value calculation."""
        # E-mini: 50 * price
        notional = es_future.get_notional_value(Decimal("1"), Decimal("5000"))
        assert notional == Decimal("250000")  # 1 contract * 5000 * 50

    def test_margin_requirement(self, es_future: FuturesSpec) -> None:
        """Test futures margin calculation."""
        initial = es_future.get_margin_requirement(Decimal("2"), is_initial=True)
        maintenance = es_future.get_margin_requirement(Decimal("2"), is_initial=False)

        assert initial == Decimal("24000")  # 2 * 12000
        assert maintenance == Decimal("20000")  # 2 * 10000

    def test_days_to_expiry(self, es_future: FuturesSpec) -> None:
        """Test days to expiry calculation."""
        days = es_future.days_to_expiry
        assert days is not None
        # Will be negative if past expiration, which is fine for test

    def test_tick_value_calculation(self) -> None:
        """Test tick value auto-calculation."""
        future = FuturesSpec(
            symbol="CLZ24",
            instrument_type=InstrumentType.FUTURE,
            underlying="CL",
            contract_size=Decimal("1000"),  # 1000 barrels
            tick_size=Decimal("0.01"),  # $0.01 per barrel
        )
        # tick_value = tick_size * contract_size = 0.01 * 1000 = $10
        assert future.tick_value == Decimal("10")


class TestOptionsSpec:
    """Tests for OptionsSpec."""

    @pytest.fixture
    def call_option(self) -> OptionsSpec:
        """Create call option spec."""
        return OptionsSpec(
            symbol="AAPL240315C00150000",
            instrument_type=InstrumentType.OPTION,
            underlying="AAPL",
            option_type=OptionType.CALL,
            strike=Decimal("150"),
            expiration_date=date(2024, 3, 15),
            greeks=OptionsGreeks(
                delta=0.55,
                gamma=0.03,
                theta=-0.15,
                vega=0.25,
            ),
        )

    def test_option_multiplier(self, call_option: OptionsSpec) -> None:
        """Test option multiplier is 100."""
        assert call_option.multiplier == Decimal("100")

    def test_intrinsic_value_itm_call(self, call_option: OptionsSpec) -> None:
        """Test intrinsic value for ITM call."""
        intrinsic = call_option.get_intrinsic_value(Decimal("160"))
        assert intrinsic == Decimal("10")  # 160 - 150

    def test_intrinsic_value_otm_call(self, call_option: OptionsSpec) -> None:
        """Test intrinsic value for OTM call."""
        intrinsic = call_option.get_intrinsic_value(Decimal("140"))
        assert intrinsic == Decimal("0")

    def test_time_value(self, call_option: OptionsSpec) -> None:
        """Test time value calculation."""
        option_price = Decimal("12")
        underlying_price = Decimal("155")

        time_value = call_option.get_time_value(option_price, underlying_price)
        # Intrinsic = 155 - 150 = 5
        # Time value = 12 - 5 = 7
        assert time_value == Decimal("7")

    def test_put_intrinsic_value(self) -> None:
        """Test put option intrinsic value."""
        put = OptionsSpec(
            symbol="AAPL240315P00150000",
            underlying="AAPL",
            option_type=OptionType.PUT,
            strike=Decimal("150"),
            expiration_date=date(2024, 3, 15),
        )

        # ITM put
        assert put.get_intrinsic_value(Decimal("140")) == Decimal("10")
        # OTM put
        assert put.get_intrinsic_value(Decimal("160")) == Decimal("0")


class TestCryptoSpec:
    """Tests for CryptoSpec."""

    def test_crypto_spec(self) -> None:
        """Test crypto specification."""
        btc = CryptoSpec(
            symbol="BTC/USD",
            base_currency="BTC",
            quote_currency="USD",
        )

        assert btc.instrument_type == InstrumentType.CRYPTO
        assert btc.is_24_7
        assert btc.is_fractional
        assert btc.min_order_qty == Decimal("0.00000001")


class TestEquityHandler:
    """Tests for EquityHandler."""

    @pytest.fixture
    def handler(self) -> EquityHandler:
        """Create equity handler."""
        return EquityHandler()

    @pytest.fixture
    def spec(self) -> EquitySpec:
        """Create equity spec."""
        return EquitySpec(
            symbol="AAPL",
            instrument_type=InstrumentType.EQUITY,
        )

    def test_margin_requirement(
        self,
        handler: EquityHandler,
        spec: EquitySpec,
    ) -> None:
        """Test Reg T margin calculation."""
        initial = handler.get_margin_requirement(
            spec,
            Decimal("100"),
            Decimal("150"),
            is_initial=True,
        )
        # 100 * 150 * 0.5 = $7,500
        assert initial == Decimal("7500")

        maintenance = handler.get_margin_requirement(
            spec,
            Decimal("100"),
            Decimal("150"),
            is_initial=False,
        )
        # 100 * 150 * 0.25 = $3,750
        assert maintenance == Decimal("3750")

    def test_position_risk(
        self,
        handler: EquityHandler,
        spec: EquitySpec,
    ) -> None:
        """Test equity position risk calculation."""
        risk = handler.get_position_risk(
            spec,
            Decimal("100"),
            Decimal("150"),
            volatility=0.25,
        )

        assert risk["notional"] == pytest.approx(15000)
        assert risk["var_95_1d"] > 0
        assert risk["expected_shortfall_95"] > risk["var_95_1d"]

    def test_validate_order(
        self,
        handler: EquityHandler,
        spec: EquitySpec,
    ) -> None:
        """Test order validation."""
        valid, reason = handler.validate_order(spec, Decimal("100"), Decimal("150"))
        assert valid

        invalid, reason = handler.validate_order(spec, Decimal("0"), Decimal("150"))
        assert not invalid
        assert "zero" in reason.lower()


class TestFuturesHandler:
    """Tests for FuturesHandler."""

    @pytest.fixture
    def handler(self) -> FuturesHandler:
        """Create futures handler."""
        return FuturesHandler()

    @pytest.fixture
    def spec(self) -> FuturesSpec:
        """Create futures spec."""
        return FuturesSpec(
            symbol="ESH24",
            underlying="ES",
            contract_size=Decimal("50"),
            expiration_date=date(2025, 3, 15),
            margin_initial=Decimal("12000"),
            margin_maintenance=Decimal("10000"),
        )

    def test_futures_margin(
        self,
        handler: FuturesHandler,
        spec: FuturesSpec,
    ) -> None:
        """Test futures margin requirement."""
        margin = handler.get_margin_requirement(
            spec,
            Decimal("2"),
            Decimal("5000"),
            is_initial=True,
        )
        assert margin == Decimal("24000")

    def test_futures_risk(
        self,
        handler: FuturesHandler,
        spec: FuturesSpec,
    ) -> None:
        """Test futures risk calculation."""
        risk = handler.get_position_risk(
            spec,
            Decimal("1"),
            Decimal("5000"),
            volatility=0.15,
        )

        # Notional = 1 * 5000 * 50 = 250000
        assert risk["notional"] == pytest.approx(250000)
        assert "days_to_expiry" in risk


class TestOptionsHandler:
    """Tests for OptionsHandler."""

    @pytest.fixture
    def handler(self) -> OptionsHandler:
        """Create options handler."""
        return OptionsHandler()

    @pytest.fixture
    def call(self) -> OptionsSpec:
        """Create call option."""
        return OptionsSpec(
            symbol="AAPL240315C00150000",
            underlying="AAPL",
            option_type=OptionType.CALL,
            strike=Decimal("150"),
            expiration_date=date(2025, 3, 15),
            greeks=OptionsGreeks(delta=0.5, gamma=0.03, theta=-0.15, vega=0.25),
        )

    def test_long_option_margin(
        self,
        handler: OptionsHandler,
        call: OptionsSpec,
    ) -> None:
        """Test long option margin (full premium)."""
        # Long 10 contracts at $5
        margin = handler.get_margin_requirement(
            call,
            Decimal("10"),
            Decimal("5"),
            is_initial=True,
        )
        # 10 * 5 * 100 = $5,000
        assert margin == Decimal("5000")

    def test_options_risk_with_greeks(
        self,
        handler: OptionsHandler,
        call: OptionsSpec,
    ) -> None:
        """Test options risk using Greeks."""
        risk = handler.get_position_risk(
            call,
            Decimal("10"),
            Decimal("5"),
            volatility=0.3,
        )

        assert "delta_risk" in risk
        assert "theta_decay_daily" in risk
        assert "vega_risk_1pct" in risk


class TestInstrumentRegistry:
    """Tests for InstrumentRegistry."""

    @pytest.fixture
    def registry(self) -> InstrumentRegistry:
        """Create registry with sample instruments."""
        registry = InstrumentRegistry()

        registry.register_spec(EquitySpec(
            symbol="AAPL",
            instrument_type=InstrumentType.EQUITY,
            sector="Technology",
        ))
        registry.register_spec(FuturesSpec(
            symbol="ESH24",
            underlying="ES",
            contract_size=Decimal("50"),
            expiration_date=date(2025, 3, 15),
            margin_initial=Decimal("12000"),
            margin_maintenance=Decimal("10000"),
        ))

        return registry

    def test_get_spec(self, registry: InstrumentRegistry) -> None:
        """Test retrieving specs."""
        aapl = registry.get_spec("AAPL")
        assert aapl is not None
        assert aapl.instrument_type == InstrumentType.EQUITY

        es = registry.get_spec("ESH24")
        assert es is not None
        assert es.instrument_type == InstrumentType.FUTURE

    def test_get_margin_unified(self, registry: InstrumentRegistry) -> None:
        """Test unified margin calculation."""
        # Equity margin
        equity_margin = registry.get_margin_requirement(
            "AAPL",
            Decimal("100"),
            Decimal("150"),
        )
        assert equity_margin > 0

        # Futures margin
        futures_margin = registry.get_margin_requirement(
            "ESH24",
            Decimal("1"),
            Decimal("5000"),
        )
        assert futures_margin == Decimal("12000")

    def test_get_position_risk(self, registry: InstrumentRegistry) -> None:
        """Test unified risk calculation."""
        risk = registry.get_position_risk(
            "AAPL",
            Decimal("100"),
            Decimal("150"),
        )
        assert "notional" in risk
        assert "var_95_1d" in risk


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_equity_spec(self) -> None:
        """Test equity spec factory."""
        spec = create_equity_spec(
            "AAPL",
            sector="Technology",
            market_cap=3_000_000_000_000,
        )
        assert spec.symbol == "AAPL"
        assert spec.sector == "Technology"
        assert spec.instrument_type == InstrumentType.EQUITY

    def test_create_futures_spec(self) -> None:
        """Test futures spec factory."""
        spec = create_futures_spec(
            "ESH24",
            underlying="ES",
            expiration_date=date(2025, 3, 15),
            contract_size=50,
            initial_margin=12000,
        )
        assert spec.symbol == "ESH24"
        assert spec.contract_size == Decimal("50")

    def test_create_option_spec(self) -> None:
        """Test option spec factory."""
        spec = create_option_spec(
            "AAPL240315C00150000",
            underlying="AAPL",
            option_type=OptionType.CALL,
            strike=150,
            expiration_date=date(2025, 3, 15),
        )
        assert spec.strike == Decimal("150")
        assert spec.option_type == OptionType.CALL
