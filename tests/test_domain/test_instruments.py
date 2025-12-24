"""Tests for domain instruments module."""

from datetime import datetime, timedelta

import pytest

from ordinis.domain.instruments import (
    BaseInstrument,
    FutureContract,
    InstrumentType,
)


class TestInstrumentType:
    """Tests for InstrumentType enum."""

    def test_instrument_type_values(self):
        """Test InstrumentType enum values."""
        assert InstrumentType.EQUITY.value == "equity"
        assert InstrumentType.FUTURE.value == "future"
        assert InstrumentType.OPTION.value == "option"
        assert InstrumentType.CRYPTO.value == "crypto"
        assert InstrumentType.FOREX.value == "forex"

    def test_instrument_type_members(self):
        """Test all InstrumentType members exist."""
        assert len(InstrumentType) == 5


class TestBaseInstrument:
    """Tests for BaseInstrument model."""

    def test_base_instrument_creation(self):
        """Test creating a base instrument with required fields."""
        instrument = BaseInstrument(
            symbol="AAPL",
            type=InstrumentType.EQUITY,
        )
        assert instrument.symbol == "AAPL"
        assert instrument.type == InstrumentType.EQUITY
        assert instrument.exchange is None
        assert instrument.currency == "USD"
        assert instrument.tick_size == 0.01
        assert instrument.lot_size == 1.0

    def test_base_instrument_with_all_fields(self):
        """Test creating a base instrument with all fields."""
        instrument = BaseInstrument(
            symbol="AAPL",
            type=InstrumentType.EQUITY,
            exchange="NASDAQ",
            currency="USD",
            tick_size=0.01,
            lot_size=100.0,
        )
        assert instrument.exchange == "NASDAQ"
        assert instrument.lot_size == 100.0

    def test_base_instrument_crypto(self):
        """Test creating a crypto instrument."""
        instrument = BaseInstrument(
            symbol="BTC/USD",
            type=InstrumentType.CRYPTO,
            exchange="COINBASE",
            tick_size=0.01,
        )
        assert instrument.type == InstrumentType.CRYPTO
        assert instrument.exchange == "COINBASE"

    def test_base_instrument_forex(self):
        """Test creating a forex instrument."""
        instrument = BaseInstrument(
            symbol="EUR/USD",
            type=InstrumentType.FOREX,
            tick_size=0.0001,
            lot_size=100000.0,
        )
        assert instrument.type == InstrumentType.FOREX
        assert instrument.tick_size == 0.0001


class TestFutureContract:
    """Tests for FutureContract model."""

    def test_future_contract_creation(self):
        """Test creating a future contract."""
        expiry = datetime.now() + timedelta(days=30)
        contract = FutureContract(
            symbol="ESZ24",
            expiry_date=expiry,
            multiplier=50.0,
            underlying_symbol="SPX",
        )
        assert contract.symbol == "ESZ24"
        assert contract.type == InstrumentType.FUTURE
        assert contract.multiplier == 50.0
        assert contract.margin_requirement == 0.1
        assert contract.underlying_symbol == "SPX"
        assert contract.settlement_method == "cash"

    def test_future_contract_physical_settlement(self):
        """Test creating a physically settled future."""
        expiry = datetime.now() + timedelta(days=60)
        contract = FutureContract(
            symbol="CLX24",
            expiry_date=expiry,
            multiplier=1000.0,
            underlying_symbol="CL",
            settlement_method="physical",
            margin_requirement=0.05,
        )
        assert contract.settlement_method == "physical"
        assert contract.margin_requirement == 0.05

    def test_days_to_expiry_future(self):
        """Test days_to_expiry calculation for future date."""
        expiry = datetime.now() + timedelta(days=45)
        contract = FutureContract(
            symbol="ESZ24",
            expiry_date=expiry,
            underlying_symbol="SPX",
        )
        # Allow some tolerance for test execution time
        assert 44 <= contract.days_to_expiry <= 46

    def test_days_to_expiry_past(self):
        """Test days_to_expiry returns 0 for expired contract."""
        expiry = datetime.now() - timedelta(days=10)
        contract = FutureContract(
            symbol="ESU24",
            expiry_date=expiry,
            underlying_symbol="SPX",
        )
        assert contract.days_to_expiry == 0

    def test_days_to_expiry_today(self):
        """Test days_to_expiry for contract expiring today."""
        expiry = datetime.now()
        contract = FutureContract(
            symbol="ESU24",
            expiry_date=expiry,
            underlying_symbol="SPX",
        )
        assert contract.days_to_expiry == 0

    def test_future_contract_type_frozen(self):
        """Test that FutureContract type is fixed to FUTURE."""
        expiry = datetime.now() + timedelta(days=30)
        contract = FutureContract(
            symbol="ESZ24",
            expiry_date=expiry,
            underlying_symbol="SPX",
        )
        # Type should always be FUTURE
        assert contract.type == InstrumentType.FUTURE
