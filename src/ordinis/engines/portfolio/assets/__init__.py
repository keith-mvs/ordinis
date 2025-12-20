"""
Multi-Asset Support - Instrument types and handlers.

This module provides extensible framework for different asset classes
including equities, futures, options, and crypto.
"""

from ordinis.engines.portfolio.assets.instrument_types import (
    CryptoSpec,
    EquityHandler,
    EquitySpec,
    ETFSpec,
    FuturesHandler,
    FuturesMonthCode,
    FuturesSpec,
    InstrumentHandler,
    InstrumentRegistry,
    InstrumentSpec,
    InstrumentType,
    OptionsGreeks,
    OptionsHandler,
    OptionsSpec,
    OptionStyle,
    OptionType,
    create_equity_spec,
    create_futures_spec,
    create_option_spec,
)

__all__ = [
    "CryptoSpec",
    "EquityHandler",
    "EquitySpec",
    "ETFSpec",
    "FuturesHandler",
    "FuturesMonthCode",
    "FuturesSpec",
    "InstrumentHandler",
    "InstrumentRegistry",
    "InstrumentSpec",
    "InstrumentType",
    "OptionsGreeks",
    "OptionsHandler",
    "OptionsSpec",
    "OptionStyle",
    "OptionType",
    "create_equity_spec",
    "create_futures_spec",
    "create_option_spec",
]
