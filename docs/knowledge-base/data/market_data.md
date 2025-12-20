---
title: Market Data Adapters
description >
Unified access to market data sources, normalizing data into MarketDataEvent Pydantic models for consistent consumption across engines
applyTo: '**'
---
____

# Module

The `ordinis.adapters.market_data` module provides unified access to various market data sources, normalizing data into `MarketDataEvent` Pydantic models for consistent consumption across engines.

## Supported Adapters

- **AlphaVantage**: Free API for equities, forex, and crypto data.
- **Finnhub**: Real-time and historical data with WebSocket support.
- **Massive**: High-frequency data for institutional use, supporting tick-level granularity and low-latency feeds for equities, futures, and options.
- **TwelveData**: Multi-asset data with global coverage.
- **Alpaca**: Commission-free API for equities and crypto market data, including real-time quotes, historical bars, and news feeds, optimized for algorithmic trading.

## Key Features

- **Normalization**: All adapters return standardized `MarketDataEvent` objects with fields like `symbol`, `timestamp`, `open`, `high`, `low`, `close`, `volume`.
- **Rate Limiting**: Built-in throttling to respect API limits.
- **Retry Logic**: Automatic retries on failures with exponential backoff.
- **Async Support**: Fully asynchronous for non-blocking operations.

## Usage Example

```python
## Integration

Market data flows into the `StreamingBus` as events, consumed by engines like `SignalEngine` for processing.
```

## Integration

Market data flows into the `StreamingBus` as events, consumed by engines like `SignalEngine` for processing.

The `ordinis.adapters.market_data` module provides unified access to various market data sources, normalizing data into `MarketDataEvent` Pydantic models for consistent consumption across engines.

## Supported Adapters

- **AlphaVantage**: Free API for equities, forex, and crypto data.
- **Finnhub**: Real-time and historical data with WebSocket support.
- **Massive**: High-frequency data for institutional use.
- **TwelveData**: Multi-asset data with global coverage.

## Key Features

- **Normalization**: All adapters return standardized `MarketDataEvent` objects with fields like `symbol`, `timestamp`, `open`, `high`, `low`, `close`, `volume`.
- **Rate Limiting**: Built-in throttling to respect API limits.
- **Retry Logic**: Automatic retries on failures with exponential backoff.
- **Async Support**: Fully asynchronous for non-blocking operations.

## Usage Example

```python
from ordinis.adapters.market_data import AlphaVantageAdapter

adapter = AlphaVantageAdapter(api_key="your_key")
data = await adapter.fetch_historical(symbol="AAPL", days=30)
# Returns list of MarketDataEvent
```

## Integration

Market data flows into the `StreamingBus` as events, consumed by engines like `SignalEngine` for processing.
