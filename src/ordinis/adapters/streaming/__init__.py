"""
WebSocket streaming adapters for real-time market data.

Provides:
- Stream protocols and data types
- WebSocket connection management with reconnection
- Polygon.io streaming provider
- Finnhub streaming provider
"""

from ordinis.adapters.streaming.finnhub_stream import FinnhubStream
from ordinis.adapters.streaming.polygon_stream import PolygonMarket, PolygonStream
from ordinis.adapters.streaming.stream_protocol import (
    BaseStreamHandler,
    BufferedStreamHandler,
    CallbackStreamHandler,
    StreamBar,
    StreamConfig,
    StreamEventType,
    StreamProvider,
    StreamQuote,
    StreamStatus,
    StreamTrade,
)
from ordinis.adapters.streaming.websocket_manager import WebSocketManager

__all__ = [
    # Handlers
    "BaseStreamHandler",
    "BufferedStreamHandler",
    "CallbackStreamHandler",
    # Providers
    "FinnhubStream",
    "PolygonMarket",
    "PolygonStream",
    # Data classes
    "StreamBar",
    "StreamConfig",
    # Enums
    "StreamEventType",
    # Protocols
    "StreamProvider",
    "StreamQuote",
    "StreamStatus",
    "StreamTrade",
    # Managers
    "WebSocketManager",
]
