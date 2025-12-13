"""
Adapters layer for Ordinis trading system.

I/O implementations (outbound + inbound) that connect to external systems.

Contains:
- storage/: Database and persistence adapters (SQLite, etc.)
- market_data/: Market data provider adapters (Polygon, IEX, Yahoo, etc.)
- alerting/: Notification channel adapters (email, desktop, etc.)
- telemetry/: Monitoring and metrics adapters
- caching/: Data caching layer for reduced API calls
"""

__all__: list[str] = []
