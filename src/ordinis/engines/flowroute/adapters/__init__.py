"""Broker adapters for FlowRoute."""

from __future__ import annotations

from typing import TYPE_CHECKING

# Lazy imports to avoid requiring all dependencies at import time.
# Use direct imports: `from ordinis.engines.flowroute.adapters.paper import PaperBrokerAdapter`

if TYPE_CHECKING:
    from .alpaca import AlpacaBrokerAdapter
    from .paper import PaperBrokerAdapter

__all__ = ["AlpacaBrokerAdapter", "PaperBrokerAdapter"]


def __getattr__(name: str):
    """Lazy load adapters to avoid import errors with missing dependencies."""
    if name == "PaperBrokerAdapter":
        from .paper import PaperBrokerAdapter

        return PaperBrokerAdapter
    if name == "AlpacaBrokerAdapter":
        from .alpaca import AlpacaBrokerAdapter

        return AlpacaBrokerAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
