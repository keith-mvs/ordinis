"""
Core protocol interfaces for Ordinis trading system.

Defines abstract contracts (ports) for Clean Architecture.
Components implement these protocols to ensure interchangeability.
"""

from ordinis.core.protocols.broker import BrokerAdapter
from ordinis.core.protocols.cost_model import CostModel
from ordinis.core.protocols.event_bus import Event, EventBus
from ordinis.core.protocols.execution import ExecutionEngine
from ordinis.core.protocols.fill_model import FillModel
from ordinis.core.protocols.risk_policy import RiskPolicy

__all__ = [
    "BrokerAdapter",
    "CostModel",
    "Event",
    "EventBus",
    "ExecutionEngine",
    "FillModel",
    "RiskPolicy",
]
