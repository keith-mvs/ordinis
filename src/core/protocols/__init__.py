"""
Core protocol interfaces for Ordinis trading system.

Defines abstract contracts (ports) for Clean Architecture.
Components implement these protocols to ensure interchangeability.
"""

from core.protocols.broker import BrokerAdapter
from core.protocols.cost_model import CostModel
from core.protocols.event_bus import Event, EventBus
from core.protocols.execution import ExecutionEngine
from core.protocols.fill_model import FillModel
from core.protocols.risk_policy import RiskPolicy

__all__ = [
    "BrokerAdapter",
    "CostModel",
    "Event",
    "EventBus",
    "ExecutionEngine",
    "FillModel",
    "RiskPolicy",
]
