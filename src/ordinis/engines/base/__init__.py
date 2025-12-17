"""Base engine framework for Ordinis.

This module provides the foundational classes and protocols that all
Ordinis engines must implement for consistent lifecycle management,
governance, and observability.

Standard Engine Template:
    engines/{engine_name}/
    ├── __init__.py              # Public API exports
    ├── core/
    │   ├── __init__.py
    │   ├── engine.py            # Main engine class (extends BaseEngine)
    │   ├── config.py            # Engine config (extends BaseEngineConfig)
    │   └── models.py            # Data models/schemas
    ├── hooks/
    │   ├── __init__.py
    │   └── governance.py        # Engine-specific governance
    └── {domain_specific}/       # Engine-specific folders

Example:
    from ordinis.engines.base import BaseEngine, BaseEngineConfig, HealthStatus

    class MyEngineConfig(BaseEngineConfig):
        custom_setting: str = "default"

    class MyEngine(BaseEngine[MyEngineConfig]):
        async def _do_initialize(self) -> None:
            # Setup resources
            pass

        async def _do_shutdown(self) -> None:
            # Cleanup
            pass

        async def _do_health_check(self) -> HealthStatus:
            return HealthStatus(level=HealthLevel.HEALTHY, message="OK")
"""

# Configuration
from ordinis.engines.base.config import (
    AIEngineConfig,
    BaseEngineConfig,
    DataEngineConfig,
    TradingEngineConfig,
)

# Engine base class
from ordinis.engines.base.engine import BaseEngine

# Governance hooks
from ordinis.engines.base.hooks import (
    BaseGovernanceHook,
    CompositeGovernanceHook,
    Decision,
    GovernanceHook,
    PreflightContext,
    PreflightResult,
)

# Data models
from ordinis.engines.base.models import (
    AuditRecord,
    EngineError,
    EngineMetrics,
    EngineState,
    HealthLevel,
    HealthStatus,
)

# Requirements framework
from ordinis.engines.base.requirements import (
    Requirement,
    RequirementCategory,
    RequirementPriority,
    RequirementRegistry,
    RequirementStatus,
    RequirementVerification,
    verifies,
)

__all__ = [
    "AIEngineConfig",
    "AuditRecord",
    # Engine
    "BaseEngine",
    # Configuration
    "BaseEngineConfig",
    "BaseGovernanceHook",
    "CompositeGovernanceHook",
    "DataEngineConfig",
    "Decision",
    "EngineError",
    "EngineMetrics",
    # Models
    "EngineState",
    # Hooks
    "GovernanceHook",
    "HealthLevel",
    "HealthStatus",
    "PreflightContext",
    "PreflightResult",
    # Requirements
    "Requirement",
    "RequirementCategory",
    "RequirementPriority",
    "RequirementRegistry",
    "RequirementStatus",
    "RequirementVerification",
    "TradingEngineConfig",
    "verifies",
]
