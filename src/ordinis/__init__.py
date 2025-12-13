"""
Ordinis - AI-driven algorithmic trading system.

Clean Architecture package structure:
- core/: Domain models, protocols, events, container
- application/: Use cases, services, strategies
- adapters/: External system integrations (storage, market data, alerting, telemetry)
- engines/: Business logic engines (cortex, flowroute, proofbench)
- interface/: User interfaces (CLI, dashboard)
- runtime/: Configuration, bootstrap, logging
"""

__version__ = "0.2.0-dev"
__author__ = "Ordinis Development Team"
