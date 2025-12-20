#!/usr/bin/env python
"""
Interactive REPL for Ordinis Development.
Usage: python -i scripts/repl_dev.py
"""

from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from ordinis.runtime import initialize, load_config

print("Loading Ordinis Dev Environment...")

# Load dev configuration
settings = load_config("configs/dev.yaml")
container = initialize(settings)

print("\n=== Ordinis Dev REPL ===")
print("Available objects:")
print("  container  - Main dependency container")
print("  settings   - Loaded configuration")
print("\nEngines:")
if hasattr(container, "signal_engine"):
    print("  container.signal_engine")
if hasattr(container, "risk_engine"):
    print("  container.risk_engine")
if hasattr(container, "execution_engine"):
    print("  container.execution_engine")
if hasattr(container, "portfolio_engine"):
    print("  container.portfolio_engine")
if hasattr(container, "analytics_engine"):
    print("  container.analytics_engine")
if hasattr(container, "cortex"):
    print("  container.cortex")
if hasattr(container, "synapse"):
    print("  container.synapse")
if hasattr(container, "helix"):
    print("  container.helix")

print("\nExample usage:")
print("  >>> await container.helix.generate(messages=[{'role': 'user', 'content': 'Hello'}])")
print("  >>> await container.portfolio_engine.get_state()")
print(
    "\nNote: Use 'await' for async methods (requires python -m asyncio or similar, or just run loops manually)"
)
