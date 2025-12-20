#!/usr/bin/env python
"""Quick test to verify tracing imports work."""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from ordinis.engines.base.engine import BaseEngine
    print("✓ BaseEngine import successful")
except Exception as e:
    print(f"✗ BaseEngine import failed: {e}")
    sys.exit(1)

try:
    from ordinis.adapters.telemetry import TracingConfig, setup_tracing, get_tracer
    print("✓ Telemetry imports successful")
except Exception as e:
    print(f"✗ Telemetry import failed: {e}")
    sys.exit(1)

try:
    from opentelemetry import trace
    tracer = trace.get_tracer("test")
    print("✓ OpenTelemetry import successful")
except Exception as e:
    print(f"✗ OpenTelemetry import failed: {e}")
    sys.exit(1)

print("\n✓ All tracing imports verified successfully!")
print("Ready to run demo_dev_system.py with OTLP endpoint http://localhost:4319")
