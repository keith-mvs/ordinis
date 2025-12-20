#!/usr/bin/env python
"""Quick test of tracing module import and setup."""
import sys
print(f"Python: {sys.executable}")

try:
    from ordinis.adapters.telemetry.tracing import setup_tracing, TracingConfig
    print("✓ Import successful!")
    
    # Test setup (won't actually connect without collector running)
    config = TracingConfig(
        service_name="ordinis-test",
        otlp_endpoint="http://localhost:4319",
        enabled=True
    )
    print(f"✓ Config created: {config.service_name} -> {config.otlp_endpoint}")
    
    result = setup_tracing(config)
    print(f"✓ Setup returned: {result}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
