#!/usr/bin/env python
"""Check OpenTelemetry imports."""

print("Checking OpenTelemetry imports...")

# Check trace exporter
try:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    print("✓ OTLPSpanExporter")
except ImportError as e:
    print(f"✗ OTLPSpanExporter: {e}")

# Check log exporter (try different import paths)
log_exporter = None
try:
    from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
    print("✓ OTLPLogExporter (from _log_exporter)")
    log_exporter = "old"
except ImportError as e:
    print(f"  OTLPLogExporter (from _log_exporter): {e}")

if not log_exporter:
    try:
        from opentelemetry.exporter.otlp.proto.http.log_exporter import OTLPLogExporter
        print("✓ OTLPLogExporter (from log_exporter)")
        log_exporter = "new"
    except ImportError as e:
        print(f"  OTLPLogExporter (from log_exporter): {e}")

if not log_exporter:
    try:
        from opentelemetry.exporter.otlp.proto.http import OTLPLogExporter
        print("✓ OTLPLogExporter (direct)")
        log_exporter = "direct"
    except ImportError as e:
        print(f"  OTLPLogExporter (direct): {e}")

# List what's available
print("\nAvailable in opentelemetry.exporter.otlp.proto.http:")
import opentelemetry.exporter.otlp.proto.http as m
for name in sorted(dir(m)):
    if not name.startswith('_'):
        print(f"  - {name}")

print(f"\nLog exporter path: {log_exporter}")
