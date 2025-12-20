#!/usr/bin/env python
"""
Ordinis Dev System Demo with OpenTelemetry Tracing.

Visualize agent execution in AI Toolkit by viewing traces at http://localhost:4319
Run this demo and open the AI Toolkit trace viewer to see:
- LLM calls (prompts, completions, latency)
- Engine orchestration flow
- Signal generation and risk evaluation
"""
import asyncio
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ordinis.adapters.telemetry import (
    TracingConfig,
    setup_tracing,
    shutdown_tracing,
    get_tracer,
)
from ordinis.runtime import initialize, load_config

# Initialize tracing for visualization
tracer = get_tracer(__name__)


async def main():
    # Setup tracing with AI Toolkit endpoint
    tracing_enabled = setup_tracing(
        TracingConfig(
            service_name="ordinis-demo",
            otlp_endpoint="http://localhost:4319",
            capture_message_content=True,
            enabled=True,
        )
    )
    if tracing_enabled:
        print("[INFO] ✓ Tracing enabled - view traces in AI Toolkit at http://localhost:4319")
    else:
        print("[WARN] Tracing not enabled - traces will not be collected")

    try:
        with tracer.start_as_current_span("demo.main") as span:
            span.set_attribute("demo.type", "dev_system")
            print("[INFO] Starting Ordinis Dev System Demo...")

            # Load dev configuration
            try:
                with tracer.start_as_current_span("demo.load_config") as config_span:
                    settings = load_config("configs/dev.yaml")
                    config_span.set_attribute("config.path", "configs/dev.yaml")
                    print("[INFO] Loaded configs/dev.yaml")
            except Exception as e:
                print(f"[ERROR] Failed to load config: {e}")
                return

            # Initialize container
            try:
                with tracer.start_as_current_span("demo.initialize_container"):
                    container = initialize(settings)
                    print("[INFO] Container initialized")
            except Exception as e:
                print(f"[ERROR] Failed to initialize container: {e}")
                return

            # Start the bus
            if hasattr(container, "bus"):
                with tracer.start_as_current_span("demo.start_bus") as bus_span:
                    await container.bus.start()
                    bus_span.set_attribute("bus.type", settings.bus.type)
                    print(f"[INFO] StreamingBus started ({settings.bus.type})")
            else:
                print("[WARN] No bus found in container")

            # Fire a single market-tick event (or run cycle)
            if hasattr(container, "orchestration"):
                with tracer.start_as_current_span("demo.orchestration_cycle") as orch_span:
                    print("[INFO] Running orchestration cycle...")
                    mock_data = {"AAPL": {"price": 150.0, "volume": 1000}}
                    orch_span.set_attribute("symbols", list(mock_data.keys()))
                    await container.orchestration.run_cycle(data=mock_data)
                    print("[INFO] Cycle complete")
            else:
                print("[WARN] No orchestration engine found")

            # Inspect portfolio
            if hasattr(container, "portfolio_engine"):
                with tracer.start_as_current_span("demo.portfolio_snapshot"):
                    state = await container.portfolio_engine.get_state()
                    print("\n=== Portfolio snapshot ===")
                    print(state)

                    # Generate analytics report
                    if hasattr(container, "analytics_engine"):
                        with tracer.start_as_current_span("demo.analytics_report"):
                            print("\n=== Generating Analytics Report ===")
                            report = await container.analytics_engine.analyze(state)
                            print(report)
            else:
                print("[WARN] No portfolio engine found")

    finally:
        # Shutdown tracing gracefully to flush all spans
        if tracing_enabled:
            shutdown_tracing()
            print("[INFO] Tracing shutdown - all spans flushed")


if __name__ == "__main__":
    asyncio.run(main())
