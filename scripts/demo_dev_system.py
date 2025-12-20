#!/usr/bin/env python
import asyncio
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from ordinis.runtime import initialize, load_config


async def main():
    print("[INFO] Starting Ordinis Dev System Demo...")

    # Load dev configuration
    try:
        settings = load_config("configs/dev.yaml")
        print("[INFO] Loaded configs/dev.yaml")
    except Exception as e:
        print(f"[ERROR] Failed to load config: {e}")
        return

    # Initialize container
    try:
        container = initialize(settings)
        print("[INFO] Container initialized")
    except Exception as e:
        print(f"[ERROR] Failed to initialize container: {e}")
        return

    # Start the bus
    if hasattr(container, "bus"):
        await container.bus.start()
        print(f"[INFO] StreamingBus started ({settings.bus.type})")
    else:
        print("[WARN] No bus found in container")

    # Fire a single market-tick event (or run cycle)
    if hasattr(container, "orchestration"):
        print("[INFO] Running orchestration cycle...")
        mock_data = {"AAPL": {"price": 150.0, "volume": 1000}}
        await container.orchestration.run_cycle(data=mock_data)
        print("[INFO] Cycle complete")
    else:
        print("[WARN] No orchestration engine found")

    # Inspect portfolio
    if hasattr(container, "portfolio_engine"):
        state = await container.portfolio_engine.get_state()
        print("\n=== Portfolio snapshot ===")
        print(state)

        # Generate analytics report
        if hasattr(container, "analytics_engine"):
            print("\n=== Generating Analytics Report ===")
            report = await container.analytics_engine.analyze(state)
            print(report)
    else:
        print("[WARN] No portfolio engine found")


if __name__ == "__main__":
    asyncio.run(main())
