"""
CodeGen Proposal CLI.

This module provides a CLI for generating code proposals based on natural language descriptions.
It simulates an LLM generating a code patch.

Usage:
    python -m ordinis.services.codegen.propose_change --desc "Add a new moving average strategy"
"""

import argparse
import asyncio


async def propose_change(description: str):
    print(f"[INFO] CodeGen Request: '{description}'")

    print("[INFO] Analyzing codebase context...")
    await asyncio.sleep(0.5)

    print("[INFO] Generating solution...")
    await asyncio.sleep(1.0)

    # Mock Proposal
    print("\n" + "=" * 40)
    print("PROPOSED CHANGE")
    print("=" * 40)

    print("File: src/ordinis/engines/signalcore/models/new_strategy.py")
    print("-" * 20)
    print("""
class NewStrategyModel(Model):
    def __init__(self, config):
        super().__init__(config)
        self.period = 50

    async def generate(self, data, timestamp):
        # Implementation of requested strategy
        pass
    """)
    print("-" * 20)
    print("\n[INFO] Proposal generated successfully.")


def main():
    parser = argparse.ArgumentParser(description="CodeGen Proposal Tool")
    parser.add_argument("--desc", type=str, required=True, help="Description of change")

    args = parser.parse_args()

    asyncio.run(propose_change(args.desc))


if __name__ == "__main__":
    main()
