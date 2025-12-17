"""
RiskGuard Validation CLI.

This module provides a CLI for validating risk rules against test signals.
It helps ensure that risk policies are correctly filtering unsafe trades.

Usage:
    python -m ordinis.engines.riskguard.validate
"""

import argparse
import asyncio
from datetime import UTC, datetime

from ordinis.engines.riskguard.core.engine import RiskGuardEngine
from ordinis.engines.riskguard.core.rules import RiskRule, RuleCategory
from ordinis.engines.signalcore.core.signal import Direction, Signal, SignalType


async def run_validation():
    print("[INFO] Starting RiskGuard Validation...")

    # 1. Initialize Engine
    engine = RiskGuardEngine()
    # await engine.initialize()  # Removed: No initialize method needed

    # 2. Define Test Rules
    # We add a specific rule for testing: Max Position Value < $10,000
    test_rule = RiskRule(
        rule_id="TEST_MAX_VALUE",
        category=RuleCategory.PRE_TRADE,
        name="Test Max Value",
        description="Reject trades over $10k",
        condition="position_value < threshold",
        threshold=10000.0,
        comparison="<",
        action_on_breach="reject",
        severity="high",
        enabled=True,
    )
    engine._rules["TEST_MAX_VALUE"] = test_rule
    print(f"[INFO] Added test rule: {test_rule.name} (Threshold: {test_rule.threshold})")

    # 3. Define Test Signals
    test_cases = [
        {
            "name": "Valid Small Trade",
            "signal": Signal(
                model_id="test_model",
                signal_type=SignalType.ENTRY,
                direction=Direction.LONG,
                score=0.8,
                timestamp=datetime.now(UTC),
                metadata={"current_price": 90.0},  # 90 * 100 = 9000 < 10000 -> PASS
                symbol="TEST",
                probability=0.8,
                expected_return=0.05,
                confidence_interval=(0.04, 0.06),
                model_version="1.0",
            ),
            "expected": "approved",
        },
        {
            "name": "Invalid Large Trade",
            "signal": Signal(
                model_id="test_model",
                signal_type=SignalType.ENTRY,
                direction=Direction.LONG,
                score=0.8,
                timestamp=datetime.now(UTC),
                metadata={"current_price": 2000.0},
                symbol="TEST",
                probability=0.8,
                expected_return=0.05,
                confidence_interval=(0.04, 0.06),
                model_version="1.0",
            ),
            "expected": "rejected",
        },
        {
            "name": "Hold Signal (Should Pass)",
            "signal": Signal(
                model_id="test_model",
                signal_type=SignalType.HOLD,
                direction=Direction.NEUTRAL,
                score=0.0,
                timestamp=datetime.now(UTC),
                metadata={"current_price": 90.0},  # Small enough to pass
                symbol="TEST",
                probability=0.5,
                expected_return=0.0,
                confidence_interval=(0.0, 0.0),
                model_version="1.0",
            ),
            "expected": "approved",  # Holds usually pass pre-trade checks or are ignored
        },
    ]

    # 4. Run Tests
    print("\n[INFO] Running Test Cases...")
    passed_count = 0

    for case in test_cases:
        signal = case["signal"]
        expected = case["expected"]
        name = case["name"]

        # We need to mock the portfolio state or context if rules depend on it
        # For this simple rule, it depends on signal metadata (price) * default size

        approved_signals, rejections = await engine.evaluate([signal])

        is_approved = len(approved_signals) > 0
        result = "approved" if is_approved else "rejected"

        status = "PASS" if result == expected else "FAIL"
        if status == "PASS":
            passed_count += 1

        print(f"[{status}] {name}")
        print(f"  Expected: {expected}, Got: {result}")
        if not is_approved:
            print(f"  Reasons: {rejections}")

    # 5. Summary
    print("\n" + "=" * 40)
    print(f"VALIDATION SUMMARY: {passed_count}/{len(test_cases)} Passed")
    print("=" * 40)


def main():
    parser = argparse.ArgumentParser(description="RiskGuard Validation Tool")
    parser.parse_args()

    asyncio.run(run_validation())


if __name__ == "__main__":
    main()
