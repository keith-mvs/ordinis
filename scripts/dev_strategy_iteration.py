import asyncio
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from ordinis.engines.riskguard.core.rules import RiskRule, RuleCategory
from ordinis.engines.signalcore.core.model import ModelConfig
from ordinis.engines.signalcore.models import (
    ADXTrendModel,
    ATRBreakoutModel,
    BollingerBandsModel,
    FundamentalValueModel,
    MACDModel,
    SentimentMomentumModel,
    StatisticalReversionModel,
    VolumeTrendModel,
)
from ordinis.engines.signalcore.models import (
    RSIMeanReversionModel as RSIModel,
)
from ordinis.runtime import initialize, load_config


async def main():
    print("[INFO] Starting Strategy Iteration Session...")

    # Configuration Flags
    ENABLE_STRICT_RISK = False  # Set to True to test risk rejection
    ENABLE_DRAWDOWN_RISK = True  # Enable the new drawdown rule
    SELECTED_STRATEGY = (
        "ALL"  # "BB", "RSI", "MACD", "ATR", "ADX", "STAT", "VOL", "FUND", "SENT", or "ALL"
    )

    # 1. Initialize Dev Environment
    settings = load_config("configs/dev.yaml")
    container = initialize(settings)
    print("[INFO] Dev Container Initialized")

    # 1.5 Initialize Database if enabled
    db = container.get_database_manager()

    try:
        if db:
            print("[INFO] Initializing Database Connection...")
            await db.initialize()
            print("[INFO] Database Connected")

        # 2. Register Strategies
        strategies = []

        if SELECTED_STRATEGY in ["BB", "ALL"]:
            strategies.append(
                BollingerBandsModel(
                    ModelConfig(model_id="bb_v1", model_type="technical", min_data_points=20)
                )
            )

        if SELECTED_STRATEGY in ["RSI", "ALL"]:
            strategies.append(
                RSIModel(ModelConfig(model_id="rsi_v1", model_type="technical", min_data_points=15))
            )

        if SELECTED_STRATEGY in ["MACD", "ALL"]:
            strategies.append(
                MACDModel(
                    ModelConfig(model_id="macd_v1", model_type="technical", min_data_points=30)
                )
            )

        if SELECTED_STRATEGY in ["ATR", "ALL"]:
            strategies.append(
                ATRBreakoutModel(
                    ModelConfig(model_id="atr_v1", model_type="technical", min_data_points=20)
                )
            )

        if SELECTED_STRATEGY in ["ADX", "ALL"]:
            strategies.append(
                ADXTrendModel(
                    ModelConfig(model_id="adx_v1", model_type="technical", min_data_points=30)
                )
            )

        if SELECTED_STRATEGY in ["STAT", "ALL"]:
            strategies.append(
                StatisticalReversionModel(
                    ModelConfig(model_id="stat_v1", model_type="technical", min_data_points=30)
                )
            )

        if SELECTED_STRATEGY in ["VOL", "ALL"]:
            strategies.append(
                VolumeTrendModel(
                    ModelConfig(model_id="vol_v1", model_type="technical", min_data_points=30)
                )
            )

        if SELECTED_STRATEGY in ["FUND", "ALL"]:
            strategies.append(
                FundamentalValueModel(
                    ModelConfig(model_id="fund_v1", model_type="fundamental", min_data_points=1)
                )
            )

        if SELECTED_STRATEGY in ["SENT", "ALL"]:
            strategies.append(
                SentimentMomentumModel(
                    ModelConfig(model_id="sent_v1", model_type="sentiment", min_data_points=1)
                )
            )

        for strat in strategies:
            container.signal_engine._registry.register(strat)

        print(f"[INFO] Registered models: {container.signal_engine.list_models()}")

        # 2.5 Add Risk Rule (Optional)
        if ENABLE_STRICT_RISK:
            print("[INFO] Adding Strict Risk Rule...")
            # from ordinis.engines.riskguard.core.rules import RiskRule
            strict_rule = RiskRule(
                rule_id="DEMO_LIMIT",
                category=RuleCategory.PRE_TRADE,
                name="Demo Limit",
                description="Strict limit for demo",
                condition="position_value < threshold",
                threshold=10000.0,
                comparison="<",
                action_on_breach="reject",
                severity="high",
                enabled=True,
            )
            container.risk_engine._rules["DEMO_LIMIT"] = strict_rule
            print("[INFO] Risk Rule Added: Max Trade Value $10,000")

        if ENABLE_DRAWDOWN_RISK:
            print("[INFO] Adding Drawdown Risk Rule...")
            drawdown_rule = RiskRule(
                rule_id="MAX_DD",
                category=RuleCategory.PRE_TRADE,
                name="Max Drawdown Protection",
                description="Halts entries if drawdown > 10%",
                condition="drawdown",
                threshold=0.10,  # 10%
                comparison="<=",
                action_on_breach="reject",
                severity="critical",
                enabled=True,
            )
            container.risk_engine._rules["MAX_DD"] = drawdown_rule
            print("[INFO] Risk Rule Added: Max Drawdown 10%")

        # 3. Create Mock Data
        # Generate data that triggers signals for all strategies
        # Increase data points to satisfy model warmup requirements (e.g. ADX needs ~60)
        dates = pd.date_range(start="2025-01-01", periods=100, tz="UTC")

        # Create a price series with a trend and volatility
        # Start at 150, trend up to 180 (stronger trend), then drop to 120
        prices = np.linspace(150, 180, 50).tolist() + np.linspace(180, 120, 50).tolist()
        prices = np.array(prices) + np.random.normal(0, 1.0, 100)  # Add more noise

        # Variable volatility for ATR/ADX
        volatility = np.random.uniform(2.0, 5.0, 100)

        # P/E Ratio: Oscillate between 10 (undervalued) and 40 (overvalued)
        pe_ratios = np.linspace(20, 40, 50).tolist() + np.linspace(40, 10, 50).tolist()

        # Sentiment: Oscillate between 0.8 (positive) and 0.2 (negative)
        sentiment_scores = (
            np.linspace(0.8, 0.9, 30).tolist()
            + np.linspace(0.9, 0.2, 40).tolist()
            + np.linspace(0.2, 0.5, 30).tolist()
        )

        mock_df = pd.DataFrame(
            {
                "open": prices,
                "high": prices + volatility,
                "low": prices - volatility,
                "close": prices,
                "volume": np.random.randint(1000, 5000, 100),
                "pe_ratio": pe_ratios,
                "sentiment_score": sentiment_scores,
                "symbol": ["AAPL"] * 100,
            },
            index=dates,
        )

        # 4. Run Cycle for each strategy
        print("\n[INFO] Running Trading Cycles...")

        cycle_data = {"AAPL": mock_df}

        # Debug: Generate signals manually to inspect
        print("\n[DEBUG] Inspecting Signals...")
        batch = await container.signal_engine.generate_batch(cycle_data)
        for sig in batch.signals:
            print(
                f"  [{sig.model_id}] Type={sig.signal_type} Direction={sig.direction} Score={sig.score:.2f}"
            )
            if sig.metadata:
                print(f"    Metadata: {sig.metadata}")

        # We run the orchestration cycle. The SignalEngine will run ALL registered models
        # and return a batch of signals.

        result = await container.orchestration.run_cycle(data=cycle_data)

        print(f"\n[INFO] Cycle Status: {result.status}")
        print(f"[INFO] Signals Approved: {result.signals_approved}")
        print(f"[INFO] Orders Submitted: {result.orders_submitted}")
        print(f"[INFO] Orders Filled: {result.orders_filled}")

        if result.errors:
            print(f"[ERROR] Cycle Errors: {result.errors}")

    finally:
        if db:
            print("[INFO] Shutting down Database Connection...")
            await db.shutdown()
            print("[INFO] Database Disconnected")


if __name__ == "__main__":
    asyncio.run(main())
