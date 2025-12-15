"""
Valuation Model for SignalCore.

This model generates trading signals based on fundamental valuation metrics
such as P/E ratio, P/B ratio, and EV/EBITDA.
"""

from datetime import datetime

import numpy as np
import pandas as pd

from ordinis.engines.signalcore.core.model import Model, ModelConfig
from ordinis.engines.signalcore.core.signal import Direction, Signal, SignalType


class ValuationModel(Model):
    """
    Fundamental valuation model that scores assets based on value metrics.

    Generates signals based on a composite score of:
    - Price to Earnings (P/E)
    - Price to Book (P/B)
    - Enterprise Value to EBITDA (EV/EBITDA)
    """

    def __init__(self, config: ModelConfig = None):
        if config is None:
            config = ModelConfig(
                model_id="valuation_model",
                model_type="fundamental",
                version="1.0.0",
                parameters={
                    "pe_weight": 0.4,
                    "pb_weight": 0.3,
                    "evebitda_weight": 0.3,
                    "buy_threshold": 70,
                    "sell_threshold": 30,
                },
                enabled=True,
                min_data_points=1,
            )
        super().__init__(config)

    def validate(self, data: pd.DataFrame) -> tuple[bool, str]:
        """
        Validate that data contains necessary fundamental columns.
        """
        # Base validation
        is_valid, msg = super().validate(data)
        if not is_valid:
            return False, msg

        # Check for fundamental columns
        # Note: In a real scenario, we might calculate these from raw data if not present
        required_fundamentals = ["pe_ratio", "pb_ratio", "ev_ebitda"]
        missing = [col for col in required_fundamentals if col not in data.columns]

        # For now, we'll be lenient if columns are missing but warn
        if missing:
            # If we can't calculate them, we might fail.
            # Here we assume if they are missing we can't run.
            # However, to make it robust for testing with OHLCV only, we might skip
            return False, f"Missing fundamental columns: {missing}"

        return True, ""

    async def generate(self, data: pd.DataFrame, timestamp: datetime) -> Signal:
        """
        Generate valuation signal.
        """
        # In a real implementation, we would fetch fundamental data if not in 'data'
        # For this implementation, we assume 'data' has the pre-calculated metrics
        # or we calculate them from price and fundamental fields if available.

        # Get the latest row
        try:
            current_data = data.loc[timestamp]
        except KeyError:
            # If exact timestamp missing, get latest
            current_data = data.iloc[-1]

        # Extract metrics (handling missing data with defaults or NaNs)
        pe = current_data.get("pe_ratio", np.nan)
        pb = current_data.get("pb_ratio", np.nan)
        evebitda = current_data.get("ev_ebitda", np.nan)

        # Normalize scores (simplified logic: lower is better for value)
        # We'll use a percentile or simple inversion for scoring 0-100
        # This is a placeholder logic. Real logic would compare to sector/history.

        # Score 0-100 where 100 is "Cheap" (Good Value)
        # Example thresholds: PE < 15 is good, > 30 is bad
        pe_score = max(0, min(100, (30 - pe) * (100 / 15))) if not np.isnan(pe) else 50

        # PB < 1.5 is good, > 5 is bad
        pb_score = max(0, min(100, (5 - pb) * (100 / 3.5))) if not np.isnan(pb) else 50

        # EV/EBITDA < 10 is good, > 20 is bad
        ev_score = max(0, min(100, (20 - evebitda) * (100 / 10))) if not np.isnan(evebitda) else 50

        # Composite Score
        weights = self.config.parameters
        composite_score = (
            pe_score * weights.get("pe_weight", 0.4)
            + pb_score * weights.get("pb_weight", 0.3)
            + ev_score * weights.get("evebitda_weight", 0.3)
        )

        # Determine Direction
        buy_threshold = weights.get("buy_threshold", 70)
        sell_threshold = weights.get("sell_threshold", 30)

        if composite_score >= buy_threshold:
            direction = Direction.LONG
            probability = composite_score / 100.0
        elif composite_score <= sell_threshold:
            direction = Direction.SHORT
            probability = (100 - composite_score) / 100.0
        else:
            direction = Direction.NEUTRAL
            probability = 0.5

        # Normalize score to [-1, 1] (50 is neutral)
        normalized_score = (composite_score - 50) / 50.0

        return Signal(
            symbol=current_data.get("symbol", "UNKNOWN"),
            timestamp=timestamp,
            signal_type=SignalType.ENTRY if direction != Direction.NEUTRAL else SignalType.HOLD,
            direction=direction,
            probability=probability,
            score=normalized_score,
            expected_return=0.0,
            confidence_interval=(0.0, 0.0),
            model_id=self.config.model_id,
            model_version=self.config.version,
            metadata={
                "composite_score": composite_score,
                "pe_score": pe_score,
                "pb_score": pb_score,
                "ev_score": ev_score,
                "raw_pe": pe,
                "raw_pb": pb,
                "raw_ev": evebitda,
            },
        )
