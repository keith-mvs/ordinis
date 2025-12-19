"""
Growth Model for SignalCore.

This model generates trading signals based on fundamental growth metrics
such as Revenue Growth, EPS Growth, and Margin Expansion.
"""

from datetime import datetime

import numpy as np
import pandas as pd

from ordinis.engines.signalcore.core.model import Model, ModelConfig
from ordinis.engines.signalcore.core.signal import Direction, Signal, SignalType


class GrowthModel(Model):
    """
    Fundamental growth model that scores assets based on growth metrics.

    Generates signals based on a composite score of:
    - Revenue Growth (YoY)
    - EPS Growth (YoY)
    - Margin Expansion (Operating Margin change)
    """

    def __init__(self, config: ModelConfig = None):
        if config is None:
            config = ModelConfig(
                model_id="growth_model",
                model_type="fundamental",
                version="1.0.0",
                parameters={
                    "revenue_weight": 0.4,
                    "eps_weight": 0.4,
                    "margin_weight": 0.2,
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
        required_fundamentals = ["revenue_growth", "eps_growth", "margin_expansion"]
        missing = [col for col in required_fundamentals if col not in data.columns]

        if missing:
            return False, f"Missing fundamental columns: {missing}"

        return True, ""

    async def generate(self, data: pd.DataFrame, timestamp: datetime) -> Signal:
        """
        Generate growth signal.
        """
        # Get the latest row
        try:
            current_data = data.loc[timestamp]
        except KeyError:
            current_data = data.iloc[-1]

        # Extract metrics
        rev_growth = current_data.get("revenue_growth", np.nan)
        eps_growth = current_data.get("eps_growth", np.nan)
        margin_exp = current_data.get("margin_expansion", np.nan)

        # Normalize scores (simplified logic: higher is better)
        # Score 0-100 where 100 is High Growth

        # Revenue Growth > 20% is great (100), < 0% is bad (0)
        rev_score = max(0, min(100, rev_growth * 5)) if not np.isnan(rev_growth) else 50

        # EPS Growth > 25% is great (100), < 0% is bad (0)
        eps_score = max(0, min(100, eps_growth * 4)) if not np.isnan(eps_growth) else 50

        # Margin Expansion > 5% is great (100), < -5% is bad (0)
        margin_score = max(0, min(100, (margin_exp + 5) * 10)) if not np.isnan(margin_exp) else 50

        # Composite Score
        weights = self.config.parameters
        composite_score = (
            rev_score * weights.get("revenue_weight", 0.4)
            + eps_score * weights.get("eps_weight", 0.4)
            + margin_score * weights.get("margin_weight", 0.2)
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
                "revenue_score": rev_score,
                "eps_score": eps_score,
                "margin_score": margin_score,
                "raw_revenue_growth": rev_growth,
                "raw_eps_growth": eps_growth,
                "raw_margin_expansion": margin_exp,
            },
        )
