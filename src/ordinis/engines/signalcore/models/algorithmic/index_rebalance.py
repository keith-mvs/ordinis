"""
Index Rebalancing Model for SignalCore.

Generates trading signals based on index rebalancing events.
"""

from datetime import datetime

import pandas as pd

from ordinis.engines.signalcore.core.model import Model, ModelConfig
from ordinis.engines.signalcore.core.signal import Direction, Signal, SignalType


class IndexRebalanceModel(Model):
    """
    Event-driven model that exploits index fund rebalancing flows.

    Generates signals when stocks are added/removed from major indices.
    """

    def __init__(self, config: ModelConfig = None):
        if config is None:
            config = ModelConfig(
                model_id="index_rebalance_model",
                model_type="algorithmic",
                version="1.0.0",
                parameters={
                    "pre_event_days": 5,
                    "post_event_days": 1,
                    "addition_strength": 0.8,
                    "deletion_strength": -0.8,
                },
                enabled=True,
                min_data_points=1,
            )
        super().__init__(config)

    def validate(self, data: pd.DataFrame) -> tuple[bool, str]:
        """Validate data contains necessary columns."""
        is_valid, msg = super().validate(data)
        if not is_valid:
            return False, msg

        # Check for event data
        if "index_event" not in data.columns and "rebalance_event" not in data.columns:
            # Will use placeholder logic
            pass

        return True, ""

    async def generate(self, data: pd.DataFrame, timestamp: datetime) -> Signal:
        """Generate index rebalancing signal."""
        try:
            current_data = data.loc[timestamp]
        except KeyError:
            current_data = data.iloc[-1]

        # Check for rebalancing event
        if "index_event" in data.columns:
            event = current_data.get("index_event", "none")
        elif "rebalance_event" in data.columns:
            event = current_data.get("rebalance_event", "none")
        elif len(data) >= 20:
            avg_volume = data["volume"].iloc[-20:-1].mean()
            current_volume = data["volume"].iloc[-1]
            if current_volume > avg_volume * 1.5:
                event = "potential_addition"
            else:
                event = "none"
        else:
            event = "none"

        # Determine signal based on event
        addition_strength = self.config.parameters.get("addition_strength", 0.8)
        deletion_strength = self.config.parameters.get("deletion_strength", -0.8)

        if event in ["addition", "potential_addition"]:
            direction = Direction.LONG
            score = addition_strength
            probability = 0.75
        elif event in ["deletion", "potential_deletion"]:
            direction = Direction.SHORT
            score = deletion_strength
            probability = 0.75
        else:
            direction = Direction.NEUTRAL
            score = 0.0
            probability = 0.5

        return Signal(
            symbol=current_data.get("symbol", "UNKNOWN"),
            timestamp=timestamp,
            signal_type=SignalType.ENTRY if direction != Direction.NEUTRAL else SignalType.HOLD,
            direction=direction,
            probability=probability,
            score=score,
            expected_return=0.0,
            confidence_interval=(0.0, 0.0),
            model_id=self.config.model_id,
            model_version=self.config.version,
            metadata={"event_type": event, "signal_strength": score},
        )
