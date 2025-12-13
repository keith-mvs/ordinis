"""
Volatility estimation models using arch library (GARCH family).

Reference: https://github.com/Nixtla/statsforecast (GARCH tutorial)
Library: https://arch.readthedocs.io/
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

import numpy as np
import pandas as pd

from ...core.model import Model, ModelConfig
from ...core.signal import Direction, Signal, SignalType

# Optional arch import with graceful degradation
try:
    from arch import arch_model

    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    arch_model = None


@dataclass
class VolatilityForecast:
    """Result of volatility forecast."""

    current_volatility: float  # Annualized
    forecast_volatility: float  # Annualized forecast
    volatility_change: float  # Percentage change
    high_volatility_regime: bool
    model_name: str
    forecast_timestamp: datetime


class GARCHVolatilityModel(Model):
    """
    GARCH-based volatility estimation model.

    Uses GARCH(p,q) for conditional volatility forecasting.
    Supports GARCH, EGARCH, and TGARCH variants.
    """

    def __init__(
        self,
        config: ModelConfig | None = None,
        p: int = 1,
        q: int = 1,
        variant: Literal["GARCH", "EGARCH", "TGARCH"] = "GARCH",
        horizon: int = 5,
        high_vol_threshold: float = 0.25,  # Annualized
    ):
        """
        Initialize GARCH volatility model.

        Args:
            config: Model configuration
            p: ARCH order (number of lagged squared returns)
            q: GARCH order (number of lagged variances)
            variant: GARCH variant (GARCH, EGARCH, TGARCH)
            horizon: Forecast horizon (days)
            high_vol_threshold: Threshold for high volatility regime (annualized)
        """
        if config is None:
            config = ModelConfig(
                model_id=f"{variant.lower()}-volatility",
                model_type="volatility",
                version="1.0.0",
                parameters={"p": p, "q": q, "variant": variant, "horizon": horizon},
                min_data_points=252,  # Need 1 year for volatility estimation
                lookback_period=504,  # 2 years
            )
        super().__init__(config)

        self.p = p
        self.q = q
        self.variant = variant
        self.horizon = horizon
        self.high_vol_threshold = high_vol_threshold

    def generate(self, data: pd.DataFrame, timestamp: datetime) -> Signal:
        """
        Generate trading signal based on volatility forecast.

        High volatility -> reduce position sizes (weak long/short signals)
        Low volatility -> normal position sizes
        Increasing volatility -> cautious
        Decreasing volatility -> opportunity

        Args:
            data: Historical OHLCV data
            timestamp: Current timestamp

        Returns:
            Signal with volatility-adjusted direction
        """
        symbol = data.attrs.get("symbol", "UNKNOWN")

        # Get volatility forecast
        vol_forecast = self._forecast(data)

        # Determine signal based on volatility regime
        if vol_forecast.high_volatility_regime:
            # High vol: reduce exposure, favor neutral
            if vol_forecast.volatility_change > 0.1:
                # Increasing high vol: defensive
                direction = Direction.NEUTRAL
                strength = 0.2
            else:
                # Stable/decreasing high vol: cautious
                direction = Direction.NEUTRAL
                strength = 0.4
        elif vol_forecast.volatility_change < -0.1:
            # Decreasing vol: opportunity
            direction = Direction.LONG
            strength = 0.6
        else:
            direction = Direction.NEUTRAL
            strength = 0.5

        # Confidence inversely related to volatility
        confidence = max(0.1, min(1.0, 1.0 - vol_forecast.forecast_volatility))

        self._last_update = timestamp

        return Signal(
            symbol=symbol,
            timestamp=timestamp,
            signal_type=SignalType.ENTRY if direction != Direction.NEUTRAL else SignalType.HOLD,
            direction=direction,
            probability=confidence,
            expected_return=0.0,  # Volatility model doesn't predict returns
            confidence_interval=(
                -vol_forecast.forecast_volatility,
                vol_forecast.forecast_volatility,
            ),
            score=strength
            if direction == Direction.LONG
            else (-strength if direction == Direction.SHORT else 0.0),
            model_id=self.config.model_id,
            model_version=self.config.version,
            metadata={
                "volatility_model": vol_forecast.model_name,
                "current_volatility": vol_forecast.current_volatility,
                "forecast_volatility": vol_forecast.forecast_volatility,
                "volatility_change": vol_forecast.volatility_change,
                "high_volatility_regime": vol_forecast.high_volatility_regime,
                "horizon": self.horizon,
            },
        )

    def _forecast(self, data: pd.DataFrame) -> VolatilityForecast:
        """
        Run GARCH volatility forecast.

        Args:
            data: Historical OHLCV data

        Returns:
            VolatilityForecast with predictions
        """
        # Calculate returns (percentage)
        returns = data["close"].pct_change().dropna() * 100  # Scale for GARCH

        if not ARCH_AVAILABLE:
            # Fallback: Rolling standard deviation
            return self._fallback_forecast(returns)

        try:
            # Create GARCH model based on variant
            if self.variant == "EGARCH":
                model = arch_model(
                    returns,
                    vol="EGARCH",
                    p=self.p,
                    q=self.q,
                    rescale=True,
                )
            elif self.variant == "TGARCH":
                model = arch_model(
                    returns,
                    vol="GARCH",
                    p=self.p,
                    o=1,  # Threshold term
                    q=self.q,
                    rescale=True,
                )
            else:  # Standard GARCH
                model = arch_model(
                    returns,
                    vol="GARCH",
                    p=self.p,
                    q=self.q,
                    rescale=True,
                )

            # Fit model
            result = model.fit(disp="off", show_warning=False)

            # Get conditional volatility (current)
            current_vol_daily = float(result.conditional_volatility.iloc[-1]) / 100
            current_vol_annual = current_vol_daily * np.sqrt(252)

            # Forecast volatility
            forecast = result.forecast(horizon=self.horizon)
            forecast_var = forecast.variance.iloc[-1].values
            forecast_vol_daily = float(np.sqrt(forecast_var[-1])) / 100
            forecast_vol_annual = forecast_vol_daily * np.sqrt(252)

            # Calculate change
            vol_change = (forecast_vol_annual - current_vol_annual) / current_vol_annual

            return VolatilityForecast(
                current_volatility=current_vol_annual,
                forecast_volatility=forecast_vol_annual,
                volatility_change=vol_change,
                high_volatility_regime=forecast_vol_annual > self.high_vol_threshold,
                model_name=f"{self.variant}({self.p},{self.q})",
                forecast_timestamp=datetime.utcnow(),
            )

        except Exception:
            # Fallback on GARCH fitting failure
            return self._fallback_forecast(returns)

    def _fallback_forecast(self, returns: pd.Series) -> VolatilityForecast:
        """Fallback volatility estimation using rolling std."""
        # Rolling volatility (20-day)
        rolling_vol = returns.rolling(20).std() / 100
        current_vol_daily = float(rolling_vol.iloc[-1])
        current_vol_annual = current_vol_daily * np.sqrt(252)

        # Simple persistence forecast
        forecast_vol_annual = current_vol_annual  # Assume persistence

        return VolatilityForecast(
            current_volatility=current_vol_annual,
            forecast_volatility=forecast_vol_annual,
            volatility_change=0.0,
            high_volatility_regime=forecast_vol_annual > self.high_vol_threshold,
            model_name="rolling_std_fallback",
            forecast_timestamp=datetime.utcnow(),
        )

    def estimate_volatility(self, data: pd.DataFrame) -> float:
        """
        Estimate current annualized volatility.

        Convenience method for use by other components (e.g., RiskGuard).

        Args:
            data: Historical OHLCV data

        Returns:
            Annualized volatility estimate
        """
        forecast = self._forecast(data)
        return forecast.current_volatility

    def describe(self) -> dict[str, Any]:
        """Get model description."""
        desc = super().describe()
        desc.update(
            {
                "garch_order": (self.p, self.q),
                "variant": self.variant,
                "horizon": self.horizon,
                "high_vol_threshold": self.high_vol_threshold,
                "arch_available": ARCH_AVAILABLE,
            }
        )
        return desc


class EGARCHVolatilityModel(GARCHVolatilityModel):
    """
    EGARCH volatility model.

    Captures asymmetric volatility response (leverage effect).
    Negative returns typically increase volatility more than positive returns.
    """

    def __init__(
        self,
        config: ModelConfig | None = None,
        p: int = 1,
        q: int = 1,
        horizon: int = 5,
        high_vol_threshold: float = 0.25,
    ):
        super().__init__(
            config=config,
            p=p,
            q=q,
            variant="EGARCH",
            horizon=horizon,
            high_vol_threshold=high_vol_threshold,
        )


class TGARCHVolatilityModel(GARCHVolatilityModel):
    """
    TGARCH (Threshold GARCH / GJR-GARCH) volatility model.

    Models asymmetric response using threshold indicator.
    """

    def __init__(
        self,
        config: ModelConfig | None = None,
        p: int = 1,
        q: int = 1,
        horizon: int = 5,
        high_vol_threshold: float = 0.25,
    ):
        super().__init__(
            config=config,
            p=p,
            q=q,
            variant="TGARCH",
            horizon=horizon,
            high_vol_threshold=high_vol_threshold,
        )
