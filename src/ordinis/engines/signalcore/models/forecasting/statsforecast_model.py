"""
Price forecasting models using Nixtla statsforecast.

Reference: https://github.com/Nixtla/statsforecast
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from ...core.model import Model, ModelConfig
from ...core.signal import Direction, Signal, SignalType

# Optional statsforecast import with graceful degradation
try:
    from statsforecast import StatsForecast
    from statsforecast.models import ARIMA, ETS, AutoARIMA

    STATSFORECAST_AVAILABLE = True
except ImportError:
    STATSFORECAST_AVAILABLE = False
    StatsForecast = None
    ARIMA = None
    AutoARIMA = None
    ETS = None


@dataclass
class ForecastResult:
    """Result of price forecast."""

    predicted_price: float
    predicted_return: float
    confidence_low: float
    confidence_high: float
    horizon: int
    model_name: str
    forecast_timestamp: datetime


class ARIMAForecastModel(Model):
    """
    ARIMA-based price forecasting model using Nixtla statsforecast.

    Uses ARIMA(p,d,q) for time series forecasting with configurable parameters.
    Falls back to simple moving average if statsforecast unavailable.
    """

    def __init__(
        self,
        config: ModelConfig | None = None,
        p: int = 1,
        d: int = 1,
        q: int = 1,
        horizon: int = 5,
        confidence_level: float = 0.95,
    ):
        """
        Initialize ARIMA forecast model.

        Args:
            config: Model configuration
            p: AR order
            d: Differencing order
            q: MA order
            horizon: Forecast horizon (days)
            confidence_level: Confidence interval level
        """
        if config is None:
            config = ModelConfig(
                model_id="arima-forecast",
                model_type="forecasting",
                version="1.0.0",
                parameters={"p": p, "d": d, "q": q, "horizon": horizon},
                min_data_points=100,
                lookback_period=252,
            )
        super().__init__(config)

        self.p = p
        self.d = d
        self.q = q
        self.horizon = horizon
        self.confidence_level = confidence_level
        self._model = None

    def generate(self, data: pd.DataFrame, timestamp: datetime) -> Signal:
        """
        Generate trading signal based on ARIMA price forecast.

        Args:
            data: Historical OHLCV data
            timestamp: Current timestamp

        Returns:
            Signal with forecast-based direction and expected return
        """
        symbol = data.attrs.get("symbol", "UNKNOWN")
        current_price = float(data["close"].iloc[-1])

        # Get forecast
        forecast = self._forecast(data)

        # Determine signal direction based on predicted return
        if forecast.predicted_return > 0.01:  # >1% expected return
            direction = Direction.LONG
            strength = min(forecast.predicted_return * 10, 1.0)
        elif forecast.predicted_return < -0.01:  # <-1% expected return
            direction = Direction.SHORT
            strength = min(abs(forecast.predicted_return) * 10, 1.0)
        else:
            direction = Direction.NEUTRAL
            strength = 0.0

        # Calculate confidence from prediction interval width
        interval_width = forecast.confidence_high - forecast.confidence_low
        confidence = max(0.0, min(1.0, 1.0 - (interval_width / current_price)))

        self._last_update = timestamp

        return Signal(
            symbol=symbol,
            timestamp=timestamp,
            signal_type=SignalType.ENTRY if direction != Direction.NEUTRAL else SignalType.HOLD,
            direction=direction,
            probability=confidence,
            expected_return=forecast.predicted_return,
            confidence_interval=(forecast.confidence_low, forecast.confidence_high),
            score=strength if direction == Direction.LONG else -strength,
            model_id=self.config.model_id,
            model_version=self.config.version,
            metadata={
                "forecast_model": forecast.model_name,
                "predicted_price": forecast.predicted_price,
                "current_price": current_price,
                "horizon": self.horizon,
            },
        )

    def _forecast(self, data: pd.DataFrame) -> ForecastResult:
        """
        Run ARIMA forecast.

        Args:
            data: Historical OHLCV data

        Returns:
            ForecastResult with predictions
        """
        current_price = float(data["close"].iloc[-1])

        if not STATSFORECAST_AVAILABLE:
            # Fallback: Simple trend extrapolation
            returns = data["close"].pct_change().dropna()
            mean_return = float(returns.tail(20).mean())
            std_return = float(returns.tail(20).std())

            predicted_price = current_price * (1 + mean_return * self.horizon)
            predicted_return = (predicted_price - current_price) / current_price

            return ForecastResult(
                predicted_price=predicted_price,
                predicted_return=predicted_return,
                confidence_low=current_price * (1 - 2 * std_return * np.sqrt(self.horizon)),
                confidence_high=current_price * (1 + 2 * std_return * np.sqrt(self.horizon)),
                horizon=self.horizon,
                model_name="simple_trend_fallback",
                forecast_timestamp=datetime.utcnow(),
            )

        # Prepare data for statsforecast (requires 'ds' and 'y' columns)
        df = pd.DataFrame(
            {
                "unique_id": "price",
                "ds": data.index,
                "y": data["close"].values,
            }
        )

        # Create and fit model
        model = StatsForecast(
            models=[ARIMA(order=(self.p, self.d, self.q))],
            freq="D",
            n_jobs=1,
        )

        # Fit and forecast
        model.fit(df)
        forecast_df = model.predict(h=self.horizon, level=[int(self.confidence_level * 100)])

        # Extract forecast values
        predicted_price = float(forecast_df["ARIMA"].iloc[-1])
        predicted_return = (predicted_price - current_price) / current_price

        # Get confidence intervals
        ci_col_lo = f"ARIMA-lo-{int(self.confidence_level * 100)}"
        ci_col_hi = f"ARIMA-hi-{int(self.confidence_level * 100)}"

        confidence_low = (
            float(forecast_df[ci_col_lo].iloc[-1])
            if ci_col_lo in forecast_df
            else predicted_price * 0.95
        )
        confidence_high = (
            float(forecast_df[ci_col_hi].iloc[-1])
            if ci_col_hi in forecast_df
            else predicted_price * 1.05
        )

        return ForecastResult(
            predicted_price=predicted_price,
            predicted_return=predicted_return,
            confidence_low=confidence_low,
            confidence_high=confidence_high,
            horizon=self.horizon,
            model_name=f"ARIMA({self.p},{self.d},{self.q})",
            forecast_timestamp=datetime.utcnow(),
        )

    def describe(self) -> dict[str, Any]:
        """Get model description."""
        desc = super().describe()
        desc.update(
            {
                "arima_order": (self.p, self.d, self.q),
                "horizon": self.horizon,
                "confidence_level": self.confidence_level,
                "statsforecast_available": STATSFORECAST_AVAILABLE,
            }
        )
        return desc


class AutoARIMAForecastModel(Model):
    """
    AutoARIMA-based price forecasting model.

    Automatically selects optimal ARIMA parameters using Nixtla statsforecast.
    """

    def __init__(
        self,
        config: ModelConfig | None = None,
        horizon: int = 5,
        season_length: int = 1,
        confidence_level: float = 0.95,
    ):
        """
        Initialize AutoARIMA forecast model.

        Args:
            config: Model configuration
            horizon: Forecast horizon (days)
            season_length: Seasonal period (1 for no seasonality)
            confidence_level: Confidence interval level
        """
        if config is None:
            config = ModelConfig(
                model_id="autoarima-forecast",
                model_type="forecasting",
                version="1.0.0",
                parameters={"horizon": horizon, "season_length": season_length},
                min_data_points=100,
                lookback_period=252,
            )
        super().__init__(config)

        self.horizon = horizon
        self.season_length = season_length
        self.confidence_level = confidence_level

    def generate(self, data: pd.DataFrame, timestamp: datetime) -> Signal:
        """
        Generate trading signal based on AutoARIMA price forecast.

        Args:
            data: Historical OHLCV data
            timestamp: Current timestamp

        Returns:
            Signal with forecast-based direction and expected return
        """
        symbol = data.attrs.get("symbol", "UNKNOWN")
        current_price = float(data["close"].iloc[-1])

        # Get forecast
        forecast = self._forecast(data)

        # Determine signal direction
        if forecast.predicted_return > 0.01:
            direction = Direction.LONG
            strength = min(forecast.predicted_return * 10, 1.0)
        elif forecast.predicted_return < -0.01:
            direction = Direction.SHORT
            strength = min(abs(forecast.predicted_return) * 10, 1.0)
        else:
            direction = Direction.NEUTRAL
            strength = 0.0

        # Calculate confidence
        interval_width = forecast.confidence_high - forecast.confidence_low
        confidence = max(0.0, min(1.0, 1.0 - (interval_width / current_price)))

        self._last_update = timestamp

        return Signal(
            symbol=symbol,
            timestamp=timestamp,
            signal_type=SignalType.ENTRY if direction != Direction.NEUTRAL else SignalType.HOLD,
            direction=direction,
            probability=confidence,
            expected_return=forecast.predicted_return,
            confidence_interval=(forecast.confidence_low, forecast.confidence_high),
            score=strength if direction == Direction.LONG else -strength,
            model_id=self.config.model_id,
            model_version=self.config.version,
            metadata={
                "forecast_model": forecast.model_name,
                "predicted_price": forecast.predicted_price,
                "current_price": current_price,
                "horizon": self.horizon,
            },
        )

    def _forecast(self, data: pd.DataFrame) -> ForecastResult:
        """Run AutoARIMA forecast."""
        current_price = float(data["close"].iloc[-1])

        if not STATSFORECAST_AVAILABLE:
            # Fallback: Exponential smoothing approximation
            returns = data["close"].pct_change().dropna()
            ewm_return = float(returns.ewm(span=20).mean().iloc[-1])
            ewm_std = float(returns.ewm(span=20).std().iloc[-1])

            predicted_price = current_price * (1 + ewm_return * self.horizon)
            predicted_return = (predicted_price - current_price) / current_price

            return ForecastResult(
                predicted_price=predicted_price,
                predicted_return=predicted_return,
                confidence_low=current_price * (1 - 2 * ewm_std * np.sqrt(self.horizon)),
                confidence_high=current_price * (1 + 2 * ewm_std * np.sqrt(self.horizon)),
                horizon=self.horizon,
                model_name="ewm_fallback",
                forecast_timestamp=datetime.utcnow(),
            )

        # Prepare data for statsforecast
        df = pd.DataFrame(
            {
                "unique_id": "price",
                "ds": data.index,
                "y": data["close"].values,
            }
        )

        # Create AutoARIMA model
        model = StatsForecast(
            models=[AutoARIMA(season_length=self.season_length)],
            freq="D",
            n_jobs=1,
        )

        # Fit and forecast
        model.fit(df)
        forecast_df = model.predict(h=self.horizon, level=[int(self.confidence_level * 100)])

        # Extract forecast values
        predicted_price = float(forecast_df["AutoARIMA"].iloc[-1])
        predicted_return = (predicted_price - current_price) / current_price

        # Get confidence intervals
        ci_col_lo = f"AutoARIMA-lo-{int(self.confidence_level * 100)}"
        ci_col_hi = f"AutoARIMA-hi-{int(self.confidence_level * 100)}"

        confidence_low = (
            float(forecast_df[ci_col_lo].iloc[-1])
            if ci_col_lo in forecast_df
            else predicted_price * 0.95
        )
        confidence_high = (
            float(forecast_df[ci_col_hi].iloc[-1])
            if ci_col_hi in forecast_df
            else predicted_price * 1.05
        )

        return ForecastResult(
            predicted_price=predicted_price,
            predicted_return=predicted_return,
            confidence_low=confidence_low,
            confidence_high=confidence_high,
            horizon=self.horizon,
            model_name="AutoARIMA",
            forecast_timestamp=datetime.utcnow(),
        )

    def describe(self) -> dict[str, Any]:
        """Get model description."""
        desc = super().describe()
        desc.update(
            {
                "horizon": self.horizon,
                "season_length": self.season_length,
                "confidence_level": self.confidence_level,
                "statsforecast_available": STATSFORECAST_AVAILABLE,
            }
        )
        return desc
