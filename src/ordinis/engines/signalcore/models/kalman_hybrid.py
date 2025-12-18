"""
Kalman Filter Hybrid Strategy.

Decomposes price into trend and residual using Kalman filter.
Mean-reverts the residual only when aligned with trend direction.

Theory:
- Kalman filter separates signal (trend) from noise (residual)
- Mean reversion on residual = fading noise, not trend
- Only trade when residual is extreme AND trend supports direction
- Avoids counter-trend mean reversion trades
"""

from dataclasses import dataclass
from datetime import datetime
import logging

import numpy as np
import pandas as pd

from ordinis.engines.signalcore.core.model import Model, ModelConfig
from ordinis.engines.signalcore.core.signal import Signal, SignalType

logger = logging.getLogger(__name__)


@dataclass
class KalmanConfig:
    """Kalman Hybrid configuration."""

    # Kalman filter parameters
    process_noise_q: float = 1e-5  # Higher = more responsive trend
    observation_noise_r: float = 1e-2  # Higher = smoother trend

    # Signal parameters
    residual_z_entry: float = 2.0  # Z-score threshold for entry
    residual_z_exit: float = 0.5  # Z-score threshold for exit
    trend_slope_min: float = 0.0001  # Minimum trend slope for confirmation
    residual_lookback: int = 100  # Lookback for residual normalization

    # Risk parameters
    atr_period: int = 14
    atr_stop_mult: float = 1.5
    atr_tp_mult: float = 2.0


@dataclass
class KalmanState:
    """Kalman filter state."""

    level: float  # Estimated trend level
    variance: float  # State uncertainty
    residual: float  # Price - trend
    residual_z: float  # Normalized residual
    trend_slope: float  # First difference of level
    confidence: float  # 1 / variance


class KalmanFilter:
    """
    Simple 1D Kalman filter for trend extraction.

    State model: x_t = x_{t-1} + w_t,  w_t ~ N(0, Q)
    Observation: z_t = x_t + v_t,      v_t ~ N(0, R)
    """

    def __init__(self, q: float = 1e-5, r: float = 1e-2):
        """
        Initialize Kalman filter.

        Args:
            q: Process noise variance (higher = more responsive)
            r: Observation noise variance (higher = smoother)
        """
        self.q = q
        self.r = r
        self.x = 0.0  # State estimate
        self.p = 1.0  # State covariance
        self.initialized = False

    def reset(self, initial_value: float):
        """Reset filter with initial value."""
        self.x = initial_value
        self.p = 1.0
        self.initialized = True

    def update(self, observation: float) -> tuple[float, float, float]:
        """
        Update filter with new observation.

        Args:
            observation: New price observation

        Returns:
            Tuple of (filtered_level, residual, variance)
        """
        if not self.initialized:
            self.reset(observation)
            return observation, 0.0, 1.0

        # Predict
        x_pred = self.x
        p_pred = self.p + self.q

        # Update
        k = p_pred / (p_pred + self.r)  # Kalman gain
        self.x = x_pred + k * (observation - x_pred)
        self.p = (1 - k) * p_pred

        residual = observation - self.x

        return self.x, residual, self.p


class KalmanHybridModel(Model):
    """
    Kalman Filter Hybrid Strategy.

    Combines Kalman trend filter with mean reversion on residuals.
    Only trades when trend direction confirms residual signal.

    Signal Logic:
    1. Run Kalman filter to extract trend level
    2. Calculate residual (price - trend)
    3. Normalize residual to z-score
    4. Calculate trend slope
    5. LONG: residual_z < -2 AND trend_slope > 0 (oversold in uptrend)
    6. SHORT: residual_z > 2 AND trend_slope < 0 (overbought in downtrend)

    Example:
        config = ModelConfig(
            model_id="kalman_hybrid",
            model_type="hybrid",
            parameters={"process_noise_q": 1e-6}
        )
        model = KalmanHybridModel(config)
        signal = await model.generate("AAPL", df, timestamp)
    """

    def __init__(self, config: ModelConfig):
        """Initialize Kalman Hybrid model."""
        super().__init__(config)
        params = config.parameters or {}

        self.kalman_config = KalmanConfig(
            process_noise_q=params.get("process_noise_q", 1e-5),
            observation_noise_r=params.get("observation_noise_r", 1e-2),
            residual_z_entry=params.get("residual_z_entry", 2.0),
            residual_z_exit=params.get("residual_z_exit", 0.5),
            trend_slope_min=params.get("trend_slope_min", 0.0001),
            residual_lookback=params.get("residual_lookback", 100),
            atr_period=params.get("atr_period", 14),
            atr_stop_mult=params.get("atr_stop_mult", 1.5),
            atr_tp_mult=params.get("atr_tp_mult", 2.0),
        )

        # Filter instance (reset for each symbol)
        self._filters: dict[str, KalmanFilter] = {}

    def _get_filter(self, symbol: str) -> KalmanFilter:
        """Get or create Kalman filter for symbol."""
        if symbol not in self._filters:
            self._filters[symbol] = KalmanFilter(
                q=self.kalman_config.process_noise_q,
                r=self.kalman_config.observation_noise_r,
            )
        return self._filters[symbol]

    def run_filter(self, prices: pd.Series, symbol: str = "default") -> pd.DataFrame:
        """
        Run Kalman filter over price series.

        Args:
            prices: Price series
            symbol: Symbol for filter instance

        Returns:
            DataFrame with filtered values
        """
        kf = KalmanFilter(
            q=self.kalman_config.process_noise_q,
            r=self.kalman_config.observation_noise_r,
        )

        levels = []
        residuals = []
        variances = []

        for price in prices:
            level, residual, var = kf.update(price)
            levels.append(level)
            residuals.append(residual)
            variances.append(var)

        df = pd.DataFrame(
            {
                "price": prices.values,
                "trend_level": levels,
                "residual": residuals,
                "state_var": variances,
            },
            index=prices.index,
        )

        # Calculate trend slope
        df["trend_slope"] = df["trend_level"].diff()

        # Normalize residual to z-score
        lookback = min(self.kalman_config.residual_lookback, len(df))
        rolling_mean = df["residual"].rolling(lookback, min_periods=20).mean()
        rolling_std = df["residual"].rolling(lookback, min_periods=20).std()
        df["residual_z"] = (df["residual"] - rolling_mean) / (rolling_std + 1e-10)

        # Confidence (inverse of variance)
        df["confidence"] = 1 / (df["state_var"] + 1e-9)

        return df

    def _calculate_atr(self, df: pd.DataFrame) -> float:
        """Calculate Average True Range."""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(self.kalman_config.atr_period).mean().iloc[-1]

    async def generate(
        self,
        symbol: str,
        data: pd.DataFrame,
        timestamp: datetime,
    ) -> Signal | None:
        """
        Generate Kalman hybrid signal.

        Args:
            symbol: Stock symbol
            data: OHLCV DataFrame
            timestamp: Current timestamp

        Returns:
            Signal if conditions met
        """
        min_bars = self.kalman_config.residual_lookback + 20
        if len(data) < min_bars:
            return None

        # Run Kalman filter
        kalman_df = self.run_filter(data["close"], symbol)

        # Get current values
        current = kalman_df.iloc[-1]
        residual_z = current["residual_z"]
        trend_slope = current["trend_slope"]
        confidence = current["confidence"]
        trend_level = current["trend_level"]

        current_price = data["close"].iloc[-1]

        # Determine signal
        signal_type = SignalType.HOLD
        direction = 0

        # LONG: Oversold residual in uptrend
        if (
            residual_z < -self.kalman_config.residual_z_entry
            and trend_slope > self.kalman_config.trend_slope_min
        ):
            signal_type = SignalType.BUY
            direction = 1

        # SHORT: Overbought residual in downtrend
        elif (
            residual_z > self.kalman_config.residual_z_entry
            and trend_slope < -self.kalman_config.trend_slope_min
        ):
            signal_type = SignalType.SELL
            direction = -1

        if signal_type == SignalType.HOLD:
            return Signal(
                signal_type=SignalType.HOLD,
                symbol=symbol,
                timestamp=timestamp,
                confidence=0.0,
                metadata={
                    "residual_z": residual_z,
                    "trend_slope": trend_slope,
                    "trend_level": trend_level,
                    "kalman_confidence": confidence,
                },
            )

        # Calculate ATR for stops
        atr = self._calculate_atr(data)

        if direction > 0:
            stop_loss = current_price - (atr * self.kalman_config.atr_stop_mult)
            take_profit = current_price + (atr * self.kalman_config.atr_tp_mult)
        else:
            stop_loss = current_price + (atr * self.kalman_config.atr_stop_mult)
            take_profit = current_price - (atr * self.kalman_config.atr_tp_mult)

        # Confidence based on z-score extremity and filter confidence
        z_extremity = min(1.0, abs(residual_z) / 4.0)
        signal_confidence = 0.4 + 0.3 * z_extremity + 0.3 * min(1.0, confidence / confidence)

        return Signal(
            signal_type=signal_type,
            symbol=symbol,
            timestamp=timestamp,
            confidence=signal_confidence,
            metadata={
                "strategy": "kalman_hybrid",
                "residual_z": residual_z,
                "trend_slope": trend_slope,
                "trend_level": trend_level,
                "kalman_confidence": confidence,
                "direction": direction,
                "entry_price": current_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "atr": atr,
            },
        )


def optimize_kalman_params(
    prices: pd.Series,
    q_range: list = [1e-7, 1e-6, 1e-5, 1e-4],
    r_range: list = [1e-3, 1e-2, 1e-1],
) -> dict:
    """
    Grid search for optimal Kalman parameters.

    Optimizes for trend smoothness vs responsiveness.
    """
    results = []

    for q in q_range:
        for r in r_range:
            kf = KalmanFilter(q=q, r=r)

            levels = []
            for price in prices:
                level, _, _ = kf.update(price)
                levels.append(level)

            levels = np.array(levels)

            # Metrics
            # 1. Trend smoothness (lower = smoother)
            trend_roughness = np.mean(np.abs(np.diff(np.diff(levels))))

            # 2. Tracking error (lower = closer to price)
            tracking_error = np.mean((prices.values - levels) ** 2)

            # 3. Trend slope Sharpe (signal quality)
            slopes = np.diff(levels)
            sharpe = np.mean(slopes) / (np.std(slopes) + 1e-10)

            results.append(
                {
                    "q": q,
                    "r": r,
                    "roughness": trend_roughness,
                    "tracking_error": tracking_error,
                    "sharpe": sharpe,
                    "score": sharpe / (tracking_error + 1e-6),
                }
            )

    results_df = pd.DataFrame(results)
    best = results_df.loc[results_df["score"].idxmax()]

    return {
        "best_q": best["q"],
        "best_r": best["r"],
        "best_score": best["score"],
        "all_results": results_df,
    }


def backtest(
    df: pd.DataFrame,
    symbol: str,
    config: KalmanConfig | None = None,
) -> dict:
    """Backtest Kalman Hybrid strategy."""
    if config is None:
        config = KalmanConfig()

    model_config = ModelConfig(
        model_id=f"kalman_backtest_{symbol}",
        model_type="hybrid",
        parameters={
            "process_noise_q": config.process_noise_q,
            "observation_noise_r": config.observation_noise_r,
            "residual_z_entry": config.residual_z_entry,
        },
    )

    model = KalmanHybridModel(model_config)

    trades = []
    position = None

    start_idx = config.residual_lookback + 30

    for i in range(start_idx, len(df)):
        window = df.iloc[: i + 1]
        timestamp = df.index[i]

        import asyncio

        signal = asyncio.get_event_loop().run_until_complete(
            model.generate(symbol, window, timestamp)
        )

        if signal is None:
            continue

        current_price = df["close"].iloc[i]

        # Check exits
        if position is not None:
            exit_reason = None

            if position["direction"] == 1:
                if current_price <= position["stop_loss"]:
                    exit_reason = "stop_loss"
                elif current_price >= position["take_profit"]:
                    exit_reason = "take_profit"
            elif current_price >= position["stop_loss"]:
                exit_reason = "stop_loss"
            elif current_price <= position["take_profit"]:
                exit_reason = "take_profit"

            # Also exit on residual normalization
            if signal.metadata.get("residual_z"):
                residual_z = signal.metadata["residual_z"]
                if (
                    position["direction"] == 1
                    and residual_z > -config.residual_z_exit
                    or position["direction"] == -1
                    and residual_z < config.residual_z_exit
                ):
                    exit_reason = "residual_normalized"

            if exit_reason:
                pnl = (current_price - position["entry"]) * position["direction"]
                pnl_pct = pnl / position["entry"] * 100
                trades.append(
                    {
                        "entry_time": position["entry_time"],
                        "exit_time": timestamp,
                        "entry_price": position["entry"],
                        "exit_price": current_price,
                        "direction": position["direction"],
                        "pnl_pct": pnl_pct,
                        "exit_reason": exit_reason,
                    }
                )
                position = None

        # New entry
        if position is None and signal.signal_type != SignalType.HOLD:
            direction = 1 if signal.signal_type == SignalType.BUY else -1
            position = {
                "entry": current_price,
                "entry_time": timestamp,
                "direction": direction,
                "stop_loss": signal.metadata["stop_loss"],
                "take_profit": signal.metadata["take_profit"],
            }

    if not trades:
        return {"trades": 0, "total_return": 0, "win_rate": 0}

    trades_df = pd.DataFrame(trades)
    winners = trades_df[trades_df["pnl_pct"] > 0]

    return {
        "trades": len(trades),
        "total_return": trades_df["pnl_pct"].sum(),
        "win_rate": len(winners) / len(trades) * 100,
        "avg_win": winners["pnl_pct"].mean() if len(winners) > 0 else 0,
        "avg_loss": trades_df[trades_df["pnl_pct"] <= 0]["pnl_pct"].mean(),
        "trades_df": trades_df,
    }
