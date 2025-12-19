"""
GARCH Volatility Breakout Strategy.

Trades volatility expansions when realized volatility exceeds
GARCH(1,1) forecast by a significant margin.

Theory:
- GARCH models lag during volatility regime changes
- When realized vol >> forecast, a vol breakout is occurring
- Trade in the direction of the price move during expansion
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
class GARCHConfig:
    """GARCH Breakout configuration."""

    garch_lookback: int = 252  # Days for GARCH estimation
    realized_window: int = 5  # Days for realized vol
    breakout_threshold: float = 2.0  # Realized/Forecast ratio
    min_forecast_vol: float = 0.01  # Minimum annualized vol
    atr_period: int = 14
    atr_stop_mult: float = 2.0
    atr_tp_mult: float = 2.5


class GARCHBreakoutModel(Model):
    """
    GARCH Volatility Breakout Strategy.

    Signal Logic:
    1. Fit GARCH(1,1) to recent returns
    2. Forecast next-period volatility
    3. Compare to realized volatility
    4. If realized > threshold × forecast, breakout detected
    5. Trade in direction of recent price move

    Example:
        config = ModelConfig(
            model_id="garch_breakout_spy",
            model_type="volatility_breakout",
            parameters={"breakout_threshold": 2.0}
        )
        model = GARCHBreakoutModel(config)
        signal = await model.generate("SPY", df, timestamp)
    """

    def __init__(self, config: ModelConfig):
        """Initialize GARCH Breakout model."""
        super().__init__(config)
        params = config.parameters or {}

        self.garch_config = GARCHConfig(
            garch_lookback=params.get("garch_lookback", 252),
            realized_window=params.get("realized_window", 5),
            breakout_threshold=params.get("breakout_threshold", 2.0),
            min_forecast_vol=params.get("min_forecast_vol", 0.01),
            atr_period=params.get("atr_period", 14),
            atr_stop_mult=params.get("atr_stop_mult", 2.0),
            atr_tp_mult=params.get("atr_tp_mult", 2.5),
        )

        self._arch_available = self._check_arch()

    def _check_arch(self) -> bool:
        """Check if arch library is available."""
        try:
            import arch

            return True
        except ImportError:
            logger.warning("arch library not installed. Using fallback EWMA volatility.")
            return False

    def _calculate_returns(self, df: pd.DataFrame) -> pd.Series:
        """Calculate log returns."""
        return np.log(df["close"] / df["close"].shift(1)).dropna()

    def _fit_garch(self, returns: pd.Series) -> tuple[float, dict]:
        """
        Fit GARCH(1,1) and forecast next-period volatility.

        Returns:
            Tuple of (forecast_vol, model_params)
        """
        if not self._arch_available:
            return self._ewma_fallback(returns)

        try:
            from arch import arch_model

            # Scale returns to percentage for numerical stability
            scaled_returns = returns * 100

            model = arch_model(scaled_returns, vol="Garch", p=1, q=1, rescale=False)
            fitted = model.fit(disp="off", show_warning=False)

            # Forecast 1 step ahead
            forecast = fitted.forecast(horizon=1)
            forecast_var = forecast.variance.iloc[-1, 0]

            # Convert back to decimal and annualize
            forecast_vol = np.sqrt(forecast_var) / 100 * np.sqrt(252)

            params = {
                "omega": fitted.params.get("omega", 0),
                "alpha": fitted.params.get("alpha[1]", 0),
                "beta": fitted.params.get("beta[1]", 0),
            }

            return forecast_vol, params

        except Exception as e:
            logger.warning(f"GARCH fitting failed: {e}. Using EWMA fallback.")
            return self._ewma_fallback(returns)

    def _ewma_fallback(self, returns: pd.Series) -> tuple[float, dict]:
        """EWMA volatility as fallback when GARCH fails."""
        ewma_var = returns.ewm(span=20).var().iloc[-1]
        forecast_vol = np.sqrt(ewma_var) * np.sqrt(252)
        return forecast_vol, {"method": "ewma"}

    def _calculate_realized_vol(self, returns: pd.Series) -> float:
        """Calculate realized volatility over recent window."""
        recent = returns.iloc[-self.garch_config.realized_window :]
        return recent.std() * np.sqrt(252)

    def _calculate_atr(self, df: pd.DataFrame) -> float:
        """Calculate Average True Range."""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(self.garch_config.atr_period).mean().iloc[-1]

    async def generate(
        self,
        symbol: str,
        data: pd.DataFrame,
        timestamp: datetime,
    ) -> Signal | None:
        """
        Generate GARCH breakout signal.

        Args:
            symbol: Stock symbol
            data: OHLCV DataFrame
            timestamp: Current timestamp

        Returns:
            Signal if breakout detected, None otherwise
        """
        if len(data) < self.garch_config.garch_lookback:
            return None

        # Calculate returns
        returns = self._calculate_returns(data)
        if len(returns) < self.garch_config.garch_lookback:
            return None

        # Fit GARCH and get forecast
        lookback_returns = returns.iloc[-self.garch_config.garch_lookback :]
        forecast_vol, garch_params = self._fit_garch(lookback_returns)

        # Ensure minimum forecast
        forecast_vol = max(forecast_vol, self.garch_config.min_forecast_vol)

        # Calculate realized volatility
        realized_vol = self._calculate_realized_vol(returns)

        # Calculate breakout ratio
        vol_ratio = realized_vol / forecast_vol

        # Check for breakout
        is_breakout = vol_ratio > self.garch_config.breakout_threshold

        if not is_breakout:
            return Signal(
                signal_type=SignalType.HOLD,
                symbol=symbol,
                timestamp=timestamp,
                confidence=0.0,
                metadata={
                    "vol_ratio": vol_ratio,
                    "forecast_vol": forecast_vol,
                    "realized_vol": realized_vol,
                },
            )

        # Determine direction from recent price move
        recent_return = returns.iloc[-self.garch_config.realized_window :].sum()
        direction = np.sign(recent_return)

        # Calculate ATR for stops
        atr = self._calculate_atr(data)
        current_price = data["close"].iloc[-1]

        if direction > 0:
            signal_type = SignalType.BUY
            stop_loss = current_price - (atr * self.garch_config.atr_stop_mult)
            take_profit = current_price + (atr * self.garch_config.atr_tp_mult)
        else:
            signal_type = SignalType.SELL
            stop_loss = current_price + (atr * self.garch_config.atr_stop_mult)
            take_profit = current_price - (atr * self.garch_config.atr_tp_mult)

        # Confidence based on breakout magnitude
        confidence = min(0.9, 0.5 + (vol_ratio - self.garch_config.breakout_threshold) * 0.2)

        return Signal(
            signal_type=signal_type,
            symbol=symbol,
            timestamp=timestamp,
            confidence=confidence,
            metadata={
                "strategy": "garch_breakout",
                "vol_ratio": vol_ratio,
                "forecast_vol": forecast_vol,
                "realized_vol": realized_vol,
                "direction": direction,
                "entry_price": current_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "atr": atr,
                "garch_params": garch_params,
            },
        )


def backtest(
    df: pd.DataFrame,
    symbol: str,
    config: GARCHConfig | None = None,
) -> dict:
    """
    Backtest GARCH Breakout strategy.

    Args:
        df: OHLCV DataFrame with DatetimeIndex
        symbol: Stock symbol
        config: Strategy configuration

    Returns:
        Backtest results dictionary
    """
    if config is None:
        config = GARCHConfig()

    model_config = ModelConfig(
        model_id=f"garch_backtest_{symbol}",
        model_type="volatility_breakout",
        parameters={
            "garch_lookback": config.garch_lookback,
            "realized_window": config.realized_window,
            "breakout_threshold": config.breakout_threshold,
        },
    )

    model = GARCHBreakoutModel(model_config)

    trades = []
    position = None

    # Need enough data for GARCH
    start_idx = config.garch_lookback + 10

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

        # Check exits for existing position
        if position is not None:
            exit_reason = None

            if position["direction"] == 1:  # Long
                if current_price <= position["stop_loss"]:
                    exit_reason = "stop_loss"
                elif current_price >= position["take_profit"]:
                    exit_reason = "take_profit"
            elif current_price >= position["stop_loss"]:
                exit_reason = "stop_loss"
            elif current_price <= position["take_profit"]:
                exit_reason = "take_profit"

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
                        "vol_ratio": position["vol_ratio"],
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
                "vol_ratio": signal.metadata["vol_ratio"],
            }

    # Calculate statistics
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
        "max_win": trades_df["pnl_pct"].max(),
        "max_loss": trades_df["pnl_pct"].min(),
        "avg_vol_ratio": trades_df["vol_ratio"].mean(),
        "trades_df": trades_df,
    }
