"""
Mutual Information Ensemble Strategy.

Weights multiple alpha signals by their mutual information with forward returns.
Signals with higher predictive information get higher weight.

Theory:
- Mutual Information: I(X;Y) measures how much knowing X reduces uncertainty in Y
- High MI = signal has predictive power for returns
- Low MI = signal is noise
- Weights recalibrated on rolling basis
- Avoids overfitting to recent performance
"""

from dataclasses import dataclass
from datetime import datetime
import logging
from typing import Callable

import numpy as np
import pandas as pd

from ordinis.engines.signalcore.core.model import Model, ModelConfig
from ordinis.engines.signalcore.core.signal import Direction, Signal, SignalType

logger = logging.getLogger(__name__)


@dataclass
class MIConfig:
    """MI Ensemble configuration."""

    # MI estimation parameters
    mi_lookback: int = 252  # Lookback for MI calculation
    mi_bins: int = 10  # Bins for discretization
    forward_period: int = 5  # Forward return period (days)

    # Weighting parameters
    min_weight: float = 0.0  # Minimum signal weight
    max_weight: float = 0.5  # Maximum signal weight (cap)
    recalc_frequency: int = 21  # Recalculate weights every N bars
    mi_decay: float = 0.9  # Exponential decay for old MI values

    # Signal parameters
    ensemble_threshold: float = 0.3  # Threshold for combined signal
    min_signals_agree: int = 2  # Minimum signals that must agree

    # Risk parameters
    atr_period: int = 14
    atr_stop_mult: float = 1.5
    atr_tp_mult: float = 2.5


def mutual_information(x: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """
    Estimate mutual information between two continuous variables.

    Uses histogram-based discretization for estimation.
    I(X;Y) = H(X) + H(Y) - H(X,Y)

    Args:
        x: First variable (signal values)
        y: Second variable (forward returns)
        bins: Number of bins for discretization

    Returns:
        Estimated mutual information in nats
    """
    # Handle edge cases
    if len(x) != len(y) or len(x) < 50:
        return 0.0

    # Remove NaN
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]

    if len(x) < 50:
        return 0.0

    # Discretize
    x_bins = np.linspace(np.percentile(x, 1), np.percentile(x, 99), bins + 1)
    y_bins = np.linspace(np.percentile(y, 1), np.percentile(y, 99), bins + 1)

    x_discrete = np.digitize(x, x_bins)
    y_discrete = np.digitize(y, y_bins)

    # Joint histogram
    joint_hist = np.histogram2d(x_discrete, y_discrete, bins=bins)[0]
    joint_prob = joint_hist / joint_hist.sum()

    # Marginal probabilities
    x_prob = joint_prob.sum(axis=1)
    y_prob = joint_prob.sum(axis=0)

    # Calculate MI
    mi = 0.0
    for i in range(len(x_prob)):
        for j in range(len(y_prob)):
            if joint_prob[i, j] > 0 and x_prob[i] > 0 and y_prob[j] > 0:
                mi += joint_prob[i, j] * np.log(joint_prob[i, j] / (x_prob[i] * y_prob[j]))

    return max(0.0, mi)


def normalized_mutual_information(x: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """
    Normalized mutual information (0 to 1 range).

    NMI = 2 * I(X;Y) / (H(X) + H(Y))
    """
    mi = mutual_information(x, y, bins)

    # Calculate entropies
    def entropy(arr):
        hist, _ = np.histogram(arr, bins=bins)
        prob = hist / hist.sum()
        prob = prob[prob > 0]
        return -np.sum(prob * np.log(prob))

    h_x = entropy(x)
    h_y = entropy(y)

    if h_x + h_y == 0:
        return 0.0

    return 2 * mi / (h_x + h_y)


@dataclass
class SignalDefinition:
    """Definition of a signal component."""

    name: str
    compute: Callable[[pd.DataFrame], pd.Series]

    def __hash__(self):
        return hash(self.name)


# Default signal library
def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """RSI signal (normalized to -1 to 1)."""
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return (rsi - 50) / 50  # Normalize to -1 to 1


def compute_momentum(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Momentum signal."""
    returns = df["close"].pct_change(period)
    return returns / returns.std()  # Z-score normalize


def compute_mean_reversion(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Mean reversion signal (negative of z-score)."""
    z = (df["close"] - df["close"].rolling(period).mean()) / df["close"].rolling(period).std()
    return -z  # Negative because high z = sell signal


def compute_volatility_breakout(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Volatility breakout signal."""
    rolling_high = df["high"].rolling(period).max()
    rolling_low = df["low"].rolling(period).min()
    range_pos = (df["close"] - rolling_low) / (rolling_high - rolling_low + 1e-10)
    return 2 * range_pos - 1  # Normalize to -1 to 1


def compute_trend_strength(df: pd.DataFrame) -> pd.Series:
    """ADX-like trend strength."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

    tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(
        axis=1
    )

    atr = tr.rolling(14).mean()
    plus_di = 100 * plus_dm.rolling(14).mean() / atr
    minus_di = 100 * minus_dm.rolling(14).mean() / atr

    dx = abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(14).mean()

    # Sign based on direction
    direction = np.sign(plus_di - minus_di)
    return adx * direction / 50  # Normalize


DEFAULT_SIGNALS = [
    SignalDefinition("rsi", compute_rsi),
    SignalDefinition("momentum", compute_momentum),
    SignalDefinition("mean_reversion", compute_mean_reversion),
    SignalDefinition("vol_breakout", compute_volatility_breakout),
    SignalDefinition("trend_strength", compute_trend_strength),
]


class MIEnsembleModel(Model):
    """
    Mutual Information Weighted Ensemble Strategy.

    Combines multiple alpha signals weighted by their mutual
    information with forward returns.

    Signal Logic:
    1. Calculate each component signal value
    2. Estimate MI between each signal and forward returns
    3. Weight signals by normalized MI
    4. Combine: ensemble = Σ(w_i * signal_i)
    5. LONG: ensemble > threshold
    6. SHORT: ensemble < -threshold

    Example:
        config = ModelConfig(
            model_id="mi_ensemble",
            model_type="ensemble",
            parameters={"mi_lookback": 252}
        )
        model = MIEnsembleModel(config)
        signal = await model.generate("AAPL", df, timestamp)
    """

    def __init__(
        self,
        config: ModelConfig,
        signals: list[SignalDefinition] | None = None,
    ):
        """Initialize MI Ensemble model."""
        super().__init__(config)
        params = config.parameters or {}

        self.mi_config = MIConfig(
            mi_lookback=params.get("mi_lookback", 252),
            mi_bins=params.get("mi_bins", 10),
            forward_period=params.get("forward_period", 5),
            min_weight=params.get("min_weight", 0.0),
            max_weight=params.get("max_weight", 0.5),
            recalc_frequency=params.get("recalc_frequency", 21),
            ensemble_threshold=params.get("ensemble_threshold", 0.3),
            min_signals_agree=params.get("min_signals_agree", 2),
            atr_period=params.get("atr_period", 14),
            atr_stop_mult=params.get("atr_stop_mult", 1.5),
            atr_tp_mult=params.get("atr_tp_mult", 2.5),
        )

        self.signals = signals or DEFAULT_SIGNALS

        # Weight cache
        self._weights: dict[str, dict[str, float]] = {}  # symbol -> signal -> weight
        self._last_recalc: dict[str, int] = {}

    def calculate_mi_weights(
        self,
        df: pd.DataFrame,
        symbol: str,
    ) -> dict[str, float]:
        """
        Calculate MI-based weights for each signal.

        IMPORTANT: Uses only PAST data to avoid lookahead bias.
        The MI is calculated on a rolling historical window where
        we know both the signal and the subsequent realized return.

        At time T, we only use data from [T - lookback - forward_period, T - forward_period]
        so that forward_period is the "gap" ensuring all returns are realized.

        Args:
            df: OHLCV DataFrame
            symbol: Symbol for caching

        Returns:
            Dictionary of signal name to weight
        """
        lookback = self.mi_config.mi_lookback
        fwd = self.mi_config.forward_period

        # Minimum required data
        min_required = lookback + fwd + 50
        if len(df) < min_required:
            # Equal weights if insufficient data
            return {sig.name: 1.0 / len(self.signals) for sig in self.signals}

        # Use only the historical window where returns are realized:
        # Signal at t, return from t to t+fwd. So we stop at -fwd from end.
        # df_hist excludes the last fwd bars where we don't know the forward return.
        df_hist = df.iloc[:-fwd]

        # Forward returns: return from t to t+fwd (now fully realized in df_hist)
        forward_ret = df_hist["close"].pct_change(fwd).shift(-fwd)

        # Further limit to lookback window to avoid using very old data
        # Take the most recent 'lookback' bars from df_hist
        if len(df_hist) > lookback:
            df_hist = df_hist.iloc[-lookback:]
            forward_ret = forward_ret.iloc[-lookback:]

        weights = {}
        mi_values = {}

        for sig_def in self.signals:
            try:
                # CRITICAL: Use df_hist (not df) to avoid lookahead bias
                # Signal values must be computed on the same historical window
                signal_values = sig_def.compute(df_hist)

                # Align and get valid portion
                valid_idx = ~(signal_values.isna() | forward_ret.isna())
                x = signal_values[valid_idx].values
                y = forward_ret[valid_idx].values

                if len(x) >= 50:
                    mi = normalized_mutual_information(x, y, self.mi_config.mi_bins)
                else:
                    mi = 0.0

                mi_values[sig_def.name] = mi

            except Exception as e:
                logger.warning(f"Error computing signal {sig_def.name}: {e}")
                mi_values[sig_def.name] = 0.0

        # Normalize weights
        total_mi = sum(mi_values.values())
        if total_mi > 0:
            for name, mi in mi_values.items():
                raw_weight = mi / total_mi
                weights[name] = np.clip(
                    raw_weight,
                    self.mi_config.min_weight,
                    self.mi_config.max_weight,
                )
        else:
            # Equal weights if no MI
            for sig_def in self.signals:
                weights[sig_def.name] = 1.0 / len(self.signals)

        # Renormalize after clipping
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}

        logger.debug(f"MI weights for {symbol}: {weights}")

        return weights

    def _calculate_atr(self, df: pd.DataFrame) -> float:
        """Calculate Average True Range."""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(self.mi_config.atr_period).mean().iloc[-1]

    async def generate(
        self,
        symbol: str,
        data: pd.DataFrame,
        timestamp: datetime,
    ) -> Signal | None:
        """
        Generate MI-weighted ensemble signal.

        Args:
            symbol: Stock symbol
            data: OHLCV DataFrame
            timestamp: Current timestamp

        Returns:
            Signal if conditions met
        """
        min_bars = self.mi_config.mi_lookback + self.mi_config.forward_period + 50
        if len(data) < min_bars:
            return None

        current_bar = len(data)

        # Recalculate weights if needed
        if (
            symbol not in self._weights
            or current_bar - self._last_recalc.get(symbol, 0) >= self.mi_config.recalc_frequency
        ):
            self._weights[symbol] = self.calculate_mi_weights(data, symbol)
            self._last_recalc[symbol] = current_bar

        weights = self._weights.get(symbol, {})

        # Calculate current signal values
        signal_values = {}
        for sig_def in self.signals:
            try:
                values = sig_def.compute(data)
                signal_values[sig_def.name] = values.iloc[-1]
            except Exception as e:
                logger.warning(f"Error computing signal {sig_def.name}: {e}")
                signal_values[sig_def.name] = 0.0

        # Weighted ensemble
        ensemble = 0.0
        signals_long = 0
        signals_short = 0

        for name, value in signal_values.items():
            weight = weights.get(name, 0.0)
            ensemble += weight * value

            if value > 0:
                signals_long += 1
            elif value < 0:
                signals_short += 1

        # Determine signal
        signal_type = SignalType.HOLD
        direction = 0

        if (
            ensemble > self.mi_config.ensemble_threshold
            and signals_long >= self.mi_config.min_signals_agree
        ):
            signal_type = SignalType.ENTRY
            direction = Direction.LONG
        elif (
            ensemble < -self.mi_config.ensemble_threshold
            and signals_short >= self.mi_config.min_signals_agree
        ):
            signal_type = SignalType.ENTRY
            direction = Direction.SHORT

        current_price = data["close"].iloc[-1]

        if signal_type == SignalType.HOLD:
            return Signal(
                signal_type=SignalType.HOLD,
                direction=Direction.NEUTRAL,
                symbol=symbol,
                timestamp=timestamp,
                confidence=0.0,
                metadata={
                    "ensemble_value": ensemble,
                    "signal_values": signal_values,
                    "weights": weights,
                    "signals_long": signals_long,
                    "signals_short": signals_short,
                },
            )

        # Calculate stops
        atr = self._calculate_atr(data)

        if direction == Direction.LONG:
            stop_loss = current_price - (atr * self.mi_config.atr_stop_mult)
            take_profit = current_price + (atr * self.mi_config.atr_tp_mult)
        else:
            stop_loss = current_price + (atr * self.mi_config.atr_stop_mult)
            take_profit = current_price - (atr * self.mi_config.atr_tp_mult)

        # Confidence based on ensemble strength and signal agreement
        ensemble_confidence = min(1.0, abs(ensemble) / 1.0)
        agreement = max(signals_long, signals_short) / len(self.signals)
        confidence = 0.5 * ensemble_confidence + 0.5 * agreement

        return Signal(
            signal_type=signal_type,
            direction=direction,
            symbol=symbol,
            timestamp=timestamp,
            confidence=confidence,
            metadata={
                "strategy": "mi_ensemble",
                "ensemble_value": ensemble,
                "direction": direction.value,
                "signal_values": signal_values,
                "weights": weights,
                "signals_long": signals_long,
                "signals_short": signals_short,
                "entry_price": current_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "atr": atr,
            },
        )


def analyze_signal_mi(
    df: pd.DataFrame,
    signals: list[SignalDefinition] = DEFAULT_SIGNALS,
    forward_periods: list[int] = [1, 5, 10, 21],
) -> pd.DataFrame:
    """
    Analyze MI for all signals across different forward periods.

    Args:
        df: OHLCV DataFrame
        signals: List of signal definitions
        forward_periods: List of forward return periods

    Returns:
        DataFrame with MI values
    """
    results = []

    for sig_def in signals:
        signal_values = sig_def.compute(df)

        for period in forward_periods:
            forward_ret = df["close"].pct_change(period).shift(-period)

            valid_idx = ~(signal_values.isna() | forward_ret.isna())
            x = signal_values[valid_idx].values
            y = forward_ret[valid_idx].values

            if len(x) >= 50:
                mi = mutual_information(x, y)
                nmi = normalized_mutual_information(x, y)
            else:
                mi = nmi = 0.0

            results.append(
                {
                    "signal": sig_def.name,
                    "forward_period": period,
                    "mi": mi,
                    "nmi": nmi,
                }
            )

    return pd.DataFrame(results)


def backtest(
    df: pd.DataFrame,
    symbol: str,
    config: MIConfig | None = None,
    signals: list[SignalDefinition] | None = None,
) -> dict:
    """Backtest MI Ensemble strategy."""
    if config is None:
        config = MIConfig()

    model_config = ModelConfig(
        model_id=f"mi_ensemble_backtest_{symbol}",
        model_type="ensemble",
        parameters={
            "mi_lookback": config.mi_lookback,
            "ensemble_threshold": config.ensemble_threshold,
        },
    )

    model = MIEnsembleModel(model_config, signals)

    trades = []
    position = None

    start_idx = config.mi_lookback + 50

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

            # Also exit on signal reversal
            if signal.metadata.get("direction"):
                sig_dir = signal.metadata["direction"]
                if (
                    position["direction"] == 1
                    and sig_dir == -1
                    or position["direction"] == -1
                    and sig_dir == 1
                ):
                    exit_reason = "signal_reversal"

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
            direction = signal.metadata.get("direction", 0)
            if direction != 0:
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
