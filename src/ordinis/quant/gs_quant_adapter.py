"""
Standalone quantitative finance functions adapted from Goldman Sachs' gs-quant library.

This module provides production-grade timeseries analytics without requiring GS API
credentials. Functions are adapted from gs-quant (Apache 2.0 license) with dependencies
on GS-specific APIs removed.

Original source: https://github.com/goldmansachs/gs-quant
License: Apache 2.0

Usage:
    from ordinis.quant.gs_quant_adapter import (
        moving_average, bollinger_bands, rsi, macd, volatility, sharpe_ratio,
        returns, max_drawdown, correlation, beta, zscores
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
import math

import numpy as np
import pandas as pd

# =============================================================================
# Helper Types and Utilities (from gs_quant.timeseries.helper)
# =============================================================================


class Returns(Enum):
    """Return calculation method."""

    SIMPLE = "simple"
    LOGARITHMIC = "logarithmic"
    ABSOLUTE = "absolute"


class AnnualizationFactor(IntEnum):
    """Trading day conventions for annualization."""

    DAILY = 252
    WEEKLY = 52
    SEMI_MONTHLY = 26
    MONTHLY = 12
    QUARTERLY = 4
    ANNUALLY = 1


@dataclass
class Window:
    """
    Rolling window specification with optional ramp-up period.

    Args:
        w: Window size (int for observations, str for time period like '22d', '1m')
        r: Ramp-up value (defaults to window size). Results during ramp-up are NaN.

    Examples:
        >>> Window(22, 10)  # 22-day window, 10-day ramp
        >>> Window('1m', '1w')  # 1 month window, 1 week ramp
    """

    w: int | str | pd.DateOffset | None = None
    r: int | str | pd.DateOffset | None = None

    def __post_init__(self) -> None:
        if self.r is None:
            self.r = self.w


def _to_offset(tenor: str) -> pd.DateOffset:
    """Convert tenor string to pandas DateOffset."""
    import re

    matcher = re.fullmatch(r"(\d+)([hdwmy])", tenor)
    if not matcher:
        raise ValueError(f"Invalid tenor: {tenor}")

    mapping = {"h": "hours", "d": "days", "w": "weeks", "m": "months", "y": "years"}
    name = mapping[matcher.group(2)]
    return pd.DateOffset(**{name: int(matcher.group(1))})


def _normalize_window(x: pd.Series, window: Window | int | str | None) -> Window:
    """Normalize window specification to Window object."""
    if isinstance(window, int):
        return Window(window, window)
    if isinstance(window, str):
        offset = _to_offset(window)
        return Window(offset, offset)
    if window is None:
        return Window(len(x), 0)
    if isinstance(window, Window):
        w = _to_offset(window.w) if isinstance(window.w, str) else window.w
        r = _to_offset(window.r) if isinstance(window.r, str) else window.r
        if w is None:
            w = len(x)
        return Window(w, r)
    return window


def _apply_ramp(x: pd.Series, window: Window) -> pd.Series:
    """Apply ramp-up period to series (set initial values to NaN)."""
    if isinstance(window.r, pd.DateOffset):
        if len(x) == 0:
            return x
        start_date = x.index[0] + window.r
        if hasattr(start_date, "date"):
            start_date = start_date.date() if hasattr(x.index[0], "date") else start_date
        return x.loc[x.index >= start_date]
    if isinstance(window.r, int):
        return x.iloc[window.r :] if window.r < len(x) else pd.Series(dtype=float)
    return x


def _get_annualization_factor(x: pd.Series) -> int:
    """Infer annualization factor from series frequency."""
    if len(x) < 2:
        return AnnualizationFactor.DAILY

    prev_idx = x.index[0]
    distances = []
    for idx in x.index[1:]:
        d = (idx - prev_idx).days if hasattr(idx - prev_idx, "days") else 1
        if d > 0:
            distances.append(d)
        prev_idx = idx

    if not distances:
        return AnnualizationFactor.DAILY

    avg = np.mean(distances)
    if avg < 2.1:
        return AnnualizationFactor.DAILY
    if 6 <= avg < 8:
        return AnnualizationFactor.WEEKLY
    if 14 <= avg < 17:
        return AnnualizationFactor.SEMI_MONTHLY
    if 25 <= avg < 35:
        return AnnualizationFactor.MONTHLY
    if 85 <= avg < 97:
        return AnnualizationFactor.QUARTERLY
    if 360 <= avg < 386:
        return AnnualizationFactor.ANNUALLY
    return AnnualizationFactor.DAILY


# =============================================================================
# Returns and Price Functions (from gs_quant.timeseries.econometrics)
# =============================================================================


def returns(
    series: pd.Series,
    obs: int = 1,
    return_type: Returns = Returns.SIMPLE,
) -> pd.Series:
    """
    Calculate returns from price series.

    Args:
        series: Time series of prices
        obs: Number of observations for return calculation (default: 1)
        return_type: Return calculation method (simple, logarithmic, absolute)

    Returns:
        Series of returns

    Examples:
        >>> prices = pd.Series([100, 102, 101, 105])
        >>> returns(prices)  # Simple returns
        >>> returns(prices, return_type=Returns.LOGARITHMIC)  # Log returns
    """
    if series.size < 1:
        return series

    shifted = series.shift(obs)

    if return_type == Returns.SIMPLE:
        return series / shifted - 1
    if return_type == Returns.LOGARITHMIC:
        return np.log(series) - np.log(shifted)
    if return_type == Returns.ABSOLUTE:
        return series - shifted
    raise ValueError(f"Unknown return type: {return_type}")


def prices(
    series: pd.Series,
    initial: float = 1.0,
    return_type: Returns = Returns.SIMPLE,
) -> pd.Series:
    """
    Calculate price levels from returns series.

    Args:
        series: Time series of returns
        initial: Initial price level (default: 1.0)
        return_type: Return type of input series

    Returns:
        Series of price levels
    """
    if series.size < 1:
        return series

    if return_type == Returns.SIMPLE:
        return (1 + series).cumprod() * initial
    if return_type == Returns.LOGARITHMIC:
        return np.exp(series).cumprod() * initial
    if return_type == Returns.ABSOLUTE:
        return series.cumsum() + initial
    raise ValueError(f"Unknown return type: {return_type}")


# =============================================================================
# Technical Indicators (from gs_quant.timeseries.technicals)
# =============================================================================


def moving_average(
    x: pd.Series,
    w: Window | int | str = 20,
) -> pd.Series:
    """
    Simple moving average over specified window.

    Args:
        x: Time series of prices
        w: Window size (int, str like '22d', or Window object)

    Returns:
        Series of moving average values

    Examples:
        >>> prices = generate_series(100)
        >>> moving_average(prices, 22)  # 22-day SMA
    """
    window = _normalize_window(x, w)
    if isinstance(window.w, pd.DateOffset):
        result = x.rolling(window.w).mean()
    else:
        result = x.rolling(window.w, min_periods=1).mean()
    return _apply_ramp(result, window)


def exponential_moving_average(
    x: pd.Series,
    span: int | None = None,
    alpha: float | None = None,
) -> pd.Series:
    """
    Exponentially weighted moving average.

    Args:
        x: Time series of prices
        span: Span for EMA calculation (mutually exclusive with alpha)
        alpha: Smoothing factor (mutually exclusive with span)

    Returns:
        Series of EMA values

    Notes:
        Either span or alpha must be provided. span=N is equivalent to alpha=2/(N+1).
    """
    if span is not None:
        return x.ewm(span=span, adjust=False).mean()
    if alpha is not None:
        return x.ewm(alpha=alpha, adjust=False).mean()
    raise ValueError("Either span or alpha must be provided")


def bollinger_bands(
    x: pd.Series,
    w: Window | int | str = 20,
    k: float = 2.0,
) -> pd.DataFrame:
    """
    Bollinger Bands with given window and width.

    Args:
        x: Time series of prices
        w: Window size for moving average and standard deviation
        k: Number of standard deviations for band width (default: 2)

    Returns:
        DataFrame with columns ['lower', 'middle', 'upper']

    Examples:
        >>> prices = generate_series(100)
        >>> bb = bollinger_bands(prices, 20, 2)
        >>> bb['upper'], bb['lower']
    """
    window = _normalize_window(x, w)
    if isinstance(window.w, pd.DateOffset):
        avg = x.rolling(window.w).mean()
        std = x.rolling(window.w).std()
    else:
        avg = x.rolling(window.w, min_periods=1).mean()
        std = x.rolling(window.w, min_periods=1).std()

    upper = avg + k * std
    lower = avg - k * std

    result = pd.DataFrame({"lower": lower, "middle": avg, "upper": upper}, index=x.index)
    return result.iloc[window.r :] if isinstance(window.r, int) else result


def rsi(
    x: pd.Series,
    w: Window | int | str = 14,
) -> pd.Series:
    """
    Relative Strength Index (RSI).

    Args:
        x: Time series of prices
        w: Window size (default: 14)

    Returns:
        Series of RSI values (0-100)

    Notes:
        RSI = 100 - (100 / (1 + RS)) where RS = avg_gain / avg_loss
    """
    window = _normalize_window(x, w)
    delta = x.diff(1)

    gains = delta.clip(lower=0)
    losses = (-delta).clip(lower=0)

    if isinstance(window.w, pd.DateOffset):
        avg_gains = gains.rolling(window.w).mean()
        avg_losses = losses.rolling(window.w).mean()
    else:
        # Use exponential smoothing (Wilder's smoothing)
        avg_gains = gains.ewm(alpha=1 / window.w, adjust=False).mean()
        avg_losses = losses.ewm(alpha=1 / window.w, adjust=False).mean()

    rs = avg_gains / avg_losses.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    return _apply_ramp(result, window)


def macd(
    x: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    Moving Average Convergence Divergence (MACD).

    Args:
        x: Time series of prices
        fast: Fast EMA period (default: 12)
        slow: Slow EMA period (default: 26)
        signal: Signal line EMA period (default: 9)

    Returns:
        DataFrame with columns ['macd', 'signal', 'histogram']

    Examples:
        >>> prices = generate_series(100)
        >>> result = macd(prices)
        >>> result['histogram']  # MACD histogram for trend signals
    """
    fast_ema = x.ewm(span=fast, adjust=False).mean()
    slow_ema = x.ewm(span=slow, adjust=False).mean()

    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return pd.DataFrame(
        {"macd": macd_line, "signal": signal_line, "histogram": histogram},
        index=x.index,
    )


# =============================================================================
# Risk Metrics (from gs_quant.timeseries.econometrics)
# =============================================================================


def volatility(
    x: pd.Series,
    w: Window | int | str = 22,
    returns_type: Returns | None = Returns.SIMPLE,
    annualization_factor: int | None = None,
) -> pd.Series:
    """
    Realized volatility of price or return series.

    Args:
        x: Time series of prices or returns
        w: Window size
        returns_type: Return calculation method. If None, x is assumed to be returns.
        annualization_factor: Override auto-detected factor (252 for daily, etc.)

    Returns:
        Annualized volatility series (as percentage, e.g., 20.0 for 20%)

    Examples:
        >>> prices = generate_series(100)
        >>> vol = volatility(prices, 22)  # 22-day rolling volatility
    """
    window = _normalize_window(x, w)

    if x.size < 1:
        return x

    ret = returns(x, return_type=returns_type) if returns_type is not None else x

    if isinstance(window.w, pd.DateOffset):
        vol = ret.rolling(window.w).std()
    else:
        vol = ret.rolling(window.w, min_periods=1).std()

    if annualization_factor is None:
        annualization_factor = _get_annualization_factor(ret)

    annualized = vol * math.sqrt(annualization_factor) * 100
    return _apply_ramp(annualized, window)


def sharpe_ratio(
    x: pd.Series,
    risk_free_rate: float = 0.0,
    w: Window | int | str | None = None,
    annualization_factor: int | None = None,
) -> pd.Series | float:
    """
    Rolling or full-period Sharpe ratio.

    Args:
        x: Time series of prices
        risk_free_rate: Annual risk-free rate (default: 0)
        w: Window size. If None, calculates over full series.
        annualization_factor: Override auto-detected factor

    Returns:
        Sharpe ratio (series if window provided, scalar if full-period)

    Notes:
        Sharpe = (annualized_return - risk_free_rate) / annualized_volatility
    """
    ret = returns(x)

    if annualization_factor is None:
        annualization_factor = _get_annualization_factor(ret)

    if w is None:
        # Full period calculation
        ann_return = ret.mean() * annualization_factor
        ann_vol = ret.std() * math.sqrt(annualization_factor)
        if ann_vol == 0:
            return np.nan
        return (ann_return - risk_free_rate) / ann_vol

    window = _normalize_window(x, w)

    if isinstance(window.w, pd.DateOffset):
        mean_ret = ret.rolling(window.w).mean()
        std_ret = ret.rolling(window.w).std()
    else:
        mean_ret = ret.rolling(window.w, min_periods=1).mean()
        std_ret = ret.rolling(window.w, min_periods=1).std()

    ann_return = mean_ret * annualization_factor
    ann_vol = std_ret * math.sqrt(annualization_factor)

    result = (ann_return - risk_free_rate) / ann_vol.replace(0, np.nan)
    return _apply_ramp(result, window)


def max_drawdown(
    x: pd.Series,
    w: Window | int | str | None = None,
) -> pd.Series:
    """
    Maximum peak-to-trough drawdown.

    Args:
        x: Time series of prices
        w: Window size. If None, uses expanding window.

    Returns:
        Drawdown as negative ratio (e.g., -0.2 for 20% drawdown)

    Examples:
        >>> prices = generate_series(100)
        >>> dd = max_drawdown(prices, 22)
    """
    if w is None:
        rolling_max = x.expanding().max()
        drawdown = x / rolling_max - 1
        return drawdown.expanding().min()

    window = _normalize_window(x, w)

    if isinstance(window.w, pd.DateOffset):
        rolling_max = x.rolling(window.w).max()
        drawdown = x / rolling_max - 1
        result = drawdown.rolling(window.w).min()
    else:
        rolling_max = x.rolling(window.w, min_periods=1).max()
        drawdown = x / rolling_max - 1
        result = drawdown.rolling(window.w, min_periods=1).min()

    return _apply_ramp(result, window)


def correlation(
    x: pd.Series,
    y: pd.Series,
    w: Window | int | str = 22,
) -> pd.Series:
    """
    Rolling correlation between two price series.

    Args:
        x: First price series
        y: Second price series
        w: Window size

    Returns:
        Rolling correlation series (-1 to 1)
    """
    window = _normalize_window(x, w)

    ret_x = returns(x)
    ret_y = returns(y)

    if isinstance(window.w, pd.DateOffset):
        result = ret_x.rolling(window.w).corr(ret_y)
    else:
        result = ret_x.rolling(window.w, min_periods=2).corr(ret_y)

    return _apply_ramp(result, window)


def beta(
    x: pd.Series,
    benchmark: pd.Series,
    w: Window | int | str = 22,
) -> pd.Series:
    """
    Rolling beta of asset vs benchmark.

    Args:
        x: Asset price series
        benchmark: Benchmark price series
        w: Window size

    Returns:
        Rolling beta series

    Notes:
        Beta = Cov(asset, benchmark) / Var(benchmark)
    """
    window = _normalize_window(x, w)

    ret_x = returns(x)
    ret_b = returns(benchmark)

    if isinstance(window.w, pd.DateOffset):
        cov = ret_x.rolling(window.w).cov(ret_b)
        var = ret_b.rolling(window.w).var()
    else:
        cov = ret_x.rolling(window.w, min_periods=2).cov(ret_b)
        var = ret_b.rolling(window.w, min_periods=2).var()

    result = cov / var.replace(0, np.nan)
    return _apply_ramp(result, window)


# =============================================================================
# Statistics (from gs_quant.timeseries.statistics)
# =============================================================================


def zscores(
    x: pd.Series,
    w: Window | int | str | None = None,
) -> pd.Series:
    """
    Rolling z-scores over given window.

    Args:
        x: Time series
        w: Window size. If None, uses full series mean and std.

    Returns:
        Z-score series (standardized values)

    Notes:
        Z = (x - mean) / std
    """
    if x.size < 1:
        return x

    if w is None:
        mean = x.mean()
        std = x.std()
        if std == 0:
            return pd.Series(0, index=x.index)
        return (x - mean) / std

    window = _normalize_window(x, w)

    if isinstance(window.w, pd.DateOffset):
        mean = x.rolling(window.w).mean()
        std = x.rolling(window.w).std()
    else:
        mean = x.rolling(window.w, min_periods=1).mean()
        std = x.rolling(window.w, min_periods=1).std()

    result = (x - mean) / std.replace(0, np.nan)
    return _apply_ramp(result, window)


def percentiles(
    x: pd.Series,
    w: Window | int | str | None = None,
) -> pd.Series:
    """
    Rolling percentile rank of each value.

    Args:
        x: Time series
        w: Window size. If None, uses full series.

    Returns:
        Percentile rank series (0-100)
    """
    from scipy.stats import percentileofscore

    if x.size < 1:
        return x

    if w is None:
        return pd.Series(
            [percentileofscore(x, val, kind="mean") for val in x],
            index=x.index,
        )

    window = _normalize_window(x, w)

    def pct_rank(arr: np.ndarray) -> float:
        if len(arr) < 2:
            return 50.0
        return percentileofscore(arr, arr[-1], kind="mean")

    if isinstance(window.w, pd.DateOffset):
        # Rolling with DateOffset
        result = x.rolling(window.w).apply(pct_rank, raw=True)
    else:
        result = x.rolling(window.w, min_periods=1).apply(pct_rank, raw=True)

    return _apply_ramp(result, window)


def rolling_std(
    x: pd.Series,
    w: Window | int | str = 22,
) -> pd.Series:
    """
    Rolling standard deviation.

    Args:
        x: Time series
        w: Window size

    Returns:
        Rolling standard deviation series
    """
    window = _normalize_window(x, w)

    if isinstance(window.w, pd.DateOffset):
        result = x.rolling(window.w).std()
    else:
        result = x.rolling(window.w, min_periods=1).std()

    return _apply_ramp(result, window)


def rolling_mean(
    x: pd.Series,
    w: Window | int | str = 22,
) -> pd.Series:
    """
    Rolling mean.

    Args:
        x: Time series
        w: Window size

    Returns:
        Rolling mean series
    """
    return moving_average(x, w)


def rolling_min(
    x: pd.Series,
    w: Window | int | str = 22,
) -> pd.Series:
    """
    Rolling minimum.

    Args:
        x: Time series
        w: Window size

    Returns:
        Rolling minimum series
    """
    window = _normalize_window(x, w)

    if isinstance(window.w, pd.DateOffset):
        result = x.rolling(window.w).min()
    else:
        result = x.rolling(window.w, min_periods=1).min()

    return _apply_ramp(result, window)


def rolling_max(
    x: pd.Series,
    w: Window | int | str = 22,
) -> pd.Series:
    """
    Rolling maximum.

    Args:
        x: Time series
        w: Window size

    Returns:
        Rolling maximum series
    """
    window = _normalize_window(x, w)

    if isinstance(window.w, pd.DateOffset):
        result = x.rolling(window.w).max()
    else:
        result = x.rolling(window.w, min_periods=1).max()

    return _apply_ramp(result, window)


# =============================================================================
# Utility Functions
# =============================================================================


def generate_series(
    length: int = 100,
    start_price: float = 100.0,
    daily_volatility: float = 0.01,
    seed: int | None = None,
) -> pd.Series:
    """
    Generate synthetic price series for testing.

    Args:
        length: Number of observations
        start_price: Starting price
        daily_volatility: Daily return volatility
        seed: Random seed for reproducibility

    Returns:
        Series with DatetimeIndex and synthetic prices
    """
    if seed is not None:
        np.random.seed(seed)

    dates = pd.date_range(end=pd.Timestamp.today(), periods=length, freq="D")
    daily_returns = np.random.normal(0, daily_volatility, length)
    prices_arr = start_price * np.cumprod(1 + daily_returns)

    return pd.Series(prices_arr, index=dates, name="price")


__all__ = [
    # Types
    "Returns",
    "Window",
    "AnnualizationFactor",
    # Returns/Prices
    "returns",
    "prices",
    # Technical Indicators
    "moving_average",
    "exponential_moving_average",
    "bollinger_bands",
    "rsi",
    "macd",
    # Risk Metrics
    "volatility",
    "sharpe_ratio",
    "max_drawdown",
    "correlation",
    "beta",
    # Statistics
    "zscores",
    "percentiles",
    "rolling_std",
    "rolling_mean",
    "rolling_min",
    "rolling_max",
    # Utilities
    "generate_series",
]
