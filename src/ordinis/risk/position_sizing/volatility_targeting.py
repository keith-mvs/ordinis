"""
Volatility Targeting Position Sizing
Production-ready implementation with multiple volatility estimators.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


class VolatilityEstimator(Enum):
    """Volatility estimation methods."""

    SIMPLE = "simple"  # Standard deviation
    EWMA = "ewma"  # Exponentially weighted
    GARCH = "garch"  # GARCH(1,1)
    PARKINSON = "parkinson"  # High-low range
    GARMAN_KLASS = "garman_klass"  # OHLC-based
    YANG_ZHANG = "yang_zhang"  # Best OHLC estimator


@dataclass
class VolatilityTargetConfig:
    """Configuration for volatility targeting."""

    target_volatility: float = 0.15  # 15% annual
    lookback_period: int = 20  # Days for vol estimation
    vol_estimator: VolatilityEstimator = VolatilityEstimator.EWMA
    ewma_halflife: int = 10  # EWMA decay
    vol_floor: float = 0.05  # Minimum vol assumption
    vol_cap: float = 0.50  # Maximum vol assumption
    max_leverage: float = 2.0  # Position limit
    min_leverage: float = 0.1  # Minimum position
    annualization_factor: float = 252.0  # Trading days


class VolatilityCalculator:
    """
    Multi-method volatility calculator.

    Provides various volatility estimators for different use cases.
    """

    def __init__(self, annualization: float = 252.0):
        self.annualization = annualization
        self.sqrt_annualization = np.sqrt(annualization)

    def simple_volatility(self, returns: np.ndarray, lookback: int = 20) -> float:
        """Simple rolling standard deviation."""
        lookback = min(len(returns), lookback)
        return np.std(returns[-lookback:]) * self.sqrt_annualization

    def ewma_volatility(self, returns: np.ndarray, halflife: int = 10) -> float:
        """Exponentially weighted moving average volatility."""
        if len(returns) < 2:
            return 0.0

        decay = 1 - np.exp(-np.log(2) / halflife)
        weights = np.array([(1 - decay) ** i for i in range(len(returns))])
        weights = weights[::-1] / weights.sum()

        mean = np.average(returns, weights=weights)
        variance = np.average((returns - mean) ** 2, weights=weights)

        return np.sqrt(variance) * self.sqrt_annualization

    def parkinson_volatility(self, high: np.ndarray, low: np.ndarray, lookback: int = 20) -> float:
        """Parkinson high-low range estimator."""
        lookback = min(len(high), lookback)

        log_hl = np.log(high[-lookback:] / low[-lookback:])
        variance = np.mean(log_hl**2) / (4 * np.log(2))

        return np.sqrt(variance) * self.sqrt_annualization

    def garman_klass_volatility(
        self,
        open_: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        lookback: int = 20,
    ) -> float:
        """Garman-Klass OHLC volatility estimator."""
        lookback = min(len(close), lookback)

        o = open_[-lookback:]
        h = high[-lookback:]
        low_vals = low[-lookback:]
        c = close[-lookback:]

        log_hl = np.log(h / low_vals)
        log_co = np.log(c / o)

        variance = 0.5 * np.mean(log_hl**2) - (2 * np.log(2) - 1) * np.mean(log_co**2)

        return np.sqrt(max(variance, 0)) * self.sqrt_annualization

    def yang_zhang_volatility(
        self,
        open_: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        lookback: int = 20,
    ) -> float:
        """Yang-Zhang volatility estimator (best OHLC estimator)."""
        if len(close) < lookback + 1:
            lookback = max(len(close) - 1, 2)

        o = open_[-lookback:]
        h = high[-lookback:]
        low_vals = low[-lookback:]
        c = close[-lookback:]
        c_prev = close[-(lookback + 1) : -1]

        # Overnight variance
        log_oc = np.log(o / c_prev)
        overnight_var = np.var(log_oc, ddof=1)

        # Open-to-close variance
        log_co = np.log(c / o)
        open_close_var = np.var(log_co, ddof=1)

        # Rogers-Satchell variance
        log_ho = np.log(h / o)
        log_lo = np.log(low_vals / o)
        log_hc = np.log(h / c)
        log_lc = np.log(low_vals / c)
        rs_var = np.mean(log_ho * log_hc + log_lo * log_lc)

        # Yang-Zhang combination
        k = 0.34 / (1.34 + (lookback + 1) / (lookback - 1))
        variance = overnight_var + k * open_close_var + (1 - k) * rs_var

        return np.sqrt(max(variance, 0)) * self.sqrt_annualization


@dataclass
class VolatilityTargetResult:
    """Result of volatility targeting calculation."""

    target_leverage: float
    current_volatility: float
    target_volatility: float
    position_scalar: float
    is_capped: bool
    vol_estimator_used: str


class VolatilityTargeting:
    """
    Volatility targeting position sizing engine.

    Dynamically scales positions to maintain target portfolio volatility.
    """

    def __init__(self, config: VolatilityTargetConfig | None = None):
        self.config = config or VolatilityTargetConfig()
        self.vol_calc = VolatilityCalculator(self.config.annualization_factor)
        self._vol_history: list[float] = []

    def calculate_position_scalar(
        self, returns: np.ndarray, ohlc: pd.DataFrame | None = None
    ) -> VolatilityTargetResult:
        """
        Calculate position scalar to achieve target volatility.

        Args:
            returns: Historical returns
            ohlc: DataFrame with Open, High, Low, Close (optional)

        Returns:
            VolatilityTargetResult with scaling information
        """
        # Estimate current volatility
        vol = self._estimate_volatility(returns, ohlc)

        # Apply floor and cap
        vol = np.clip(vol, self.config.vol_floor, self.config.vol_cap)

        # Calculate target leverage
        raw_leverage = self.config.target_volatility / vol

        # Apply leverage limits
        is_capped = False
        if raw_leverage > self.config.max_leverage:
            raw_leverage = self.config.max_leverage
            is_capped = True
        elif raw_leverage < self.config.min_leverage:
            raw_leverage = self.config.min_leverage
            is_capped = True

        # Track history
        self._vol_history.append(vol)

        return VolatilityTargetResult(
            target_leverage=raw_leverage,
            current_volatility=vol,
            target_volatility=self.config.target_volatility,
            position_scalar=raw_leverage,
            is_capped=is_capped,
            vol_estimator_used=self.config.vol_estimator.value,
        )

    def _estimate_volatility(self, returns: np.ndarray, ohlc: pd.DataFrame | None = None) -> float:
        """Estimate volatility using configured method."""
        estimator = self.config.vol_estimator
        lookback = self.config.lookback_period

        # Simple volatility
        if estimator == VolatilityEstimator.SIMPLE:
            return self.vol_calc.simple_volatility(returns, lookback)

        # EWMA volatility
        if estimator == VolatilityEstimator.EWMA:
            return self.vol_calc.ewma_volatility(returns, self.config.ewma_halflife)

        # OHLC-based estimators - fallback to EWMA if no OHLC data
        ohlc_estimators = {
            VolatilityEstimator.PARKINSON,
            VolatilityEstimator.GARMAN_KLASS,
            VolatilityEstimator.YANG_ZHANG,
        }
        if estimator in ohlc_estimators and ohlc is not None:
            # Convert to numpy arrays for type safety
            high_arr = np.asarray(ohlc["High"].values)
            low_arr = np.asarray(ohlc["Low"].values)
            open_arr = np.asarray(ohlc["Open"].values)
            close_arr = np.asarray(ohlc["Close"].values)

            # Calculate based on specific estimator
            estimator_map = {
                VolatilityEstimator.PARKINSON: lambda: self.vol_calc.parkinson_volatility(
                    high_arr, low_arr, lookback
                ),
                VolatilityEstimator.GARMAN_KLASS: lambda: self.vol_calc.garman_klass_volatility(
                    open_arr, high_arr, low_arr, close_arr, lookback
                ),
                VolatilityEstimator.YANG_ZHANG: lambda: self.vol_calc.yang_zhang_volatility(
                    open_arr, high_arr, low_arr, close_arr, lookback
                ),
            }
            return estimator_map[estimator]()

        # Default fallback for unknown estimators or missing OHLC
        return self.vol_calc.ewma_volatility(returns, self.config.ewma_halflife)

    def calculate_position_size(
        self,
        account_value: float,
        asset_price: float,
        returns: np.ndarray,
        ohlc: pd.DataFrame | None = None,
    ) -> dict[str, float | bool]:
        """
        Calculate position size in shares/contracts.

        Args:
            account_value: Total account value
            asset_price: Current asset price
            returns: Historical returns
            ohlc: OHLC data (optional)

        Returns:
            Dictionary with position sizing details
        """
        result = self.calculate_position_scalar(returns, ohlc)

        target_notional = account_value * result.target_leverage
        shares = target_notional / asset_price

        return {
            "shares": shares,
            "notional": target_notional,
            "leverage": result.target_leverage,
            "current_vol": result.current_volatility,
            "target_vol": result.target_volatility,
            "is_capped": result.is_capped,
        }


class MultiAssetVolatilityTargeting:
    """
    Volatility targeting for multi-asset portfolios.

    Combines individual asset volatility targeting with
    portfolio-level risk budgeting.
    """

    def __init__(
        self,
        target_volatility: float = 0.10,
        correlation_lookback: int = 60,
        max_concentration: float = 0.30,
    ):
        self.target_volatility = target_volatility
        self.correlation_lookback = correlation_lookback
        self.max_concentration = max_concentration

    def calculate_weights(
        self, returns_df: pd.DataFrame, base_weights: np.ndarray | None = None
    ) -> dict[str, float]:
        """
        Calculate volatility-targeted weights for portfolio.

        Args:
            returns_df: DataFrame of asset returns
            base_weights: Starting weights (equal weight if None)

        Returns:
            Dictionary of asset weights
        """
        n_assets = len(returns_df.columns)

        if base_weights is None:
            base_weights = np.ones(n_assets) / n_assets

        # Calculate covariance matrix
        cov_matrix = returns_df.iloc[-self.correlation_lookback :].cov() * 252

        # Portfolio volatility at base weights
        port_vol = np.sqrt(base_weights @ cov_matrix.values @ base_weights)

        # Scale to target volatility
        leverage = self.target_volatility / port_vol
        scaled_weights = base_weights * leverage

        # Apply concentration limits
        scaled_weights = np.clip(scaled_weights, -self.max_concentration, self.max_concentration)

        # Create output dictionary
        weights = {col: scaled_weights[i] for i, col in enumerate(returns_df.columns)}

        weights["_total_leverage"] = np.sum(np.abs(scaled_weights))
        weights["_portfolio_vol"] = port_vol * leverage

        return weights


class AdaptiveVolatilityTargeting:
    """
    Adaptive volatility targeting with regime detection.

    Adjusts target volatility based on market regime.
    """

    def __init__(
        self,
        base_target: float = 0.12,
        low_vol_target: float = 0.15,
        high_vol_target: float = 0.08,
        regime_threshold: float = 0.20,
    ):
        self.base_target = base_target
        self.low_vol_target = low_vol_target
        self.high_vol_target = high_vol_target
        self.regime_threshold = regime_threshold

    def get_adaptive_target(self, current_vol: float, vol_percentile: float) -> float:
        """
        Get adaptive target based on current volatility regime.

        Args:
            current_vol: Current realized volatility
            vol_percentile: Percentile of current vol vs history

        Returns:
            Adaptive target volatility
        """
        if vol_percentile > 0.8:
            # High vol regime - reduce target
            return self.high_vol_target
        if vol_percentile < 0.2:
            # Low vol regime - increase target
            return self.low_vol_target
        # Normal regime
        return self.base_target
