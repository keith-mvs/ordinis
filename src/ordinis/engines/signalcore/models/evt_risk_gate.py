"""
EVT Risk Gate - Extreme Value Theory Tail Risk Management.

Uses Generalized Pareto Distribution to estimate tail risk and
dynamically adjust position sizes when tail risk is elevated.

Theory:
- Normal VaR underestimates tail risk (fat tails)
- GPD provides better tail estimation
- Shape parameter ξ indicates tail heaviness
- High ξ or VaR → reduce exposure
"""

from dataclasses import dataclass
from datetime import datetime
import logging

import numpy as np
import pandas as pd
from scipy.stats import genpareto

from ordinis.engines.signalcore.core.model import Model, ModelConfig
from ordinis.engines.signalcore.core.signal import Signal, SignalType

logger = logging.getLogger(__name__)


@dataclass
class EVTConfig:
    """EVT Risk Gate configuration."""

    lookback: int = 252  # Days for tail estimation
    threshold_quantile: float = 0.95  # Percentile for tail threshold
    var_confidence: float = 0.99  # VaR confidence level
    min_tail_obs: int = 25  # Minimum exceedances required
    var_alert_pct: float = 0.03  # VaR threshold for alert (3%)
    xi_alert: float = 0.3  # Shape parameter threshold
    position_reduction: float = 0.5  # Reduce to 50% when alert


@dataclass
class EVTResult:
    """EVT analysis result."""

    threshold: float
    shape_xi: float
    scale: float
    var: float
    cvar: float
    tail_count: int
    is_alert: bool
    position_multiplier: float


class EVTRiskGate(Model):
    """
    Extreme Value Theory Risk Gate.

    This is an OVERLAY strategy - it modifies position sizes of other
    strategies rather than generating its own entry signals.

    Signal Logic:
    1. Fit GPD to loss tail (exceedances over threshold)
    2. Estimate VaR and CVaR at 99% confidence
    3. Monitor shape parameter ξ (tail heaviness)
    4. If VaR > limit or ξ > threshold, reduce position sizes

    Usage:
        # As overlay on existing strategy
        base_signal = base_strategy.generate(symbol, df, ts)
        evt_result = evt_gate.analyze(df)
        adjusted_size = base_size * evt_result.position_multiplier
    """

    def __init__(self, config: ModelConfig):
        """Initialize EVT Risk Gate."""
        super().__init__(config)
        params = config.parameters or {}

        self.evt_config = EVTConfig(
            lookback=params.get("lookback", 252),
            threshold_quantile=params.get("threshold_quantile", 0.95),
            var_confidence=params.get("var_confidence", 0.99),
            min_tail_obs=params.get("min_tail_obs", 25),
            var_alert_pct=params.get("var_alert_pct", 0.03),
            xi_alert=params.get("xi_alert", 0.3),
            position_reduction=params.get("position_reduction", 0.5),
        )

    def _calculate_returns(self, df: pd.DataFrame) -> pd.Series:
        """Calculate returns from price data."""
        return df["close"].pct_change().dropna()

    def _fit_gpd(self, losses: np.ndarray) -> tuple[float, float, float]:
        """
        Fit Generalized Pareto Distribution to exceedances.

        Args:
            losses: Array of losses (positive values)

        Returns:
            Tuple of (shape_xi, location, scale)
        """
        try:
            # Fit GPD with location fixed at 0 (for exceedances)
            shape, loc, scale = genpareto.fit(losses, floc=0)
            return shape, loc, scale
        except Exception as e:
            logger.warning(f"GPD fitting failed: {e}")
            return 0.0, 0.0, np.std(losses)

    def _calculate_var(
        self,
        threshold: float,
        shape: float,
        scale: float,
        n_obs: int,
        n_exceed: int,
    ) -> float:
        """
        Calculate VaR using GPD tail estimation.

        VaR_α = u + (σ/ξ) × [(n/N_u × (1-α))^(-ξ) - 1]
        """
        alpha = self.evt_config.var_confidence
        exceed_prob = n_exceed / n_obs

        if abs(shape) < 1e-6:
            # Exponential case (ξ → 0)
            var = threshold + scale * np.log(exceed_prob / (1 - alpha))
        else:
            var = threshold + (scale / shape) * ((exceed_prob / (1 - alpha)) ** shape - 1)

        return max(var, 0)

    def _calculate_cvar(
        self,
        var: float,
        threshold: float,
        shape: float,
        scale: float,
    ) -> float:
        """
        Calculate CVaR (Expected Shortfall) from GPD parameters.

        CVaR = VaR + (σ + ξ × (VaR - u)) / (1 - ξ)
        """
        if shape >= 1:
            # Infinite mean case
            return var * 2

        cvar = var + (scale + shape * (var - threshold)) / (1 - shape)
        return max(cvar, var)

    def analyze(self, df: pd.DataFrame) -> EVTResult:
        """
        Analyze tail risk using EVT.

        Args:
            df: OHLCV DataFrame

        Returns:
            EVTResult with tail metrics and position multiplier
        """
        returns = self._calculate_returns(df)

        if len(returns) < self.evt_config.lookback:
            lookback_returns = returns
        else:
            lookback_returns = returns.iloc[-self.evt_config.lookback :]

        # Convert to losses (negative returns become positive)
        losses = -lookback_returns

        # Calculate threshold
        threshold = np.percentile(losses, self.evt_config.threshold_quantile * 100)

        # Get exceedances
        exceedances = losses[losses > threshold] - threshold
        n_exceed = len(exceedances)

        if n_exceed < self.evt_config.min_tail_obs:
            # Not enough tail data, use conservative estimate
            logger.debug(f"Only {n_exceed} tail observations, using conservative estimate")
            return EVTResult(
                threshold=threshold,
                shape_xi=0.2,  # Conservative assumption
                scale=float(np.std(losses)),
                var=float(np.percentile(losses, 99)),
                cvar=float(np.percentile(losses, 99) * 1.2),
                tail_count=n_exceed,
                is_alert=False,
                position_multiplier=1.0,
            )

        # Fit GPD
        shape, loc, scale = self._fit_gpd(exceedances.values)

        # Calculate VaR and CVaR
        var = self._calculate_var(threshold, shape, scale, len(lookback_returns), n_exceed)
        cvar = self._calculate_cvar(var, threshold, shape, scale)

        # Determine alert status
        is_alert = (var > self.evt_config.var_alert_pct) or (shape > self.evt_config.xi_alert)

        # Calculate position multiplier
        if is_alert:
            # Gradual reduction based on severity
            var_excess = max(0, var - self.evt_config.var_alert_pct)
            xi_excess = max(0, shape - self.evt_config.xi_alert)

            severity = min(1.0, var_excess * 10 + xi_excess * 2)
            position_multiplier = 1.0 - severity * (1.0 - self.evt_config.position_reduction)
            position_multiplier = max(0.1, position_multiplier)  # Never go below 10%
        else:
            position_multiplier = 1.0

        return EVTResult(
            threshold=float(threshold),
            shape_xi=float(shape),
            scale=float(scale),
            var=float(var),
            cvar=float(cvar),
            tail_count=n_exceed,
            is_alert=is_alert,
            position_multiplier=float(position_multiplier),
        )

    async def generate(
        self,
        symbol: str,
        data: pd.DataFrame,
        timestamp: datetime,
    ) -> Signal | None:
        """
        Generate risk gate signal.

        Note: This returns a HOLD signal with risk metadata.
        The position_multiplier should be used to scale other signals.
        """
        evt_result = self.analyze(data)

        return Signal(
            signal_type=SignalType.HOLD,  # Risk gate doesn't generate trades
            symbol=symbol,
            timestamp=timestamp,
            confidence=1.0 - evt_result.position_multiplier,  # Higher = more risk
            metadata={
                "strategy": "evt_risk_gate",
                "threshold": evt_result.threshold,
                "shape_xi": evt_result.shape_xi,
                "scale": evt_result.scale,
                "var_99": evt_result.var,
                "cvar_99": evt_result.cvar,
                "tail_count": evt_result.tail_count,
                "is_alert": evt_result.is_alert,
                "position_multiplier": evt_result.position_multiplier,
            },
        )


class EVTGatedStrategy:
    """
    Wrapper to apply EVT risk gate to any base strategy.

    Example:
        base = ATROptimizedRSIModel(config)
        gated = EVTGatedStrategy(base, evt_config)
        signal = await gated.generate(symbol, df, ts)
        # signal.metadata["adjusted_size"] contains risk-adjusted size
    """

    def __init__(
        self,
        base_strategy: Model,
        evt_config: ModelConfig | None = None,
    ):
        """
        Initialize gated strategy.

        Args:
            base_strategy: Underlying strategy model
            evt_config: EVT Risk Gate configuration
        """
        self.base = base_strategy

        if evt_config is None:
            evt_config = ModelConfig(
                model_id="evt_gate",
                model_type="risk_overlay",
            )

        self.gate = EVTRiskGate(evt_config)

    async def generate(
        self,
        symbol: str,
        data: pd.DataFrame,
        timestamp: datetime,
    ) -> Signal | None:
        """
        Generate risk-adjusted signal.
        """
        # Get base signal
        base_signal = await self.base.generate(symbol, data, timestamp)

        if base_signal is None:
            return None

        # Get risk assessment
        evt_result = self.gate.analyze(data)

        # Adjust signal with risk info
        base_signal.metadata["evt_var"] = evt_result.var
        base_signal.metadata["evt_xi"] = evt_result.shape_xi
        base_signal.metadata["evt_alert"] = evt_result.is_alert
        base_signal.metadata["position_multiplier"] = evt_result.position_multiplier
        base_signal.metadata["adjusted_confidence"] = (
            base_signal.confidence * evt_result.position_multiplier
        )

        return base_signal


def rolling_evt_analysis(
    returns: pd.Series,
    window: int = 252,
    step: int = 21,
) -> pd.DataFrame:
    """
    Rolling EVT analysis over time.

    Args:
        returns: Return series
        window: Rolling window size
        step: Step size between calculations

    Returns:
        DataFrame with rolling EVT metrics
    """
    config = ModelConfig(model_id="evt_rolling", model_type="risk")
    gate = EVTRiskGate(config)

    results = []
    dates = []

    for i in range(window, len(returns), step):
        window_returns = returns.iloc[i - window : i]

        # Create minimal DataFrame for analyze()
        df = pd.DataFrame({"close": (1 + window_returns).cumprod()})

        result = gate.analyze(df)
        results.append(
            {
                "var_99": result.var,
                "cvar_99": result.cvar,
                "shape_xi": result.shape_xi,
                "is_alert": result.is_alert,
                "position_mult": result.position_multiplier,
            }
        )
        dates.append(returns.index[i - 1])

    return pd.DataFrame(results, index=dates)
