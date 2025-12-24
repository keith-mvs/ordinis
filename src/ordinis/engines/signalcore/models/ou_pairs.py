"""
Ornstein-Uhlenbeck Pairs Trading Strategy.

Estimates OU process parameters from cointegrated pairs.
Trades mean reversion when spread deviates by half-life-calibrated thresholds.

Theory:
- Cointegrated pairs have a stationary spread
- Spread follows OU process: dS = θ(μ - S)dt + σdW
- θ = mean reversion speed, μ = long-run mean, σ = volatility
- Half-life = ln(2)/θ tells expected time to mean revert
- Entry at 2σ, exit at 0.5σ with half-life sizing
"""

from dataclasses import dataclass
from datetime import datetime
import logging

import numpy as np
import pandas as pd

from ordinis.engines.signalcore.core.model import Model, ModelConfig
from ordinis.engines.signalcore.core.signal import Direction, Signal, SignalType

logger = logging.getLogger(__name__)


@dataclass
class OUConfig:
    """OU Pairs configuration."""

    # Cointegration parameters
    coint_lookback: int = 252  # Lookback for cointegration test
    coint_pvalue: float = 0.05  # Required p-value for cointegration
    hedge_lookback: int = 60  # Lookback for hedge ratio estimation

    # OU process parameters
    ou_lookback: int = 60  # Lookback for OU parameter estimation
    min_halflife: int = 2  # Minimum half-life (days)
    max_halflife: int = 60  # Maximum half-life (days)

    # Signal parameters
    entry_z: float = 2.0  # Z-score for entry
    exit_z: float = 0.5  # Z-score for exit
    stop_z: float = 4.0  # Z-score for stop loss

    # Risk parameters
    max_position_pct: float = 0.1  # Max position size per pair


@dataclass
class OUParams:
    """Estimated OU process parameters."""

    theta: float  # Mean reversion speed
    mu: float  # Long-run mean
    sigma: float  # Process volatility
    halflife: float  # ln(2)/theta
    r_squared: float  # Regression R-squared


@dataclass
class PairStats:
    """Statistics for a trading pair."""

    asset_a: str
    asset_b: str
    hedge_ratio: float
    spread: pd.Series
    spread_z: float
    ou_params: OUParams
    coint_pvalue: float
    is_valid: bool


def estimate_ou_params(spread: pd.Series) -> OUParams:
    """
    Estimate Ornstein-Uhlenbeck parameters using OLS.

    Regression: ΔS_t = θ(μ - S_{t-1}) + ε_t
    Equivalent: ΔS_t = α + β*S_{t-1} + ε_t
    where θ = -β, μ = α/θ

    Args:
        spread: Spread time series

    Returns:
        OUParams with estimated parameters
    """
    spread = spread.dropna()
    if len(spread) < 30:
        return OUParams(theta=0.0, mu=0.0, sigma=0.0, halflife=float("inf"), r_squared=0.0)

    delta_s = spread.diff().iloc[1:]
    s_lag = spread.iloc[:-1].values
    y = delta_s.values

    # OLS regression: ΔS = α + β*S_lag
    X = np.column_stack([np.ones(len(s_lag)), s_lag])

    try:
        betas, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return OUParams(theta=0.0, mu=0.0, sigma=0.0, halflife=float("inf"), r_squared=0.0)

    alpha, beta = betas

    # theta must be positive for mean reversion
    theta = -beta
    if theta <= 0:
        return OUParams(theta=0.0, mu=0.0, sigma=0.0, halflife=float("inf"), r_squared=0.0)

    mu = alpha / theta if theta > 0 else spread.mean()

    # Residual volatility
    predicted = X @ betas
    sigma = np.std(y - predicted)

    # Half-life in same units as data
    halflife = np.log(2) / theta

    # R-squared
    ss_res = np.sum((y - predicted) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return OUParams(
        theta=theta,
        mu=mu,
        sigma=sigma,
        halflife=halflife,
        r_squared=r_squared,
    )


def test_cointegration(y1: pd.Series, y2: pd.Series) -> tuple[float, float]:
    """
    Test for cointegration using Engle-Granger two-step method.

    Returns:
        Tuple of (p-value, hedge_ratio)
    """
    try:
        from statsmodels.tsa.stattools import adfuller, coint

        # Cointegration test
        score, pvalue, _ = coint(y1, y2)

        # Hedge ratio from OLS
        X = np.column_stack([np.ones(len(y2)), y2.values])
        betas, _, _, _ = np.linalg.lstsq(X, y1.values, rcond=None)
        hedge_ratio = betas[1]

        return pvalue, hedge_ratio
    except ImportError:
        # Fallback without statsmodels
        corr = y1.corr(y2)
        hedge_ratio = y1.std() / y2.std() * np.sign(corr)
        # Approximate p-value (not rigorous)
        spread = y1 - hedge_ratio * y2
        spread_normalized = (spread - spread.mean()) / spread.std()
        # Dickey-Fuller approximation
        ar_coef = spread.autocorr(1)
        pvalue = 1 - abs(ar_coef)  # Higher autocorr = higher pvalue
        return pvalue, hedge_ratio


class OUPairsModel(Model):
    """
    Ornstein-Uhlenbeck Pairs Trading Strategy.

    Uses cointegration to find mean-reverting pairs, then
    estimates OU process parameters for optimal entry/exit.

    Signal Logic:
    1. Test cointegration between asset pair
    2. Calculate spread: S = A - β*B (β = hedge ratio)
    3. Estimate OU parameters (θ, μ, σ)
    4. Calculate z-score of spread
    5. LONG spread: z < -2 (buy A, sell B)
    6. SHORT spread: z > 2 (sell A, buy B)

    Example:
        config = ModelConfig(
            model_id="ou_pairs",
            model_type="pairs",
            parameters={"entry_z": 2.0}
        )
        model = OUPairsModel(config)
        signal = await model.generate_pair_signal("AAPL", "MSFT", df_a, df_b)
    """

    def __init__(self, config: ModelConfig):
        """Initialize OU Pairs model."""
        super().__init__(config)
        params = config.parameters or {}

        self.ou_config = OUConfig(
            coint_lookback=params.get("coint_lookback", 252),
            coint_pvalue=params.get("coint_pvalue", 0.05),
            hedge_lookback=params.get("hedge_lookback", 60),
            ou_lookback=params.get("ou_lookback", 60),
            min_halflife=params.get("min_halflife", 2),
            max_halflife=params.get("max_halflife", 60),
            entry_z=params.get("entry_z", 2.0),
            exit_z=params.get("exit_z", 0.5),
            stop_z=params.get("stop_z", 4.0),
        )

        # Cache for pair statistics
        self._pair_cache: dict[str, PairStats] = {}

    def analyze_pair(
        self,
        symbol_a: str,
        symbol_b: str,
        prices_a: pd.Series,
        prices_b: pd.Series,
    ) -> PairStats:
        """
        Analyze a potential trading pair.

        Args:
            symbol_a: First symbol
            symbol_b: Second symbol
            prices_a: Price series for symbol A
            prices_b: Price series for symbol B

        Returns:
            PairStats with analysis results
        """
        # Align series
        combined = pd.DataFrame({"a": prices_a, "b": prices_b}).dropna()
        if len(combined) < self.ou_config.coint_lookback:
            return PairStats(
                asset_a=symbol_a,
                asset_b=symbol_b,
                hedge_ratio=0.0,
                spread=pd.Series(dtype=float),
                spread_z=0.0,
                ou_params=OUParams(0, 0, 0, float("inf"), 0),
                coint_pvalue=1.0,
                is_valid=False,
            )

        y1 = combined["a"]
        y2 = combined["b"]

        # Test cointegration
        coint_pval, hedge_ratio = test_cointegration(y1, y2)

        # Calculate spread
        spread = y1 - hedge_ratio * y2

        # Estimate OU parameters
        ou_params = estimate_ou_params(spread.iloc[-self.ou_config.ou_lookback :])

        # Calculate current z-score
        recent_spread = spread.iloc[-self.ou_config.hedge_lookback :]
        spread_mean = recent_spread.mean()
        spread_std = recent_spread.std()
        current_z = (spread.iloc[-1] - spread_mean) / (spread_std + 1e-10)

        # Check validity
        is_valid = (
            coint_pval < self.ou_config.coint_pvalue
            and self.ou_config.min_halflife <= ou_params.halflife <= self.ou_config.max_halflife
            and ou_params.theta > 0
        )

        return PairStats(
            asset_a=symbol_a,
            asset_b=symbol_b,
            hedge_ratio=hedge_ratio,
            spread=spread,
            spread_z=current_z,
            ou_params=ou_params,
            coint_pvalue=coint_pval,
            is_valid=is_valid,
        )

    async def generate(
        self,
        symbol: str,
        data: pd.DataFrame,
        timestamp: datetime,
    ) -> Signal | None:
        """
        Generate signal (not used for pairs - use generate_pair_signal).

        For pairs trading, call generate_pair_signal with both assets.
        """
        logger.warning("OUPairsModel.generate() not used for pairs. Use generate_pair_signal().")
        return None

    async def generate_pair_signal(
        self,
        symbol_a: str,
        symbol_b: str,
        prices_a: pd.Series,
        prices_b: pd.Series,
        timestamp: datetime,
    ) -> Signal | None:
        """
        Generate pairs trading signal.

        Args:
            symbol_a: First symbol
            symbol_b: Second symbol
            prices_a: Price series for A
            prices_b: Price series for B
            timestamp: Current timestamp

        Returns:
            Signal for the pair trade
        """
        pair_stats = self.analyze_pair(symbol_a, symbol_b, prices_a, prices_b)

        if not pair_stats.is_valid:
            return None

        z = pair_stats.spread_z
        halflife = pair_stats.ou_params.halflife

        # Determine signal
        signal_type = SignalType.HOLD
        direction = Direction.NEUTRAL
        spread_direction = 0

        if z < -self.ou_config.entry_z:
            # Spread too low - long spread (buy A, sell B)
            signal_type = SignalType.ENTRY
            direction = Direction.LONG
            spread_direction = 1
        elif z > self.ou_config.entry_z:
            # Spread too high - short spread (sell A, buy B)
            signal_type = SignalType.ENTRY
            direction = Direction.SHORT
            spread_direction = -1

        if signal_type == SignalType.HOLD:
            return Signal(
                symbol=f"{symbol_a}/{symbol_b}",
                timestamp=timestamp,
                signal_type=SignalType.HOLD,
                direction=Direction.NEUTRAL,
                probability=0.0,
                score=0.0,
                model_id=self.config.model_id,
                model_version=self.config.version,
                confidence=0.0,
                metadata={
                    "spread_z": z,
                    "halflife": halflife,
                    "is_valid_pair": pair_stats.is_valid,
                },
            )

        # Confidence based on z-score extremity and OU params quality
        z_confidence = min(1.0, abs(z) / 4.0)
        halflife_confidence = 1.0 - (halflife / self.ou_config.max_halflife)
        r2_confidence = pair_stats.ou_params.r_squared

        confidence = 0.3 * z_confidence + 0.3 * halflife_confidence + 0.4 * r2_confidence

        # Expected holding period
        expected_hold = int(halflife * 1.5)

        score = min(1.0, abs(z) / max(self.ou_config.stop_z, 1e-10))
        score = score if direction == Direction.LONG else -score

        return Signal(
            symbol=f"{symbol_a}/{symbol_b}",
            timestamp=timestamp,
            signal_type=signal_type,
            direction=direction,
            probability=confidence,
            score=float(score),
            model_id=self.config.model_id,
            model_version=self.config.version,
            confidence=confidence,
            metadata={
                "strategy": "ou_pairs",
                "asset_a": symbol_a,
                "asset_b": symbol_b,
                "hedge_ratio": pair_stats.hedge_ratio,
                "spread_z": z,
                "spread_direction": spread_direction,
                "ou_theta": pair_stats.ou_params.theta,
                "ou_mu": pair_stats.ou_params.mu,
                "ou_sigma": pair_stats.ou_params.sigma,
                "halflife": halflife,
                "r_squared": pair_stats.ou_params.r_squared,
                "coint_pvalue": pair_stats.coint_pvalue,
                "exit_z": self.ou_config.exit_z * spread_direction,
                "stop_z": self.ou_config.stop_z * -spread_direction,
                "expected_hold_days": expected_hold,
            },
        )


def find_cointegrated_pairs(
    prices_df: pd.DataFrame,
    pvalue_threshold: float = 0.05,
) -> list[tuple[str, str, float, float]]:
    """
    Find all cointegrated pairs in a universe.

    Args:
        prices_df: DataFrame with columns as symbols, prices as values
        pvalue_threshold: Maximum p-value for cointegration

    Returns:
        List of (symbol_a, symbol_b, pvalue, hedge_ratio) tuples
    """
    symbols = prices_df.columns.tolist()
    pairs = []

    for i, sym_a in enumerate(symbols):
        for sym_b in symbols[i + 1 :]:
            pval, hedge = test_cointegration(prices_df[sym_a], prices_df[sym_b])
            if pval < pvalue_threshold:
                pairs.append((sym_a, sym_b, pval, hedge))

    # Sort by p-value (best first)
    pairs.sort(key=lambda x: x[2])

    return pairs


def backtest(
    prices_a: pd.Series,
    prices_b: pd.Series,
    symbol_a: str = "A",
    symbol_b: str = "B",
    config: OUConfig | None = None,
) -> dict:
    """Backtest OU Pairs strategy on a single pair."""
    if config is None:
        config = OUConfig()

    model_config = ModelConfig(
        model_id=f"ou_pairs_backtest_{symbol_a}_{symbol_b}",
        model_type="pairs",
        parameters={
            "entry_z": config.entry_z,
            "exit_z": config.exit_z,
            "stop_z": config.stop_z,
        },
    )

    model = OUPairsModel(model_config)

    # Align series
    combined = pd.DataFrame({"a": prices_a, "b": prices_b}).dropna()
    prices_a = combined["a"]
    prices_b = combined["b"]

    trades = []
    position = None

    start_idx = config.coint_lookback

    for i in range(start_idx, len(combined)):
        window_a = prices_a.iloc[: i + 1]
        window_b = prices_b.iloc[: i + 1]
        timestamp = combined.index[i]

        pair_stats = model.analyze_pair(symbol_a, symbol_b, window_a, window_b)

        if not pair_stats.is_valid:
            continue

        z = pair_stats.spread_z

        # Check exits
        if position is not None:
            exit_reason = None

            # Exit on mean reversion
            if (
                position["direction"] == 1
                and z >= -config.exit_z
                or position["direction"] == -1
                and z <= config.exit_z
            ):
                exit_reason = "mean_reverted"

            # Stop loss
            if (
                position["direction"] == 1
                and z < -config.stop_z
                or position["direction"] == -1
                and z > config.stop_z
            ):
                exit_reason = "stop_loss"

            if exit_reason:
                current_spread = pair_stats.spread.iloc[-1]
                entry_spread = position["entry_spread"]
                pnl = (current_spread - entry_spread) * position["direction"]
                pnl_pct = pnl / abs(entry_spread) * 100 if entry_spread != 0 else 0

                trades.append(
                    {
                        "entry_time": position["entry_time"],
                        "exit_time": timestamp,
                        "direction": position["direction"],
                        "entry_z": position["entry_z"],
                        "exit_z": z,
                        "pnl_pct": pnl_pct,
                        "exit_reason": exit_reason,
                        "halflife": position["halflife"],
                    }
                )
                position = None

        # New entry
        if position is None:
            if z < -config.entry_z:
                position = {
                    "entry_spread": pair_stats.spread.iloc[-1],
                    "entry_time": timestamp,
                    "entry_z": z,
                    "direction": 1,
                    "halflife": pair_stats.ou_params.halflife,
                }
            elif z > config.entry_z:
                position = {
                    "entry_spread": pair_stats.spread.iloc[-1],
                    "entry_time": timestamp,
                    "entry_z": z,
                    "direction": -1,
                    "halflife": pair_stats.ou_params.halflife,
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
        "avg_halflife": trades_df["halflife"].mean(),
        "trades_df": trades_df,
    }
