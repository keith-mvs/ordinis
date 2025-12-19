"""
Advanced Position Sizing and Portfolio Rebalancing Engine.

Implements production-grade position sizing with:
- Kelly Criterion variants
- Volatility-adjusted sizing
- Portfolio optimization (Mean-Variance, Risk Parity)
- Correlation-aware allocation
- Dynamic rebalancing triggers

Step 5 of Trade Enhancement Roadmap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum, auto
import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SizingMethod(Enum):
    """Position sizing methods."""
    
    FIXED_DOLLAR = auto()  # Fixed dollar amount per trade
    FIXED_PERCENT = auto()  # Fixed % of portfolio
    VOLATILITY_ADJUSTED = auto()  # Adjust for asset volatility
    KELLY = auto()  # Kelly Criterion
    HALF_KELLY = auto()  # Conservative Kelly
    RISK_PARITY = auto()  # Equal risk contribution
    MEAN_VARIANCE = auto()  # Markowitz optimization
    EQUAL_WEIGHT = auto()  # Equal allocation


class RebalanceTrigger(Enum):
    """Portfolio rebalance triggers."""
    
    SCHEDULED = auto()  # Time-based (daily, weekly, monthly)
    THRESHOLD = auto()  # Drift threshold exceeded
    SIGNAL = auto()  # New trading signal
    VOLATILITY = auto()  # Volatility regime change
    DRAWDOWN = auto()  # Max drawdown reached


@dataclass
class PositionSizeResult:
    """Result of position sizing calculation."""
    
    symbol: str
    shares: int
    notional_value: Decimal
    weight: float  # Portfolio weight
    risk_contribution: float  # Contribution to portfolio risk
    sizing_method: SizingMethod
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_valid(self) -> bool:
        return self.shares > 0 and self.confidence > 0


@dataclass
class PortfolioAllocation:
    """Target portfolio allocation."""
    
    positions: dict[str, PositionSizeResult]
    total_notional: Decimal
    expected_return: float
    expected_risk: float  # Portfolio volatility
    sharpe_ratio: float
    max_position_weight: float
    diversification_ratio: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def position_count(self) -> int:
        return len(self.positions)
        
    def get_weights(self) -> dict[str, float]:
        """Get position weights."""
        return {symbol: pos.weight for symbol, pos in self.positions.items()}


@dataclass
class RebalanceAction:
    """Required rebalance action."""
    
    symbol: str
    current_shares: int
    target_shares: int
    shares_delta: int
    action: str  # "buy", "sell", or "hold"
    priority: int  # Execution priority
    reason: str
    
    @property
    def requires_trade(self) -> bool:
        return self.shares_delta != 0


@dataclass
class RebalancePlan:
    """Complete rebalance plan."""
    
    trigger: RebalanceTrigger
    actions: list[RebalanceAction]
    current_allocation: PortfolioAllocation
    target_allocation: PortfolioAllocation
    estimated_turnover: float  # % of portfolio traded
    estimated_cost: Decimal
    priority: int = 1
    expires_at: datetime | None = None
    
    @property
    def requires_action(self) -> bool:
        return any(a.requires_trade for a in self.actions)


@dataclass
class SizingConfig:
    """Configuration for position sizing."""
    
    method: SizingMethod = SizingMethod.VOLATILITY_ADJUSTED
    max_position_pct: float = 0.10  # Max 10% per position
    min_position_pct: float = 0.01  # Min 1% per position
    max_portfolio_risk: float = 0.15  # Max 15% annual volatility
    risk_free_rate: float = 0.05  # 5% risk-free rate
    kelly_fraction: float = 0.5  # Fractional Kelly
    vol_lookback_days: int = 20  # Volatility calculation period
    vol_target: float = 0.15  # Target volatility (annualized)
    correlation_lookback_days: int = 60
    min_history_days: int = 30


@dataclass
class RebalanceConfig:
    """Configuration for rebalancing."""
    
    trigger: RebalanceTrigger = RebalanceTrigger.THRESHOLD
    threshold_pct: float = 0.05  # 5% drift triggers rebalance
    min_interval_hours: int = 4  # Minimum hours between rebalances
    max_turnover_pct: float = 0.20  # Max 20% turnover per rebalance
    priority_by_drift: bool = True
    exclude_positions_below: Decimal = Decimal("100")  # Min position value


class VolatilityCalculator:
    """Calculate various volatility measures."""
    
    @staticmethod
    def realized_volatility(
        returns: pd.Series,
        window: int = 20,
        annualize: bool = True,
    ) -> pd.Series:
        """Calculate realized volatility."""
        vol = returns.rolling(window=window).std()
        if annualize:
            vol *= np.sqrt(252)
        return vol
        
    @staticmethod
    def parkinson_volatility(
        high: pd.Series,
        low: pd.Series,
        window: int = 20,
        annualize: bool = True,
    ) -> pd.Series:
        """Parkinson volatility estimator (uses high/low)."""
        log_hl = np.log(high / low)
        vol = np.sqrt((log_hl ** 2).rolling(window=window).mean() / (4 * np.log(2)))
        if annualize:
            vol *= np.sqrt(252)
        return vol
        
    @staticmethod
    def garman_klass_volatility(
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 20,
        annualize: bool = True,
    ) -> pd.Series:
        """Garman-Klass volatility estimator."""
        log_hl = np.log(high / low) ** 2
        log_co = np.log(close / open_) ** 2
        
        gk = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
        vol = np.sqrt(gk.rolling(window=window).mean())
        
        if annualize:
            vol *= np.sqrt(252)
        return vol
        
    @staticmethod
    def ewma_volatility(
        returns: pd.Series,
        span: int = 20,
        annualize: bool = True,
    ) -> pd.Series:
        """EWMA volatility (more responsive)."""
        vol = returns.ewm(span=span).std()
        if annualize:
            vol *= np.sqrt(252)
        return vol


class PositionSizer:
    """
    Advanced position sizing engine.
    
    Implements multiple sizing methods with risk management.
    
    Example:
        >>> sizer = PositionSizer(config)
        >>> sizer.set_portfolio_value(100000)
        >>> sizer.set_returns(symbol="AAPL", returns=returns_series)
        >>> 
        >>> size = sizer.calculate_size(
        ...     symbol="AAPL",
        ...     current_price=150.0,
        ...     signal_strength=0.8,
        ... )
    """
    
    def __init__(self, config: SizingConfig | None = None) -> None:
        """Initialize position sizer."""
        self.config = config or SizingConfig()
        self._portfolio_value = Decimal("0")
        self._returns: dict[str, pd.Series] = {}
        self._prices: dict[str, pd.DataFrame] = {}
        self._vol_calc = VolatilityCalculator()
        
    def set_portfolio_value(self, value: Decimal | float) -> None:
        """Set current portfolio value."""
        self._portfolio_value = Decimal(str(value))
        
    def set_returns(self, symbol: str, returns: pd.Series) -> None:
        """Set return series for a symbol."""
        self._returns[symbol] = returns
        
    def set_price_data(self, symbol: str, data: pd.DataFrame) -> None:
        """Set OHLCV price data for a symbol."""
        self._prices[symbol] = data
        
        # Calculate returns if not already set
        if symbol not in self._returns:
            self._returns[symbol] = data["close"].pct_change().dropna()
            
    def calculate_size(
        self,
        symbol: str,
        current_price: float,
        signal_strength: float = 1.0,
        win_rate: float | None = None,
        avg_win_loss_ratio: float | None = None,
        method: SizingMethod | None = None,
    ) -> PositionSizeResult:
        """
        Calculate position size.
        
        Args:
            symbol: Trading symbol
            current_price: Current asset price
            signal_strength: Signal confidence (0-1)
            win_rate: Historical win rate (for Kelly)
            avg_win_loss_ratio: Avg win / avg loss (for Kelly)
            method: Override default sizing method
            
        Returns:
            PositionSizeResult with calculated size
        """
        method = method or self.config.method
        
        if method == SizingMethod.FIXED_DOLLAR:
            return self._fixed_dollar_size(symbol, current_price, signal_strength)
        elif method == SizingMethod.FIXED_PERCENT:
            return self._fixed_percent_size(symbol, current_price, signal_strength)
        elif method == SizingMethod.VOLATILITY_ADJUSTED:
            return self._volatility_adjusted_size(symbol, current_price, signal_strength)
        elif method in (SizingMethod.KELLY, SizingMethod.HALF_KELLY):
            return self._kelly_size(
                symbol, current_price, signal_strength,
                win_rate or 0.5, avg_win_loss_ratio or 1.5,
                half_kelly=(method == SizingMethod.HALF_KELLY),
            )
        else:
            return self._fixed_percent_size(symbol, current_price, signal_strength)
            
    def _fixed_dollar_size(
        self,
        symbol: str,
        price: float,
        signal_strength: float,
    ) -> PositionSizeResult:
        """Fixed dollar amount sizing."""
        base_amount = float(self._portfolio_value) * 0.05  # 5% default
        amount = base_amount * signal_strength
        shares = int(amount / price)
        
        return PositionSizeResult(
            symbol=symbol,
            shares=shares,
            notional_value=Decimal(str(shares * price)),
            weight=float(shares * price) / float(self._portfolio_value),
            risk_contribution=0.0,  # Not calculated
            sizing_method=SizingMethod.FIXED_DOLLAR,
            confidence=signal_strength,
            metadata={"base_amount": base_amount},
        )
        
    def _fixed_percent_size(
        self,
        symbol: str,
        price: float,
        signal_strength: float,
    ) -> PositionSizeResult:
        """Fixed percentage of portfolio sizing."""
        base_pct = 0.05  # 5% default
        target_pct = min(base_pct * signal_strength, self.config.max_position_pct)
        target_pct = max(target_pct, self.config.min_position_pct)
        
        amount = float(self._portfolio_value) * target_pct
        shares = int(amount / price)
        
        return PositionSizeResult(
            symbol=symbol,
            shares=shares,
            notional_value=Decimal(str(shares * price)),
            weight=target_pct,
            risk_contribution=target_pct,  # Approximate
            sizing_method=SizingMethod.FIXED_PERCENT,
            confidence=signal_strength,
            metadata={"target_pct": target_pct},
        )
        
    def _volatility_adjusted_size(
        self,
        symbol: str,
        price: float,
        signal_strength: float,
    ) -> PositionSizeResult:
        """Volatility-adjusted position sizing."""
        returns = self._returns.get(symbol)
        
        if returns is None or len(returns) < self.config.vol_lookback_days:
            # Fall back to fixed percent
            return self._fixed_percent_size(symbol, price, signal_strength)
            
        # Calculate recent volatility
        vol = self._vol_calc.realized_volatility(
            returns,
            window=self.config.vol_lookback_days,
            annualize=True,
        )
        current_vol = vol.iloc[-1]
        
        if np.isnan(current_vol) or current_vol == 0:
            current_vol = 0.20  # Default 20% vol
            
        # Size inversely proportional to volatility
        vol_scalar = self.config.vol_target / current_vol
        base_pct = 0.05 * vol_scalar * signal_strength
        
        # Apply limits
        target_pct = min(base_pct, self.config.max_position_pct)
        target_pct = max(target_pct, self.config.min_position_pct)
        
        amount = float(self._portfolio_value) * target_pct
        shares = int(amount / price)
        
        # Risk contribution estimate
        risk_contrib = target_pct * current_vol
        
        return PositionSizeResult(
            symbol=symbol,
            shares=shares,
            notional_value=Decimal(str(shares * price)),
            weight=target_pct,
            risk_contribution=risk_contrib,
            sizing_method=SizingMethod.VOLATILITY_ADJUSTED,
            confidence=signal_strength,
            metadata={
                "volatility": float(current_vol),
                "vol_scalar": float(vol_scalar),
                "target_pct": target_pct,
            },
        )
        
    def _kelly_size(
        self,
        symbol: str,
        price: float,
        signal_strength: float,
        win_rate: float,
        win_loss_ratio: float,
        half_kelly: bool = True,
    ) -> PositionSizeResult:
        """Kelly Criterion position sizing."""
        # Kelly formula: f* = (bp - q) / b
        # where b = odds (win/loss ratio), p = win prob, q = lose prob
        b = win_loss_ratio
        p = win_rate
        q = 1 - p
        
        kelly_fraction = (b * p - q) / b
        
        # Apply fractional Kelly
        if half_kelly or self.config.kelly_fraction < 1.0:
            kelly_fraction *= self.config.kelly_fraction
            
        # Adjust by signal strength
        kelly_fraction *= signal_strength
        
        # Apply limits
        kelly_fraction = max(0, min(kelly_fraction, self.config.max_position_pct))
        
        amount = float(self._portfolio_value) * kelly_fraction
        shares = int(amount / price)
        
        return PositionSizeResult(
            symbol=symbol,
            shares=shares,
            notional_value=Decimal(str(shares * price)),
            weight=kelly_fraction,
            risk_contribution=kelly_fraction,  # Approximate
            sizing_method=SizingMethod.HALF_KELLY if half_kelly else SizingMethod.KELLY,
            confidence=signal_strength,
            metadata={
                "raw_kelly": float((b * p - q) / b),
                "applied_kelly": kelly_fraction,
                "win_rate": win_rate,
                "win_loss_ratio": win_loss_ratio,
            },
        )


class PortfolioOptimizer:
    """
    Portfolio optimization engine.
    
    Implements Mean-Variance, Risk Parity, and other optimization methods.
    """
    
    def __init__(
        self,
        config: SizingConfig | None = None,
        max_iterations: int = 1000,
    ) -> None:
        """Initialize optimizer."""
        self.config = config or SizingConfig()
        self.max_iterations = max_iterations
        self._returns: dict[str, pd.Series] = {}
        
    def set_returns(self, symbol: str, returns: pd.Series) -> None:
        """Set returns for a symbol."""
        self._returns[symbol] = returns
        
    def optimize_mean_variance(
        self,
        symbols: list[str],
        target_return: float | None = None,
        risk_aversion: float = 1.0,
    ) -> dict[str, float]:
        """
        Mean-Variance optimization (Markowitz).
        
        Args:
            symbols: Symbols to include
            target_return: Target portfolio return (None = max Sharpe)
            risk_aversion: Risk aversion parameter
            
        Returns:
            Dictionary of symbol -> weight
        """
        # Build returns matrix
        returns_df = pd.DataFrame({
            s: self._returns[s] for s in symbols if s in self._returns
        }).dropna()
        
        if len(returns_df) < self.config.min_history_days:
            # Not enough data, use equal weight
            return self._equal_weight(symbols)
            
        # Calculate expected returns and covariance
        expected_returns = returns_df.mean() * 252  # Annualized
        cov_matrix = returns_df.cov() * 252
        
        n = len(symbols)
        
        # Simple optimization: maximize Sharpe ratio
        # Using analytical solution for max Sharpe (no constraints)
        inv_cov = np.linalg.inv(cov_matrix.values)
        excess_returns = (expected_returns - self.config.risk_free_rate).values
        
        raw_weights = inv_cov @ excess_returns
        weights = raw_weights / np.sum(np.abs(raw_weights))  # Normalize
        
        # Apply position limits
        weights = np.clip(weights, -self.config.max_position_pct, self.config.max_position_pct)
        weights = weights / np.sum(np.abs(weights))  # Re-normalize
        
        return dict(zip(symbols, weights))
        
    def optimize_risk_parity(
        self,
        symbols: list[str],
    ) -> dict[str, float]:
        """
        Risk Parity optimization.
        
        Equal risk contribution from each asset.
        """
        returns_df = pd.DataFrame({
            s: self._returns[s] for s in symbols if s in self._returns
        }).dropna()
        
        if len(returns_df) < self.config.min_history_days:
            return self._equal_weight(symbols)
            
        # Calculate volatilities
        vols = returns_df.std() * np.sqrt(252)
        
        # Inverse volatility weighting (simple risk parity)
        inv_vols = 1 / vols
        weights = inv_vols / inv_vols.sum()
        
        return dict(zip(symbols, weights.values))
        
    def _equal_weight(self, symbols: list[str]) -> dict[str, float]:
        """Equal weight allocation."""
        n = len(symbols)
        weight = 1.0 / n if n > 0 else 0
        return {s: weight for s in symbols}
        
    def calculate_portfolio_metrics(
        self,
        weights: dict[str, float],
    ) -> dict[str, float]:
        """Calculate portfolio risk/return metrics."""
        symbols = list(weights.keys())
        w = np.array([weights[s] for s in symbols])
        
        returns_df = pd.DataFrame({
            s: self._returns[s] for s in symbols if s in self._returns
        }).dropna()
        
        expected_returns = returns_df.mean() * 252
        cov_matrix = returns_df.cov() * 252
        
        # Portfolio return
        port_return = np.dot(w, expected_returns)
        
        # Portfolio volatility
        port_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        
        # Sharpe ratio
        sharpe = (port_return - self.config.risk_free_rate) / port_vol if port_vol > 0 else 0
        
        # Diversification ratio
        asset_vols = np.sqrt(np.diag(cov_matrix))
        weighted_vol = np.dot(np.abs(w), asset_vols)
        div_ratio = weighted_vol / port_vol if port_vol > 0 else 1
        
        return {
            "expected_return": float(port_return),
            "volatility": float(port_vol),
            "sharpe_ratio": float(sharpe),
            "diversification_ratio": float(div_ratio),
        }


class RebalanceEngine:
    """
    Portfolio rebalancing engine.
    
    Monitors drift and generates rebalance plans.
    """
    
    def __init__(
        self,
        config: RebalanceConfig | None = None,
    ) -> None:
        """Initialize rebalance engine."""
        self.config = config or RebalanceConfig()
        self._target_weights: dict[str, float] = {}
        self._current_positions: dict[str, int] = {}
        self._current_prices: dict[str, float] = {}
        self._last_rebalance: datetime | None = None
        
    def set_target_weights(self, weights: dict[str, float]) -> None:
        """Set target portfolio weights."""
        self._target_weights = weights
        
    def set_current_positions(self, positions: dict[str, int]) -> None:
        """Set current positions (symbol -> shares)."""
        self._current_positions = positions
        
    def set_current_prices(self, prices: dict[str, float]) -> None:
        """Set current prices."""
        self._current_prices = prices
        
    def calculate_drift(self, portfolio_value: float) -> dict[str, float]:
        """
        Calculate drift from target weights.
        
        Returns:
            Dictionary of symbol -> drift (current - target)
        """
        drift = {}
        
        for symbol in set(self._target_weights.keys()) | set(self._current_positions.keys()):
            target = self._target_weights.get(symbol, 0)
            
            shares = self._current_positions.get(symbol, 0)
            price = self._current_prices.get(symbol, 0)
            current_value = shares * price
            current_weight = current_value / portfolio_value if portfolio_value > 0 else 0
            
            drift[symbol] = current_weight - target
            
        return drift
        
    def check_rebalance_needed(self, portfolio_value: float) -> tuple[bool, str]:
        """
        Check if rebalance is needed.
        
        Returns:
            (needs_rebalance, reason)
        """
        # Check minimum interval
        if self._last_rebalance:
            elapsed = (datetime.utcnow() - self._last_rebalance).total_seconds() / 3600
            if elapsed < self.config.min_interval_hours:
                return False, f"Too soon ({elapsed:.1f}h < {self.config.min_interval_hours}h)"
                
        # Check drift
        drift = self.calculate_drift(portfolio_value)
        max_drift = max(abs(d) for d in drift.values()) if drift else 0
        
        if max_drift > self.config.threshold_pct:
            return True, f"Drift exceeded threshold ({max_drift:.1%} > {self.config.threshold_pct:.1%})"
            
        return False, "No rebalance needed"
        
    def generate_plan(
        self,
        portfolio_value: float,
        trigger: RebalanceTrigger = RebalanceTrigger.THRESHOLD,
    ) -> RebalancePlan:
        """
        Generate rebalance plan.
        
        Args:
            portfolio_value: Current portfolio value
            trigger: Trigger for this rebalance
            
        Returns:
            RebalancePlan with actions
        """
        actions: list[RebalanceAction] = []
        
        drift = self.calculate_drift(portfolio_value)
        
        # Sort by absolute drift (prioritize larger drifts)
        sorted_symbols = sorted(
            drift.keys(),
            key=lambda s: abs(drift[s]),
            reverse=True,
        )
        
        total_turnover = Decimal("0")
        
        for priority, symbol in enumerate(sorted_symbols):
            target_weight = self._target_weights.get(symbol, 0)
            current_shares = self._current_positions.get(symbol, 0)
            price = self._current_prices.get(symbol, 0)
            
            if price == 0:
                continue
                
            target_value = portfolio_value * target_weight
            target_shares = int(target_value / price)
            
            shares_delta = target_shares - current_shares
            
            if shares_delta == 0:
                action_type = "hold"
            elif shares_delta > 0:
                action_type = "buy"
            else:
                action_type = "sell"
                
            # Check turnover limit
            trade_value = abs(shares_delta * price)
            if float(total_turnover + Decimal(str(trade_value))) / portfolio_value > self.config.max_turnover_pct:
                continue  # Skip to avoid exceeding turnover
                
            total_turnover += Decimal(str(trade_value))
            
            actions.append(RebalanceAction(
                symbol=symbol,
                current_shares=current_shares,
                target_shares=target_shares,
                shares_delta=shares_delta,
                action=action_type,
                priority=priority,
                reason=f"Drift: {drift[symbol]:+.1%}",
            ))
            
        estimated_turnover = float(total_turnover) / portfolio_value if portfolio_value > 0 else 0
        
        # Create allocation snapshots (simplified)
        current_allocation = PortfolioAllocation(
            positions={},  # Would populate from current state
            total_notional=Decimal(str(portfolio_value)),
            expected_return=0.0,
            expected_risk=0.0,
            sharpe_ratio=0.0,
            max_position_weight=0.0,
            diversification_ratio=1.0,
        )
        
        target_allocation = PortfolioAllocation(
            positions={},  # Would populate from target
            total_notional=Decimal(str(portfolio_value)),
            expected_return=0.0,
            expected_risk=0.0,
            sharpe_ratio=0.0,
            max_position_weight=max(self._target_weights.values()) if self._target_weights else 0,
            diversification_ratio=1.0,
        )
        
        return RebalancePlan(
            trigger=trigger,
            actions=actions,
            current_allocation=current_allocation,
            target_allocation=target_allocation,
            estimated_turnover=estimated_turnover,
            estimated_cost=total_turnover * Decimal("0.001"),  # Rough estimate
            expires_at=datetime.utcnow() + timedelta(hours=4),
        )
        
    def mark_rebalanced(self) -> None:
        """Mark that rebalance was executed."""
        self._last_rebalance = datetime.utcnow()
