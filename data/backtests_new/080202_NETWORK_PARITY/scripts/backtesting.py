#!/usr/bin/env python3
"""
Backtesting Engine for Network Parity Strategy

GPU-accelerated backtesting with transaction cost modeling and
comprehensive performance metrics.

Author: Ordinis Quantitative Research
Version: 1.0.0
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

# GPU support
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = np  # type: ignore

from config import (
    NetworkParityParams,
    TransactionCostConfig,
    BacktestingConfig,
)
from data_pipeline import DataPipelineResult, compute_correlation_matrix

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Trade:
    """Record of a single trade."""

    symbol: str
    entry_date: datetime
    exit_date: datetime | None = None
    entry_price: float = 0.0
    exit_price: float = 0.0
    direction: int = 0  # 1 = long, -1 = short
    shares: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "entry_date": self.entry_date.isoformat() if self.entry_date else None,
            "exit_date": self.exit_date.isoformat() if self.exit_date else None,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "direction": self.direction,
            "shares": self.shares,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
            "exit_reason": self.exit_reason,
        }


@dataclass
class Position:
    """Current position in a symbol."""

    symbol: str
    shares: float = 0.0
    entry_price: float = 0.0
    entry_date: datetime | None = None
    direction: int = 0
    highest_price: float = 0.0  # For trailing stop
    target_weight: float = 0.0

    @property
    def is_open(self) -> bool:
        """Check if position is open."""
        return self.shares != 0


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    symbol: str | None = None  # None for portfolio-level
    params: dict[str, Any] = field(default_factory=dict)

    # Performance metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    num_trades: int = 0
    avg_trade_pnl: float = 0.0
    avg_trade_pnl_pct: float = 0.0

    # Detailed data
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    daily_returns: list[float] = field(default_factory=list)
    positions_history: list[dict] = field(default_factory=list)

    # Network-specific metrics
    avg_centrality: float = 0.0
    network_density: float = 0.0
    rebalance_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "symbol": self.symbol,
            "params": self.params,
            "metrics": {
                "total_return": self.total_return,
                "annualized_return": self.annualized_return,
                "sharpe_ratio": self.sharpe_ratio,
                "sortino_ratio": self.sortino_ratio,
                "max_drawdown": self.max_drawdown,
                "win_rate": self.win_rate,
                "profit_factor": self.profit_factor,
                "num_trades": self.num_trades,
                "avg_trade_pnl": self.avg_trade_pnl,
                "avg_trade_pnl_pct": self.avg_trade_pnl_pct,
            },
            "network_metrics": {
                "avg_centrality": self.avg_centrality,
                "network_density": self.network_density,
                "rebalance_count": self.rebalance_count,
            },
            "trades": [t.to_dict() for t in self.trades[:50]],  # Limit for JSON size
            "equity_curve_sample": self.equity_curve[:100] if self.equity_curve else [],
        }

    def compute_score(
        self,
        return_weight: float = 0.40,
        sortino_weight: float = 0.35,
        win_rate_weight: float = 0.15,
        drawdown_penalty: float = 0.10,
    ) -> float:
        """
        Compute composite score for optimization.

        Args:
            return_weight: Weight for total return
            sortino_weight: Weight for Sortino ratio
            win_rate_weight: Weight for win rate
            drawdown_penalty: Weight for drawdown penalty

        Returns:
            Composite score
        """
        # Penalize if too few trades
        if self.num_trades < 5:
            return -np.inf

        # Penalize excessive drawdown
        if self.max_drawdown > 0.30:
            return -np.inf

        score = (
            return_weight * self.total_return
            + sortino_weight * (self.sortino_ratio / 10.0)  # Normalize
            + win_rate_weight * self.win_rate
            - drawdown_penalty * abs(self.max_drawdown)
        )

        return score


# =============================================================================
# NETWORK ANALYSIS
# =============================================================================

def compute_eigenvector_centrality(
    adj_matrix: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Compute eigenvector centrality using power iteration.

    Args:
        adj_matrix: Adjacency matrix (n x n)
        max_iter: Maximum iterations
        tol: Convergence tolerance

    Returns:
        Centrality scores (n,)
    """
    n = adj_matrix.shape[0]
    xp = cp if GPU_AVAILABLE else np

    # Transfer to GPU if available
    A = xp.asarray(adj_matrix)

    # Initialize with uniform
    c = xp.ones(n) / np.sqrt(n)

    for _ in range(max_iter):
        c_new = A @ c
        norm = xp.linalg.norm(c_new)
        if norm > 0:
            c_new = c_new / norm

        # Check convergence
        if xp.linalg.norm(c_new - c) < tol:
            break
        c = c_new

    # Transfer back to CPU
    if GPU_AVAILABLE:
        return cp.asnumpy(c)
    return c


def compute_degree_centrality(adj_matrix: np.ndarray) -> np.ndarray:
    """
    Compute degree centrality.

    Args:
        adj_matrix: Adjacency matrix (n x n)

    Returns:
        Centrality scores (n,)
    """
    n = adj_matrix.shape[0]
    degrees = adj_matrix.sum(axis=1)
    return degrees / (n - 1)


def build_correlation_network(
    returns: pd.DataFrame,
    threshold: float = 0.3,
    method: str = "eigenvector",
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """
    Build correlation network and compute centrality.

    Args:
        returns: Returns DataFrame (rows = dates, columns = symbols)
        threshold: Minimum absolute correlation for edge
        method: Centrality method (eigenvector, degree)

    Returns:
        (adjacency_matrix, centrality_scores, symbol_centrality_dict)
    """
    # Compute correlation matrix
    corr = returns.corr().values
    n = corr.shape[0]

    # Build adjacency matrix
    adj = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if abs(corr[i, j]) >= threshold:
                adj[i, j] = 1
                adj[j, i] = 1

    # Compute centrality
    if method == "eigenvector":
        centrality = compute_eigenvector_centrality(adj)
    else:  # degree
        centrality = compute_degree_centrality(adj)

    # Map to symbols
    symbols = returns.columns.tolist()
    symbol_centrality = {sym: centrality[i] for i, sym in enumerate(symbols)}

    return adj, centrality, symbol_centrality


def compute_network_weights(
    centrality: dict[str, float],
    decay: float = 0.5,
    min_weight: float = 0.02,
    max_weight: float = 0.30,
    epsilon: float = 0.1,
) -> dict[str, float]:
    """
    Compute portfolio weights inversely proportional to centrality.

    Args:
        centrality: Symbol -> centrality score
        decay: Power for inverse weighting
        min_weight: Minimum position weight
        max_weight: Maximum position weight
        epsilon: Small constant to avoid division by zero

    Returns:
        Symbol -> target weight
    """
    # Inverse centrality
    raw_weights = {}
    for sym, c in centrality.items():
        raw_weights[sym] = (c + epsilon) ** (-decay)

    # Normalize
    total = sum(raw_weights.values())
    weights = {sym: w / total for sym, w in raw_weights.items()}

    # Apply bounds
    for sym in weights:
        weights[sym] = max(min_weight, min(max_weight, weights[sym]))

    # Re-normalize after bounding
    total = sum(weights.values())
    weights = {sym: w / total for sym, w in weights.items()}

    return weights


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

class BacktestEngine:
    """
    Backtesting engine for Network Parity strategy.

    Supports GPU acceleration and comprehensive transaction cost modeling.
    """

    def __init__(
        self,
        params: NetworkParityParams,
        config: BacktestingConfig | None = None,
    ):
        """
        Initialize backtest engine.

        Args:
            params: Strategy parameters
            config: Backtesting configuration
        """
        self.params = params
        self.config = config or BacktestingConfig()

        # State
        self.capital = self.config.initial_capital
        self.positions: dict[str, Position] = {}
        self.trades: list[Trade] = []
        self.equity_curve: list[float] = []
        self.daily_returns: list[float] = []

        # Network state
        self.current_weights: dict[str, float] = {}
        self.current_centrality: dict[str, float] = {}
        self.last_rebalance_idx = -999
        self.rebalance_count = 0

    def reset(self) -> None:
        """Reset engine state."""
        self.capital = self.config.initial_capital
        self.positions.clear()
        self.trades.clear()
        self.equity_curve.clear()
        self.daily_returns.clear()
        self.current_weights.clear()
        self.current_centrality.clear()
        self.last_rebalance_idx = -999
        self.rebalance_count = 0

    def _compute_transaction_cost(self, value: float) -> float:
        """Compute transaction cost for a trade value."""
        return value * self.config.transaction_costs.total_cost_pct

    def _get_portfolio_value(self, prices: dict[str, float]) -> float:
        """Compute current portfolio value."""
        value = self.capital
        for sym, pos in self.positions.items():
            if pos.is_open and sym in prices:
                value += pos.shares * prices[sym]
        return value

    def _compute_z_scores(
        self,
        prices: pd.DataFrame,
        lookback: int = 20,
    ) -> pd.DataFrame:
        """Compute rolling z-scores for mean reversion signals."""
        rolling_mean = prices.rolling(lookback).mean()
        rolling_std = prices.rolling(lookback).std()
        z_scores = (prices - rolling_mean) / (rolling_std + 1e-10)
        return z_scores

    def _check_stop_conditions(
        self,
        pos: Position,
        current_price: float,
    ) -> str | None:
        """
        Check if stop-loss, take-profit, or trailing stop triggered.

        Returns:
            Exit reason or None
        """
        if not pos.is_open:
            return None

        pnl_pct = (current_price - pos.entry_price) / pos.entry_price * pos.direction

        # Stop loss
        if pnl_pct <= -self.params.stop_loss_pct:
            return "stop_loss"

        # Take profit
        if pnl_pct >= self.params.take_profit_pct:
            return "take_profit"

        # Trailing stop
        if current_price > pos.highest_price:
            pos.highest_price = current_price
        elif pos.direction == 1:  # Long position
            from_high = (pos.highest_price - current_price) / pos.highest_price
            if from_high >= self.params.trailing_stop_pct:
                return "trailing_stop"

        return None

    def run(
        self,
        data: DataPipelineResult,
        symbols: list[str] | None = None,
    ) -> BacktestResult:
        """
        Run backtest on provided data.

        Args:
            data: Loaded market data
            symbols: Optional subset of symbols to trade

        Returns:
            BacktestResult with performance metrics
        """
        self.reset()

        # Get symbols to trade
        if symbols is None:
            symbols = list(data.data.keys())

        if len(symbols) < 2:
            logger.warning("Need at least 2 symbols for correlation network")
            return BacktestResult(params=self.params.to_dict())

        # Get daily returns for network construction
        returns_df = data.get_returns_matrix("1D")

        # Align all data to common index
        # For simplicity, use the first symbol's index
        first_sym = symbols[0]
        if first_sym not in data.data:
            return BacktestResult(params=self.params.to_dict())

        price_data = {}
        for sym in symbols:
            if sym in data.data:
                md = data.data[sym]
                daily = md.to_daily()
                price_data[sym] = daily.bars["close"]

        # Align to common dates
        price_df = pd.DataFrame(price_data).dropna()
        if len(price_df) < self.params.corr_lookback + 10:
            logger.warning(f"Insufficient data: {len(price_df)} bars")
            return BacktestResult(params=self.params.to_dict())

        n_bars = len(price_df)
        dates = price_df.index.tolist()

        # Compute z-scores for signals
        z_scores = self._compute_z_scores(price_df, self.params.momentum_lookback)

        # Main backtest loop
        for i in range(self.params.corr_lookback, n_bars):
            current_date = dates[i]
            prices = {sym: price_df[sym].iloc[i] for sym in symbols if sym in price_df.columns}

            # Check for rebalance
            if i - self.last_rebalance_idx >= self.params.recalc_frequency:
                # Get lookback returns
                lookback_returns = returns_df.iloc[max(0, i - self.params.corr_lookback):i]

                if len(lookback_returns) >= 10:
                    # Build network and compute weights
                    _, _, self.current_centrality = build_correlation_network(
                        lookback_returns,
                        threshold=self.params.corr_threshold,
                        method=self.params.centrality_method,
                    )

                    self.current_weights = compute_network_weights(
                        self.current_centrality,
                        decay=self.params.weight_decay,
                        min_weight=self.params.min_weight,
                        max_weight=self.params.max_weight,
                    )

                    self.last_rebalance_idx = i
                    self.rebalance_count += 1

            # Portfolio value before trading
            portfolio_value = self._get_portfolio_value(prices)

            # Process each symbol
            for sym in symbols:
                if sym not in prices or sym not in z_scores.columns:
                    continue

                current_price = prices[sym]
                current_z = z_scores[sym].iloc[i]

                # Check existing position
                if sym in self.positions and self.positions[sym].is_open:
                    pos = self.positions[sym]

                    # Check stop conditions
                    exit_reason = self._check_stop_conditions(pos, current_price)

                    # Check z-score exit
                    if exit_reason is None and abs(current_z) <= self.params.z_score_exit:
                        exit_reason = "z_score_exit"

                    if exit_reason:
                        # Close position
                        pnl = (current_price - pos.entry_price) * pos.shares * pos.direction
                        pnl -= self._compute_transaction_cost(abs(pos.shares * current_price))

                        trade = Trade(
                            symbol=sym,
                            entry_date=pos.entry_date,
                            exit_date=current_date,
                            entry_price=pos.entry_price,
                            exit_price=current_price,
                            direction=pos.direction,
                            shares=pos.shares,
                            pnl=pnl,
                            pnl_pct=pnl / (pos.entry_price * pos.shares) if pos.shares > 0 else 0,
                            exit_reason=exit_reason,
                        )
                        self.trades.append(trade)
                        self.capital += pos.shares * current_price + pnl
                        self.positions[sym] = Position(symbol=sym)

                else:
                    # Check for entry signal
                    target_weight = self.current_weights.get(sym, 0)

                    if target_weight > 0 and abs(current_z) >= self.params.z_score_entry:
                        # Determine direction (mean reversion: opposite of z-score)
                        direction = -1 if current_z > 0 else 1

                        # Calculate position size
                        target_value = portfolio_value * target_weight
                        shares = target_value / current_price
                        cost = self._compute_transaction_cost(target_value)

                        if self.capital >= target_value + cost:
                            self.positions[sym] = Position(
                                symbol=sym,
                                shares=shares,
                                entry_price=current_price,
                                entry_date=current_date,
                                direction=direction,
                                highest_price=current_price,
                                target_weight=target_weight,
                            )
                            self.capital -= target_value + cost

            # Record equity
            portfolio_value = self._get_portfolio_value(prices)
            self.equity_curve.append(portfolio_value)

            if len(self.equity_curve) > 1:
                daily_ret = (portfolio_value - self.equity_curve[-2]) / self.equity_curve[-2]
                self.daily_returns.append(daily_ret)

        # Compute final metrics
        return self._compute_metrics(symbols)

    def _compute_metrics(self, symbols: list[str]) -> BacktestResult:
        """Compute performance metrics from backtest results."""
        result = BacktestResult(
            params=self.params.to_dict(),
            trades=self.trades.copy(),
            equity_curve=self.equity_curve.copy(),
            daily_returns=self.daily_returns.copy(),
        )

        if not self.equity_curve:
            return result

        # Basic returns
        initial = self.config.initial_capital
        final = self.equity_curve[-1] if self.equity_curve else initial
        result.total_return = (final - initial) / initial
        result.num_trades = len(self.trades)

        # Annualized return (assume 252 trading days)
        n_days = len(self.equity_curve)
        if n_days > 1:
            result.annualized_return = (1 + result.total_return) ** (252 / n_days) - 1

        # Sharpe ratio
        if self.daily_returns:
            returns_arr = np.array(self.daily_returns)
            mean_ret = np.mean(returns_arr)
            std_ret = np.std(returns_arr)
            if std_ret > 0:
                result.sharpe_ratio = np.sqrt(252) * mean_ret / std_ret

            # Sortino ratio (only downside volatility)
            downside = returns_arr[returns_arr < 0]
            if len(downside) > 0:
                downside_std = np.std(downside)
                if downside_std > 0:
                    result.sortino_ratio = np.sqrt(252) * mean_ret / downside_std

        # Maximum drawdown
        if self.equity_curve:
            equity = np.array(self.equity_curve)
            peak = np.maximum.accumulate(equity)
            drawdown = (peak - equity) / peak
            result.max_drawdown = float(np.max(drawdown))

        # Trade statistics
        if self.trades:
            wins = [t for t in self.trades if t.pnl > 0]
            losses = [t for t in self.trades if t.pnl <= 0]

            result.win_rate = len(wins) / len(self.trades)

            total_wins = sum(t.pnl for t in wins) if wins else 0
            total_losses = abs(sum(t.pnl for t in losses)) if losses else 0

            if total_losses > 0:
                result.profit_factor = total_wins / total_losses

            result.avg_trade_pnl = np.mean([t.pnl for t in self.trades])
            result.avg_trade_pnl_pct = np.mean([t.pnl_pct for t in self.trades])

        # Network metrics
        if self.current_centrality:
            result.avg_centrality = np.mean(list(self.current_centrality.values()))
        result.rebalance_count = self.rebalance_count

        return result


def run_single_symbol_backtest(
    data: DataPipelineResult,
    symbol: str,
    params: NetworkParityParams,
    config: BacktestingConfig | None = None,
) -> BacktestResult:
    """
    Run backtest for a single symbol (for per-symbol reporting).

    Args:
        data: Loaded market data
        symbol: Symbol to backtest
        params: Strategy parameters
        config: Backtesting configuration

    Returns:
        BacktestResult for the symbol
    """
    # For single symbol, use simplified momentum/mean-reversion
    config = config or BacktestingConfig()

    if symbol not in data.data:
        return BacktestResult(symbol=symbol, params=params.to_dict())

    md = data.data[symbol]
    daily = md.to_daily()
    prices = daily.bars["close"]

    if len(prices) < params.momentum_lookback + 10:
        return BacktestResult(symbol=symbol, params=params.to_dict())

    # Compute z-scores
    rolling_mean = prices.rolling(params.momentum_lookback).mean()
    rolling_std = prices.rolling(params.momentum_lookback).std()
    z_scores = (prices - rolling_mean) / (rolling_std + 1e-10)

    # Simple backtest
    capital = config.initial_capital
    position = 0.0
    entry_price = 0.0
    entry_date = None
    trades = []
    equity_curve = []

    dates = prices.index.tolist()
    n = len(prices)

    for i in range(params.momentum_lookback, n):
        current_price = prices.iloc[i]
        current_z = z_scores.iloc[i]
        current_date = dates[i]

        if position == 0:
            # Entry signal
            if abs(current_z) >= params.z_score_entry:
                direction = -1 if current_z > 0 else 1
                position_value = capital * params.max_weight
                shares = position_value / current_price
                cost = position_value * config.transaction_costs.total_cost_pct

                if capital >= position_value + cost:
                    position = shares * direction
                    entry_price = current_price
                    entry_date = current_date
                    capital -= position_value + cost

        else:
            # Exit signal
            should_exit = False
            exit_reason = ""

            pnl_pct = (current_price - entry_price) / entry_price * np.sign(position)

            if pnl_pct <= -params.stop_loss_pct:
                should_exit = True
                exit_reason = "stop_loss"
            elif pnl_pct >= params.take_profit_pct:
                should_exit = True
                exit_reason = "take_profit"
            elif abs(current_z) <= params.z_score_exit:
                should_exit = True
                exit_reason = "z_score_exit"

            if should_exit:
                exit_value = abs(position) * current_price
                cost = exit_value * config.transaction_costs.total_cost_pct
                pnl = (current_price - entry_price) * position - cost

                trades.append(Trade(
                    symbol=symbol,
                    entry_date=entry_date,
                    exit_date=current_date,
                    entry_price=entry_price,
                    exit_price=current_price,
                    direction=int(np.sign(position)),
                    shares=abs(position),
                    pnl=pnl,
                    pnl_pct=pnl / (entry_price * abs(position)) if position != 0 else 0,
                    exit_reason=exit_reason,
                ))

                capital += exit_value + pnl
                position = 0

        # Portfolio value
        pv = capital + abs(position) * current_price if position != 0 else capital
        equity_curve.append(pv)

    # Compute result
    result = BacktestResult(
        symbol=symbol,
        params=params.to_dict(),
        trades=trades,
        equity_curve=equity_curve,
    )

    if equity_curve:
        initial = config.initial_capital
        final = equity_curve[-1]
        result.total_return = (final - initial) / initial
        result.num_trades = len(trades)

        if len(equity_curve) > 1:
            returns = np.diff(equity_curve) / equity_curve[:-1]
            result.daily_returns = returns.tolist()

            if len(returns) > 0:
                mean_ret = np.mean(returns)
                std_ret = np.std(returns)
                if std_ret > 0:
                    result.sharpe_ratio = np.sqrt(252) * mean_ret / std_ret

                downside = returns[returns < 0]
                if len(downside) > 0:
                    result.sortino_ratio = np.sqrt(252) * mean_ret / np.std(downside)

            peak = np.maximum.accumulate(equity_curve)
            drawdown = (peak - equity_curve) / peak
            result.max_drawdown = float(np.max(drawdown))

        if trades:
            wins = [t for t in trades if t.pnl > 0]
            result.win_rate = len(wins) / len(trades) if trades else 0
            result.avg_trade_pnl = np.mean([t.pnl for t in trades])

    return result


if __name__ == "__main__":
    from data_pipeline import DataPipeline
    from config import NetworkParityParams, BacktestingConfig

    print(f"GPU Available: {GPU_AVAILABLE}")

    # Load test data
    pipeline = DataPipeline()
    test_symbols = ["RIOT", "MARA", "SOFI", "HOOD", "GME", "PLUG"]
    result = pipeline.load_universe(test_symbols, aggregate_mins=5)

    if result.n_symbols >= 2:
        # Run backtest
        params = NetworkParityParams()
        config = BacktestingConfig()

        engine = BacktestEngine(params, config)
        bt_result = engine.run(result, test_symbols)

        print(f"\nBacktest Results:")
        print(f"  Total Return: {bt_result.total_return:.2%}")
        print(f"  Sharpe Ratio: {bt_result.sharpe_ratio:.2f}")
        print(f"  Sortino Ratio: {bt_result.sortino_ratio:.2f}")
        print(f"  Max Drawdown: {bt_result.max_drawdown:.2%}")
        print(f"  Win Rate: {bt_result.win_rate:.1%}")
        print(f"  Trades: {bt_result.num_trades}")
        print(f"  Rebalances: {bt_result.rebalance_count}")

        # Compute score
        score = bt_result.compute_score()
        print(f"  Optimization Score: {score:.4f}")
    else:
        print(f"Insufficient symbols loaded: {result.symbols_loaded}")
