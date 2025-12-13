"""Performance analytics for backtest results.

Provides comprehensive performance metrics including risk-adjusted returns,
drawdown analysis, and trade statistics.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ..core.portfolio import Trade


@dataclass
class PerformanceMetrics:
    """Container for performance metrics.

    Attributes:
        # Returns
        total_return: Total return (%)
        annualized_return: Annualized return (%)

        # Risk metrics
        volatility: Annualized volatility (%)
        downside_deviation: Downside deviation (%)

        # Risk-adjusted metrics
        sharpe_ratio: Sharpe ratio
        sortino_ratio: Sortino ratio
        calmar_ratio: Calmar ratio

        # Drawdown metrics
        max_drawdown: Maximum drawdown (%)
        avg_drawdown: Average drawdown (%)
        max_drawdown_duration: Max drawdown duration (days)

        # Trade statistics
        num_trades: Number of trades
        win_rate: Win rate (%)
        profit_factor: Profit factor
        avg_win: Average win ($)
        avg_loss: Average loss ($)
        largest_win: Largest win ($)
        largest_loss: Largest loss ($)
        avg_trade_duration: Average trade duration (days)

        # Additional metrics
        expectancy: Expected value per trade
        recovery_factor: Recovery factor (total return / max drawdown)
        equity_final: Final equity value
    """

    # Returns
    total_return: float
    annualized_return: float

    # Risk metrics
    volatility: float
    downside_deviation: float

    # Risk-adjusted metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Drawdown metrics
    max_drawdown: float
    avg_drawdown: float
    max_drawdown_duration: float

    # Trade statistics
    num_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    avg_trade_duration: float

    # Additional metrics
    expectancy: float
    recovery_factor: float
    equity_final: float


class PerformanceAnalyzer:
    """Analyzes backtest performance and generates metrics.

    Calculates risk-adjusted returns, drawdown analysis, and trade statistics.
    """

    def __init__(self, risk_free_rate: float = 0.02):
        """Initialize performance analyzer.

        Args:
            risk_free_rate: Annual risk-free rate (default: 2%)
        """
        self.risk_free_rate = risk_free_rate

    def analyze(
        self,
        equity_curve: list[tuple],
        trades: list["Trade"],
        initial_capital: float,
    ) -> PerformanceMetrics:
        """Analyze backtest performance.

        Args:
            equity_curve: List of (timestamp, equity) tuples
            trades: List of completed trades
            initial_capital: Starting capital

        Returns:
            PerformanceMetrics object
        """
        if not equity_curve:
            return self._empty_metrics()

        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(equity_curve, columns=["timestamp", "equity"])
        df.set_index("timestamp", inplace=True)
        # Remove duplicates (keep last value for each timestamp)
        df = df[~df.index.duplicated(keep="last")]

        # Calculate returns
        returns = df["equity"].pct_change().dropna()

        # Return metrics
        total_return = self._total_return(df["equity"].iloc[-1], initial_capital)
        annualized_return = self._annualized_return(df, initial_capital)

        # Risk metrics
        volatility = self._volatility(returns)
        downside_deviation = self._downside_deviation(returns, self.risk_free_rate)

        # Risk-adjusted metrics
        sharpe_ratio = self._sharpe_ratio(returns, self.risk_free_rate)
        sortino_ratio = self._sortino_ratio(returns, self.risk_free_rate)

        # Drawdown metrics
        max_dd, avg_dd, max_dd_duration = self._drawdown_analysis(df["equity"])

        calmar_ratio = annualized_return / abs(max_dd) if max_dd != 0 else 0.0

        # Trade statistics
        if trades:
            trade_stats = self._trade_statistics(trades)
        else:
            trade_stats = {
                "num_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "largest_win": 0.0,
                "largest_loss": 0.0,
                "avg_trade_duration": 0.0,
                "expectancy": 0.0,
            }

        # Recovery factor
        recovery_factor = total_return / abs(max_dd) if max_dd != 0 else 0.0

        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            downside_deviation=downside_deviation,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_dd,
            avg_drawdown=avg_dd,
            max_drawdown_duration=max_dd_duration,
            num_trades=trade_stats["num_trades"],
            win_rate=trade_stats["win_rate"],
            profit_factor=trade_stats["profit_factor"],
            avg_win=trade_stats["avg_win"],
            avg_loss=trade_stats["avg_loss"],
            largest_win=trade_stats["largest_win"],
            largest_loss=trade_stats["largest_loss"],
            avg_trade_duration=trade_stats["avg_trade_duration"],
            expectancy=trade_stats["expectancy"],
            recovery_factor=recovery_factor,
            equity_final=df["equity"].iloc[-1],
        )

    def _total_return(self, final_equity: float, initial_capital: float) -> float:
        """Calculate total return percentage.

        Args:
            final_equity: Final equity value
            initial_capital: Initial capital

        Returns:
            Total return as percentage
        """
        return ((final_equity - initial_capital) / initial_capital) * 100

    def _annualized_return(self, equity_df: pd.DataFrame, initial_capital: float) -> float:
        """Calculate annualized return.

        Args:
            equity_df: DataFrame with equity values
            initial_capital: Initial capital

        Returns:
            Annualized return as percentage
        """
        if len(equity_df) < 2:
            return 0.0

        # Calculate number of years
        time_delta = equity_df.index[-1] - equity_df.index[0]
        years = time_delta.total_seconds() / (365.25 * 24 * 3600)

        if years == 0:
            return 0.0

        # Annualized return using CAGR formula
        final_equity = equity_df["equity"].iloc[-1]
        cagr = ((final_equity / initial_capital) ** (1 / years) - 1) * 100

        return cagr

    def _volatility(self, returns: pd.Series) -> float:
        """Calculate annualized volatility.

        Args:
            returns: Series of returns

        Returns:
            Annualized volatility as percentage
        """
        if len(returns) == 0:
            return 0.0

        # Assume daily returns, annualize with sqrt(252)
        daily_vol = returns.std()
        annualized_vol = daily_vol * np.sqrt(252) * 100

        return annualized_vol

    def _downside_deviation(self, returns: pd.Series, risk_free: float) -> float:
        """Calculate downside deviation (semi-standard deviation).

        Args:
            returns: Series of returns
            risk_free: Annual risk-free rate

        Returns:
            Annualized downside deviation as percentage
        """
        if len(returns) == 0:
            return 0.0

        # Daily risk-free rate
        daily_rf = risk_free / 252

        # Only consider returns below risk-free rate
        downside_returns = returns[returns < daily_rf]

        if len(downside_returns) == 0:
            return 0.0

        # Calculate semi-standard deviation
        downside_dev = downside_returns.std()
        annualized_dd = downside_dev * np.sqrt(252) * 100

        return annualized_dd

    def _sharpe_ratio(self, returns: pd.Series, risk_free: float) -> float:
        """Calculate Sharpe ratio.

        Args:
            returns: Series of returns
            risk_free: Annual risk-free rate

        Returns:
            Sharpe ratio
        """
        if len(returns) == 0 or returns.std() == 0:
            return 0.0

        # Daily risk-free rate
        daily_rf = risk_free / 252

        # Excess returns
        excess_returns = returns - daily_rf

        # Sharpe ratio (annualized)
        sharpe = (excess_returns.mean() / returns.std()) * np.sqrt(252)

        return sharpe

    def _sortino_ratio(self, returns: pd.Series, risk_free: float) -> float:
        """Calculate Sortino ratio.

        Args:
            returns: Series of returns
            risk_free: Annual risk-free rate

        Returns:
            Sortino ratio
        """
        if len(returns) == 0:
            return 0.0

        # Daily risk-free rate
        daily_rf = risk_free / 252

        # Excess returns
        excess_returns = returns - daily_rf

        # Downside returns
        downside_returns = returns[returns < daily_rf]

        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0

        # Sortino ratio (annualized)
        sortino = (excess_returns.mean() / downside_returns.std()) * np.sqrt(252)

        return sortino

    def _drawdown_analysis(self, equity: pd.Series) -> tuple[float, float, float]:
        """Analyze drawdowns.

        Args:
            equity: Series of equity values

        Returns:
            Tuple of (max_drawdown, avg_drawdown, max_drawdown_duration)
        """
        if len(equity) == 0:
            return 0.0, 0.0, 0.0

        # Calculate running maximum
        running_max = equity.cummax()

        # Calculate drawdown
        drawdown = ((equity - running_max) / running_max) * 100

        # Max drawdown
        max_dd = drawdown.min()

        # Average drawdown (only consider drawdown periods)
        dd_periods = drawdown[drawdown < 0]
        avg_dd = dd_periods.mean() if len(dd_periods) > 0 else 0.0

        # Max drawdown duration
        # Find periods where drawdown is at maximum
        is_drawdown = drawdown < -0.01  # Consider > 0.01% as drawdown

        if not is_drawdown.any():
            return max_dd, avg_dd, 0.0

        # Calculate duration of each drawdown period
        dd_starts = is_drawdown & ~is_drawdown.shift(1, fill_value=False)
        dd_ends = ~is_drawdown & is_drawdown.shift(1, fill_value=False)

        max_duration = 0.0
        current_start = None

        for idx in equity.index:
            if dd_starts.at[idx]:
                current_start = idx
            elif dd_ends.at[idx] and current_start is not None:
                duration = (idx - current_start).total_seconds() / (24 * 3600)
                max_duration = max(max_duration, duration)
                current_start = None

        # Check if still in drawdown at end
        if current_start is not None:
            duration = (equity.index[-1] - current_start).total_seconds() / (24 * 3600)
            max_duration = max(max_duration, duration)

        return max_dd, avg_dd, max_duration

    def _trade_statistics(self, trades: list["Trade"]) -> dict:
        """Calculate trade statistics.

        Args:
            trades: List of completed trades

        Returns:
            Dictionary of trade statistics
        """
        if not trades:
            return {
                "num_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "largest_win": 0.0,
                "largest_loss": 0.0,
                "avg_trade_duration": 0.0,
                "expectancy": 0.0,
            }

        # Separate winners and losers
        winners = [t for t in trades if t.is_winner]
        losers = [t for t in trades if t.is_loser]

        # Win rate
        win_rate = (len(winners) / len(trades)) * 100

        # Profit factor
        gross_profit = sum(t.pnl for t in winners)
        gross_loss = abs(sum(t.pnl for t in losers))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Average win/loss
        avg_win = gross_profit / len(winners) if winners else 0.0
        avg_loss = gross_loss / len(losers) if losers else 0.0

        # Largest win/loss
        largest_win = max((t.pnl for t in winners), default=0.0)
        largest_loss = min((t.pnl for t in losers), default=0.0)

        # Average trade duration (in days)
        avg_duration = sum(t.duration for t in trades) / len(trades)
        avg_duration_days = avg_duration / (24 * 3600)

        # Expectancy
        expectancy = sum(t.pnl for t in trades) / len(trades)

        return {
            "num_trades": len(trades),
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "largest_win": largest_win,
            "largest_loss": largest_loss,
            "avg_trade_duration": avg_duration_days,
            "expectancy": expectancy,
        }

    def _empty_metrics(self) -> PerformanceMetrics:
        """Return empty metrics when no data available.

        Returns:
            PerformanceMetrics with all zeros
        """
        return PerformanceMetrics(
            total_return=0.0,
            annualized_return=0.0,
            volatility=0.0,
            downside_deviation=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            max_drawdown=0.0,
            avg_drawdown=0.0,
            max_drawdown_duration=0.0,
            num_trades=0,
            win_rate=0.0,
            profit_factor=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            largest_win=0.0,
            largest_loss=0.0,
            avg_trade_duration=0.0,
            expectancy=0.0,
            recovery_factor=0.0,
            equity_final=0.0,
        )
