"""
Signal Analysis Framework - Individual Signal Testing Infrastructure
This script provides the foundation for testing individual signals in isolation.

Usage:
    python signal_analysis_framework.py --signal momentum --lookback 6 --universe large
"""

from dataclasses import dataclass
import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SignalConfig:
    """Configuration for signal testing"""

    name: str
    lookback_periods: list[int]  # For momentum: [3, 6, 12] months
    update_frequency: str  # 'daily' or 'quarterly'
    direction: str  # 'positive' (higher=better) or 'inverse' (lower=better)
    min_observations: int = 20  # Minimum data points required


@dataclass
class BacktestResults:
    """Container for backtest results"""

    signal_name: str
    period: str
    total_return: float
    annual_return: float
    annual_volatility: float
    sharpe_ratio: float
    information_ratio: float
    win_rate: float
    max_drawdown: float
    avg_drawdown: float
    profit_factor: float
    trades: pd.DataFrame


class SignalCalculator:
    """Base class for signal calculations"""

    def __init__(self, config: SignalConfig):
        self.config = config
        self.signal_values = {}

    def calculate_pe_signal(self, prices: pd.Series, earnings: pd.Series) -> pd.Series:
        """
        Calculate P/E ratio signal (inverse: lower P/E = higher signal)

        Args:
            prices: Daily close prices
            earnings: Trailing 12-month EPS

        Returns:
            Signal values (0-100 scale, higher = stronger buy signal)
        """
        pe_ratio = prices / earnings

        # Handle edge cases
        pe_ratio = pe_ratio.replace([np.inf, -np.inf], np.nan)
        pe_ratio = pe_ratio[pe_ratio > 0]  # Exclude negative/zero earnings

        # Percentile rank (0-100)
        pe_percentile = pe_ratio.rank(pct=True) * 100

        # Inverse: low P/E should be high signal
        pe_signal = 100 - pe_percentile

        return pe_signal

    def calculate_momentum_signal(self, prices: pd.Series, lookback_months: int) -> pd.Series:
        """
        Calculate momentum signal (positive: higher recent return = higher signal)

        Args:
            prices: Daily close prices
            lookback_months: Lookback period in months

        Returns:
            Signal values (0-100 scale)
        """
        lookback_days = lookback_months * 21

        # Calculate returns
        returns = prices.pct_change(periods=lookback_days)

        # Percentile rank
        momentum_percentile = returns.rank(pct=True) * 100

        return momentum_percentile

    def calculate_earnings_growth_signal(
        self, current_eps: pd.Series, prior_eps: pd.Series
    ) -> pd.Series:
        """
        Calculate earnings growth signal

        Args:
            current_eps: Current year EPS
            prior_eps: Prior year EPS

        Returns:
            Signal values (0-100 scale)
        """
        # Calculate growth rate
        growth = (current_eps - prior_eps) / prior_eps.abs()

        # Handle edge cases
        growth = growth.replace([np.inf, -np.inf], np.nan)
        growth = growth[prior_eps > 0]  # Exclude negative prior earnings

        # Percentile rank
        growth_percentile = growth.rank(pct=True) * 100

        return growth_percentile


class SignalBacktester:
    """Backtests individual signals"""

    def __init__(self, min_portfolio_size: int = 30):
        self.min_portfolio_size = min_portfolio_size

    def run_signal_backtest(
        self,
        prices: pd.DataFrame,  # Columns: tickers, Index: dates
        signal_values: pd.DataFrame,  # Signal values (0-100 scale)
        holding_period_days: int = 30,
        rebalance_frequency: str = "monthly",
        transaction_cost_bps: int = 10,
        long_percentile: tuple[int, int] = (90, 100),  # Top 10%
        short_percentile: tuple[int, int] = (0, 10),  # Bottom 10%
    ) -> BacktestResults:
        """
        Run backtest on signal

        Args:
            prices: Historical prices (tickers as columns)
            signal_values: Signal rankings (0-100)
            holding_period_days: Days to hold position
            rebalance_frequency: 'daily', 'weekly', 'monthly', 'quarterly'
            transaction_cost_bps: Round-trip cost in basis points
            long_percentile: Percentile range for long portfolio
            short_percentile: Percentile range for short portfolio

        Returns:
            BacktestResults with full performance metrics
        """

        # Initialize tracking
        daily_returns = []
        trades = []
        portfolio_values = [1.0]
        peak_value = 1.0

        # Determine rebalance dates
        rebalance_dates = self._get_rebalance_dates(prices.index, rebalance_frequency)

        for date_idx, date in enumerate(prices.index):
            # Check if rebalance needed
            if date in rebalance_dates:
                # Get signal values on this date
                signals = signal_values.loc[date]

                # Skip if insufficient data
                if signals.isna().sum() > len(signals) * 0.5:
                    continue

                # Select long and short portfolios
                long_tickers = self._select_portfolio(signals, percentile_range=long_percentile)
                short_tickers = self._select_portfolio(signals, percentile_range=short_percentile)

                # Skip if insufficient portfolios
                if len(long_tickers) < self.min_portfolio_size // 2:
                    continue

                # Calculate returns for holding period
                for hold_date_idx in range(
                    date_idx + 1, min(date_idx + holding_period_days + 1, len(prices))
                ):
                    hold_date = prices.index[hold_date_idx]

                    # Calculate returns
                    long_return = self._calculate_portfolio_return(
                        prices, long_tickers, date, hold_date, transaction_cost_bps
                    )

                    short_return = self._calculate_portfolio_return(
                        prices, short_tickers, date, hold_date, transaction_cost_bps
                    )

                    spread_return = long_return - short_return
                    daily_returns.append(spread_return)

                    # Track trade
                    trades.append(
                        {
                            "entry_date": date,
                            "exit_date": hold_date,
                            "long_return": long_return,
                            "short_return": short_return,
                            "spread_return": spread_return,
                            "long_tickers": len(long_tickers),
                            "short_tickers": len(short_tickers),
                        }
                    )

                    # Update equity curve
                    if daily_returns:
                        new_value = portfolio_values[-1] * (1 + spread_return)
                        portfolio_values.append(new_value)
                        peak_value = max(peak_value, new_value)

        # Calculate metrics
        results = self._calculate_metrics(daily_returns, portfolio_values, pd.DataFrame(trades))

        return results

    def _select_portfolio(self, signals: pd.Series, percentile_range: tuple[int, int]) -> list[str]:
        """Select stocks in percentile range"""
        valid_signals = signals.dropna()
        if len(valid_signals) == 0:
            return []

        min_signal = np.percentile(valid_signals, percentile_range[0])
        max_signal = np.percentile(valid_signals, percentile_range[1])

        selected = valid_signals[
            (valid_signals >= min_signal) & (valid_signals <= max_signal)
        ].index.tolist()

        return selected

    def _calculate_portfolio_return(
        self,
        prices: pd.DataFrame,
        tickers: list[str],
        start_date,
        end_date,
        transaction_cost_bps: int,
    ) -> float:
        """Calculate equal-weight portfolio return"""

        if not tickers or start_date not in prices.index or end_date not in prices.index:
            return 0.0

        start_prices = prices.loc[start_date, tickers]
        end_prices = prices.loc[end_date, tickers]

        # Handle missing data
        valid_mask = ~(start_prices.isna() | end_prices.isna())
        if valid_mask.sum() < len(tickers) * 0.5:
            return 0.0

        # Calculate return
        returns = (end_prices - start_prices) / start_prices
        portfolio_return = returns[valid_mask].mean()

        # Apply transaction costs (entry and exit)
        transaction_cost = 2 * (transaction_cost_bps / 10000)
        net_return = portfolio_return - transaction_cost

        return net_return

    def _get_rebalance_dates(self, dates: pd.DatetimeIndex, frequency: str) -> list:
        """Get rebalance dates based on frequency"""

        rebalance_dates = []
        last_rebalance = dates[0]

        for date in dates:
            if frequency == "daily":
                rebalance_dates.append(date)
            elif frequency == "weekly":
                if (date - last_rebalance).days >= 7:
                    rebalance_dates.append(date)
                    last_rebalance = date
            elif frequency == "monthly":
                if date.month != last_rebalance.month:
                    rebalance_dates.append(date)
                    last_rebalance = date
            elif frequency == "quarterly":
                if date.quarter != last_rebalance.quarter:
                    rebalance_dates.append(date)
                    last_rebalance = date

        return rebalance_dates

    def _calculate_metrics(
        self, daily_returns: list[float], portfolio_values: list[float], trades: pd.DataFrame
    ) -> dict:
        """Calculate comprehensive backtest metrics"""

        if not daily_returns or len(portfolio_values) < 2:
            return {}

        returns_array = np.array(daily_returns)
        portfolio_array = np.array(portfolio_values)

        # Basic metrics
        total_return = (portfolio_array[-1] - 1) * 100
        num_years = len(daily_returns) / 252  # Approximate years
        annual_return = (
            (((portfolio_array[-1]) ** (1 / max(num_years, 0.5)) - 1) * 100) if num_years > 0 else 0
        )
        annual_volatility = np.std(returns_array) * np.sqrt(252) * 100

        # Risk metrics
        sharpe_ratio = (annual_return / annual_volatility * 100) if annual_volatility > 0 else 0

        # Drawdown
        running_max = np.maximum.accumulate(portfolio_array)
        drawdowns = (portfolio_array - running_max) / running_max
        max_drawdown = np.min(drawdowns) * 100

        # Win rate
        win_rate = (
            (returns_array > 0).sum() / len(returns_array) * 100 if len(returns_array) > 0 else 0
        )

        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "annual_volatility": annual_volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "num_trades": len(trades) if not trades.empty else 0,
            "trades": trades,
        }


def main():
    """Example usage"""
    logger.info("Signal Analysis Framework loaded successfully")
    logger.info("Use this module as a library for signal testing")

    # Example configuration
    config = SignalConfig(
        name="momentum_6m", lookback_periods=[6], update_frequency="daily", direction="positive"
    )
    logger.info(f"Example config: {config}")


if __name__ == "__main__":
    main()
