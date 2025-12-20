"""
Covered Call Strategy Backtest

Backtests covered call strategy on historical stock data.
Simulates monthly covered call selling with realistic pricing.

Usage:
    python examples/backtest_covered_call.py

Results saved to: data/backtest_results/

Author: Ordinis Project
"""

from datetime import datetime, timedelta
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.engines.optionscore.pricing.black_scholes import (
    BlackScholesEngine,
    OptionType,
    PricingParameters,
)
from src.engines.optionscore.pricing.greeks import GreeksCalculator
from src.strategies.options.covered_call import CoveredCallStrategy


class CoveredCallBacktest:
    """
    Covered call backtest engine.

    Simulates monthly covered call selling on a stock position.
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        shares_per_contract: int = 100,
        strike_otm_pct: float = 0.05,
        days_to_expiration: int = 30,
        risk_free_rate: float = 0.03,
        assumed_iv: float = 0.25,
    ):
        """
        Initialize backtest.

        Args:
            initial_capital: Starting capital
            shares_per_contract: Shares covered per call (default 100)
            strike_otm_pct: Strike distance above stock (default 5%)
            days_to_expiration: Days until option expiration (default 30)
            risk_free_rate: Risk-free rate for pricing (default 3%)
            assumed_iv: Assumed implied volatility (default 25%)
        """
        self.initial_capital = initial_capital
        self.shares_per_contract = shares_per_contract
        self.strike_otm_pct = strike_otm_pct
        self.days_to_expiration = days_to_expiration
        self.risk_free_rate = risk_free_rate
        self.assumed_iv = assumed_iv

        # Initialize pricing engines
        self.pricing_engine = BlackScholesEngine()
        self.greeks_calc = GreeksCalculator(self.pricing_engine)
        self.strategy = CoveredCallStrategy(name="Backtest CC")

        # Tracking
        self.trades = []
        self.equity_curve = []

    def price_call_option(self, stock_price: float, strike: float, time_to_expiry: float) -> dict:
        """
        Price a call option using Black-Scholes.

        Args:
            stock_price: Current stock price
            strike: Option strike price
            time_to_expiry: Time to expiration in years

        Returns:
            Dictionary with price and Greeks
        """
        if time_to_expiry <= 0:
            # Expired - intrinsic value only
            return {
                "price": max(stock_price - strike, 0),
                "delta": 1.0 if stock_price > strike else 0.0,
                "gamma": 0.0,
                "theta": 0.0,
                "vega": 0.0,
            }

        params = PricingParameters(
            S=stock_price,
            K=strike,
            T=time_to_expiry,
            r=self.risk_free_rate,
            sigma=self.assumed_iv,
            q=0.0,
        )

        price = self.pricing_engine.price_call(params)
        greeks = self.greeks_calc.all_greeks(params, OptionType.CALL)

        return {
            "price": price,
            "delta": greeks["delta"],
            "gamma": greeks["gamma"],
            "theta": greeks["theta"],
            "vega": greeks["vega"],
        }

    def run_backtest(self, data: pd.DataFrame, symbol: str = "AAPL") -> dict:
        """
        Run covered call backtest.

        Args:
            data: DataFrame with OHLCV data (Date index, close column)
            symbol: Stock symbol

        Returns:
            Dictionary with backtest results
        """
        print(f"\n{'='*80}")
        print(f"Covered Call Backtest: {symbol}".center(80))
        print(f"{'='*80}\n")

        # Ensure data is sorted
        data = data.sort_index()

        # Calculate number of contracts based on capital
        initial_price = data["close"].iloc[0]
        num_contracts = int(self.initial_capital / (initial_price * self.shares_per_contract))
        shares_owned = num_contracts * self.shares_per_contract
        remaining_cash = self.initial_capital - (shares_owned * initial_price)

        print("Initial Setup:")
        print(f"  Capital:           ${self.initial_capital:,.2f}")
        print(f"  Stock Price:       ${initial_price:.2f}")
        print(f"  Shares Purchased:  {shares_owned:,}")
        print(f"  Contracts:         {num_contracts}")
        print(f"  Stock Cost:        ${shares_owned * initial_price:,.2f}")
        print(f"  Remaining Cash:    ${remaining_cash:,.2f}")
        print(f"  Strike OTM:        {self.strike_otm_pct*100:.0f}%")
        print(f"  DTE:               {self.days_to_expiration} days")
        print()

        # State tracking
        position_open = False
        position_entry_date = None
        position_entry_price = None
        position_strike = None
        position_premium = None
        position_expiry_date = None

        total_premium_collected = 0.0
        total_trades = 0
        assignments = 0
        expired_otm = 0

        # Process each date
        for current_date, row in data.iterrows():
            current_price = row["close"]

            # Check if we should open a new position
            if not position_open:
                # Sell covered call
                position_entry_date = current_date
                position_entry_price = current_price
                position_strike = current_price * (1 + self.strike_otm_pct)
                position_expiry_date = current_date + timedelta(days=self.days_to_expiration)

                # Price the call option
                time_to_expiry = self.days_to_expiration / 365.0
                option_data = self.price_call_option(current_price, position_strike, time_to_expiry)
                position_premium = option_data["price"]

                # Record trade
                total_premium_collected += (
                    position_premium * num_contracts * self.shares_per_contract
                )
                total_trades += 1
                position_open = True

                self.trades.append(
                    {
                        "entry_date": position_entry_date,
                        "entry_price": position_entry_price,
                        "strike": position_strike,
                        "premium": position_premium,
                        "expiry_date": position_expiry_date,
                        "delta": option_data["delta"],
                        "status": "open",
                    }
                )

            # Check if position should be closed (expiration or assignment risk)
            if position_open and current_date >= position_expiry_date:
                # Option expired
                if current_price > position_strike:
                    # ITM - stock called away
                    assignments += 1
                    realized_pnl = (
                        position_strike - position_entry_price
                    ) * shares_owned + position_premium * num_contracts * self.shares_per_contract
                    outcome = "assigned"

                    # "Buy back" shares at current price for next cycle
                    shares_owned = num_contracts * self.shares_per_contract

                else:
                    # OTM - expired worthless (we keep premium)
                    expired_otm += 1
                    realized_pnl = (
                        current_price - position_entry_price
                    ) * shares_owned + position_premium * num_contracts * self.shares_per_contract
                    outcome = "expired"

                # Update trade record
                self.trades[-1]["exit_date"] = current_date
                self.trades[-1]["exit_price"] = current_price
                self.trades[-1]["outcome"] = outcome
                self.trades[-1]["pnl"] = realized_pnl
                self.trades[-1]["status"] = "closed"

                position_open = False

            # Calculate current equity
            stock_value = shares_owned * current_price
            total_equity = stock_value + remaining_cash

            self.equity_curve.append(
                {
                    "date": current_date,
                    "stock_price": current_price,
                    "stock_value": stock_value,
                    "cash": remaining_cash,
                    "total_equity": total_equity,
                }
            )

        # Calculate performance metrics
        equity_df = pd.DataFrame(self.equity_curve).set_index("date")
        trades_df = pd.DataFrame(self.trades)

        final_equity = equity_df["total_equity"].iloc[-1]
        total_return = (final_equity - self.initial_capital) / self.initial_capital
        total_premium_pct = total_premium_collected / self.initial_capital

        # Equity curve stats
        equity_df["returns"] = equity_df["total_equity"].pct_change()
        daily_returns = equity_df["returns"].dropna()
        sharpe_ratio = (
            (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252))
            if len(daily_returns) > 0
            else 0
        )

        # Max drawdown
        equity_df["cummax"] = equity_df["total_equity"].cummax()
        equity_df["drawdown"] = (equity_df["total_equity"] - equity_df["cummax"]) / equity_df[
            "cummax"
        ]
        max_drawdown = equity_df["drawdown"].min()

        # Print results
        print(f"\n{'='*80}")
        print("Backtest Results".center(80))
        print(f"{'='*80}\n")

        print("Performance:")
        print(f"  Initial Capital:       ${self.initial_capital:,.2f}")
        print(f"  Final Equity:          ${final_equity:,.2f}")
        print(f"  Total Return:          {total_return*100:.2f}%")
        print(
            f"  Premium Collected:     ${total_premium_collected:,.2f} ({total_premium_pct*100:.2f}%)"
        )
        print(f"  Max Drawdown:          {max_drawdown*100:.2f}%")
        print(f"  Sharpe Ratio:          {sharpe_ratio:.2f}")
        print()

        print("Trade Statistics:")
        print(f"  Total Trades:          {total_trades}")
        print(f"  Assignments:           {assignments} ({assignments/total_trades*100:.1f}%)")
        print(f"  Expired OTM:           {expired_otm} ({expired_otm/total_trades*100:.1f}%)")
        print(f"  Avg Premium/Trade:     ${total_premium_collected/total_trades:.2f}")
        print()

        if len(trades_df) > 0:
            closed_trades = trades_df[trades_df["status"] == "closed"]
            if len(closed_trades) > 0:
                avg_pnl = closed_trades["pnl"].mean()
                win_rate = (closed_trades["pnl"] > 0).sum() / len(closed_trades)
                print("Closed Trades:")
                print(f"  Average P&L:           ${avg_pnl:,.2f}")
                print(f"  Win Rate:              {win_rate*100:.1f}%")
                print()

        return {
            "initial_capital": self.initial_capital,
            "final_equity": final_equity,
            "total_return": total_return,
            "total_premium": total_premium_collected,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "total_trades": total_trades,
            "assignments": assignments,
            "expired_otm": expired_otm,
            "equity_curve": equity_df,
            "trades": trades_df,
        }

    def save_results(self, results: dict, symbol: str, output_dir: str = "data/backtest_results"):
        """Save backtest results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save equity curve
        equity_file = output_path / f"{symbol}_covered_call_equity_{timestamp}.csv"
        results["equity_curve"].to_csv(equity_file)
        print(f"Equity curve saved to: {equity_file}")

        # Save trades
        trades_file = output_path / f"{symbol}_covered_call_trades_{timestamp}.csv"
        results["trades"].to_csv(trades_file, index=False)
        print(f"Trades saved to: {trades_file}")

        # Save summary
        summary_file = output_path / f"{symbol}_covered_call_summary_{timestamp}.txt"
        with open(summary_file, "w") as f:
            f.write(f"Covered Call Backtest Summary - {symbol}\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("Performance:\n")
            f.write(f"  Initial Capital:    ${results['initial_capital']:,.2f}\n")
            f.write(f"  Final Equity:       ${results['final_equity']:,.2f}\n")
            f.write(f"  Total Return:       {results['total_return']*100:.2f}%\n")
            f.write(f"  Premium Collected:  ${results['total_premium']:,.2f}\n")
            f.write(f"  Sharpe Ratio:       {results['sharpe_ratio']:.2f}\n")
            f.write(f"  Max Drawdown:       {results['max_drawdown']*100:.2f}%\n\n")
            f.write("Trade Statistics:\n")
            f.write(f"  Total Trades:       {results['total_trades']}\n")
            f.write(f"  Assignments:        {results['assignments']}\n")
            f.write(f"  Expired OTM:        {results['expired_otm']}\n")

        print(f"Summary saved to: {summary_file}")
        print()


def main():
    """Run covered call backtest on AAPL sample data."""
    # Load AAPL historical data
    data_file = Path("data/historical/AAPL_historical.csv")

    if not data_file.exists():
        print(f"Error: Data file not found: {data_file}")
        print("Please ensure AAPL historical data exists in data/historical/")
        return

    print("Loading data...")
    data = pd.read_csv(data_file, index_col=0, parse_dates=True)

    # Use last 2 years for faster backtest
    data = data.tail(504)  # ~2 years of daily data

    print(f"Loaded {len(data)} bars from {data.index[0].date()} to {data.index[-1].date()}")

    # Run backtest
    backtest = CoveredCallBacktest(
        initial_capital=100000.0,
        shares_per_contract=100,
        strike_otm_pct=0.05,  # 5% OTM
        days_to_expiration=30,  # Monthly
        risk_free_rate=0.03,
        assumed_iv=0.25,  # 25% IV
    )

    results = backtest.run_backtest(data, symbol="AAPL")

    # Save results
    backtest.save_results(results, symbol="AAPL")

    print("=" * 80)
    print("Backtest Complete!".center(80))
    print("=" * 80)


if __name__ == "__main__":
    main()
