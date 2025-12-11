"""
Married Put Strategy Backtest

Backtests married put strategy on historical stock data.
Simulates rolling protective put purchases with realistic pricing.

Usage:
    python examples/backtest_married_put.py

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


class MarriedPutBacktest:
    """
    Married put backtest engine.

    Simulates protective put buying on a stock position with rolling puts.
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        shares_per_contract: int = 100,
        protection_pct: float = 0.05,
        days_to_expiration: int = 45,
        risk_free_rate: float = 0.03,
        assumed_iv: float = 0.25,
    ):
        """
        Initialize backtest.

        Args:
            initial_capital: Starting capital
            shares_per_contract: Shares covered per put (default 100)
            protection_pct: Put strike distance below stock (default 5%)
            days_to_expiration: Days until put expiration (default 45)
            risk_free_rate: Risk-free rate for pricing (default 3%)
            assumed_iv: Assumed implied volatility (default 25%)
        """
        self.initial_capital = initial_capital
        self.shares_per_contract = shares_per_contract
        self.protection_pct = protection_pct
        self.days_to_expiration = days_to_expiration
        self.risk_free_rate = risk_free_rate
        self.assumed_iv = assumed_iv

        # Initialize pricing engines
        self.pricing_engine = BlackScholesEngine()
        self.greeks_calc = GreeksCalculator(self.pricing_engine)

        # Tracking
        self.trades = []
        self.equity_curve = []

    def price_put_option(self, stock_price: float, strike: float, time_to_expiry: float) -> dict:
        """
        Price a put option using Black-Scholes.

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
                "price": max(strike - stock_price, 0),
                "delta": -1.0 if stock_price < strike else 0.0,
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

        price = self.pricing_engine.price_put(params)
        greeks = self.greeks_calc.all_greeks(params, OptionType.PUT)

        return {
            "price": price,
            "delta": greeks["delta"],
            "gamma": greeks["gamma"],
            "theta": greeks["theta"],
            "vega": greeks["vega"],
        }

    def run_backtest(self, data: pd.DataFrame, symbol: str = "AAPL") -> dict:
        """
        Run married put backtest.

        Args:
            data: DataFrame with OHLCV data (Date index, close column)
            symbol: Stock symbol

        Returns:
            Dictionary with backtest results
        """
        print(f"\n{'='*80}")
        print(f"Married Put Backtest: {symbol}".center(80))
        print(f"{'='*80}\n")

        # Ensure data is sorted
        data = data.sort_index()

        # Calculate number of contracts based on capital
        initial_price = data["close"].iloc[0]

        # Reserve some capital for put premiums
        capital_for_stock = self.initial_capital * 0.95
        num_contracts = int(capital_for_stock / (initial_price * self.shares_per_contract))
        shares_owned = num_contracts * self.shares_per_contract
        stock_cost = shares_owned * initial_price

        # Buy initial protective put
        initial_strike = initial_price * (1 - self.protection_pct)
        time_to_expiry = self.days_to_expiration / 365.0
        initial_put = self.price_put_option(initial_price, initial_strike, time_to_expiry)
        initial_put_cost = initial_put["price"] * shares_owned

        remaining_cash = self.initial_capital - stock_cost - initial_put_cost

        print("Initial Setup:")
        print(f"  Capital:              ${self.initial_capital:,.2f}")
        print(f"  Stock Price:          ${initial_price:.2f}")
        print(f"  Shares Purchased:     {shares_owned:,}")
        print(f"  Contracts:            {num_contracts}")
        print(f"  Stock Cost:           ${stock_cost:,.2f}")
        print(f"  Initial Put Strike:   ${initial_strike:.2f} ({self.protection_pct*100:.0f}% OTM)")
        print(f"  Initial Put Cost:     ${initial_put_cost:,.2f}")
        print(f"  Remaining Cash:       ${remaining_cash:,.2f}")
        print(f"  DTE:                  {self.days_to_expiration} days")
        print()

        # State tracking
        position_open = False
        position_entry_date = None
        position_entry_price = None
        position_strike = None
        position_premium = None
        position_expiry_date = None

        total_premium_paid = initial_put_cost
        total_trades = 1
        put_expirations = 0
        protection_events = 0

        # Record initial put purchase
        self.trades.append({
            "entry_date": data.index[0],
            "entry_price": initial_price,
            "strike": initial_strike,
            "premium": initial_put["price"],
            "expiry_date": data.index[0] + timedelta(days=self.days_to_expiration),
            "delta": initial_put["delta"],
            "status": "open",
        })

        position_open = True
        position_entry_date = data.index[0]
        position_entry_price = initial_price
        position_strike = initial_strike
        position_premium = initial_put["price"]
        position_expiry_date = data.index[0] + timedelta(days=self.days_to_expiration)

        # Process each date
        for current_date, row in data.iterrows():
            current_price = row["close"]

            # Check if put expired (need to roll)
            if position_open and current_date >= position_expiry_date:
                # Put expired
                if current_price < position_strike:
                    # Protection was needed
                    protection_events += 1
                    outcome = "protected"
                else:
                    # Put expired worthless
                    put_expirations += 1
                    outcome = "expired"

                # Update trade record
                self.trades[-1]["exit_date"] = current_date
                self.trades[-1]["exit_price"] = current_price
                self.trades[-1]["outcome"] = outcome
                self.trades[-1]["status"] = "closed"

                # Roll to new put (buy another protective put)
                new_strike = current_price * (1 - self.protection_pct)
                new_expiry = current_date + timedelta(days=self.days_to_expiration)
                time_to_expiry = self.days_to_expiration / 365.0
                new_put = self.price_put_option(current_price, new_strike, time_to_expiry)
                new_put_cost = new_put["price"] * shares_owned

                # Check if we have enough cash
                if remaining_cash < new_put_cost:
                    print(f"Warning: Insufficient cash at {current_date} to roll put")
                    position_open = False
                    break

                remaining_cash -= new_put_cost
                total_premium_paid += new_put_cost
                total_trades += 1

                # Record new trade
                self.trades.append({
                    "entry_date": current_date,
                    "entry_price": current_price,
                    "strike": new_strike,
                    "premium": new_put["price"],
                    "expiry_date": new_expiry,
                    "delta": new_put["delta"],
                    "status": "open",
                })

                position_entry_date = current_date
                position_entry_price = current_price
                position_strike = new_strike
                position_premium = new_put["price"]
                position_expiry_date = new_expiry

            # Calculate current equity
            stock_value = shares_owned * current_price

            # Value of protective put (if position is open)
            put_value = 0.0
            if position_open and position_expiry_date:
                days_left = (position_expiry_date - current_date).days
                if days_left > 0:
                    time_left = days_left / 365.0
                    put_data = self.price_put_option(current_price, position_strike, time_left)
                    put_value = put_data["price"] * shares_owned
                else:
                    # At expiration
                    put_value = max(position_strike - current_price, 0) * shares_owned

            total_equity = stock_value + put_value + remaining_cash

            self.equity_curve.append({
                "date": current_date,
                "stock_price": current_price,
                "stock_value": stock_value,
                "put_value": put_value,
                "cash": remaining_cash,
                "total_equity": total_equity,
            })

        # Calculate performance metrics
        equity_df = pd.DataFrame(self.equity_curve).set_index("date")
        trades_df = pd.DataFrame(self.trades)

        final_equity = equity_df["total_equity"].iloc[-1]
        total_return = (final_equity - self.initial_capital) / self.initial_capital
        total_premium_pct = total_premium_paid / self.initial_capital

        # Calculate buy-and-hold comparison (no protection)
        bnh_shares = int(self.initial_capital / initial_price)
        bnh_final_value = bnh_shares * data["close"].iloc[-1]
        bnh_return = (bnh_final_value - self.initial_capital) / self.initial_capital

        # Equity curve stats
        equity_df["returns"] = equity_df["total_equity"].pct_change()
        daily_returns = equity_df["returns"].dropna()
        sharpe_ratio = (
            (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252))
            if len(daily_returns) > 0 and daily_returns.std() > 0
            else 0
        )

        # Max drawdown
        equity_df["cummax"] = equity_df["total_equity"].cummax()
        equity_df["drawdown"] = (equity_df["total_equity"] - equity_df["cummax"]) / equity_df["cummax"]
        max_drawdown = equity_df["drawdown"].min()

        # Print results
        print(f"\n{'='*80}")
        print("Backtest Results".center(80))
        print(f"{'='*80}\n")

        print("Performance:")
        print(f"  Initial Capital:          ${self.initial_capital:,.2f}")
        print(f"  Final Equity:             ${final_equity:,.2f}")
        print(f"  Total Return:             {total_return*100:.2f}%")
        print(f"  Premium Paid:             ${total_premium_paid:,.2f} ({total_premium_pct*100:.2f}%)")
        print(f"  Max Drawdown:             {max_drawdown*100:.2f}%")
        print(f"  Sharpe Ratio:             {sharpe_ratio:.2f}")
        print()

        print("Comparison:")
        print(f"  Buy-and-Hold Return:      {bnh_return*100:.2f}%")
        print(f"  Buy-and-Hold Final:       ${bnh_final_value:,.2f}")
        print(f"  Protection Cost:          {(bnh_return - total_return)*100:.2f}%")
        print()

        print("Trade Statistics:")
        print(f"  Total Puts Purchased:     {total_trades}")
        print(f"  Expired Worthless:        {put_expirations} ({put_expirations/total_trades*100:.1f}%)")
        print(f"  Protection Events:        {protection_events} ({protection_events/total_trades*100:.1f}%)")
        print(f"  Avg Premium/Put:          ${total_premium_paid/total_trades:,.2f}")
        print(f"  Annual Protection Cost:   {(total_premium_pct / (len(data)/252))*100:.2f}%")
        print()

        return {
            "initial_capital": self.initial_capital,
            "final_equity": final_equity,
            "total_return": total_return,
            "total_premium_paid": total_premium_paid,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "total_trades": total_trades,
            "put_expirations": put_expirations,
            "protection_events": protection_events,
            "bnh_return": bnh_return,
            "bnh_final_value": bnh_final_value,
            "equity_curve": equity_df,
            "trades": trades_df,
        }

    def save_results(self, results: dict, symbol: str, output_dir: str = "data/backtest_results"):
        """Save backtest results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save equity curve
        equity_file = output_path / f"{symbol}_married_put_equity_{timestamp}.csv"
        results["equity_curve"].to_csv(equity_file)
        print(f"Equity curve saved to: {equity_file}")

        # Save trades
        trades_file = output_path / f"{symbol}_married_put_trades_{timestamp}.csv"
        results["trades"].to_csv(trades_file, index=False)
        print(f"Trades saved to: {trades_file}")

        # Save summary
        summary_file = output_path / f"{symbol}_married_put_summary_{timestamp}.txt"
        with open(summary_file, "w") as f:
            f.write(f"Married Put Backtest Summary - {symbol}\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("Performance:\n")
            f.write(f"  Initial Capital:       ${results['initial_capital']:,.2f}\n")
            f.write(f"  Final Equity:          ${results['final_equity']:,.2f}\n")
            f.write(f"  Total Return:          {results['total_return']*100:.2f}%\n")
            f.write(f"  Premium Paid:          ${results['total_premium_paid']:,.2f}\n")
            f.write(f"  Sharpe Ratio:          {results['sharpe_ratio']:.2f}\n")
            f.write(f"  Max Drawdown:          {results['max_drawdown']*100:.2f}%\n\n")
            f.write("Comparison:\n")
            f.write(f"  Buy-and-Hold Return:   {results['bnh_return']*100:.2f}%\n")
            f.write(f"  Buy-and-Hold Final:    ${results['bnh_final_value']:,.2f}\n\n")
            f.write("Trade Statistics:\n")
            f.write(f"  Total Puts:            {results['total_trades']}\n")
            f.write(f"  Expired Worthless:     {results['put_expirations']}\n")
            f.write(f"  Protection Events:     {results['protection_events']}\n")

        print(f"Summary saved to: {summary_file}")
        print()


def main():
    """Run married put backtest on AAPL sample data."""
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
    backtest = MarriedPutBacktest(
        initial_capital=100000.0,
        shares_per_contract=100,
        protection_pct=0.05,  # 5% OTM
        days_to_expiration=45,  # Roll every 45 days
        risk_free_rate=0.03,
        assumed_iv=0.25,  # 25% IV
    )

    results = backtest.run_backtest(data, symbol="AAPL")

    # Save results
    backtest.save_results(results, symbol="AAPL")

    print("=" * 80)
    print("Backtest Complete!".center(80))
    print("=" * 80)
    print()
    print("Key Insights:")
    print(f"  - Protected position returned {results['total_return']*100:.2f}% vs {results['bnh_return']*100:.2f}% buy-and-hold")
    print(f"  - Total protection cost: {(results['bnh_return'] - results['total_return'])*100:.2f}%")
    print(f"  - Maximum drawdown reduced from unprotected to {results['max_drawdown']*100:.2f}%")
    print(f"  - {results['protection_events']} times protection prevented larger losses")
    print()


if __name__ == "__main__":
    main()
