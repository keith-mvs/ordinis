#!/usr/bin/env python3
"""
Strike Price Analysis Template

Template for comparing different strike price configurations for options strategies.
Helps evaluate cost-benefit tradeoffs across various strike selections.

Usage:
    from strike_analyzer import StrikeAnalyzer

    analyzer = StrikeAnalyzer(underlying_price=450, strategy_type='vertical_spread')
    analyzer.add_configuration(long_strike=445, short_strike=455, ...)
    results = analyzer.compare()

Customize for your strategy by:
1. Modify add_configuration() to accept strategy-specific parameters
2. Update calculate_metrics() with strategy-specific calculations
3. Implement find_optimal() with strategy-specific scoring logic

Author: Ordinis-1 Project
Version: 1.0.0
Python: 3.11+
"""

from typing import Any

import pandas as pd

# TODO: Import your strategy calculator
# from strategy_calculator import StrategyTemplate


class StrikeAnalyzer:
    """
    Analyzes and compares multiple strike price configurations.

    TODO: Customize this class for your specific strategy

    Attributes:
        underlying_price: Current stock price
        strategy_type: Type of strategy being analyzed
        configurations: List of strike configurations to compare

    Example (Vertical Spread):
        >>> analyzer = StrikeAnalyzer(underlying_price=450)
        >>> analyzer.add_configuration(
        ...     long_strike=445, short_strike=455,
        ...     long_premium=8.50, short_premium=3.20,
        ...     label="$10 Spread"
        ... )
        >>> results = analyzer.compare()
    """

    def __init__(self, underlying_price: float, strategy_type: str = "generic"):
        """
        Initialize strike analyzer.

        Args:
            underlying_price: Current stock price
            strategy_type: Strategy type for labeling (optional)
        """
        self.underlying_price = underlying_price
        self.strategy_type = strategy_type
        self.configurations: list[dict] = []
        self.positions: list[Any] = []  # TODO: Type hint with your strategy class

    def add_configuration(self, label: str | None = None, **kwargs) -> None:
        """
        Add a strike configuration to compare.

        TODO: Customize parameters for your strategy

        Args:
            label: Custom label for this configuration
            **kwargs: Strategy-specific parameters

        Examples for different strategies:
            # Vertical Spread:
            add_configuration(long_strike=445, short_strike=455,
                            long_premium=8.50, short_premium=3.20,
                            label="$10 Spread")

            # Stock + Option:
            add_configuration(stock_shares=100, option_strike=43,
                            option_premium=2.10, label="$43 Strike")

            # Straddle:
            add_configuration(strike=450, call_premium=5.50,
                            put_premium=5.25, label="ATM Straddle")

            # Iron Condor:
            add_configuration(short_put=440, long_put=435,
                            short_call=460, long_call=465,
                            label="$5 Wide Wings")
        """
        # TODO: Extract strategy-specific parameters
        # Example for vertical spread:
        # long_strike = kwargs.get('long_strike')
        # short_strike = kwargs.get('short_strike')

        # Auto-generate label if not provided
        if label is None:
            # TODO: Create meaningful label based on strike configuration
            label = f"Config {len(self.configurations) + 1}"

        # TODO: Create position using your strategy calculator
        # Example:
        # from strategy_calculator import StrategyTemplate
        # position = StrategyTemplate(
        #     underlying_price=self.underlying_price,
        #     long_strike=long_strike,
        #     short_strike=short_strike,
        #     ...
        # )
        # self.positions.append(position)

        # Store configuration data
        config_data = {"label": label}
        config_data.update(kwargs)
        self.configurations.append(config_data)

    def calculate_metrics(self) -> pd.DataFrame:
        """
        Calculate comparison metrics for all configurations.

        TODO: Customize metrics for your strategy

        Returns:
            DataFrame with comprehensive comparison metrics

        Common metrics to include:
        - Label/Description
        - Strike prices
        - Total cost/credit
        - Max profit
        - Max loss
        - Breakeven(s)
        - Risk/Reward ratio
        - Probability of profit (if calculable)
        - P/L at key price points
        """
        if not self.configurations:
            raise ValueError("No configurations added. Use add_configuration() first.")

        results = []

        for config in self.configurations:
            # TODO: Calculate metrics for this configuration
            # If you have position objects:
            # position = self.positions[i]
            # metrics = position.get_metrics_summary()

            metrics = {
                "Label": config["label"],
                # TODO: Add strategy-specific metrics
                # Examples:
                # 'Long Strike': config.get('long_strike'),
                # 'Short Strike': config.get('short_strike'),
                # 'Spread Width': config.get('short_strike') - config.get('long_strike'),
                # 'Total Cost': position.total_cost,
                # 'Max Profit': position.max_profit,
                # 'Max Loss': position.max_loss,
                # 'Breakeven': position.breakeven_price,
                # 'R:R Ratio': position.risk_reward_ratio,
            }

            # TODO: Calculate P/L at key price points
            # Example price points (±20%, ±10%, unchanged)
            # price_points = [
            #     (self.underlying_price * pct, f'{int((pct-1)*100):+d}%')
            #     for pct in [0.80, 0.90, 1.00, 1.10, 1.20]
            # ]
            #
            # for price, label in price_points:
            #     pl = position.calculate_pl_at_price(price)
            #     metrics[f'P/L at {label}'] = pl

            results.append(metrics)

        return pd.DataFrame(results)

    def compare(self, sort_by: str | None = None, ascending: bool = False) -> pd.DataFrame:
        """
        Compare all configurations with optional sorting.

        Args:
            sort_by: Column name to sort by (e.g., 'Max Profit', 'R:R Ratio')
            ascending: Sort ascending if True, descending if False

        Returns:
            Sorted DataFrame with comparison results
        """
        df = self.calculate_metrics()

        if sort_by and sort_by in df.columns:
            df = df.sort_values(by=sort_by, ascending=ascending)

        return df

    def find_optimal(self, optimization_criteria: str = "risk_reward") -> dict:
        """
        Find optimal configuration based on specified criteria.

        TODO: Customize optimization logic for your strategy

        Args:
            optimization_criteria: How to rank configurations
                Options: 'risk_reward', 'max_profit', 'min_risk', 'best_value'

        Returns:
            Dictionary with optimal configuration details

        Example optimization approaches:
        - 'risk_reward': Highest max_profit / max_loss ratio
        - 'max_profit': Highest potential profit
        - 'min_risk': Lowest maximum loss
        - 'best_value': Best protection/profit per dollar spent
        - 'probability': Highest probability of profit (if calculated)
        """
        df = self.calculate_metrics()

        # TODO: Implement optimization logic
        # Example:
        # if optimization_criteria == 'risk_reward':
        #     best_idx = df['R:R Ratio'].idxmax()
        # elif optimization_criteria == 'max_profit':
        #     best_idx = df['Max Profit'].idxmax()
        # elif optimization_criteria == 'min_risk':
        #     best_idx = df['Max Loss'].idxmin()
        # else:
        #     raise ValueError(f"Unknown criteria: {optimization_criteria}")

        # Placeholder: return first configuration
        best_idx = 0
        best_row = df.iloc[best_idx]

        return {
            "index": int(best_idx),
            "label": best_row["Label"],
            "criteria": optimization_criteria,
            "metrics": best_row.to_dict(),
            # TODO: Add formatted recommendation
            # 'recommendation': f"{best_row['Label']} offers best {optimization_criteria} ..."
        }

    def analyze_tradeoffs(self) -> dict[str, Any]:
        """
        Analyze cost-benefit tradeoffs across configurations.

        TODO: Implement strategy-specific tradeoff analysis

        Returns:
            Dictionary with tradeoff insights

        Example tradeoffs to analyze:
        - Cost vs. Protection (stock+option strategies)
        - Cost vs. Profit Potential (debit spreads)
        - Credit vs. Risk (credit spreads)
        - Spread Width vs. Probability (butterflies, condors)
        - Time vs. Cost (expiration analysis)
        """
        df = self.calculate_metrics()

        # TODO: Calculate correlations and tradeoffs
        # Examples:
        # - Correlation between cost and protection
        # - Efficiency ratios (profit per dollar risked)
        # - Diminishing returns analysis

        return {
            "num_configurations": len(df),
            # TODO: Add strategy-specific tradeoff metrics
            # 'cost_protection_correlation': correlation,
            # 'efficiency_range': (min_efficiency, max_efficiency),
            # 'sweet_spot': optimal_configuration_label,
        }

    def export_comparison(self, filename: str, format: str = "csv") -> None:
        """
        Export comparison results to file.

        Args:
            filename: Output filename
            format: File format ('csv', 'excel', 'json')
        """
        df = self.calculate_metrics()

        if format == "csv":
            df.to_csv(filename, index=False)
        elif format == "excel":
            df.to_excel(filename, index=False)
        elif format == "json":
            df.to_json(filename, orient="records", indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")

        print(f"Comparison exported to {filename}")

    def print_summary(self) -> None:
        """Print formatted comparison summary."""
        df = self.calculate_metrics()

        print(f"\n{'=' * 70}")
        print(f"Strike Configuration Analysis - {self.strategy_type}")
        print(f"Underlying Price: ${self.underlying_price:.2f}")
        print(f"Configurations Analyzed: {len(df)}")
        print(f"{'=' * 70}\n")

        # Display DataFrame with formatting
        pd.options.display.float_format = "{:,.2f}".format
        print(df.to_string(index=False))
        print()

        # TODO: Add summary statistics
        # print("\nSummary Statistics:")
        # print(f"  Average Max Profit: ${df['Max Profit'].mean():.2f}")
        # print(f"  Average Max Loss: ${df['Max Loss'].mean():.2f}")
        # print(f"  Best R:R Ratio: {df['R:R Ratio'].max():.2f}")


def compare_strikes_quick(
    underlying_price: float, strike_configs: list[dict], sort_by: str = "risk_reward"
) -> pd.DataFrame:
    """
    Quick comparison function for strike configurations.

    Convenience function for one-off comparisons without creating analyzer object.

    Args:
        underlying_price: Current stock price
        strike_configs: List of strike configuration dicts
        sort_by: Column to sort results by

    Returns:
        Comparison DataFrame

    Example:
        >>> configs = [
        ...     {'long_strike': 445, 'short_strike': 450, 'label': '$5 Spread'},
        ...     {'long_strike': 445, 'short_strike': 455, 'label': '$10 Spread'},
        ... ]
        >>> results = compare_strikes_quick(450, configs)
    """
    analyzer = StrikeAnalyzer(underlying_price)

    for config in strike_configs:
        analyzer.add_configuration(**config)

    return analyzer.compare(sort_by=sort_by)


if __name__ == "__main__":
    # TODO: Add example usage for your strategy
    print("Strike Analyzer Template - Example Usage\n")
    print("=" * 50)
    print("\nThis is a template. Customize for your specific strategy.")
    print("\nExample workflow:")
    print("1. Create analyzer: analyzer = StrikeAnalyzer(underlying_price=450)")
    print("2. Add configs: analyzer.add_configuration(strike1=..., strike2=...)")
    print("3. Compare: results = analyzer.compare()")
    print("4. Find optimal: best = analyzer.find_optimal('risk_reward')")
    print("\n" + "=" * 50)
