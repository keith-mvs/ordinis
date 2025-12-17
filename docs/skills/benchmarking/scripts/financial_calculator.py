#!/usr/bin/env python3
"""
Financial Benchmarking Calculator

Calculates key financial metrics and valuation multiples for company analysis.
Supports both absolute values and comparative analysis across peer groups.
"""

from dataclasses import dataclass
import json


@dataclass
class FinancialData:
    """Company financial data structure"""

    # Income Statement
    revenue: float
    cogs: float = 0.0
    operating_expenses: float = 0.0
    ebitda: float = 0.0
    ebit: float = 0.0
    net_income: float = 0.0

    # Balance Sheet
    total_assets: float = 0.0
    current_assets: float = 0.0
    inventory: float = 0.0
    accounts_receivable: float = 0.0
    cash: float = 0.0
    total_liabilities: float = 0.0
    current_liabilities: float = 0.0
    total_debt: float = 0.0
    shareholders_equity: float = 0.0

    # Cash Flow
    operating_cash_flow: float = 0.0
    capex: float = 0.0

    # Market Data
    market_cap: float = 0.0
    shares_outstanding: float = 0.0

    # Operational
    num_employees: int | None = None

    # Period Information
    period_label: str = "Current"


class FinancialCalculator:
    """Calculate financial metrics and ratios"""

    @staticmethod
    def gross_margin(data: FinancialData) -> float | None:
        """Calculate gross margin percentage"""
        if data.revenue == 0:
            return None
        return ((data.revenue - data.cogs) / data.revenue) * 100

    @staticmethod
    def ebitda_margin(data: FinancialData) -> float | None:
        """Calculate EBITDA margin percentage"""
        if data.revenue == 0:
            return None
        return (data.ebitda / data.revenue) * 100

    @staticmethod
    def net_profit_margin(data: FinancialData) -> float | None:
        """Calculate net profit margin percentage"""
        if data.revenue == 0:
            return None
        return (data.net_income / data.revenue) * 100

    @staticmethod
    def roe(data: FinancialData) -> float | None:
        """Calculate Return on Equity"""
        if data.shareholders_equity == 0:
            return None
        return (data.net_income / data.shareholders_equity) * 100

    @staticmethod
    def roa(data: FinancialData) -> float | None:
        """Calculate Return on Assets"""
        if data.total_assets == 0:
            return None
        return (data.net_income / data.total_assets) * 100

    @staticmethod
    def current_ratio(data: FinancialData) -> float | None:
        """Calculate current ratio (liquidity measure)"""
        if data.current_liabilities == 0:
            return None
        return data.current_assets / data.current_liabilities

    @staticmethod
    def debt_to_equity(data: FinancialData) -> float | None:
        """Calculate debt-to-equity ratio"""
        if data.shareholders_equity == 0:
            return None
        return data.total_debt / data.shareholders_equity

    @staticmethod
    def asset_turnover(data: FinancialData) -> float | None:
        """Calculate asset turnover ratio"""
        if data.total_assets == 0:
            return None
        return data.revenue / data.total_assets

    @staticmethod
    def revenue_per_employee(data: FinancialData) -> float | None:
        """Calculate revenue per employee"""
        if not data.num_employees or data.num_employees == 0:
            return None
        return data.revenue / data.num_employees

    @staticmethod
    def free_cash_flow(data: FinancialData) -> float:
        """Calculate free cash flow"""
        return data.operating_cash_flow - data.capex

    @staticmethod
    def enterprise_value(data: FinancialData) -> float:
        """Calculate enterprise value"""
        return data.market_cap + data.total_debt - data.cash


class ValuationCalculator:
    """Calculate valuation multiples"""

    @staticmethod
    def pe_ratio(data: FinancialData) -> float | None:
        """Calculate Price-to-Earnings ratio"""
        if data.net_income <= 0:
            return None
        return data.market_cap / data.net_income

    @staticmethod
    def ps_ratio(data: FinancialData) -> float | None:
        """Calculate Price-to-Sales ratio"""
        if data.revenue == 0:
            return None
        return data.market_cap / data.revenue

    @staticmethod
    def pb_ratio(data: FinancialData) -> float | None:
        """Calculate Price-to-Book ratio"""
        if data.shareholders_equity <= 0:
            return None
        return data.market_cap / data.shareholders_equity

    @staticmethod
    def ev_revenue(data: FinancialData) -> float | None:
        """Calculate EV/Revenue multiple"""
        if data.revenue == 0:
            return None
        ev = FinancialCalculator.enterprise_value(data)
        return ev / data.revenue

    @staticmethod
    def ev_ebitda(data: FinancialData) -> float | None:
        """Calculate EV/EBITDA multiple"""
        if data.ebitda <= 0:
            return None
        ev = FinancialCalculator.enterprise_value(data)
        return ev / data.ebitda

    @staticmethod
    def fcf_yield(data: FinancialData) -> float | None:
        """Calculate Free Cash Flow yield percentage"""
        if data.market_cap == 0:
            return None
        fcf = FinancialCalculator.free_cash_flow(data)
        return (fcf / data.market_cap) * 100

    @staticmethod
    def ev_fcf(data: FinancialData) -> float | None:
        """Calculate EV/FCF multiple"""
        fcf = FinancialCalculator.free_cash_flow(data)
        if fcf <= 0:
            return None
        ev = FinancialCalculator.enterprise_value(data)
        return ev / fcf


def calculate_all_metrics(data: FinancialData) -> dict[str, float | None]:
    """Calculate all available financial metrics for a company"""
    calc = FinancialCalculator()
    val = ValuationCalculator()

    return {
        # Profitability Metrics
        "gross_margin_%": calc.gross_margin(data),
        "ebitda_margin_%": calc.ebitda_margin(data),
        "net_profit_margin_%": calc.net_profit_margin(data),
        "roe_%": calc.roe(data),
        "roa_%": calc.roa(data),
        # Efficiency Metrics
        "asset_turnover": calc.asset_turnover(data),
        "revenue_per_employee": calc.revenue_per_employee(data),
        # Liquidity & Solvency
        "current_ratio": calc.current_ratio(data),
        "debt_to_equity": calc.debt_to_equity(data),
        # Cash Flow
        "free_cash_flow": calc.free_cash_flow(data),
        "enterprise_value": calc.enterprise_value(data),
        # Valuation Multiples
        "pe_ratio": val.pe_ratio(data),
        "ps_ratio": val.ps_ratio(data),
        "pb_ratio": val.pb_ratio(data),
        "ev_revenue": val.ev_revenue(data),
        "ev_ebitda": val.ev_ebitda(data),
        "fcf_yield_%": val.fcf_yield(data),
        "ev_fcf": val.ev_fcf(data),
    }


def calculate_growth_rate(current: float, prior: float) -> float | None:
    """Calculate period-over-period growth rate"""
    if prior == 0:
        return None
    return ((current - prior) / prior) * 100


def calculate_cagr(ending: float, beginning: float, years: float) -> float | None:
    """Calculate Compound Annual Growth Rate"""
    if beginning <= 0 or years <= 0:
        return None
    return (((ending / beginning) ** (1 / years)) - 1) * 100


def benchmark_against_peers(
    target_metrics: dict[str, float], peer_metrics: list[dict[str, float]]
) -> dict[str, dict[str, float]]:
    """
    Compare target company metrics against peer group

    Returns statistics for each metric including min, max, median, mean,
    and target company's percentile ranking
    """
    results = {}

    for metric in target_metrics:
        values = [p[metric] for p in peer_metrics if metric in p and p[metric] is not None]

        if not values:
            continue

        values.sort()
        n = len(values)
        target_val = target_metrics[metric]

        # Calculate statistics
        stats = {
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / n,
            "median": values[n // 2] if n % 2 else (values[n // 2 - 1] + values[n // 2]) / 2,
            "target": target_val,
        }

        # Calculate target's percentile
        if target_val is not None:
            below = sum(1 for v in values if v < target_val)
            stats["percentile"] = (below / n) * 100

        results[metric] = stats

    return results


if __name__ == "__main__":
    # Example usage
    example_company = FinancialData(
        revenue=100_000_000,
        cogs=40_000_000,
        ebitda=25_000_000,
        net_income=15_000_000,
        total_assets=80_000_000,
        shareholders_equity=50_000_000,
        operating_cash_flow=20_000_000,
        capex=5_000_000,
        market_cap=200_000_000,
        total_debt=20_000_000,
        cash=10_000_000,
        num_employees=500,
    )

    metrics = calculate_all_metrics(example_company)
    print(json.dumps(metrics, indent=2, default=str))
