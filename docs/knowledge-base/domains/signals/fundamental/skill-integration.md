# Fundamental Analysis - Skill Integration

**Section**: 02_signals/fundamental
**Last Updated**: 2025-12-12
**Source Skills**:
- [benchmarking](../../../../.claude/skills/benchmarking/SKILL.md)
- [financial-analysis](../../../../.claude/skills/financial-analysis/SKILL.md)

---

## Overview

This document integrates the `benchmarking` and `financial-analysis` Claude skills with the fundamental analysis knowledge base section. These skills provide interactive analysis capabilities and production-ready modeling tools.

---

## Skill Capabilities

### Benchmarking Skill

**Purpose**: Compare performance and valuation metrics across peer groups

**Key Features**:
- Peer group construction methodology
- Financial statement normalization
- Valuation multiple analysis
- Percentile ranking and positioning
- Investment opportunity identification

**Use Cases**:
- Evaluating acquisition targets
- Assessing portfolio company performance
- Identifying undervalued opportunities
- Competitive analysis

### Financial Analysis Skill

**Purpose**: Generate production-ready financial models with audit trails

**Key Features**:
- DCF valuation models
- Three-statement models
- Budget vs. actual analysis
- Scenario and sensitivity analysis
- Professional Excel output with validation

**Use Cases**:
- Company valuation
- Financial planning and forecasting
- Investment due diligence
- Portfolio company analysis

---

## Benchmarking Methodology

### 1. Peer Group Construction

```python
def construct_peer_group(target_company: dict, candidate_pool: list) -> list:
    """
    Build comparable peer group using multi-factor criteria.

    Criteria:
    - Industry classification (NAICS/SIC codes)
    - Revenue range (0.5x to 2x target)
    - Business model similarity
    - Geographic overlap
    - Growth stage alignment

    Target: 8-15 peers for statistical validity
    """
    peers = []

    for candidate in candidate_pool:
        # Industry match
        industry_match = (
            candidate['naics'] == target_company['naics'] or
            candidate['sic_3digit'] == target_company['sic_3digit']
        )

        # Size match (0.5x to 2x revenue)
        size_match = (
            target_company['revenue'] * 0.5 <= candidate['revenue'] <=
            target_company['revenue'] * 2.0
        )

        # Business model similarity (qualitative score)
        model_match = candidate['business_model_score'] >= 0.7

        if industry_match and size_match and model_match:
            peers.append(candidate)

    return peers[:15]  # Cap at 15 peers
```

### 2. Financial Data Normalization

```python
def normalize_financials(raw_data: dict) -> dict:
    """
    Normalize financial statements for peer comparison.

    Adjustments:
    - Remove one-time items (restructuring, litigation, etc.)
    - Standardize fiscal periods to calendar year
    - Convert to TTM figures
    - Apply consistent accounting treatments
    """
    normalized = raw_data.copy()

    # Remove one-time items
    one_time_items = [
        'restructuring_charges',
        'asset_writedowns',
        'litigation_settlements',
        'acquisition_costs'
    ]

    for item in one_time_items:
        if item in normalized:
            normalized['operating_income'] += normalized.get(item, 0)
            normalized['net_income'] += normalized.get(item, 0)

    # Calculate normalized EBITDA
    normalized['ebitda_normalized'] = (
        normalized['operating_income'] +
        normalized['depreciation'] +
        normalized['amortization']
    )

    return normalized
```

### 3. Core Metrics Calculation

```python
def calculate_benchmarking_metrics(financials: dict) -> dict:
    """
    Calculate comprehensive performance and valuation metrics.
    """
    metrics = {}

    # Profitability
    metrics['gross_margin'] = financials['gross_profit'] / financials['revenue']
    metrics['ebitda_margin'] = financials['ebitda'] / financials['revenue']
    metrics['operating_margin'] = financials['operating_income'] / financials['revenue']
    metrics['net_margin'] = financials['net_income'] / financials['revenue']

    # Returns
    metrics['roe'] = financials['net_income'] / financials['equity']
    metrics['roa'] = financials['net_income'] / financials['total_assets']
    metrics['roic'] = financials['nopat'] / financials['invested_capital']

    # Growth
    metrics['revenue_growth_yoy'] = (
        financials['revenue'] / financials['revenue_prior'] - 1
    )
    metrics['ebitda_growth_yoy'] = (
        financials['ebitda'] / financials['ebitda_prior'] - 1
    )

    # Efficiency
    metrics['asset_turnover'] = financials['revenue'] / financials['total_assets']
    metrics['revenue_per_employee'] = financials['revenue'] / financials['employees']

    # Financial Health
    metrics['current_ratio'] = financials['current_assets'] / financials['current_liabilities']
    metrics['debt_to_equity'] = financials['total_debt'] / financials['equity']
    metrics['interest_coverage'] = financials['ebit'] / financials['interest_expense']

    # Valuation (requires market data)
    if 'enterprise_value' in financials:
        metrics['ev_ebitda'] = financials['enterprise_value'] / financials['ebitda']
        metrics['ev_revenue'] = financials['enterprise_value'] / financials['revenue']

    if 'market_cap' in financials:
        metrics['pe_ratio'] = financials['market_cap'] / financials['net_income']
        metrics['price_to_book'] = financials['market_cap'] / financials['book_value']
        metrics['fcf_yield'] = financials['free_cash_flow'] / financials['market_cap']

    return metrics
```

### 4. Percentile Ranking

```python
def calculate_percentile_ranking(
    target_metrics: dict,
    peer_metrics: list,
    metric_name: str
) -> dict:
    """
    Calculate target company's percentile ranking vs peers.
    """
    import numpy as np

    peer_values = [p[metric_name] for p in peer_metrics if metric_name in p]
    target_value = target_metrics[metric_name]

    # Calculate percentile
    percentile = (sum(v < target_value for v in peer_values) / len(peer_values)) * 100

    # Statistical context
    return {
        'value': target_value,
        'percentile': percentile,
        'peer_median': np.median(peer_values),
        'peer_25th': np.percentile(peer_values, 25),
        'peer_75th': np.percentile(peer_values, 75),
        'outperforms_median': target_value > np.median(peer_values),
        'position': (
            'Top Quartile' if percentile >= 75 else
            'Above Median' if percentile >= 50 else
            'Below Median' if percentile >= 25 else
            'Bottom Quartile'
        )
    }
```

### 5. Implied Valuation Range

```python
def calculate_implied_valuation(
    target_financials: dict,
    peer_multiples: list,
    primary_metric: str = 'ebitda'
) -> dict:
    """
    Calculate implied valuation range using peer multiples.
    """
    import numpy as np

    if primary_metric == 'ebitda':
        multiples = [p['ev_ebitda'] for p in peer_multiples]
        base_value = target_financials['ebitda']
    elif primary_metric == 'revenue':
        multiples = [p['ev_revenue'] for p in peer_multiples]
        base_value = target_financials['revenue']
    else:
        raise ValueError(f"Unsupported metric: {primary_metric}")

    return {
        'low': base_value * np.percentile(multiples, 25),
        'median': base_value * np.median(multiples),
        'high': base_value * np.percentile(multiples, 75),
        'applied_multiples': {
            '25th': np.percentile(multiples, 25),
            'median': np.median(multiples),
            '75th': np.percentile(multiples, 75)
        },
        'base_metric': primary_metric,
        'base_value': base_value
    }
```

---

## Financial Modeling Framework

### DCF Model Structure

```python
def create_dcf_model(
    company_name: str,
    base_financials: dict,
    assumptions: dict,
    projection_years: int = 5
) -> dict:
    """
    Create comprehensive DCF valuation model.

    Args:
        company_name: Target company name
        base_financials: Current year financials
        assumptions: Growth rates, margins, WACC, terminal growth
        projection_years: Forecast horizon

    Returns:
        Dictionary with projections, valuation, and sensitivity
    """
    model = {
        'company': company_name,
        'assumptions': assumptions,
        'projections': [],
        'valuation': {}
    }

    # Project financials
    revenue = base_financials['revenue']
    for year in range(1, projection_years + 1):
        revenue *= (1 + assumptions['revenue_growth'])
        ebitda = revenue * assumptions['ebitda_margin']
        capex = revenue * assumptions['capex_pct']
        nwc_change = revenue * assumptions['nwc_pct']

        fcf = ebitda * (1 - assumptions['tax_rate']) - capex - nwc_change

        model['projections'].append({
            'year': year,
            'revenue': revenue,
            'ebitda': ebitda,
            'fcf': fcf
        })

    # Terminal value (Gordon Growth)
    terminal_fcf = model['projections'][-1]['fcf'] * (1 + assumptions['terminal_growth'])
    terminal_value = terminal_fcf / (assumptions['wacc'] - assumptions['terminal_growth'])

    # Present value calculations
    pv_fcfs = sum(
        p['fcf'] / ((1 + assumptions['wacc']) ** p['year'])
        for p in model['projections']
    )
    pv_terminal = terminal_value / ((1 + assumptions['wacc']) ** projection_years)

    model['valuation'] = {
        'pv_fcfs': pv_fcfs,
        'pv_terminal': pv_terminal,
        'enterprise_value': pv_fcfs + pv_terminal,
        'equity_value': pv_fcfs + pv_terminal - base_financials['net_debt'],
        'per_share': (pv_fcfs + pv_terminal - base_financials['net_debt']) / base_financials['shares']
    }

    return model
```

### Three-Statement Integration

```python
def build_three_statement_model(
    historical_data: dict,
    assumptions: dict,
    projection_years: int = 5
) -> dict:
    """
    Build integrated three-statement financial model.

    Income Statement -> Balance Sheet -> Cash Flow
    with automatic balancing.
    """
    model = {
        'income_statement': [],
        'balance_sheet': [],
        'cash_flow': [],
        'validation': {}
    }

    # Build each statement with linkages
    for year in range(1, projection_years + 1):
        # Income Statement
        revenue = project_revenue(historical_data, assumptions, year)
        cogs = revenue * (1 - assumptions['gross_margin'])
        opex = revenue * assumptions['opex_pct']
        ebit = revenue - cogs - opex
        interest = calculate_interest(debt_schedule, year)
        ebt = ebit - interest
        taxes = ebt * assumptions['tax_rate']
        net_income = ebt - taxes

        model['income_statement'].append({
            'year': year,
            'revenue': revenue,
            'gross_profit': revenue - cogs,
            'ebit': ebit,
            'net_income': net_income
        })

        # Balance Sheet (with auto-balance)
        assets = calculate_assets(revenue, assumptions)
        liabilities = calculate_liabilities(revenue, assumptions)
        equity_needed = assets - liabilities  # Plug to balance

        model['balance_sheet'].append({
            'year': year,
            'total_assets': assets,
            'total_liabilities': liabilities,
            'equity': equity_needed
        })

        # Cash Flow Statement
        ocf = calculate_operating_cash_flow(net_income, working_capital_changes)
        icf = -calculate_capex(revenue, assumptions)
        fcf = calculate_financing_cash_flow(debt_changes, dividends)

        model['cash_flow'].append({
            'year': year,
            'operating': ocf,
            'investing': icf,
            'financing': fcf,
            'net_change': ocf + icf + fcf
        })

    # Validation checks
    model['validation'] = {
        'balance_sheet_balances': all(
            abs(bs['total_assets'] - bs['total_liabilities'] - bs['equity']) < 0.01
            for bs in model['balance_sheet']
        ),
        'cash_reconciles': True  # Add reconciliation logic
    }

    return model
```

### Sensitivity Analysis

```python
def create_sensitivity_matrix(
    base_model: dict,
    variable1: tuple,  # (name, min, max, steps)
    variable2: tuple
) -> dict:
    """
    Create two-way sensitivity analysis matrix.
    """
    import numpy as np

    v1_name, v1_min, v1_max, v1_steps = variable1
    v2_name, v2_min, v2_max, v2_steps = variable2

    v1_range = np.linspace(v1_min, v1_max, v1_steps)
    v2_range = np.linspace(v2_min, v2_max, v2_steps)

    matrix = np.zeros((v1_steps, v2_steps))

    for i, v1 in enumerate(v1_range):
        for j, v2 in enumerate(v2_range):
            # Re-run model with adjusted assumptions
            adjusted_assumptions = base_model['assumptions'].copy()
            adjusted_assumptions[v1_name] = v1
            adjusted_assumptions[v2_name] = v2

            result = recalculate_valuation(base_model, adjusted_assumptions)
            matrix[i, j] = result['equity_value']

    return {
        'matrix': matrix,
        'variable1': {'name': v1_name, 'values': v1_range},
        'variable2': {'name': v2_name, 'values': v2_range},
        'base_case': base_model['valuation']['equity_value']
    }
```

---

## Industry-Specific Metrics

### Technology / SaaS

```python
SAAS_METRICS = {
    # Growth
    'arr_growth': lambda data: (data['arr'] / data['arr_prior']) - 1,
    'mrr_growth': lambda data: (data['mrr'] / data['mrr_prior']) - 1,

    # Retention
    'net_revenue_retention': lambda data: data['expansion_mrr'] / data['beginning_mrr'],
    'gross_revenue_retention': lambda data: 1 - (data['churned_mrr'] / data['beginning_mrr']),
    'logo_retention': lambda data: 1 - (data['churned_customers'] / data['beginning_customers']),

    # Unit Economics
    'ltv': lambda data: data['arpu'] / data['monthly_churn_rate'],
    'cac': lambda data: data['sales_marketing'] / data['new_customers'],
    'ltv_cac_ratio': lambda data: data['ltv'] / data['cac'],
    'cac_payback_months': lambda data: data['cac'] / (data['arpu'] * data['gross_margin']),

    # Rule of 40
    'rule_of_40': lambda data: data['revenue_growth_pct'] + data['ebitda_margin_pct'],

    # Magic Number
    'magic_number': lambda data: (
        (data['arr'] - data['arr_prior']) / data['sales_marketing_prior']
    )
}
```

### Financial Services

```python
FINANCIAL_METRICS = {
    # Profitability
    'net_interest_margin': lambda data: data['net_interest_income'] / data['avg_earning_assets'],
    'efficiency_ratio': lambda data: data['noninterest_expense'] / (
        data['net_interest_income'] + data['noninterest_income']
    ),
    'roa': lambda data: data['net_income'] / data['avg_total_assets'],
    'roe': lambda data: data['net_income'] / data['avg_equity'],

    # Credit Quality
    'npl_ratio': lambda data: data['nonperforming_loans'] / data['total_loans'],
    'loan_loss_reserve': lambda data: data['allowance'] / data['total_loans'],
    'net_charge_off_ratio': lambda data: data['net_charge_offs'] / data['avg_loans'],

    # Capital
    'tier1_capital_ratio': lambda data: data['tier1_capital'] / data['risk_weighted_assets'],
    'cet1_ratio': lambda data: data['cet1_capital'] / data['risk_weighted_assets'],
    'leverage_ratio': lambda data: data['tier1_capital'] / data['total_assets']
}
```

---

## Skill Cross-References

### Detailed Reference Materials

- [references/financial_metrics.md](../../../../.claude/skills/benchmarking/references/financial_metrics.md) - Complete metric formulas
- [references/analysis_framework.md](../../../../.claude/skills/benchmarking/references/analysis_framework.md) - Eight-phase methodology
- [references/financial-ratios.md](../../../../.claude/skills/financial-analysis/references/financial-ratios.md) - Ratio calculations
- [references/modeling-standards.md](../../../../.claude/skills/financial-analysis/references/modeling-standards.md) - Professional standards

### KB Section Mapping

| Skill Component | KB Section |
|-----------------|------------|
| Profitability Metrics | [README.md#quality-metrics](README.md#3-quality-metrics) |
| Valuation Multiples | [README.md#valuation-metrics](README.md#2-valuation-metrics) |
| Growth Analysis | [README.md#growth-metrics](README.md#4-growth-metrics) |
| Sector Analysis | [README.md#sector-analysis](README.md#5-sector-analysis) |

---

## Dependencies

```
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
openpyxl>=3.1.0  # For Excel model generation
scipy>=1.10.0
```

---

## Academic References

1. **Damodaran, A.**: "Investment Valuation" - DCF and relative valuation
2. **Koller, T. et al.**: "Valuation: Measuring and Managing the Value of Companies" (McKinsey)
3. **Piotroski, J.**: "Value Investing: The Use of Historical Financial Statement Information"
4. **Graham, B. & Dodd, D.**: "Security Analysis" - Foundational text

---

**Template**: KB Skills Integration v1.0
