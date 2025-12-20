# Financial and Vendor Risk

**Section**: 03_risk/financial
**Last Updated**: 2025-12-12
**Version**: 1.0

---

## Definition and Scope

**Financial risk** encompasses the potential for monetary losses arising from market movements, credit events, liquidity constraints, and funding issues.

**Vendor risk** addresses exposures arising from third-party relationships, including service providers, counterparties, and suppliers.

---

## 1. Financial Risk Categories

### 1.1 Market Risk

**Definition**: Risk of losses from adverse movements in market prices.

| Risk Type | Source | Example |
|-----------|--------|---------|
| **Equity risk** | Stock price movements | Position losses |
| **Interest rate risk** | Rate changes | Bond price impact |
| **Currency risk** | FX movements | Cross-border exposure |
| **Commodity risk** | Commodity prices | Hedging exposure |
| **Volatility risk** | Implied vol changes | Options positions |

**Measurement Methods**:
```python
# Value at Risk (VaR)
def calculate_var(returns: pd.Series, confidence: float = 0.95) -> float:
    return returns.quantile(1 - confidence)

# Expected Shortfall (ES)
def calculate_es(returns: pd.Series, confidence: float = 0.95) -> float:
    var = calculate_var(returns, confidence)
    return returns[returns <= var].mean()

# Greeks for derivatives
GREEKS = {
    'delta': 'First-order price sensitivity',
    'gamma': 'Second-order price sensitivity',
    'vega': 'Volatility sensitivity',
    'theta': 'Time decay',
    'rho': 'Interest rate sensitivity'
}
```

### 1.2 Credit Risk

**Definition**: Risk of loss from counterparty failure to meet obligations.

| Component | Definition | Measure |
|-----------|------------|---------|
| **Probability of Default (PD)** | Likelihood of default | Rating, CDS spread |
| **Loss Given Default (LGD)** | Loss severity if default | Recovery rate |
| **Exposure at Default (EAD)** | Exposure when default occurs | Current + potential |
| **Expected Loss (EL)** | EL = PD × LGD × EAD | Dollar amount |

**Counterparty Categories**:
| Category | Examples | Risk Profile |
|----------|----------|--------------|
| Prime brokers | Goldman, Morgan Stanley | Lower (systemically important) |
| Exchanges | NYSE, CME | Lower (regulated) |
| Clearinghouses | DTCC, OCC | Lower (risk mutualization) |
| OTC counterparties | Hedge funds, corporates | Higher (bilateral) |
| Service vendors | Data, technology | Variable |

### 1.3 Liquidity Risk

**Definition**: Risk of inability to meet cash flow needs or execute transactions at reasonable cost.

| Type | Description | Impact |
|------|-------------|--------|
| **Funding liquidity** | Ability to fund obligations | Forced selling |
| **Market liquidity** | Ability to transact | Price impact |
| **Asset liquidity** | Time to convert to cash | Delays |

**Liquidity Metrics**:
```python
LIQUIDITY_METRICS = {
    'cash_buffer': 'Days of operating expenses covered',
    'bid_ask_spread': 'Transaction cost indicator',
    'market_depth': 'Volume at best prices',
    'time_to_liquidate': 'Days to exit positions',
    'concentration': 'Position size vs ADV'
}
```

### 1.4 Funding Risk

**Definition**: Risk of inability to obtain funding at acceptable cost.

| Source | Risk | Mitigation |
|--------|------|------------|
| Credit facilities | Covenant breach | Headroom monitoring |
| Prime brokerage | Margin calls | Excess margin |
| Repo markets | Haircut changes | Diversification |
| Capital markets | Access closure | Staggered maturities |

---

## 2. Vendor Risk Categories

### 2.1 Vendor Dependency Risk

**Definition**: Risk from reliance on third-party providers.

| Dependency Level | Definition | Examples |
|------------------|------------|----------|
| **Critical** | Business cannot operate without | Core trading platform |
| **High** | Significant impact if unavailable | Market data, OMS |
| **Medium** | Operational degradation | Analytics, reporting |
| **Low** | Minor inconvenience | Non-essential tools |

### 2.2 Vendor Service Risk

**Definition**: Risk of vendor failing to deliver as expected.

| Risk | Description | Indicators |
|------|-------------|------------|
| **Performance** | SLA breaches | Uptime, latency metrics |
| **Quality** | Service degradation | Error rates, complaints |
| **Capacity** | Unable to scale | Peak load failures |
| **Support** | Inadequate response | Resolution times |

### 2.3 Vendor Stability Risk

**Definition**: Risk of vendor financial or operational failure.

| Factor | Warning Signs |
|--------|---------------|
| Financial health | Revenue decline, cash burn |
| Ownership changes | M&A activity, PE involvement |
| Key personnel | Executive departures |
| Customer base | Large customer losses |
| Technology | Underinvestment, legacy issues |

### 2.4 Concentration Risk

**Definition**: Risk from over-reliance on limited vendors.

| Area | Concentration Concern |
|------|----------------------|
| Single vendor | No alternative if fails |
| Single technology | Platform dependency |
| Single geography | Regional disruption |
| Single individual | Key contact risk |

---

## 3. Early Warning Signals

### 3.1 Financial Risk Indicators

| Indicator | Green | Amber | Red |
|-----------|-------|-------|-----|
| VaR utilization | < 60% | 60-80% | > 80% |
| Counterparty rating | A or better | BBB | BB or lower |
| Margin excess | > 50% | 20-50% | < 20% |
| Liquidity coverage | > 120% | 100-120% | < 100% |
| Concentration | < 10% single | 10-20% | > 20% |

### 3.2 Vendor Risk Indicators

| Indicator | Green | Amber | Red |
|-----------|-------|-------|-----|
| SLA performance | > 99% | 95-99% | < 95% |
| Financial rating | Stable | On watch | Downgrade |
| Contract issues | 0 | 1-3/quarter | > 3/quarter |
| Security incidents | 0 | Any disclosed | Any affecting us |
| Exit readiness | Tested | Documented | Undocumented |

---

## 4. Impact Analysis

### 4.1 Financial Risk Impacts

| Scenario | Potential Impact | Recovery Time |
|----------|------------------|---------------|
| Market stress (10% move) | -5 to -20% portfolio | N/A (mark-to-market) |
| Counterparty default | 0-100% exposure | Months to years |
| Liquidity crunch | Forced sales at discount | Days to weeks |
| Margin call cascade | Position liquidation | Immediate |

### 4.2 Vendor Risk Impacts

| Scenario | Potential Impact | Recovery Time |
|----------|------------------|---------------|
| Critical vendor outage | Trading halt | Hours to days |
| Data vendor failure | Decision impairment | Hours |
| Security breach at vendor | Data exposure, regulatory | Months |
| Vendor bankruptcy | Service termination | Weeks to months |

---

## 5. Mitigation Strategies

### 5.1 Market Risk Mitigation

| Strategy | Description | Implementation |
|----------|-------------|----------------|
| **Diversification** | Spread exposure | Portfolio construction |
| **Hedging** | Offset exposures | Derivatives overlay |
| **Limits** | Cap exposures | Position limits, VaR limits |
| **Stress testing** | Measure tail risk | Scenario analysis |
| **Dynamic management** | Adjust to conditions | Regime-based sizing |

### 5.2 Credit Risk Mitigation

| Strategy | Description | Implementation |
|----------|-------------|----------------|
| **Counterparty limits** | Cap exposure per entity | Credit limits |
| **Collateralization** | Secure obligations | Margin, CSA |
| **Netting** | Offset exposures | Master agreements |
| **Credit derivatives** | Transfer risk | CDS, guarantees |
| **Central clearing** | Mutualize risk | CCP membership |

### 5.3 Liquidity Risk Mitigation

| Strategy | Description | Implementation |
|----------|-------------|----------------|
| **Cash buffers** | Maintain reserves | Operating cash policy |
| **Credit facilities** | Backup funding | Committed lines |
| **Asset liquidity** | Tradeable holdings | Liquidity constraints |
| **Liability management** | Stagger maturities | Funding ladder |

### 5.4 Vendor Risk Mitigation

| Strategy | Description | Implementation |
|----------|-------------|----------------|
| **Due diligence** | Pre-contract assessment | Vendor evaluation process |
| **Contractual protections** | Legal safeguards | SLAs, termination rights |
| **Monitoring** | Ongoing oversight | Performance reviews |
| **Contingency planning** | Exit readiness | Alternative identification |
| **Diversification** | Multiple providers | Dual sourcing |

---

## 6. Vendor Management Framework

### 6.1 Vendor Lifecycle

```
1. Identification & Selection
   └── Requirements, RFP, evaluation

2. Due Diligence
   └── Financial, operational, security

3. Contracting
   └── SLA, terms, exit provisions

4. Onboarding
   └── Integration, testing, go-live

5. Ongoing Management
   └── Performance monitoring, reviews

6. Exit/Renewal
   └── Evaluation, transition planning
```

### 6.2 Due Diligence Framework

| Area | Assessment Elements |
|------|---------------------|
| **Financial** | Financials, credit rating, stability |
| **Operational** | Capacity, resilience, continuity |
| **Security** | Controls, certifications, breaches |
| **Compliance** | Regulatory, contractual adherence |
| **Strategic** | Roadmap, investment, market position |

### 6.3 Contractual Requirements

| Clause | Purpose |
|--------|---------|
| **SLA** | Performance standards, remedies |
| **Security requirements** | Protection obligations |
| **Audit rights** | Verification capability |
| **Termination** | Exit triggers, notice periods |
| **Transition assistance** | Orderly exit support |
| **Insurance** | Coverage requirements |
| **Indemnification** | Liability allocation |

---

## 7. Concentration Management

### 7.1 Exposure Limits

| Dimension | Limit | Rationale |
|-----------|-------|-----------|
| Single counterparty | ≤ 10% of capital | Default impact |
| Sector | ≤ 25% of exposure | Correlation |
| Geography | ≤ 30% of exposure | Regional risk |
| Single vendor | Critical = 0, High ≤ 2 | Dependency |
| Technology platform | Documented alternatives | Lock-in |

### 7.2 Diversification Strategies

| Strategy | Application |
|----------|-------------|
| Multiple prime brokers | Execution, financing |
| Multiple data vendors | Price, reference data |
| Multi-cloud | Infrastructure |
| Dual banking | Cash, payments |

---

## 8. Real-World Examples

### 8.1 Archegos Capital (2021)

**Event**: Concentrated positions, margin calls, prime broker losses.

**Impact**: $10B+ losses across prime brokers.

**Root Causes**:
- Concentrated equity positions
- Multiple prime brokers unaware of total exposure
- Inadequate margin requirements
- Poor counterparty risk management

**Lessons**:
- Counterparty exposure aggregation
- Position transparency
- Adequate initial margin
- Stress testing concentrated positions

### 8.2 TSB IT Migration (2018)

**Event**: Failed core banking system migration.

**Impact**: Weeks of service disruption, £370M+ costs.

**Root Causes**:
- Inadequate testing
- Poor contingency planning
- Underestimated complexity
- Vendor coordination failures

**Lessons**:
- Comprehensive testing
- Rollback capability
- Phased migration
- Vendor accountability

---

## 9. Residual Risk

Financial and vendor risks remain due to:

- Market unpredictability
- Counterparty opacity
- Vendor dependency necessity
- Concentration trade-offs
- Black swan events

**Acceptance requires**:
- Limits within appetite
- Controls operating effectively
- Monitoring in place
- Contingency plans tested
- Adequate capital/reserves

---

## Cross-References

- [Market Risk (Trading)](../README.md) - Existing trading-focused risk content
- [Advanced Risk Methods](../advanced_risk_methods.md) - VaR, CVaR, stress testing
- [Operational Risk](../operational/operational_risk.md)
- [Risk Governance](../frameworks/risk_governance.md)

---

**Template**: Enterprise Risk Management v1.0
