# Emerging and Systemic Risk

**Section**: 03_risk/emerging
**Last Updated**: 2025-12-12
**Version**: 1.0

---

## Definition and Scope

**Emerging risks** are new or evolving risks that are difficult to quantify due to limited data, uncertain characteristics, or rapidly changing nature.

**Systemic risks** are risks that can trigger a breakdown of an entire system or market, often through interconnected failures and contagion effects.

---

## 1. Emerging Risk Categories

### 1.1 Technology Disruption

**Definition**: Risks arising from transformative technological changes.

| Technology | Opportunity | Risk |
|------------|-------------|------|
| **AI/ML** | Alpha generation, automation | Model risk, job displacement |
| **Quantum computing** | Optimization, cryptography | Security vulnerabilities |
| **Blockchain/DeFi** | Settlement, transparency | Regulatory, operational |
| **Cloud computing** | Scalability, cost | Concentration, security |
| **Edge computing** | Latency reduction | Complexity, control |

**AI-Specific Risks**:
| Risk | Description |
|------|-------------|
| Model opacity | "Black box" decision-making |
| Bias amplification | Systematic errors at scale |
| Adversarial attacks | Deliberate model manipulation |
| Data dependency | Quality and availability |
| Regulatory uncertainty | Evolving AI governance |

### 1.2 Climate and Environmental

**Definition**: Risks related to climate change and environmental factors.

| Risk Type | Description | Trading Impact |
|-----------|-------------|----------------|
| **Physical** | Extreme weather, disasters | Infrastructure, supply chains |
| **Transition** | Policy changes, carbon pricing | Sector valuations |
| **Liability** | Climate-related litigation | Company exposure |
| **Stranded assets** | Obsolete fossil fuel investments | Portfolio impact |

**Climate Risk Transmission**:
```
Physical Event
    │
    ├── Direct damage (infrastructure)
    │       │
    │       └── Business interruption
    │
    ├── Supply chain disruption
    │       │
    │       └── Input costs, availability
    │
    └── Market repricing
            │
            └── Asset value changes
```

### 1.3 Geopolitical

**Definition**: Risks from international political developments.

| Risk | Examples | Impact |
|------|----------|--------|
| **Trade conflicts** | Tariffs, sanctions | Supply chains, markets |
| **Political instability** | Regime change, unrest | Market volatility |
| **Deglobalization** | Reshoring, fragmentation | Cost structures |
| **Cyber warfare** | State-sponsored attacks | Infrastructure |
| **Resource competition** | Energy, rare earths | Commodity markets |

### 1.4 Demographic and Social

**Definition**: Risks from population and social changes.

| Trend | Description | Impact |
|-------|-------------|--------|
| **Aging populations** | Dependency ratios | Interest rates, healthcare |
| **Urbanization** | City concentration | Infrastructure, housing |
| **Inequality** | Wealth concentration | Political risk, demand |
| **Labor shifts** | Gig economy, remote work | Productivity, real estate |
| **Generational change** | Different preferences | Consumption patterns |

### 1.5 Pandemic and Health

**Definition**: Risks from disease outbreaks and health crises.

| Phase | Immediate Risks | Longer-term Risks |
|-------|-----------------|-------------------|
| **Outbreak** | Market volatility, liquidity | Supply chain restructuring |
| **Response** | Policy uncertainty | Fiscal sustainability |
| **Recovery** | Uneven recovery | Behavioral changes |
| **Endemic** | Ongoing disruptions | Healthcare costs |

---

## 2. Systemic Risk Categories

### 2.1 Financial System Risk

**Definition**: Risk of cascading failures through the financial system.

| Mechanism | Description | Example |
|-----------|-------------|---------|
| **Counterparty contagion** | Default chains | Lehman Brothers 2008 |
| **Market contagion** | Correlated selling | Flash crashes |
| **Liquidity spirals** | Forced deleveraging | LTCM 1998 |
| **Funding contagion** | Credit market freeze | 2008 credit crisis |
| **Infrastructure failure** | Central system outage | SWIFt disruption |

**Interconnectedness Metrics**:
```python
SYSTEMIC_RISK_INDICATORS = {
    'counterparty_concentration': 'Exposure to systemically important institutions',
    'market_correlation': 'Asset correlation in stress periods',
    'liquidity_dependency': 'Reliance on market functioning',
    'leverage_levels': 'System-wide leverage',
    'central_clearing_concentration': 'CCP dependencies'
}
```

### 2.2 Infrastructure Risk

**Definition**: Risk of critical infrastructure failure.

| Infrastructure | Failure Mode | Cascading Impact |
|----------------|--------------|------------------|
| **Power grid** | Outage | All systems affected |
| **Internet** | Connectivity loss | Communication, data |
| **Financial market infrastructure** | Exchange/clearing outage | Trading halt |
| **Cloud providers** | Major outage | Widespread service impact |
| **Payment systems** | Settlement failure | Liquidity crisis |

### 2.3 Cyber Systemic Risk

**Definition**: Risk of cyber events with systemic impact.

| Scenario | Mechanism | Impact |
|----------|-----------|--------|
| **Critical infrastructure attack** | SCADA/ICS compromise | Physical damage |
| **Financial system attack** | Core banking compromise | Trust breakdown |
| **Supply chain attack** | Widespread software compromise | Mass exploitation |
| **DNS/BGP attack** | Internet routing manipulation | Global disruption |

---

## 3. Horizon Scanning

### 3.1 Scanning Framework

```
1. Source Identification
   └── Academic, industry, government, media

2. Signal Detection
   └── Weak signals, trend indicators

3. Analysis
   └── Probability, impact, velocity

4. Prioritization
   └── Watch, monitor, action

5. Response Planning
   └── Scenarios, contingencies

6. Integration
   └── Risk register, strategy
```

### 3.2 Scanning Sources

| Source Type | Examples |
|-------------|----------|
| **Academic** | Research journals, think tanks |
| **Industry** | Trade publications, conferences |
| **Government** | Policy papers, agency reports |
| **Technology** | Patent filings, VC investments |
| **Media** | Investigative journalism |
| **Networks** | Industry associations, peer groups |

### 3.3 Emerging Risk Register

| Risk | Probability | Impact | Velocity | Status |
|------|-------------|--------|----------|--------|
| AI model failure at scale | Medium | High | Fast | Monitor |
| Quantum crypto break | Low (near-term) | Critical | Fast | Watch |
| Climate physical event | Medium | High | Variable | Monitor |
| Major cyber infrastructure attack | Medium | Critical | Immediate | Action |
| Geopolitical conflict escalation | Medium | High | Fast | Monitor |

---

## 4. Early Warning Signals

### 4.1 Systemic Stress Indicators

| Indicator | Normal | Elevated | Crisis |
|-----------|--------|----------|--------|
| VIX | < 20 | 20-30 | > 30 |
| TED spread | < 50 bps | 50-100 bps | > 100 bps |
| Credit spreads (IG) | < 150 bps | 150-300 bps | > 300 bps |
| Interbank lending | Flowing | Cautious | Frozen |
| Central bank liquidity | Normal | Elevated | Emergency |

### 4.2 Emerging Risk Signals

| Signal Type | Examples |
|-------------|----------|
| **Regulatory activity** | Consultations, speeches |
| **Incident frequency** | Increasing near-misses |
| **Technology adoption** | Accelerating uptake |
| **Investment flows** | Capital movement patterns |
| **Expert warnings** | Specialist concerns |

---

## 5. Impact Analysis

### 5.1 Systemic Event Impact

| Scenario | Market Impact | Duration |
|----------|---------------|----------|
| Major financial institution failure | -20% to -50% | Months |
| Infrastructure outage (extended) | Trading halt | Days-weeks |
| Geopolitical crisis | -10% to -30% | Weeks-months |
| Pandemic (severe) | -30% to -50% | Months-years |
| Cyber infrastructure attack | Uncertain | Days-months |

### 5.2 Business Impact

| Dimension | Systemic Event Impact |
|-----------|----------------------|
| Trading operations | Potential complete halt |
| Counterparty relationships | Default exposure |
| Liquidity | Severe constraints |
| Funding | Access uncertainty |
| Regulatory | Enhanced scrutiny |

---

## 6. Mitigation Strategies

### 6.1 Emerging Risk Management

| Strategy | Description | Implementation |
|----------|-------------|----------------|
| **Horizon scanning** | Early detection | Formal process |
| **Scenario planning** | Explore possibilities | Regular exercises |
| **Optionality** | Maintain flexibility | Strategic choices |
| **Diversification** | Reduce concentration | Portfolio construction |
| **Insurance** | Risk transfer | Coverage review |

### 6.2 Systemic Risk Management

| Strategy | Description | Implementation |
|----------|-------------|----------------|
| **Counterparty diversification** | Multiple relationships | Limit setting |
| **Stress testing** | Systemic scenarios | Regular testing |
| **Liquidity buffers** | Maintain reserves | Policy minimums |
| **Contingency planning** | Response procedures | Documented playbooks |
| **Coordination** | Industry collaboration | Working groups |

### 6.3 Technology Risk Mitigation

| Strategy | Description | Implementation |
|----------|-------------|----------------|
| **Multi-cloud** | Avoid single provider | Architecture design |
| **Crypto agility** | Prepare for quantum | Algorithm flexibility |
| **AI governance** | Model risk management | Review frameworks |
| **Resilience design** | Assume failures | Fault tolerance |

---

## 7. Scenario Planning

### 7.1 Scenario Framework

```
1. Identify Critical Uncertainties
   └── Key drivers with uncertain outcomes

2. Develop Scenario Matrix
   └── 2x2 based on two key uncertainties

3. Flesh Out Scenarios
   └── Narrative, implications, indicators

4. Test Strategy
   └── How does current strategy perform?

5. Identify Strategic Options
   └── What would we do in each scenario?

6. Define Signposts
   └── What would tell us which scenario is unfolding?
```

### 7.2 Sample Scenarios

| Scenario | Description | Trading Impact |
|----------|-------------|----------------|
| **Tech acceleration** | AI/quantum accelerate | Model obsolescence risk |
| **Deglobalization** | Trade fragmentation | Regional focus needed |
| **Climate shock** | Major physical event | Sector rotation |
| **Cyber crisis** | Infrastructure attack | Operational halt |
| **Pandemic recurrence** | New pathogen | Volatility, remote ops |

---

## 8. Governance for Emerging Risk

### 8.1 Roles and Responsibilities

| Role | Responsibility |
|------|----------------|
| **Board** | Strategic risk oversight |
| **CEO** | Emerging risk escalation |
| **CRO** | Framework and monitoring |
| **Strategy** | Competitive implications |
| **Technology** | Technology risk assessment |
| **Operations** | Operational resilience |

### 8.2 Reporting Framework

| Report | Frequency | Content |
|--------|-----------|---------|
| Emerging risk horizon | Quarterly | New and evolving risks |
| Systemic risk assessment | Semi-annual | Interconnection analysis |
| Scenario exercise results | Annual | Strategic implications |
| Deep dive | As needed | Specific risk investigation |

---

## 9. Real-World Examples

### 9.1 Global Financial Crisis (2008)

**Type**: Systemic financial crisis.

**Mechanism**:
- Subprime mortgage exposure
- Complex securitization
- Counterparty contagion
- Liquidity collapse

**Lessons**:
- Interconnection awareness
- Stress testing importance
- Liquidity reserves
- Regulatory evolution

### 9.2 COVID-19 Pandemic (2020)

**Type**: Emerging pandemic risk materialized.

**Impact**:
- Market volatility spike
- Remote operations test
- Supply chain disruption
- Policy response

**Lessons**:
- Business continuity preparation
- Technology infrastructure
- Scenario planning value
- Adaptation capability

---

## 10. Residual Risk

Emerging and systemic risks remain due to:

- Inherent unpredictability
- Black swan events
- Complex system dynamics
- Limited historical data
- Interconnected nature

**Acceptance requires**:
- Active horizon scanning
- Scenario planning capability
- Stress testing program
- Contingency plans
- Adequate capital/reserves

---

## Cross-References

- [Strategic Risk](../strategic/strategic_risk.md)
- [Technical Risk](../technical/technical_risk.md)
- [Risk Governance](../frameworks/risk_governance.md)
- [Advanced Risk Methods](../advanced_risk_methods.md)

---

**Template**: Enterprise Risk Management v1.0
