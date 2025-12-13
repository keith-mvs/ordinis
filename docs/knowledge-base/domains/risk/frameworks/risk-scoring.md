# Risk Scoring, Prioritization, and Heat Maps

**Section**: 03_risk/frameworks
**Last Updated**: 2025-12-12
**Version**: 1.0

---

## Overview

Risk scoring provides a standardized method for comparing and prioritizing risks across different categories. This document defines scoring methodologies, prioritization frameworks, and visualization techniques.

---

## 1. Scoring Methodologies

### 1.1 Simple Multiplicative Score

The most common approach: multiply probability by impact.

```python
def simple_risk_score(probability: int, impact: int) -> int:
    """
    Simple P × I scoring.
    Range: 1-25 (with 5-point scales)
    """
    return probability * impact
```

**Advantages**: Simple, intuitive, widely understood
**Limitations**: Doesn't differentiate high-P/low-I from low-P/high-I

### 1.2 Weighted Factor Score

Incorporate additional dimensions with weights.

```python
def weighted_risk_score(
    probability: int,
    impact: int,
    velocity: int,
    detectability: int,
    weights: Dict[str, float] = None
) -> float:
    """
    Multi-factor weighted score.
    """
    weights = weights or {
        'probability': 0.30,
        'impact': 0.40,
        'velocity': 0.15,
        'detectability': 0.15
    }

    return (
        probability * weights['probability'] +
        impact * weights['impact'] +
        velocity * weights['velocity'] +
        detectability * weights['detectability']
    ) * 5  # Scale to 0-25 range
```

### 1.3 Logarithmic Score (High-Impact Focus)

For contexts where extreme impacts require emphasis.

```python
import math

def log_risk_score(probability: int, impact: int) -> float:
    """
    Logarithmic scoring emphasizes high impact.
    Differentiates catastrophic from major risks.
    """
    # Exponential impact scale
    impact_factor = 2 ** (impact - 1)  # 1, 2, 4, 8, 16
    return probability * math.log2(1 + impact_factor)
```

---

## 2. Prioritization Framework

### 2.1 Priority Matrix

| Score Range | Priority Level | Response Timeframe | Escalation |
|-------------|----------------|-------------------|------------|
| 20-25 | Critical | Immediate (24h) | Executive/Board |
| 15-19 | High | Short-term (1 week) | Senior Management |
| 9-14 | Medium | Medium-term (1 month) | Risk Manager |
| 4-8 | Low | Long-term (quarterly) | Risk Owner |
| 1-3 | Minimal | Annual review | Risk Owner |

### 2.2 Multi-Criteria Prioritization

When scores are equal, apply tiebreakers:

```python
def prioritize_risks(risks: List[Risk]) -> List[Risk]:
    """
    Multi-criteria prioritization.
    Primary: Risk score
    Secondary: Velocity
    Tertiary: Controllability (inverse)
    """
    return sorted(
        risks,
        key=lambda r: (
            -r.risk_score,       # Higher score first
            -r.velocity,         # Faster velocity first
            r.controllability    # Less controllable first
        )
    )
```

### 2.3 Forced Ranking

For resource allocation, force-rank top risks:

| Rank | Risk | Score | Resource Allocation |
|------|------|-------|---------------------|
| 1 | Market crash exposure | 25 | 30% of risk budget |
| 2 | System outage | 20 | 20% of risk budget |
| 3 | Regulatory violation | 16 | 15% of risk budget |
| ... | ... | ... | ... |

---

## 3. Heat Maps

### 3.1 Standard 5×5 Heat Map

```
                    IMPACT
           1     2     3     4     5
         ┌─────┬─────┬─────┬─────┬─────┐
       5 │  5  │ 10  │ 15  │ 20  │ 25  │  ← Almost Certain
P        ├─────┼─────┼─────┼─────┼─────┤
R      4 │  4  │  8  │ 12  │ 16  │ 20  │  ← Likely
O        ├─────┼─────┼─────┼─────┼─────┤
B      3 │  3  │  6  │  9  │ 12  │ 15  │  ← Possible
         ├─────┼─────┼─────┼─────┼─────┤
       2 │  2  │  4  │  6  │  8  │ 10  │  ← Unlikely
         ├─────┼─────┼─────┼─────┼─────┤
       1 │  1  │  2  │  3  │  4  │  5  │  ← Rare
         └─────┴─────┴─────┴─────┴─────┘
           Negl  Minor Mod   Major Catast

Color Legend:
 ■ Critical (20-25): Red
 ■ High (15-19): Orange
 ■ Medium (9-14): Yellow
 ■ Low (4-8): Light Green
 ■ Minimal (1-3): Green
```

### 3.2 Asymmetric Heat Map

For contexts where low-probability/high-impact risks require special attention:

```
                    IMPACT
           1     2     3     4     5
         ┌─────┬─────┬─────┬─────┬─────┐
       5 │  M  │  H  │  C  │  C  │  C  │
P        ├─────┼─────┼─────┼─────┼─────┤
R      4 │  L  │  M  │  H  │  C  │  C  │
O        ├─────┼─────┼─────┼─────┼─────┤
B      3 │  L  │  M  │  M  │  H  │  C  │
         ├─────┼─────┼─────┼─────┼─────┤
       2 │  L  │  L  │  M  │  H  │  H  │  ← Special zone
         ├─────┼─────┼─────┼─────┼─────┤
       1 │  L  │  L  │  M  │  H  │  H  │  ← Low prob, high impact
         └─────┴─────┴─────┴─────┴─────┘

L = Low, M = Medium, H = High, C = Critical
```

### 3.3 Heat Map Generation (Python)

```python
import matplotlib.pyplot as plt
import numpy as np

def generate_heat_map(risks: List[Dict], output_path: str):
    """
    Generate visual heat map from risk data.
    """
    # Create 5x5 grid
    grid = np.zeros((5, 5))
    labels = [['' for _ in range(5)] for _ in range(5)]

    for risk in risks:
        p = risk['probability'] - 1  # 0-indexed
        i = risk['impact'] - 1
        grid[4-p, i] += 1  # Flip for display
        labels[4-p][i] += f"{risk['id']}\n"

    fig, ax = plt.subplots(figsize=(10, 8))

    # Color mapping
    colors = plt.cm.RdYlGn_r(np.linspace(0, 1, 25))
    score_grid = np.array([
        [5, 10, 15, 20, 25],
        [4, 8, 12, 16, 20],
        [3, 6, 9, 12, 15],
        [2, 4, 6, 8, 10],
        [1, 2, 3, 4, 5]
    ])

    im = ax.imshow(score_grid, cmap='RdYlGn_r')

    # Add risk IDs
    for i in range(5):
        for j in range(5):
            if labels[i][j]:
                ax.text(j, i, labels[i][j], ha='center', va='center')

    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(['Negligible', 'Minor', 'Moderate', 'Major', 'Catastrophic'])
    ax.set_yticklabels(['Almost Certain', 'Likely', 'Possible', 'Unlikely', 'Rare'])
    ax.set_xlabel('Impact')
    ax.set_ylabel('Probability')

    plt.colorbar(im, label='Risk Score')
    plt.title('Risk Heat Map')
    plt.savefig(output_path)
```

---

## 4. Risk Thresholds and Tolerances

### 4.1 Risk Appetite Thresholds

| Category | Low Tolerance | Medium Tolerance | High Tolerance |
|----------|---------------|------------------|----------------|
| Financial | < $100K | $100K - $1M | > $1M |
| Operational | < 4 hours downtime | 4-24 hours | > 24 hours |
| Reputational | Internal only | Local media | National media |
| Compliance | Minor finding | Moderate finding | Material weakness |

### 4.2 Tolerance by Risk Category

```python
RISK_TOLERANCES = {
    'strategic': {
        'low': 6,
        'medium': 12,
        'high': 20,
        'critical': 25
    },
    'operational': {
        'low': 4,
        'medium': 9,
        'high': 16,
        'critical': 25
    },
    'compliance': {
        'low': 3,
        'medium': 6,
        'high': 12,
        'critical': 20
    }
}
```

### 4.3 Breach Response

| Threshold Breach | Required Action |
|------------------|-----------------|
| Low → Medium | Document in risk register |
| Medium → High | Develop treatment plan within 30 days |
| High → Critical | Immediate escalation, emergency response |
| Above appetite | Executive review, potential business decision |

---

## 5. Trend Analysis

### 5.1 Risk Score Trending

| Trend | Indicator | Interpretation |
|-------|-----------|----------------|
| **Improving** | Score decreasing ≥10% | Controls effective |
| **Stable** | Score change <10% | No significant change |
| **Deteriorating** | Score increasing ≥10% | Attention needed |
| **Volatile** | Score fluctuating | Environment unstable |

### 5.2 Trend Visualization

```python
def plot_risk_trend(risk_id: str, scores: List[Tuple[date, int]]):
    """
    Visualize risk score over time.
    """
    dates = [s[0] for s in scores]
    values = [s[1] for s in scores]

    plt.figure(figsize=(10, 5))
    plt.plot(dates, values, marker='o')
    plt.axhline(y=15, color='orange', linestyle='--', label='High threshold')
    plt.axhline(y=20, color='red', linestyle='--', label='Critical threshold')
    plt.fill_between(dates, 0, values, alpha=0.3)
    plt.title(f'Risk Trend: {risk_id}')
    plt.xlabel('Date')
    plt.ylabel('Risk Score')
    plt.legend()
    plt.show()
```

---

## 6. Aggregation Methods

### 6.1 Portfolio-Level Risk Score

```python
def aggregate_risk_scores(risks: List[Risk], method: str = 'weighted') -> float:
    """
    Aggregate individual risks to portfolio level.
    """
    if method == 'max':
        # Conservative: use maximum risk
        return max(r.score for r in risks)

    elif method == 'sum':
        # Additive: total exposure
        return sum(r.score for r in risks)

    elif method == 'weighted':
        # Weight by impact
        total_weight = sum(r.impact for r in risks)
        return sum(r.score * r.impact / total_weight for r in risks)

    elif method == 'sqrt_sum':
        # Diversification benefit
        return math.sqrt(sum(r.score**2 for r in risks))
```

### 6.2 Category Aggregation

| Category | Individual Risks | Aggregated Score | Method |
|----------|------------------|------------------|--------|
| Strategic | 5 risks | 18 | Max |
| Operational | 12 risks | 45 | Sum (capped) |
| Technical | 8 risks | 14 | Weighted |
| Compliance | 6 risks | 16 | Max |

---

## 7. Reporting Formats

### 7.1 Executive Summary

```markdown
## Risk Summary (Q4 2025)

| Category | Critical | High | Medium | Low | Trend |
|----------|----------|------|--------|-----|-------|
| Strategic | 0 | 2 | 3 | 1 | → |
| Operational | 1 | 4 | 8 | 12 | ↓ |
| Technical | 0 | 1 | 5 | 7 | ↑ |
| Compliance | 0 | 1 | 2 | 3 | → |
| **Total** | **1** | **8** | **18** | **23** | → |

**Key Changes:**
- 1 new Critical risk (System stability)
- 3 risks reduced from High to Medium
- 2 risks closed (remediated)
```

### 7.2 Detailed Risk Register View

| ID | Name | Category | P | I | Score | Owner | Status | Trend |
|----|------|----------|---|---|-------|-------|--------|-------|
| R001 | Market crash | Strategic | 3 | 5 | 15 | CRO | Monitor | → |
| R002 | System outage | Technical | 4 | 4 | 16 | CTO | Mitigate | ↓ |
| R003 | Data breach | Security | 2 | 5 | 10 | CISO | Monitor | → |

---

## Cross-References

- [Risk Taxonomy](risk_taxonomy.md)
- [Risk Assessment Methodology](risk_assessment_methodology.md)
- [Risk Governance Framework](risk_governance.md)

---

**Template**: Enterprise Risk Management v1.0
