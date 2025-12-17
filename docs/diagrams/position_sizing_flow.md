# Position Sizing Flow Diagrams

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Position Sizing in Ordinis                │
└─────────────────────────────────────────────────────────────┘
                           │
                           ├──────────────────┬──────────────────────┐
                           │                  │                      │
                           ▼                  ▼                      ▼
            ┌───────────────────┐  ┌──────────────────┐  ┌──────────────────┐
            │ Portfolio Engine  │  │ PortfolioOpt Eng │  │ Risk Management  │
            │  (Rebalancing)    │  │  (Optimization)  │  │   (Constraints)  │
            └───────────────────┘  └──────────────────┘  └──────────────────┘
                     │                      │                      │
                     │                      │                      │
            Weight-Based            Optimization-Based      Multi-Layer
            Allocation              Position Sizing         Validation
```

## Portfolio Engine Flow

### Target Allocation Strategy

```
Input: Current Positions + Prices
         │
         ▼
┌─────────────────────────┐
│ Calculate Total Value   │
│  = Σ(shares × price)    │
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│ Calculate Current       │
│ Weights                 │
│  w_i = value_i / total  │
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│ Compare to Targets      │
│  drift = w_i - target_i │
└─────────────────────────┘
         │
         ▼
    ┌────────┐
    │ Drift  │
    │ > 5%?  │
    └────────┘
      │    │
      No   Yes
      │    │
      ▼    ▼
   Skip  ┌──────────────────────┐
         │ Calculate Adjustment │
         │  target_$ = total×w  │
         │  adjust_$ = target-cur│
         │  shares = adjust_$/p │
         └──────────────────────┘
                 │
                 ▼
         ┌──────────────────┐
         │ Rebalance Order  │
         │  (BUY or SELL)   │
         └──────────────────┘
```

### Risk Parity Strategy

```
Input: Historical Returns (252 days)
         │
         ▼
┌─────────────────────────┐
│ Calculate Volatilities  │
│  σ_i = std(R_i)×√252    │
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│ Inverse Vol Weighting   │
│  w_i = 1/σ_i            │
│  w_i = w_i/Σ(w_j)       │
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│ Apply Constraints       │
│  w_min ≤ w_i ≤ w_max    │
│  Renormalize: Σw_i = 1  │
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│ Calculate Risk Contrib  │
│  RC_i = w_i × σ_i       │
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│ Generate Orders         │
│  (Same as Target Alloc) │
└─────────────────────────┘
```

### Signal-Driven Strategy

```
Input: Trading Signals (symbol, signal, confidence)
         │
         ▼
    ┌───────────┐
    │  Filter   │───Yes─→ Remove
    │ signal<0? │
    └───────────┘
         │No
         ▼
┌──────────────────────────┐
│ Method Selection:        │
│                          │
│ PROPORTIONAL:            │
│   w_i ∝ |s_i| × c_i      │
│                          │
│ BINARY:                  │
│   w_i = 1/N (equal)      │
│                          │
│ RANKED:                  │
│   w_i ∝ rank(s_i × c_i)  │
└──────────────────────────┘
         │
         ▼
┌──────────────────────────┐
│ Apply Cash Buffer        │
│  invested = 1 - buffer   │
│  w_i = w_i × invested    │
└──────────────────────────┘
         │
         ▼
┌──────────────────────────┐
│ Apply Min/Max Weights    │
│  clip(w_i, w_min, w_max) │
└──────────────────────────┘
         │
         ▼
┌──────────────────────────┐
│ Generate Orders          │
└──────────────────────────┘
```

## PortfolioOpt Engine Flow

### Mean-CVaR Optimization

```
Input: Historical Returns + Target Return
         │
         ▼
┌────────────────────────────────┐
│ Calculate Statistics           │
│  μ = E[R]  (expected returns)  │
│  Σ = Cov(R) (covariance)       │
└────────────────────────────────┘
         │
         ▼
┌────────────────────────────────┐
│ Formulate Optimization Problem │
│                                 │
│  minimize:                      │
│    (1-λ)×(-μᵀw) + λ×CVaR(w)    │
│                                 │
│  subject to:                    │
│    Σw_i = 1                     │
│    0 ≤ w_i ≤ max_weight         │
│    μᵀw ≥ target_return          │
└────────────────────────────────┘
         │
         ▼
    ┌─────────┐
    │ GPU     │
    │Available│
    └─────────┘
      │     │
      Yes   No
      │     │
      ▼     ▼
   cuOpt  CVXPY
   (GPU)  (CPU)
      │     │
      └──┬──┘
         ▼
┌─────────────────────┐
│ Optimal Weights w*  │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│ Validate Constraints│
│  • Concentration    │
│  • Diversification  │
│  • CVaR limit       │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│ Return Weights +    │
│ Risk Metrics        │
└─────────────────────┘
```

## Execution Pipeline Flow

### Signal to Position

```
Signal Input
         │
         ▼
┌──────────────────────────┐
│ Base Position Size       │
│  base = equity × 5%      │
└──────────────────────────┘
         │
         ▼
┌──────────────────────────┐
│ Confidence Adjustment    │
│  adj = prob × |score|    │
│  size = base × adj       │
└──────────────────────────┘
         │
         ▼
┌──────────────────────────┐
│ Apply Maximum Cap        │
│  max = equity × 15%      │
│  size = min(size, max)   │
└──────────────────────────┘
         │
         ▼
┌──────────────────────────┐
│ Convert to Quantity      │
│  qty = size / price      │
└──────────────────────────┘
         │
         ▼
┌──────────────────────────┐
│ Risk Guard Validation    │
│  • Position limit        │
│  • Exposure limit        │
│  • Min position size     │
└──────────────────────────┘
         │
         ├──Rejected─→ Skip
         │
         ▼Approved
┌──────────────────────────┐
│ Execute Order            │
└──────────────────────────┘
```

## Risk Management Layers

```
┌─────────────────────────────────────────────────┐
│              Position Size Request               │
└─────────────────────────────────────────────────┘
                      │
                      ▼
         ┌────────────────────────┐
         │   Layer 1: Engine      │
         │   • max_weight         │
         │   • max_concentration  │
         │   • min_diversification│
         └────────────────────────┘
                      │
                      ▼
         ┌────────────────────────┐
         │   Layer 2: RiskGuard   │
         │   • max_position_pct   │
         │   • min_position_size  │
         │   • max_exposure       │
         └────────────────────────┘
                      │
                      ▼
         ┌────────────────────────┐
         │   Layer 3: Governance  │
         │   • daily_trade_limit  │
         │   • max_drawdown       │
         │   • approval_threshold │
         └────────────────────────┘
                      │
                      ├──Passed──→ Execute
                      │
                      └──Failed──→ Reject/Resize
```

## Regime-Adaptive Scaling

```
Market Regime Detection
         │
         ▼
    ┌──────────┐
    │ Trending │
    │ Ranging  │
    │ Volatile │
    └──────────┘
         │
         ├──────────┬──────────┬──────────┐
         │          │          │          │
         ▼          ▼          ▼          ▼
    Trending    Ranging   Volatile  Sideways
      ×1.2        ×0.8      ×0.5      ×1.0
         │          │          │          │
         └──────────┴─────┬────┴──────────┘
                          ▼
              ┌─────────────────────┐
              │ Base Position Size  │
              │      ×Multiplier    │
              └─────────────────────┘
                          │
                          ▼
              ┌─────────────────────┐
              │ Volatility Adjust   │
              │  ÷(1+volatility)    │
              └─────────────────────┘
                          │
                          ▼
              ┌─────────────────────┐
              │ Confidence Scaling  │
              │  ×regime_confidence │
              └─────────────────────┘
                          │
                          ▼
              ┌─────────────────────┐
              │ Final Position Size │
              └─────────────────────┘
```

## Integration: Hybrid Approach

```
┌────────────────────┐
│ Historical Returns │
└────────────────────┘
         │
         ▼
┌────────────────────┐
│ PortfolioOpt       │
│  Mean-CVaR Optim   │
└────────────────────┘
         │
         ▼
┌────────────────────┐
│ Optimal Weights    │
│  w* = {AAPL: 0.35} │
│       {MSFT: 0.40} │
│       {GOOGL:0.25} │
└────────────────────┘
         │
         ▼
┌────────────────────┐
│ Convert to Targets │
│ TargetAllocation[] │
└────────────────────┘
         │
         ▼
┌────────────────────┐
│ Portfolio Engine   │
│ Target Alloc Strat │
└────────────────────┘
         │
         ▼
┌────────────────────┐
│ Current Positions  │
│ + Prices           │
└────────────────────┘
         │
         ▼
┌────────────────────┐
│ Generate Rebalance │
│ Decisions          │
└────────────────────┘
         │
         ▼
┌────────────────────┐
│ Execute Orders     │
└────────────────────┘
```

## Configuration Hierarchy

```
┌──────────────────────────────────┐
│     System-Level Defaults        │
│  (pyproject.toml, configs/*.yaml)│
└──────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────┐
│     Engine Configuration         │
│  (PortfolioEngineConfig,         │
│   PortfolioOptEngineConfig)      │
└──────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────┐
│     Strategy Parameters          │
│  (TargetAllocation, RiskParity,  │
│   SignalDriven configs)          │
└──────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────┐
│     Runtime Overrides            │
│  (Function arguments,            │
│   API parameters)                │
└──────────────────────────────────┘
```

## Key Decision Points

### When to Use Each Approach

```
┌────────────────────────────────────────┐
│        Position Sizing Method          │
└────────────────────────────────────────┘
                    │
                    │
    ┌───────────────┼───────────────┐
    │               │               │
    ▼               ▼               ▼
Fixed          Risk-Based      Optimization
Allocation     Allocation      -Based
    │               │               │
    │               │               │
    ▼               ▼               ▼
Target         Risk Parity    PortfolioOpt
Allocation     Strategy       Mean-CVaR
    │               │               │
    │               │               │
When:          When:          When:
• Simple       • Diverse      • Large
• Stable         assets         universe
• Low          • Risk           (>20 assets)
  turnover       focus        • Risk/return
• Clear        • Volatility     optimization
  targets        varies       • Constraints
                              • Computation
                                available
```
