# SignalCore Trading System Architecture

## Overview

The **SignalCore Trading System** is an AI-driven, automated trading research and execution platform built on a hybrid, LLM-orchestrated, rule-constrained protocol. The system is composed of five conceptual engines (subsystems) that define roles, responsibilities, and boundaries.

**Design Philosophy**: These engines define logical separations of concern. In implementation (e.g., within Claude Code, tools, MCP servers/clients, or microservices), they may be combined, split, or refactored as long as their **core functions and auditability are preserved**.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         SIGNALCORE TRADING SYSTEM                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                     CORTEX ORCHESTRATION ENGINE                          │   │
│   │                         (LLM Layer)                                      │   │
│   │  Research • Strategy Design • Parameter Proposals • Natural Language     │   │
│   └────────────────────────────────┬────────────────────────────────────────┘   │
│                                    │                                             │
│              configures / reviews  │  does NOT place trades                     │
│                                    ▼                                             │
│   ┌──────────────────────────────────────────────────────────────────────────┐  │
│   │                                                                           │  │
│   │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐      │  │
│   │  │   SIGNALCORE    │───▶│    RISKGUARD    │───▶│    FLOWROUTE    │      │  │
│   │  │   ML ENGINE     │    │   RULE ENGINE   │    │EXECUTION ENGINE │      │  │
│   │  │                 │    │                 │    │                 │      │  │
│   │  │ Numerical       │    │ Deterministic   │    │ Broker API      │      │  │
│   │  │ Signal Layer    │    │ Risk Layer      │    │ Routing Layer   │      │  │
│   │  └─────────────────┘    └─────────────────┘    └─────────────────┘      │  │
│   │         │                       │                       │                │  │
│   │         │                       │                       │                │  │
│   │         └───────────────────────┴───────────────────────┘                │  │
│   │                                 │                                         │  │
│   │                                 ▼                                         │  │
│   │                     ┌─────────────────────┐                              │  │
│   │                     │     PROOFBENCH      │                              │  │
│   │                     │ VALIDATION ENGINE   │                              │  │
│   │                     │                     │                              │  │
│   │                     │ Training • Testing  │                              │  │
│   │                     │ Backtesting • Eval  │                              │  │
│   │                     └─────────────────────┘                              │  │
│   │                                                                           │  │
│   └──────────────────────────────────────────────────────────────────────────┘  │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Engine Definitions

### 1. Cortex Orchestration Engine (LLM Layer)

**Purpose**: Provides the intelligent, adaptive layer for research, strategy design, and human-like reasoning without directly executing trades.

**Core Functions**:
| Function | Description |
|----------|-------------|
| Research & Summarization | Synthesize market research, academic papers, and data into actionable insights |
| Hypothesis Generation | Propose trading hypotheses based on market conditions and historical patterns |
| Strategy Documentation | Generate and maintain strategy specifications in structured formats |
| Parameter Proposals | Suggest parameter configurations for SignalCore models |
| Natural Language Interface | Enable conversational interaction for system configuration and monitoring |
| Review & Audit | Assess and critique outputs from downstream engines |

**Critical Constraints**:
- **Does NOT place trades** or output final position sizes
- **Does NOT bypass** RiskGuard rule checks
- Configures and reviews downstream engines only
- All proposals are advisory until validated by ProofBench

**Implementation Flexibility**:
```
In Claude Code, Cortex may be realized as:
├── One or more MCP tools/skills
├── Prompt chains with structured outputs
├── Integration hooks to other engines
└── NOT a single monolithic service
```

**Interfaces**:
```python
@dataclass
class CortexOutput:
    """Standard output format from Cortex engine."""
    output_type: Literal['research', 'hypothesis', 'strategy_spec', 'param_proposal', 'review']
    content: dict
    confidence: float  # 0.0 to 1.0
    reasoning: str     # Audit trail
    requires_validation: bool = True
    metadata: dict = field(default_factory=dict)


class CortexEngine(Protocol):
    """Protocol defining Cortex interface."""

    def research(
        self,
        query: str,
        sources: list[str],
        context: dict
    ) -> CortexOutput:
        """Conduct research and return synthesized findings."""
        ...

    def propose_strategy(
        self,
        hypothesis: str,
        constraints: dict
    ) -> CortexOutput:
        """Generate strategy specification from hypothesis."""
        ...

    def propose_parameters(
        self,
        strategy_id: str,
        performance_data: dict
    ) -> CortexOutput:
        """Suggest parameter adjustments based on performance."""
        ...

    def review_output(
        self,
        engine: str,
        output: dict
    ) -> CortexOutput:
        """Review and critique output from another engine."""
        ...
```

---

### 2. SignalCore ML Engine (Numerical Signal Layer)

**Purpose**: Generates quantitative trade signals using explicit, testable numerical models.

**Core Functions**:
| Function | Description |
|----------|-------------|
| Supervised Learning | Train models on historical data with known outcomes |
| Time-Series Analysis | ARIMA, GARCH, cointegration, and regime-aware models |
| Factor Models | Multi-factor alpha and risk decomposition |
| Regime Detection | Unsupervised methods for market state identification |
| Anomaly Detection | Identify unusual market conditions or data issues |
| Signal Scoring | Output calibrated probabilities, scores, or expected returns |

**Output Specification**:
```python
@dataclass
class Signal:
    """Quantitative signal output from SignalCore."""
    symbol: str
    timestamp: datetime
    signal_type: Literal['entry', 'exit', 'scale', 'hold']
    direction: Literal['long', 'short', 'neutral']

    # Quantitative outputs (NOT direct orders)
    probability: float          # Probability of favorable outcome
    expected_return: float      # Point estimate of expected return
    confidence_interval: tuple  # (lower, upper) bounds
    score: float               # Composite signal strength (-1 to +1)

    # Model attribution
    model_id: str
    model_version: str
    feature_contributions: dict  # For explainability

    # Metadata
    regime: str                 # Current detected regime
    data_quality: float         # Input data quality score
    staleness: timedelta        # Age of underlying data


@dataclass
class SignalBatch:
    """Collection of signals for portfolio-level decisions."""
    timestamp: datetime
    signals: list[Signal]
    universe: list[str]
    regime_state: dict
    correlation_matrix: Optional[np.ndarray]
    portfolio_context: dict
```

**Critical Constraints**:
- Outputs are **quantitative signals**, not direct orders
- All models must be **testable and reproducible**
- Model inputs/outputs must be **well-defined and traceable**
- No hidden state that bypasses auditability

**Implementation Flexibility**:
```
SignalCore may be realized as:
├── Standalone model-serving service
├── Collection of model endpoints (REST/gRPC)
├── Embedded in analytics pipeline
├── Hybrid: Some models local, some cloud
└── Any combination with defined I/O contracts
```

**Model Registry**:
```python
@dataclass
class ModelRegistration:
    """Registration entry for a SignalCore model."""
    model_id: str
    model_type: Literal['supervised', 'timeseries', 'factor', 'regime', 'ensemble']
    version: str
    training_date: datetime
    training_config: dict

    # Performance metadata
    validation_metrics: dict
    proofbench_report_id: str

    # Operational metadata
    input_schema: dict
    output_schema: dict
    latency_p99_ms: float
    memory_mb: float

    # Lifecycle
    status: Literal['experimental', 'staging', 'production', 'deprecated']
    approved_by: str
    approval_date: datetime
```

---

### 3. RiskGuard Rule Engine (Risk & Constraint Layer)

**Purpose**: Applies deterministic, auditable rules to all trading decisions before execution.

**Core Functions**:
| Function | Description |
|----------|-------------|
| Entry/Exit Logic | Rule-based filtering of SignalCore outputs |
| Per-Trade Risk Limits | Position sizing, max loss, stop placement |
| Portfolio Limits | Max positions, concentration, sector exposure |
| Correlation Constraints | Cross-asset correlation limits |
| Kill Switch Criteria | Emergency halt conditions |
| Sanity Checks | Price, size, liquidity, connectivity validation |

**Rule Categories**:
```python
class RuleCategory(Enum):
    """Categories of risk rules."""
    PRE_TRADE = "pre_trade"           # Before order generation
    ORDER_VALIDATION = "order_val"     # Before order submission
    POSITION_LIMIT = "position"        # Position-level constraints
    PORTFOLIO_LIMIT = "portfolio"      # Portfolio-level constraints
    KILL_SWITCH = "kill_switch"        # Emergency halt conditions
    SANITY_CHECK = "sanity"           # Data/connectivity validation


@dataclass
class RiskRule:
    """Definition of a single risk rule."""
    rule_id: str
    category: RuleCategory
    name: str
    description: str

    # Rule specification
    condition: str              # Human-readable condition
    expression: str             # Evaluable expression
    threshold: float
    comparison: Literal['<', '<=', '>', '>=', '==', '!=']

    # Actions
    action_on_breach: Literal['reject', 'resize', 'warn', 'halt']
    severity: Literal['low', 'medium', 'high', 'critical']

    # Audit
    enabled: bool = True
    last_modified: datetime = None
    modified_by: str = None


@dataclass
class RiskCheckResult:
    """Result of risk rule evaluation."""
    rule_id: str
    passed: bool
    current_value: float
    threshold: float
    message: str
    action_taken: str
    timestamp: datetime
```

**Standard Rule Set**:
```python
STANDARD_RISK_RULES = {
    # Per-trade limits
    "max_position_pct": RiskRule(
        rule_id="RT001",
        category=RuleCategory.PRE_TRADE,
        name="Max Position Size",
        description="Maximum position as percentage of equity",
        expression="position_value / portfolio_equity",
        threshold=0.10,
        comparison="<=",
        action_on_breach="resize",
        severity="high"
    ),

    "max_risk_per_trade": RiskRule(
        rule_id="RT002",
        category=RuleCategory.PRE_TRADE,
        name="Max Risk Per Trade",
        description="Maximum capital at risk per trade",
        expression="(entry_price - stop_price) * quantity / portfolio_equity",
        threshold=0.01,
        comparison="<=",
        action_on_breach="resize",
        severity="high"
    ),

    # Portfolio limits
    "max_positions": RiskRule(
        rule_id="RP001",
        category=RuleCategory.PORTFOLIO_LIMIT,
        name="Max Open Positions",
        description="Maximum number of concurrent positions",
        expression="count(open_positions)",
        threshold=10,
        comparison="<=",
        action_on_breach="reject",
        severity="medium"
    ),

    "max_sector_concentration": RiskRule(
        rule_id="RP002",
        category=RuleCategory.PORTFOLIO_LIMIT,
        name="Max Sector Concentration",
        description="Maximum exposure to single sector",
        expression="sector_exposure / portfolio_equity",
        threshold=0.30,
        comparison="<=",
        action_on_breach="reject",
        severity="high"
    ),

    "max_correlation_exposure": RiskRule(
        rule_id="RP003",
        category=RuleCategory.PORTFOLIO_LIMIT,
        name="Max Correlated Exposure",
        description="Maximum exposure to highly correlated assets (r > 0.7)",
        expression="correlated_exposure / portfolio_equity",
        threshold=0.40,
        comparison="<=",
        action_on_breach="warn",
        severity="medium"
    ),

    # Kill switches
    "daily_loss_limit": RiskRule(
        rule_id="RK001",
        category=RuleCategory.KILL_SWITCH,
        name="Daily Loss Limit",
        description="Maximum daily loss before halt",
        expression="daily_pnl / portfolio_equity",
        threshold=-0.03,
        comparison=">=",
        action_on_breach="halt",
        severity="critical"
    ),

    "max_drawdown": RiskRule(
        rule_id="RK002",
        category=RuleCategory.KILL_SWITCH,
        name="Max Drawdown",
        description="Maximum drawdown from peak before halt",
        expression="(equity - peak_equity) / peak_equity",
        threshold=-0.15,
        comparison=">=",
        action_on_breach="halt",
        severity="critical"
    ),

    # Sanity checks
    "price_deviation": RiskRule(
        rule_id="RS001",
        category=RuleCategory.SANITY_CHECK,
        name="Price Deviation Check",
        description="Order price deviation from last trade",
        expression="abs(order_price - last_price) / last_price",
        threshold=0.05,
        comparison="<=",
        action_on_breach="reject",
        severity="high"
    ),

    "liquidity_check": RiskRule(
        rule_id="RS002",
        category=RuleCategory.SANITY_CHECK,
        name="Liquidity Check",
        description="Order size relative to average daily volume",
        expression="order_quantity / avg_daily_volume",
        threshold=0.01,
        comparison="<=",
        action_on_breach="resize",
        severity="medium"
    )
}
```

**RiskGuard Interface**:
```python
class RiskGuardEngine(Protocol):
    """Protocol defining RiskGuard interface."""

    def evaluate_signal(
        self,
        signal: Signal,
        portfolio_state: PortfolioState,
        market_state: MarketState
    ) -> tuple[bool, list[RiskCheckResult], Optional[Signal]]:
        """
        Evaluate signal against all applicable rules.

        Returns:
            - passed: Whether signal passed all critical checks
            - results: List of all rule check results
            - adjusted_signal: Signal adjusted by resize rules (if any)
        """
        ...

    def evaluate_order(
        self,
        order: Order,
        portfolio_state: PortfolioState
    ) -> tuple[bool, list[RiskCheckResult]]:
        """
        Final validation before order submission.
        """
        ...

    def check_kill_switches(
        self,
        portfolio_state: PortfolioState,
        market_state: MarketState
    ) -> tuple[bool, Optional[str]]:
        """
        Check if any kill switch conditions are triggered.

        Returns:
            - triggered: Whether a kill switch was triggered
            - reason: Explanation if triggered
        """
        ...

    def get_available_capacity(
        self,
        symbol: str,
        portfolio_state: PortfolioState
    ) -> dict:
        """
        Calculate available capacity for new position.

        Returns dict with:
            - max_shares: Maximum shares allowed
            - max_value: Maximum position value allowed
            - limiting_rule: Which rule is the binding constraint
        """
        ...
```

**Implementation Flexibility**:
```
RiskGuard may be realized as:
├── Dedicated MCP server with rule evaluation
├── Embedded in execution workflows
├── Configurable rule sets (YAML/JSON)
├── Claude Code-inspectable configuration
└── Any combination maintaining determinism
```

---

### 4. ProofBench Validation Engine (Training & Evaluation Layer)

**Purpose**: Manages all model training, backtesting, and validation with rigorous statistical standards.

**Core Functions**:
| Function | Description |
|----------|-------------|
| Data Management | Train/validation/test splits with point-in-time accuracy |
| Walk-Forward Testing | Rolling window optimization and validation |
| Transaction Cost Modeling | Realistic slippage, commissions, and market impact |
| Performance Reporting | Standardized metrics (CAGR, Sharpe, drawdown, etc.) |
| Regime Analysis | Performance breakdown by market regime |
| Overfitting Detection | Parameter sensitivity and stability analysis |

**Validation Protocol**:
```python
@dataclass
class ValidationConfig:
    """Configuration for ProofBench validation run."""
    # Data splits
    train_ratio: float = 0.60
    validation_ratio: float = 0.20
    test_ratio: float = 0.20

    # Walk-forward settings
    walk_forward_enabled: bool = True
    n_walk_forward_windows: int = 5
    in_sample_ratio: float = 0.70

    # Transaction costs
    commission_per_share: float = 0.005
    slippage_bps: float = 5.0
    market_impact_coefficient: float = 0.1

    # Minimum requirements
    min_trades: int = 100
    min_time_period_days: int = 252

    # Statistical tests
    monte_carlo_runs: int = 1000
    parameter_sensitivity_pct: float = 0.10


@dataclass
class ValidationReport:
    """Comprehensive validation report from ProofBench."""
    report_id: str
    strategy_id: str
    model_id: str
    generated_at: datetime
    config: ValidationConfig

    # In-sample metrics
    in_sample_metrics: PerformanceMetrics

    # Out-of-sample metrics
    out_of_sample_metrics: PerformanceMetrics

    # Walk-forward results
    walk_forward_results: list[WalkForwardWindow]
    walk_forward_degradation: float  # OOS vs IS performance ratio

    # Monte Carlo analysis
    monte_carlo_results: MonteCarloResults

    # Regime analysis
    regime_performance: dict[str, PerformanceMetrics]

    # Parameter sensitivity
    sensitivity_analysis: SensitivityReport

    # Overall assessment
    passed_validation: bool
    failure_reasons: list[str]
    recommendations: list[str]

    # Approval
    approved: bool = False
    approved_by: Optional[str] = None
    approval_notes: Optional[str] = None


@dataclass
class PerformanceMetrics:
    """Standard performance metrics."""
    # Returns
    total_return: float
    cagr: float
    monthly_returns: list[float]

    # Risk-adjusted
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Drawdowns
    max_drawdown: float
    max_drawdown_duration_days: int
    avg_drawdown: float

    # Trade statistics
    total_trades: int
    win_rate: float
    profit_factor: float
    expectancy: float
    avg_win: float
    avg_loss: float

    # Tail risk
    var_95: float
    cvar_95: float

    # Stability
    return_stability: float  # R² of cumulative returns
    monthly_hit_rate: float  # % of profitable months
```

**Validation Criteria**:
```python
VALIDATION_CRITERIA = {
    # Minimum performance thresholds
    "min_sharpe_oos": 1.0,          # Minimum OOS Sharpe ratio
    "min_profit_factor": 1.2,        # Minimum profit factor
    "max_drawdown": -0.25,           # Maximum acceptable drawdown

    # Robustness requirements
    "max_walk_forward_degradation": 0.30,  # Max IS vs OOS degradation
    "min_monte_carlo_5th_pct": 0.0,        # 5th percentile must be profitable
    "max_parameter_sensitivity": 0.20,      # Max change with 10% param shift

    # Statistical significance
    "min_trades": 100,
    "min_win_rate": 0.35,
    "min_months": 24,

    # Regime robustness
    "profitable_in_regimes": 0.75  # Must be profitable in 75% of regimes
}
```

**ProofBench Interface**:
```python
class ProofBenchEngine(Protocol):
    """Protocol defining ProofBench interface."""

    def run_backtest(
        self,
        strategy: Strategy,
        data: MarketData,
        config: ValidationConfig
    ) -> ValidationReport:
        """Run comprehensive backtest with all validation steps."""
        ...

    def run_walk_forward(
        self,
        strategy: Strategy,
        data: MarketData,
        param_grid: dict,
        config: ValidationConfig
    ) -> list[WalkForwardWindow]:
        """Run walk-forward optimization and validation."""
        ...

    def run_monte_carlo(
        self,
        trades: list[Trade],
        n_simulations: int,
        initial_capital: float
    ) -> MonteCarloResults:
        """Run Monte Carlo simulation on trade sequence."""
        ...

    def analyze_regimes(
        self,
        equity_curve: pd.Series,
        regime_labels: pd.Series
    ) -> dict[str, PerformanceMetrics]:
        """Analyze performance by market regime."""
        ...

    def check_overfitting(
        self,
        in_sample_metrics: PerformanceMetrics,
        out_of_sample_metrics: PerformanceMetrics,
        sensitivity_results: SensitivityReport
    ) -> tuple[bool, list[str]]:
        """
        Detect potential overfitting.

        Returns:
            - is_overfit: Whether overfitting is detected
            - indicators: List of overfitting indicators found
        """
        ...
```

**Implementation Flexibility**:
```
ProofBench may be realized as:
├── Backtesting toolkit/pipeline
├── Jupyter notebook workflows
├── Automated CI/CD validation
├── Batch processing jobs
└── Any combination with consistent standards
```

---

### 5. FlowRoute Execution Engine (Broker & Routing Layer)

**Purpose**: Translates approved trading intents into executed orders with full auditability.

**Core Functions**:
| Function | Description |
|----------|-------------|
| Order Translation | Convert RiskGuard-approved intents to broker-compatible orders |
| Smart Routing | Select optimal execution venue and order type |
| Retry Handling | Manage failures, timeouts, and partial fills |
| State Reconciliation | Sync internal state with broker positions |
| Event Logging | Complete audit trail of all order lifecycle events |
| Post-Trade Analysis | Execution quality measurement |

**Order Lifecycle**:
```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          ORDER LIFECYCLE (FlowRoute)                          │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  Intent         Validated        Submitted       Acknowledged     Filled     │
│  ┌─────┐       ┌─────────┐      ┌─────────┐     ┌────────────┐  ┌───────┐  │
│  │From │──────▶│RiskGuard│─────▶│ Broker  │────▶│   Broker   │─▶│Execute│  │
│  │Risk │       │ Passed  │      │   API   │     │   Confirm  │  │  Fill │  │
│  │Guard│       └─────────┘      └─────────┘     └────────────┘  └───────┘  │
│  └─────┘            │                │                │             │        │
│                     │                │                │             │        │
│                     ▼                ▼                ▼             ▼        │
│              ┌──────────────────────────────────────────────────────────┐   │
│              │                    EVENT LOG                              │   │
│              │  order_created → validated → submitted → acked → filled   │   │
│              └──────────────────────────────────────────────────────────┘   │
│                                                                               │
│  Exception Handling:                                                          │
│  ├── Timeout → Retry with backoff (max 3 attempts)                          │
│  ├── Rejection → Log reason, notify, do not retry                           │
│  ├── Partial Fill → Update state, handle remainder per strategy             │
│  └── Connection Lost → Halt new orders, reconcile on reconnect              │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Order and Event Types**:
```python
class OrderStatus(Enum):
    """Order lifecycle states."""
    CREATED = "created"
    VALIDATED = "validated"
    PENDING_SUBMIT = "pending_submit"
    SUBMITTED = "submitted"
    ACKNOWLEDGED = "acknowledged"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    ERROR = "error"


@dataclass
class OrderIntent:
    """Trading intent from RiskGuard."""
    intent_id: str
    symbol: str
    side: Literal['buy', 'sell']
    quantity: int
    order_type: Literal['market', 'limit', 'stop', 'stop_limit']
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: Literal['day', 'gtc', 'ioc', 'fok'] = 'day'

    # Source tracking
    signal_id: str
    strategy_id: str

    # Risk parameters (from RiskGuard)
    max_slippage_pct: float = 0.01
    max_fill_time_seconds: int = 60


@dataclass
class ExecutionEvent:
    """Audit event for order lifecycle."""
    event_id: str
    order_id: str
    event_type: str
    timestamp: datetime

    # Event details
    status_before: Optional[OrderStatus]
    status_after: OrderStatus
    details: dict

    # Error handling
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    retry_count: int = 0

    # External references
    broker_order_id: Optional[str] = None
    broker_response: Optional[dict] = None


@dataclass
class Fill:
    """Execution fill record."""
    fill_id: str
    order_id: str
    symbol: str
    side: str

    # Fill details
    quantity: int
    price: float
    commission: float

    # Timing
    timestamp: datetime
    latency_ms: float

    # Quality metrics
    slippage_bps: float  # vs signal price
    vs_arrival_bps: float  # vs price at order creation
    vs_vwap_bps: float  # vs VWAP during execution window
```

**FlowRoute Interface**:
```python
class FlowRouteEngine(Protocol):
    """Protocol defining FlowRoute interface."""

    async def submit_order(
        self,
        intent: OrderIntent
    ) -> tuple[str, OrderStatus]:
        """
        Submit order to broker.

        Returns:
            - order_id: Internal order identifier
            - status: Initial order status
        """
        ...

    async def cancel_order(
        self,
        order_id: str,
        reason: str
    ) -> bool:
        """Cancel pending order."""
        ...

    async def get_order_status(
        self,
        order_id: str
    ) -> OrderStatus:
        """Get current order status."""
        ...

    async def reconcile_positions(
        self
    ) -> ReconciliationReport:
        """
        Reconcile internal state with broker positions.

        Should be called:
        - At market open
        - After any connectivity issues
        - At end of day
        """
        ...

    def get_execution_quality(
        self,
        order_id: str
    ) -> ExecutionQualityReport:
        """Analyze execution quality for an order."""
        ...

    async def emergency_close_all(
        self,
        reason: str
    ) -> list[Fill]:
        """Emergency close all positions (kill switch action)."""
        ...
```

**Broker Adapter Pattern**:
```python
class BrokerAdapter(Protocol):
    """Abstract broker connection interface."""

    async def connect(self) -> bool:
        """Establish connection to broker."""
        ...

    async def disconnect(self) -> None:
        """Close broker connection."""
        ...

    async def submit_order(self, order: BrokerOrder) -> BrokerResponse:
        """Submit order via broker API."""
        ...

    async def cancel_order(self, broker_order_id: str) -> BrokerResponse:
        """Cancel order via broker API."""
        ...

    async def get_positions(self) -> list[BrokerPosition]:
        """Get current positions from broker."""
        ...

    async def get_account(self) -> BrokerAccount:
        """Get account information."""
        ...

    def is_connected(self) -> bool:
        """Check connection status."""
        ...


# Concrete implementations
class SchwabAdapter(BrokerAdapter):
    """Charles Schwab API adapter."""
    ...

class IBKRAdapter(BrokerAdapter):
    """Interactive Brokers API adapter."""
    ...

class AlpacaAdapter(BrokerAdapter):
    """Alpaca API adapter."""
    ...
```

**Implementation Flexibility**:
```
FlowRoute may be realized as:
├── Execution adapters per broker
├── Job handlers for async execution
├── Webhook receivers for fill notifications
├── Integration points Claude Code can supervise
└── Any combination with complete audit logging
```

---

## Inter-Engine Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            DATA FLOW PROTOCOL                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────┐                                                                    │
│  │ Market  │─── Raw Data ───┐                                                   │
│  │  Data   │                │                                                   │
│  └─────────┘                ▼                                                   │
│                      ┌─────────────┐                                            │
│                      │ SignalCore  │                                            │
│                      │  ML Engine  │                                            │
│                      └──────┬──────┘                                            │
│                             │                                                   │
│                    Signal (probabilities,                                       │
│                    scores, expected returns)                                    │
│                             │                                                   │
│                             ▼                                                   │
│  ┌─────────┐        ┌─────────────┐                                            │
│  │ Cortex  │◀ ─ ─ ─▶│  RiskGuard  │                                            │
│  │(review) │        │ Rule Engine │                                            │
│  └─────────┘        └──────┬──────┘                                            │
│                             │                                                   │
│               OrderIntent (approved,                                            │
│                sized, with risk params)                                         │
│                             │                                                   │
│                             ▼                                                   │
│                      ┌─────────────┐        ┌─────────────┐                    │
│                      │  FlowRoute  │───────▶│   Broker    │                    │
│                      │  Execution  │◀───────│    API      │                    │
│                      └──────┬──────┘        └─────────────┘                    │
│                             │                                                   │
│                     Fills, Events,                                              │
│                   Execution Reports                                             │
│                             │                                                   │
│                             ▼                                                   │
│                      ┌─────────────┐                                            │
│                      │ ProofBench  │                                            │
│                      │ (analysis)  │                                            │
│                      └─────────────┘                                            │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Message Types

```python
# Inter-engine message protocol
@dataclass
class EngineMessage:
    """Standard message format between engines."""
    message_id: str
    source_engine: Literal['cortex', 'signalcore', 'riskguard', 'flowroute', 'proofbench']
    target_engine: Literal['cortex', 'signalcore', 'riskguard', 'flowroute', 'proofbench']
    message_type: str
    timestamp: datetime
    payload: dict

    # Tracing
    correlation_id: str  # Links related messages
    causation_id: Optional[str]  # ID of message that triggered this one

    # Quality
    schema_version: str
    checksum: str


# Example message flows
MESSAGE_FLOWS = {
    "signal_to_risk": {
        "source": "signalcore",
        "target": "riskguard",
        "type": "signal_evaluation_request",
        "payload_schema": Signal
    },
    "risk_to_execution": {
        "source": "riskguard",
        "target": "flowroute",
        "type": "order_intent",
        "payload_schema": OrderIntent
    },
    "execution_to_analysis": {
        "source": "flowroute",
        "target": "proofbench",
        "type": "execution_report",
        "payload_schema": ExecutionEvent
    },
    "cortex_to_signalcore": {
        "source": "cortex",
        "target": "signalcore",
        "type": "parameter_update",
        "payload_schema": ParameterProposal
    }
}
```

---

## Auditability Requirements

All engines must maintain complete audit trails:

```python
@dataclass
class AuditRecord:
    """Standard audit record format."""
    record_id: str
    timestamp: datetime
    engine: str

    # What happened
    action: str
    inputs: dict
    outputs: dict

    # Why it happened
    triggering_event: str
    decision_rationale: str

    # Who/what was involved
    actor: str  # Engine, model, or human
    affected_entities: list[str]

    # Result
    success: bool
    error: Optional[str]

    # Immutability
    checksum: str
    previous_record_id: Optional[str]  # Chain linking


AUDIT_REQUIREMENTS = {
    "retention_days": 2555,  # 7 years for financial records
    "immutability": True,
    "searchable": True,
    "exportable": ["json", "csv", "parquet"],
    "real_time_access": True
}
```

---

## Implementation Notes

### Deployment Flexibility

The five-engine architecture is **logical**, not physical. Valid implementations include:

| Pattern | Description | When to Use |
|---------|-------------|-------------|
| Monolith | All engines in single process | Development, small scale |
| Modular Monolith | Engines as modules with clear boundaries | Early production |
| Microservices | Engines as separate services | Large scale, team separation |
| Hybrid | Some engines combined, others separate | Pragmatic production |
| Serverless | Engines as functions/lambdas | Variable load patterns |

### Claude Code Integration

```
SignalCore Trading System integration with Claude Code:

├── MCP Servers
│   ├── signalcore-mcp: Model serving and signal generation
│   ├── riskguard-mcp: Rule evaluation and limit checking
│   └── flowroute-mcp: Order execution and broker integration
│
├── Tools/Skills
│   ├── Research tools (Cortex functions)
│   ├── Backtest tools (ProofBench functions)
│   └── Monitoring tools (Cross-engine)
│
├── Hooks
│   ├── Pre-trade validation hooks
│   ├── Post-execution logging hooks
│   └── Kill switch trigger hooks
│
└── Configuration
    ├── Risk rules (YAML/JSON, Claude Code inspectable)
    ├── Strategy specs (YAML, Claude Code modifiable)
    └── Engine parameters (versioned, auditable)
```

---

## Summary

The SignalCore Trading System provides a robust, auditable framework for automated trading through five specialized engines:

| Engine | Layer | Primary Responsibility |
|--------|-------|----------------------|
| **Cortex** | LLM | Intelligence, research, strategy design |
| **SignalCore** | ML | Quantitative signal generation |
| **RiskGuard** | Rules | Deterministic risk constraints |
| **FlowRoute** | Execution | Broker integration, order management |
| **ProofBench** | Validation | Training, testing, performance analysis |

**Key Principles**:
1. **Separation of concerns** with clear interfaces
2. **Auditability** at every decision point
3. **Flexibility** in implementation while preserving function
4. **Safety** through layered constraints
5. **Transparency** for human oversight and review
