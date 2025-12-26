#=====
metadata:
  title: AI Governance
  description: Governance framework for AI-embedded trading systems
  version: 1.0.1
  last_modified_date: 2025-12-15
  tags: ["governance", "policies", "standards"]

#=============================================================================
# PRINCIPLES
#=============================================================================
principles:
  transparency: true
  accuracy: true
  accountability: true
  privacy: true
  security: true
  reliability: true
  excellence: true
  quality: true
    
#==============================================================================
# SECURITY
#==============================================================================
      
api_security:
  require_api_keys: true
  ip_whitelisting: false  # Set to true for production

#==============================================================================
# RISK MANAGEMENT POLICIES 
#============================================================================   
  
network:
  allow_external_connections: true  # For market data
  require_tls_version: "1.3"
  validate_certificates: true

secrets_management:
  never_log_secrets: true
  encrypt_config_files: true
    
#==============================================================================
#OPERATIONAL POLICIES
#==============================================================================
    
  monitoring:
    system_health:
      enabled: true
      track_uptime: true
      target_uptime_pct: 99.9
      
    performance:
      track_latency: true
      track_throughput: true
      sla_p95_latency_ms: 500  # General system target; LLM calls may exceed this
      target_p95_latency_ms: 200  # Aggressive target for core trading operations (see README KPIs)

#==============================================================================
#SAFETY CONTROLS
#==============================================================================

safety:
  guardrails:
    enabled: true
    output_validation: true # Ensure outputs are within expected ranges
    content_filtering: false  # Not needed for trading system
    
  testing:
    test_edge_cases: true
    continuous_validation: true
    accurate_model_performance: true
    accurate_drift_detection: true
    accurate_performance_metrics: true
    historical_data: true

# ==============================================================================
# DOCUMENTATION REQUIREMENTS
# ==============================================================================
documentation:
  document_style_standards:
    markdown: true
    name: "google"
    url: "https://google.github.io/styleguide/docguide/style.html"
    code_blocks: true
    content:
      include_code: true
      include_images: true
      include_tables: true
      include_links: true
      include_code_blocks:
        enabled: true
        use_fenced_code_blocks: true
        declare_language: true
      include_code_comments: true
      include_code_docstrings: true
      include_code_annotations: true
      include_code_metadata: true
      include_code_metadata_fields:
        - "title"
        - "tags/keywords"
        - "references"
      use_standard_nomenclature: true
      use_consistent_formatting: true
      use_consistent_terminology: true
  
#=============================================================================
#FILENAME CONVENTIONS
#=============================================================================
    
filename:
  markdown_filename_standards:
    use_standard_volume_names: true
    use_standard_prefixes: true
    use_standard_suffixes: true
    use_uppercase: true
    use_underscores: true
    no_spaces: true
    no_symbols: true
  
  code_filename_standards:
    use_lowercase: true

#==============================================================================
#REVIEW & UPDATES
#==============================================================================
    
  metrics:
    track_compliance_rate: true
    track_violation_count: true
    track_override_frequency: true

#==============================================================================
#DESIGN SPECIFICATIONS
#==============================================================================
# It is not recommended to include detailed system design (such as architecture diagrams, component interactions, or implementation specifics) directly in this governance YAML file. Instead, reference external design documentation here for traceability and clarity.

design:
    reference_docs:
        - "docs/architecture/PRODUCTION_ARCHITECTURE.md"
        - "docs/architecture/LAYERED_SYSTEM_ARCHITECTURE.md"
        - "docs/POSITION_SIZING_LOGIC.md"
        - "README.md"

#==============================================================================
# OUTPERFORMING THE MARKET
#==============================================================================
# Performance thresholds and benchmarks that strategies must meet before
# production deployment. Governance preflight checks enforce these limits.

market_outperformance:
  enabled: true
  benchmark_index: "SPY"  # S&P 500 ETF as primary benchmark
  
  # Minimum acceptable performance thresholds (backtest & live)
  thresholds:
    sharpe_ratio:
      minimum: 1.5
      target: 2.0
      description: "Risk-adjusted return; (mean - risk_free) / std_dev"
      
    sortino_ratio:
      minimum: 2.0
      target: 3.0
      description: "Downside risk-adjusted return; penalizes only negative volatility"
      
    calmar_ratio:
      minimum: 2.0
      target: 3.0
      description: "CAGR / Max Drawdown"
      
    max_drawdown_pct:
      maximum: 20.0
      target: 15.0
      description: "Largest peak-to-trough loss; must not exceed this limit"
      
    profit_factor:
      minimum: 1.8
      target: 2.0
      description: "Gross profit / gross loss"
      
    win_rate_pct:
      minimum: 50.0
      target: 58.0
      description: "Winning trades / total trades"
      
    avg_win_loss_ratio:
      minimum: 1.2
      target: 1.5
      description: "Average win / average loss"
      
  # Alpha generation requirements
  alpha_generation:
    require_positive_alpha: true
    minimum_alpha_pct: 0.0
    target_alpha_pct: 5.0
    statistical_significance: 0.05  # p-value threshold
    description: "Excess return over benchmark; must be positive and significant"
    
  # Information ratio (alpha / tracking error)
  information_ratio:
    minimum: 0.5
    target: 1.0
    description: "(Strategy - Benchmark) / tracking_error"
    
  # Beta constraints for market-neutral strategies
  beta_constraints:
    market_neutral_max_beta: 0.3
    directional_max_beta: 1.5
    description: "Regression slope vs market; near 0 for market-neutral"
    
  # Risk metrics
  risk_metrics:
    var_95_max_pct: 2.0        # 1-day 95% VaR as % of equity
    cvar_95_max_pct: 3.0       # Expected shortfall beyond VaR
    max_leverage: 5.0          # Gross exposure / equity
    max_concentration: 0.15    # Herfindahl index (diversification)
    
  # Transaction cost awareness
  transaction_costs:
    max_slippage_bps: 5        # Basis points
    max_cost_ratio_pct: 10.0   # Transaction costs / gross profit
    max_annual_turnover_pct: 150.0
    
  # Pre-deployment acceptance criteria (all must pass)
  acceptance_criteria:
    require_all_thresholds: true
    minimum_backtest_periods: 3      # Number of benchmark packs
    governance_pass_rate_pct: 100.0  # No policy violations allowed
    simulation_uptime_pct: 99.9
    e2e_latency_p95_ms: 200
    
  # Shadow mode requirements before live trading
  shadow_mode:
    enabled: true
    minimum_duration_days: 7
    compare_vs_production: true
    auto_promote_on_pass: false  # Requires human approval

#==============================================================================
# VERIFIABLE PERFORMANCE REQUIREMENTS
#==============================================================================
# Formal, testable requirements for strategy performance validation.
# Each requirement has a unique ID, acceptance criteria, and verification method.

performance_requirements:

  # --- RISK-ADJUSTED RETURN REQUIREMENTS ---
  
  - id: PERF-RAR-001
    title: Sharpe Ratio Minimum Threshold
    category: risk_adjusted_return
    requirement: "Strategy Sharpe ratio SHALL be ≥ 1.5 over rolling 252-day periods."
    acceptance_criteria:
      - "sharpe_ratio >= 1.5"
      - "Measured using daily returns, annualized"
      - "Risk-free rate: 3-month T-bill yield"
    verification_method: "Automated: analytics.calculate_sharpe(returns, rf_rate)"
    test_command: "pytest tests/test_analytics/test_metrics.py::test_sharpe_threshold -v"
    governance_check: "governance.preflight() with operation='sharpe_validation'"
    severity: critical
    blocks_deployment: true

  - id: PERF-RAR-002
    title: Sortino Ratio Minimum Threshold
    category: risk_adjusted_return
    requirement: "Strategy Sortino ratio SHALL be ≥ 2.0 over rolling 252-day periods."
    acceptance_criteria:
      - "sortino_ratio >= 2.0"
      - "Downside deviation calculated using negative returns only"
      - "Target return: risk-free rate"
    verification_method: "Automated: analytics.calculate_sortino(returns, target)"
    test_command: "pytest tests/test_analytics/test_metrics.py::test_sortino_threshold -v"
    governance_check: "governance.preflight() with operation='sortino_validation'"
    severity: high
    blocks_deployment: true

  - id: PERF-RAR-003
    title: Calmar Ratio Minimum Threshold
    category: risk_adjusted_return
    requirement: "Strategy Calmar ratio SHALL be ≥ 2.0 (CAGR / Max Drawdown)."
    acceptance_criteria:
      - "calmar_ratio >= 2.0"
      - "CAGR annualized over backtest period"
      - "Max drawdown measured peak-to-trough"
    verification_method: "Automated: analytics.calculate_calmar(equity_curve)"
    test_command: "pytest tests/test_analytics/test_metrics.py::test_calmar_threshold -v"
    governance_check: "governance.preflight() with operation='calmar_validation'"
    severity: high
    blocks_deployment: true

  # --- DRAWDOWN REQUIREMENTS ---

  - id: PERF-DD-001
    title: Maximum Drawdown Limit
    category: drawdown
    requirement: "Strategy maximum drawdown SHALL NOT exceed 20% of peak equity."
    acceptance_criteria:
      - "max_drawdown_pct <= 20.0"
      - "Measured as (peak - trough) / peak × 100"
      - "Applies to both backtest and live trading"
    verification_method: "Automated: analytics.calculate_max_drawdown(equity_curve)"
    test_command: "pytest tests/test_analytics/test_metrics.py::test_max_drawdown_limit -v"
    governance_check: "governance.preflight() with operation='drawdown_check'"
    severity: critical
    blocks_deployment: true
    live_action: "Kill switch activation if breached"

  - id: PERF-DD-002
    title: Drawdown Recovery Time
    category: drawdown
    requirement: "Strategy SHALL recover from 10%+ drawdowns within 60 trading days (median)."
    acceptance_criteria:
      - "median_recovery_days <= 60 for drawdowns >= 10%"
      - "Measured across all historical drawdown events"
    verification_method: "Automated: analytics.calculate_recovery_periods(equity_curve)"
    test_command: "pytest tests/test_analytics/test_metrics.py::test_recovery_time -v"
    governance_check: "governance.preflight() with operation='recovery_analysis'"
    severity: medium
    blocks_deployment: false

  # --- PROFITABILITY REQUIREMENTS ---

  - id: PERF-PF-001
    title: Profit Factor Minimum
    category: profitability
    requirement: "Strategy profit factor SHALL be ≥ 1.8 (gross profit / gross loss)."
    acceptance_criteria:
      - "profit_factor >= 1.8"
      - "Calculated as sum(winning_trades) / abs(sum(losing_trades))"
    verification_method: "Automated: analytics.calculate_profit_factor(trades)"
    test_command: "pytest tests/test_analytics/test_metrics.py::test_profit_factor -v"
    governance_check: "governance.preflight() with operation='profit_factor_check'"
    severity: high
    blocks_deployment: true

  - id: PERF-PF-002
    title: Win Rate Minimum
    category: profitability
    requirement: "Strategy win rate SHALL be ≥ 50%."
    acceptance_criteria:
      - "win_rate_pct >= 50.0"
      - "Calculated as winning_trades / total_trades × 100"
      - "Minimum 100 trades for statistical validity"
    verification_method: "Automated: analytics.calculate_win_rate(trades)"
    test_command: "pytest tests/test_analytics/test_metrics.py::test_win_rate -v"
    governance_check: "governance.preflight() with operation='win_rate_check'"
    severity: high
    blocks_deployment: true

  - id: PERF-PF-003
    title: Average Win/Loss Ratio
    category: profitability
    requirement: "Strategy average win/loss ratio SHALL be ≥ 1.2."
    acceptance_criteria:
      - "avg_win_loss_ratio >= 1.2"
      - "Calculated as mean(winning_trade_pnl) / abs(mean(losing_trade_pnl))"
    verification_method: "Automated: analytics.calculate_win_loss_ratio(trades)"
    test_command: "pytest tests/test_analytics/test_metrics.py::test_win_loss_ratio -v"
    governance_check: "governance.preflight() with operation='win_loss_check'"
    severity: medium
    blocks_deployment: true

  # --- ALPHA GENERATION REQUIREMENTS ---

  - id: PERF-ALPHA-001
    title: Positive Alpha vs Benchmark
    category: alpha_generation
    requirement: "Strategy SHALL generate positive alpha versus SPY benchmark."
    acceptance_criteria:
      - "alpha > 0.0"
      - "Measured via CAPM regression: r_strategy = alpha + beta * r_benchmark + epsilon"
      - "Alpha must be annualized"
    verification_method: "Automated: analytics.calculate_alpha(strategy_returns, benchmark_returns)"
    test_command: "pytest tests/test_analytics/test_alpha.py::test_positive_alpha -v"
    governance_check: "governance.preflight() with operation='alpha_validation'"
    severity: critical
    blocks_deployment: true

  - id: PERF-ALPHA-002
    title: Alpha Statistical Significance
    category: alpha_generation
    requirement: "Strategy alpha SHALL be statistically significant (p-value < 0.05)."
    acceptance_criteria:
      - "alpha_p_value < 0.05"
      - "Determined via t-test on regression alpha coefficient"
      - "Minimum 252 daily observations required"
    verification_method: "Automated: analytics.test_alpha_significance(returns, benchmark)"
    test_command: "pytest tests/test_analytics/test_alpha.py::test_alpha_significance -v"
    governance_check: "governance.preflight() with operation='alpha_significance'"
    severity: high
    blocks_deployment: true

  - id: PERF-ALPHA-003
    title: Information Ratio Minimum
    category: alpha_generation
    requirement: "Strategy information ratio SHALL be ≥ 0.5."
    acceptance_criteria:
      - "information_ratio >= 0.5"
      - "Calculated as (strategy_return - benchmark_return) / tracking_error"
      - "Annualized tracking error"
    verification_method: "Automated: analytics.calculate_information_ratio(strategy, benchmark)"
    test_command: "pytest tests/test_analytics/test_alpha.py::test_information_ratio -v"
    governance_check: "governance.preflight() with operation='ir_validation'"
    severity: high
    blocks_deployment: true

  # --- RISK METRIC REQUIREMENTS ---

  - id: PERF-RISK-001
    title: Value at Risk (VaR) Limit
    category: risk_metrics
    requirement: "Strategy 1-day 95% VaR SHALL NOT exceed 2% of portfolio equity."
    acceptance_criteria:
      - "var_95_pct <= 2.0"
      - "Calculated using historical simulation (1000+ days)"
      - "Verified daily during live trading"
    verification_method: "Automated: risk.calculate_var(returns, confidence=0.95)"
    test_command: "pytest tests/test_risk/test_var.py::test_var_limit -v"
    governance_check: "governance.preflight() with operation='var_check'"
    severity: critical
    blocks_deployment: true
    live_action: "Position reduction if approaching limit"

  - id: PERF-RISK-002
    title: Conditional VaR (CVaR) Limit
    category: risk_metrics
    requirement: "Strategy 95% CVaR (Expected Shortfall) SHALL NOT exceed 3% of equity."
    acceptance_criteria:
      - "cvar_95_pct <= 3.0"
      - "Calculated as mean of losses beyond VaR"
    verification_method: "Automated: risk.calculate_cvar(returns, confidence=0.95)"
    test_command: "pytest tests/test_risk/test_var.py::test_cvar_limit -v"
    governance_check: "governance.preflight() with operation='cvar_check'"
    severity: critical
    blocks_deployment: true

  - id: PERF-RISK-003
    title: Maximum Leverage Limit
    category: risk_metrics
    requirement: "Strategy gross leverage SHALL NOT exceed 5.0x equity."
    acceptance_criteria:
      - "gross_leverage <= 5.0"
      - "Calculated as sum(abs(positions)) / equity"
      - "Checked before each trade execution"
    verification_method: "Automated: risk.calculate_leverage(positions, equity)"
    test_command: "pytest tests/test_risk/test_leverage.py::test_max_leverage -v"
    governance_check: "governance.preflight() with operation='leverage_check'"
    severity: critical
    blocks_deployment: true
    live_action: "Order rejection if would breach limit"

  - id: PERF-RISK-004
    title: Portfolio Concentration Limit
    category: risk_metrics
    requirement: "No single position SHALL exceed 15% of portfolio equity."
    acceptance_criteria:
      - "max_position_weight <= 0.15"
      - "Measured as position_value / total_equity"
    verification_method: "Automated: risk.check_concentration(positions, equity)"
    test_command: "pytest tests/test_risk/test_concentration.py::test_position_limit -v"
    governance_check: "governance.preflight() with operation='concentration_check'"
    severity: high
    blocks_deployment: true

  # --- BETA REQUIREMENTS ---

  - id: PERF-BETA-001
    title: Market-Neutral Beta Constraint
    category: beta_exposure
    requirement: "Market-neutral strategies SHALL maintain |beta| ≤ 0.3."
    acceptance_criteria:
      - "abs(portfolio_beta) <= 0.3"
      - "Measured via rolling 60-day regression vs SPY"
      - "Applies only to strategies tagged 'market_neutral'"
    verification_method: "Automated: analytics.calculate_beta(strategy, benchmark)"
    test_command: "pytest tests/test_analytics/test_beta.py::test_market_neutral_beta -v"
    governance_check: "governance.preflight() with operation='beta_check'"
    severity: high
    blocks_deployment: true
    strategy_tags: ["market_neutral"]

  - id: PERF-BETA-002
    title: Directional Strategy Beta Limit
    category: beta_exposure
    requirement: "Directional strategies SHALL maintain beta ≤ 1.5."
    acceptance_criteria:
      - "portfolio_beta <= 1.5"
      - "Prevents excessive market exposure"
    verification_method: "Automated: analytics.calculate_beta(strategy, benchmark)"
    test_command: "pytest tests/test_analytics/test_beta.py::test_directional_beta -v"
    governance_check: "governance.preflight() with operation='beta_check'"
    severity: medium
    blocks_deployment: false
    strategy_tags: ["directional", "long_only", "long_short"]

  # --- TRANSACTION COST REQUIREMENTS ---

  - id: PERF-TC-001
    title: Maximum Slippage
    category: transaction_costs
    requirement: "Average trade slippage SHALL NOT exceed 5 basis points."
    acceptance_criteria:
      - "avg_slippage_bps <= 5"
      - "Measured as (execution_price - signal_price) / signal_price × 10000"
      - "Tracked per trade, aggregated daily"
    verification_method: "Automated: analytics.calculate_slippage(trades)"
    test_command: "pytest tests/test_execution/test_slippage.py::test_slippage_limit -v"
    governance_check: "governance.preflight() with operation='slippage_check'"
    severity: medium
    blocks_deployment: false

  - id: PERF-TC-002
    title: Transaction Cost Ratio
    category: transaction_costs
    requirement: "Total transaction costs SHALL NOT exceed 10% of gross profit."
    acceptance_criteria:
      - "transaction_cost_ratio <= 0.10"
      - "Includes commissions, slippage, market impact"
      - "Calculated over rolling 30-day periods"
    verification_method: "Automated: analytics.calculate_cost_ratio(trades)"
    test_command: "pytest tests/test_execution/test_costs.py::test_cost_ratio -v"
    governance_check: "governance.preflight() with operation='cost_ratio_check'"
    severity: high
    blocks_deployment: true

  - id: PERF-TC-003
    title: Annual Turnover Limit
    category: transaction_costs
    requirement: "Strategy annual turnover SHALL NOT exceed 150%."
    acceptance_criteria:
      - "annual_turnover_pct <= 150.0"
      - "Calculated as sum(trade_values) / avg_portfolio_value × 100"
    verification_method: "Automated: analytics.calculate_turnover(trades, equity)"
    test_command: "pytest tests/test_execution/test_turnover.py::test_turnover_limit -v"
    governance_check: "governance.preflight() with operation='turnover_check'"
    severity: medium
    blocks_deployment: false

  # --- OPERATIONAL REQUIREMENTS ---

  - id: PERF-OPS-001
    title: End-to-End Latency
    category: operational
    requirement: "Signal-to-execution latency SHALL be ≤ 200ms at p95."
    acceptance_criteria:
      - "e2e_latency_p95_ms <= 200"
      - "Measured from signal generation to order acknowledgement"
      - "Excludes market data latency"
    verification_method: "Automated: telemetry.measure_latency(trace_id)"
    test_command: "pytest tests/test_performance/test_latency.py::test_e2e_latency -v"
    governance_check: "governance.preflight() with operation='latency_check'"
    severity: high
    blocks_deployment: true

  - id: PERF-OPS-002
    title: System Uptime
    category: operational
    requirement: "Trading system uptime SHALL be ≥ 99.9% during market hours."
    acceptance_criteria:
      - "uptime_pct >= 99.9"
      - "Measured during NYSE regular hours (9:30-16:00 ET)"
      - "Excludes scheduled maintenance windows"
    verification_method: "Automated: monitoring.calculate_uptime(start, end)"
    test_command: "pytest tests/test_operations/test_uptime.py::test_uptime_sla -v"
    governance_check: "governance.preflight() with operation='uptime_check'"
    severity: critical
    blocks_deployment: true

  - id: PERF-OPS-003
    title: Governance Pass Rate
    category: operational
    requirement: "Strategy SHALL have 100% governance preflight pass rate in shadow mode."
    acceptance_criteria:
      - "governance_pass_rate_pct == 100.0"
      - "No policy violations during 7-day shadow period"
      - "All preflight checks must pass"
    verification_method: "Automated: governance.get_pass_rate(strategy_id, period)"
    test_command: "pytest tests/test_governance/test_compliance.py::test_pass_rate -v"
    governance_check: "governance.preflight() with operation='compliance_check'"
    severity: critical
    blocks_deployment: true

  # --- SHADOW MODE REQUIREMENTS ---

  - id: PERF-SHADOW-001
    title: Shadow Mode Duration
    category: shadow_mode
    requirement: "New strategies SHALL run in shadow mode for minimum 7 calendar days."
    acceptance_criteria:
      - "shadow_duration_days >= 7"
      - "Clock starts from first signal generation"
      - "Resets if strategy code changes"
    verification_method: "Automated: deployment.get_shadow_duration(strategy_id)"
    test_command: "pytest tests/test_deployment/test_shadow.py::test_duration -v"
    governance_check: "governance.preflight() with operation='shadow_duration'"
    severity: critical
    blocks_deployment: true

  - id: PERF-SHADOW-002
    title: Shadow Mode Performance Comparison
    category: shadow_mode
    requirement: "Shadow mode signals SHALL be compared against production baseline."
    acceptance_criteria:
      - "Simulated PnL tracked for all shadow signals"
      - "Comparison metrics: Sharpe, max DD, win rate"
      - "Must meet or exceed production performance"
    verification_method: "Automated: shadow.compare_performance(strategy_id)"
    test_command: "pytest tests/test_deployment/test_shadow.py::test_comparison -v"
    governance_check: "governance.preflight() with operation='shadow_comparison'"
    severity: high
    blocks_deployment: true

  - id: PERF-SHADOW-003
    title: Human Approval for Promotion
    category: shadow_mode
    requirement: "Promotion from shadow to live SHALL require human approval."
    acceptance_criteria:
      - "auto_promote_on_pass == false"
      - "Approval recorded in audit log with approver_id"
      - "Minimum reviewer level: engineering_lead"
    verification_method: "Manual: Review approval record in audit log"
    test_command: "N/A - Manual verification required"
    governance_check: "governance.preflight() with operation='promotion_approval'"
    severity: critical
    blocks_deployment: true

  # --- BACKTEST VALIDATION REQUIREMENTS ---

  - id: PERF-BT-001
    title: Minimum Backtest Periods
    category: backtest_validation
    requirement: "Strategy SHALL be backtested across minimum 3 distinct market regimes."
    acceptance_criteria:
      - "backtest_periods >= 3"
      - "Must include: bull market, bear market, sideways/volatile"
      - "Minimum 252 trading days per period"
    verification_method: "Automated: backtest.validate_periods(results)"
    test_command: "pytest tests/test_backtest/test_validation.py::test_periods -v"
    governance_check: "governance.preflight() with operation='backtest_validation'"
    severity: high
    blocks_deployment: true

  - id: PERF-BT-002
    title: Out-of-Sample Validation
    category: backtest_validation
    requirement: "Strategy SHALL demonstrate consistent performance on out-of-sample data."
    acceptance_criteria:
      - "OOS Sharpe ratio >= 80% of in-sample Sharpe"
      - "OOS max drawdown <= 120% of in-sample max drawdown"
      - "Minimum 20% of data reserved for OOS testing"
    verification_method: "Automated: backtest.validate_oos(results)"
    test_command: "pytest tests/test_backtest/test_oos.py::test_consistency -v"
    governance_check: "governance.preflight() with operation='oos_validation'"
    severity: high
    blocks_deployment: true

  - id: PERF-BT-003
    title: Walk-Forward Validation
    category: backtest_validation
    requirement: "Strategy SHALL pass walk-forward optimization with stable parameters."
    acceptance_criteria:
      - "Parameter stability: <20% variance across windows"
      - "Minimum 5 walk-forward windows"
      - "No parameter overfitting detected"
    verification_method: "Automated: backtest.walk_forward_analysis(strategy)"
    test_command: "pytest tests/test_backtest/test_walkforward.py::test_stability -v"
    governance_check: "governance.preflight() with operation='walkforward_validation'"
    severity: medium
    blocks_deployment: false

python_standards:
  - id: GOV-PY-001
    title: Temporal Data Standardization
    standard: ISO 8601
    requirement: "All Python services must use ISO 8601 strings for date/time exchange."
    implementation_guidance: "Utilize datetime.fromisoformat() and datetime.isoformat() from the Python standard library (available since Python 3.7, with extended format support in Python 3.11+). Avoid custom strftime formatting for interchange."
    enforcement: "Code reviews must verify ISO 8601 compliance for all datetime serialization."
    severity: high

  - id: GOV-PY-002
    title: Geographic and Monetary Data Integrity
    standard: "ISO 3166, ISO 4217, ISO 639"
    requirement: "Country codes, currency symbols, and language identifiers must be validated against ISO standards."
    implementation_guidance: "Use the [pycountry](https://pypi.org/project/pycountry/) library to resolve and validate codes. Do not hardcode custom mapping dictionaries for these values."
    enforcement: "Validation functions must be tested against official ISO code sets during CI/CD."
    severity: medium

  - id: GOV-PY-003
    title: Code Quality and Maintainability
    standard: "PEP 8"
    requirement: "All Python source code must adhere to the PEP 8 style guide."
    implementation_guidance: "Configure [Ruff](https://github.com/astral-sh/ruff) as the primary linter with [Flake8](https://flake8.pycqa.org/) as fallback for legacy systems. Maximum line length: 100 characters. Use consistent import sorting via isort or Ruff's import rules."
    enforcement: "Automated linting must pass during the CI/CD pipeline. No merge without clean lint status."
    severity: medium

  - id: GOV-PY-004
    title: Secure Coding Practices
    standard: "ISO/IEC 24772-4"
    requirement: "Applications must be scanned for Python-specific language vulnerabilities (e.g., insecure deserialization, shell injection, SQL injection, path traversal)."
    implementation_guidance: "Configure [Bandit](https://github.com/PyCQA/bandit) with high and medium severity checks enabled. Address all critical findings before production deployment."
    enforcement: "Mandatory Bandit scans on all pull requests. Critical findings block merge."
    severity: critical

  - id: GOV-PY-005
    title: Supply Chain Security
    standard: "ISO/IEC 27001, NIST SP 800-218"
    requirement: "All third-party Python dependencies must be pinned with cryptographic hashes and scanned for known vulnerabilities."
    implementation_guidance: "Use requirements.txt with --hash or pyproject.toml with lock files (poetry.lock, uv.lock, or requirements.lock). Scan using [pip-audit](https://pypi.org/project/pip-audit/) or [Safety](https://pypi.org/project/safety/). Implement automated dependency updates via Dependabot or Renovate."
    enforcement: "CI/CD pipeline must fail on high or critical severity vulnerabilities. Monthly dependency update review required."
    severity: critical

  - id: GOV-PY-006
    title: Type Safety and Static Analysis
    standard: "PEP 484, PEP 604"
    requirement: "All public interfaces and complex functions must include type hints. Type checking must be enforced via static analysis."
    implementation_guidance: "Use [mypy](https://mypy-lang.org/) in strict mode or [Pyright](https://github.com/microsoft/pyright) for type checking. Minimum coverage: 80% of functions with type annotations."
    enforcement: "Type checking must pass in CI/CD. New code without type hints requires documented justification."
    severity: high

  - id: GOV-PY-007
    title: Testing Standards
    standard: "ISO/IEC 29119-4"
    requirement: "All Python services must maintain minimum test coverage of 80% for critical paths and 70% overall."
    implementation_guidance: "Use [pytest](https://pytest.org/) as the standard testing framework. Organize tests following the Arrange-Act-Assert pattern. Include unit, integration, and end-to-end tests as appropriate. Use [pytest-cov](https://pypi.org/project/pytest-cov/) for coverage reporting."
    enforcement: "CI/CD pipeline must measure and enforce coverage thresholds. Pull requests reducing coverage below thresholds require approval exception."
    severity: high

  - id: GOV-PY-008
    title: Documentation Standards
    standard: "PEP 257"
    requirement: "All modules, classes, and public functions must include docstrings following PEP 257 conventions."
    implementation_guidance: "Use Google-style or NumPy-style docstrings consistently across the codebase. Generate API documentation using [Sphinx](https://www.sphinx-doc.org/) with autodoc extension. Include type information in docstrings for clarity."
    enforcement: "Documentation coverage must be measured via [interrogate](https://pypi.org/project/interrogate/) or similar tools. Minimum 90% docstring coverage required."
    severity: medium

  - id: GOV-PY-009
    title: Dependency Management and Reproducibility
    standard: "PEP 621, PEP 665"
    requirement: "All Python projects must use isolated virtual environments and maintain reproducible dependency specifications."
    implementation_guidance: "Create virtual environments using venv, virtualenv, or [uv](https://github.com/astral-sh/uv). Specify dependencies in pyproject.toml with lock files for production deployments. Use Poetry, pip-tools, or uv for deterministic builds."
    enforcement: "Production deployments must use lock files. Development environments must be documented in README with setup instructions."
    severity: high

  - id: GOV-PY-010
    title: Logging and Observability
    standard: "ISO/IEC 25010"
    requirement: "Applications must implement structured logging with appropriate log levels and PII protection."
    implementation_guidance: "Use Python's standard logging module with structured formatters (JSON preferred). Configure appropriate log levels: DEBUG for development, INFO for production operations, WARNING for concerning events, ERROR for failures, CRITICAL for system-threatening conditions. Never log sensitive data (passwords, tokens, PII) without masking."
    enforcement: "Code reviews must verify logging practices. Automated scans for common PII patterns in log statements required."
    severity: high

  - id: GOV-PY-011
    title: Error Handling and Exception Management
    standard: "ISO/IEC 24765"
    requirement: "Applications must implement consistent error handling with appropriate exception hierarchies and recovery mechanisms."
    implementation_guidance: "Define custom exception classes inheriting from appropriate base exceptions. Avoid bare except clauses. Use context managers for resource management. Log exceptions with full context before re-raising or handling. Implement graceful degradation for non-critical failures."
    enforcement: "Code reviews must verify exception handling patterns. Bandit scans will flag bare except usage."
    severity: medium

ai_llm_standards:
  - id: GOV-AI-001
    title: LLM Output Validation and Content Safety
    standard: "AI Safety Best Practices, NIST AI RMF"
    requirement: "All LLM outputs must be validated for safety, relevance, and correctness before use in trading decisions or code generation."
    implementation_guidance: "Implement multi-layer validation: (1) Content safety filtering using Nemoguard (nvidia/llama-3.1-nemoguard-8b-content-safety), (2) Output schema validation for structured responses, (3) Semantic relevance checks against input context, (4) Human-in-the-loop review for code generation affecting trading logic. Log all validation failures with trace_id for audit."
    enforcement: "All LLM calls through Helix must pass content safety checks. Code generation outputs require approval before merge. Zero tolerance for content safety violations."
    severity: critical

  - id: GOV-AI-002
    title: LLM Rate Limiting and Cost Controls
    standard: "Operational Cost Management"
    requirement: "LLM API calls must be rate-limited and cost-monitored to prevent runaway usage and ensure system availability."
    implementation_guidance: "Implement per-component rate limits: Cortex (code analysis) - 100 req/hour, Helix (general) - 500 req/hour, CodeGenService - 50 req/hour. Use exponential backoff for retries. Set monthly cost alerts at $500, $1000, $2000 thresholds. Cache responses where appropriate (TTL: 1 hour for static queries). Track token usage per trace_id."
    enforcement: "Rate limiters must be configured in Helix engine. Cost monitoring dashboards required in production. Exceeded rate limits logged as WARNING; cost threshold breaches escalated to engineering lead."
    severity: high

  - id: GOV-AI-003
    title: Prompt Injection Protection and Input Sanitization
    standard: "OWASP LLM Top 10"
    requirement: "All user inputs and external data fed to LLMs must be sanitized to prevent prompt injection attacks."
    implementation_guidance: "Implement input validation: (1) Escape or remove special characters and control sequences, (2) Use structured prompts with clear role separation (system/user/assistant), (3) Validate input length (max 4000 chars for user prompts), (4) Strip executable code patterns from free-text inputs, (5) Use separate system prompts that cannot be overridden by user input. For Synapse RAG, sanitize retrieved documents before inclusion in context."
    enforcement: "All external inputs must pass sanitization before LLM submission. Automated tests for common injection patterns required. Failed injection attempts logged as SECURITY events with full context."
    severity: critical

  - id: GOV-AI-004
    title: Temporal Data Format Compliance for AI Tracing
    standard: "ISO 8601, GOV-PY-001"
    requirement: "All AI system traces, logs, and telemetry must use ISO 8601 datetime strings, not Unix epoch timestamps."
    implementation_guidance: "Update trace logging in artifacts/traces/ to serialize timestamps using datetime.isoformat(). Format: 'YYYY-MM-DDTHH:MM:SS.ffffffZ' (UTC). Ensure backward compatibility by accepting both formats during transition period (90 days). Update trace ingestion pipelines to normalize timestamps to ISO 8601."
    enforcement: "New trace logs must use ISO 8601 immediately. Legacy traces have 90-day conversion period. CI/CD linting checks for timestamp format compliance."
    severity: high

  - id: GOV-AI-005
    title: LLM Model Version Control and Governance
    standard: "Model Governance Best Practices"
    requirement: "All LLM models used in production must be versioned, documented, and approved through governance review."
    implementation_guidance: "Maintain model registry in configs/helix_models.yaml with version pinning. Document model selection rationale (cost, latency, quality). Require governance approval for: (1) New model providers, (2) Model version upgrades affecting trading logic, (3) Changes to default models. Log model_id and provider in all trace records. Test new models in shadow mode before production promotion."
    enforcement: "Model changes require pull request approval from engineering lead. Shadow mode testing mandatory for 7 days minimum. Trace logs must include model metadata for audit."
    severity: high

severity_definitions:
  critical:
    description: "Security vulnerabilities, data integrity risks, or compliance violations that could result in significant financial, legal, or reputational damage."
    remediation_timeline: "Immediate (within 24 hours for production, before merge for new code)"
    escalation: "Requires senior engineering and security team approval for any exceptions."
  
  high:
    description: "Significant quality, maintainability, or operational risks that could impact system reliability or development velocity."
    remediation_timeline: "Within 1 week for production systems, within sprint for new development"
    escalation: "Requires engineering lead approval for exceptions."
  
  medium:
    description: "Code quality or maintainability concerns that should be addressed but do not pose immediate operational risk."
    remediation_timeline: "Within 2 sprints or next major release"
    escalation: "Team lead discretion for exception handling."

compliance_notes:
  - "All standards apply to new development immediately upon adoption."
  - "Existing codebases have 90-day grace period for high/medium severity items, 30-day for critical."
  - "Automated tooling configuration files must be maintained in version control."
  - "Standard exceptions require documented justification and time-bound remediation plan."
  - "Annual review of standards required to align with emerging standards and tooling."