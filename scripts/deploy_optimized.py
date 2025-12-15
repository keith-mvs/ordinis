"""
One-command deployment script - sets up everything for live trading.

Usage:
    python scripts/deploy_optimized.py --phase 1 --capital 1000 --paper
    python scripts/deploy_optimized.py --phase 2 --capital 5000 --paper
    python scripts/deploy_optimized.py --phase 3 --capital 100000 --live
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys


def create_deployment_checklist(phase: int, capital: int, is_paper: bool):
    """Create deployment checklist for given phase."""

    checklist = {
        "phase": phase,
        "capital": capital,
        "trading_mode": "paper" if is_paper else "live",
        "deployment_date": datetime.now().isoformat(),
        "checklist": [],
    }

    # Phase-specific checklists
    if phase == 1:
        checklist["checklist"] = [
            {"item": "Data feed connectivity", "status": "pending", "critical": True},
            {"item": "Signal generation working", "status": "pending", "critical": True},
            {"item": "First trade executed", "status": "pending", "critical": True},
            {"item": "Daily P&L < -2% limit breached", "status": "ok", "critical": False},
            {"item": "Sharpe ratio >= 1.0", "status": "pending", "critical": True},
            {"item": "Max drawdown < 10%", "status": "pending", "critical": True},
            {"item": "5+ trades executed", "status": "pending", "critical": True},
            {"item": "No data quality issues", "status": "pending", "critical": False},
            {"item": "Monitoring dashboard active", "status": "pending", "critical": True},
        ]
    elif phase == 2:
        checklist["checklist"] = [
            {
                "item": "Phase 1 complete with passing metrics",
                "status": "pending",
                "critical": True,
            },
            {"item": "Scale to $5,000", "status": "pending", "critical": True},
            {"item": "Ensemble voting working", "status": "pending", "critical": True},
            {"item": "All sectors represented", "status": "pending", "critical": False},
            {"item": "20+ trades executed", "status": "pending", "critical": True},
            {"item": "Sharpe ratio maintained", "status": "pending", "critical": True},
            {"item": "Slippage < 5 bps on average", "status": "pending", "critical": False},
            {"item": "Commission tracking accurate", "status": "pending", "critical": True},
        ]
    elif phase == 3:
        checklist["checklist"] = [
            {
                "item": "Phase 2 complete with metrics maintained",
                "status": "pending",
                "critical": True,
            },
            {"item": "Risk controls fully tested", "status": "pending", "critical": True},
            {"item": "Circuit breakers operational", "status": "pending", "critical": True},
            {"item": "Daily loss limits enforced", "status": "pending", "critical": True},
            {"item": "Compliance audit complete", "status": "pending", "critical": True},
            {"item": "Scale to $100,000", "status": "pending", "critical": True},
            {"item": "Production monitoring active", "status": "pending", "critical": True},
            {"item": "Monthly retraining process defined", "status": "pending", "critical": True},
            {"item": "Alert thresholds configured", "status": "pending", "critical": True},
        ]

    return checklist


def setup_directories():
    """Set up required directory structure."""

    directories = [
        "config",
        "config/brokers",
        "config/data_providers",
        "logs",
        "logs/trading",
        "logs/signals",
        "logs/errors",
        "data",
        "data/cache",
        "reports",
        "reports/daily",
        "reports/weekly",
        "reports/monthly",
        "detailed_backtest_results",
        "deployment",
    ]

    for directory in directories:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

    return directories


def create_deployment_config(phase: int, capital: int, is_paper: bool):
    """Create phase-specific deployment configuration."""

    config = {
        "phase": phase,
        "capital": capital,
        "mode": "paper_trading" if is_paper else "live_trading",
        "created_at": datetime.now().isoformat(),
        "phase_config": {
            "phase_1": {
                "capital": 1000,
                "max_position_size": 100,
                "position_size_multiplier": 1.0,
                "risk_level": "conservative",
                "monitoring_frequency": "real-time",
                "review_frequency": "daily",
                "scaling_condition": "sharpe_ratio >= 1.0 and max_drawdown <= 10%",
                "next_phase_date": "T+7",
            },
            "phase_2": {
                "capital": 5000,
                "max_position_size": 500,
                "position_size_multiplier": 1.5,
                "risk_level": "moderate",
                "monitoring_frequency": "every 30 minutes",
                "review_frequency": "daily",
                "scaling_condition": "sharpe_ratio maintained and consistent returns",
                "next_phase_date": "T+14",
            },
            "phase_3": {
                "capital": 100000,
                "max_position_size": 10000,
                "position_size_multiplier": 2.0,
                "risk_level": "full_deployment",
                "monitoring_frequency": "hourly",
                "review_frequency": "daily with weekly deep dive",
                "scaling_condition": "all metrics maintained at scale",
                "next_phase_date": "ongoing",
            },
        },
        "ensemble_weights": {
            "FundamentalModel": 0.20,
            "IchimokuModel": 0.22,
            "VolumeProfileModel": 0.20,
            "AlgorithmicModel": 0.18,
            "SentimentModel": 0.12,
            "ChartPatternModel": 0.08,
        },
        "risk_parameters": {
            "daily_loss_limit_pct": -0.02,
            "max_drawdown_limit_pct": -0.20,
            "stop_loss_pct": 0.08,
            "take_profit_pct": 0.15,
            "max_correlation": 0.7,
        },
        "data_providers": {
            "primary": "alpha_vantage",
            "secondary": "polygon",
            "fallback": "yfinance",
            "update_frequency_seconds": 300,
        },
        "monitoring": {
            "log_level": "INFO",
            "enable_daily_reports": True,
            "enable_weekly_reports": True,
            "alert_on_critical_events": True,
            "dashboard_refresh_frequency_seconds": 60,
        },
        "retraining": {
            "enabled": True,
            "frequency_days": 30,
            "minimum_trades_required": 50,
            "ic_recalculation_enabled": True,
        },
    }

    return config[f"phase_{phase}"]


def generate_deployment_scripts(phase: int):
    """Generate phase-specific deployment scripts."""

    scripts = {}

    if phase == 1:
        scripts["start"] = """
#!/bin/bash
echo "Starting Phase 1 Paper Trading..."
python src/ordinis/deployment/paper_trader.py \\
  --config config/production_optimized_v1.json \\
  --capital 1000 \\
  --phase 1 \\
  --paper_trading
        """
        scripts["monitor"] = """
#!/bin/bash
echo "Monitoring Phase 1 Performance..."
python src/ordinis/deployment/monitor_trading.py \\
  --phase 1 \\
  --frequency 60 \\
  --report_daily
        """
        scripts["check_readiness"] = """
#!/bin/bash
echo "Checking Phase 1 Readiness..."
python scripts/validation_test.py \\
  --check_phase_1_metrics \\
  --alert_on_issues
        """

    elif phase == 2:
        scripts["start"] = """
#!/bin/bash
echo "Starting Phase 2 Scaled Paper Trading..."
python src/ordinis/deployment/paper_trader.py \\
  --config config/production_optimized_v1.json \\
  --capital 5000 \\
  --phase 2 \\
  --paper_trading
        """
        scripts["scale_positions"] = """
#!/bin/bash
echo "Scaling Position Sizes..."
python src/ordinis/deployment/scale_positions.py \\
  --scale_multiplier 1.5 \\
  --preserve_exposure
        """
        scripts["ensemble_validation"] = """
#!/bin/bash
echo "Validating Ensemble Strategy..."
python src/ordinis/deployment/validate_ensemble.py \\
  --all_strategies \\
  --compare_vs_phase_1
        """

    elif phase == 3:
        scripts["start"] = """
#!/bin/bash
echo "Starting Phase 3 FULL DEPLOYMENT..."
python src/ordinis/deployment/live_trader.py \\
  --config config/production_optimized_v1.json \\
  --capital 100000 \\
  --phase 3 \\
  --enable_all_risk_controls
        """
        scripts["risk_controls"] = """
#!/bin/bash
echo "Activating All Risk Controls..."
python src/ordinis/deployment/activate_risk_controls.py \\
  --daily_loss_limit -2000 \\
  --max_drawdown_limit -20000 \\
  --activate_circuit_breakers
        """
        scripts["compliance"] = """
#!/bin/bash
echo "Running Compliance Audit..."
python src/ordinis/deployment/compliance_audit.py \\
  --full_audit \\
  --generate_report
        """

    return scripts


def main():
    """Main deployment orchestration."""

    parser = argparse.ArgumentParser(description="Deploy optimized trading system")
    parser.add_argument(
        "--phase", type=int, choices=[1, 2, 3], required=True, help="Deployment phase"
    )
    parser.add_argument("--capital", type=int, required=True, help="Capital to deploy")
    parser.add_argument("--paper", action="store_true", help="Paper trading mode")
    parser.add_argument("--live", action="store_true", help="Live trading mode")
    parser.add_argument("--generate-only", action="store_true", help="Generate config only")

    args = parser.parse_args()

    if not args.paper and not args.live:
        print("ERROR: Must specify either --paper or --live")
        sys.exit(1)

    is_paper = args.paper
    phase = args.phase
    capital = args.capital

    print("=" * 80)
    print(f"DEPLOYMENT ORCHESTRATION - PHASE {phase}")
    print("=" * 80)
    print()

    # Setup directories
    print("Setting up directory structure...")
    directories = setup_directories()
    print(f"✓ Created {len(directories)} directories")
    print()

    # Generate checklist
    print(f"Creating Phase {phase} checklist...")
    checklist = create_deployment_checklist(phase, capital, is_paper)

    checklist_path = Path("deployment") / f"phase_{phase}_checklist.json"
    with open(checklist_path, "w") as f:
        json.dump(checklist, f, indent=2)

    print(f"✓ Checklist saved to {checklist_path}")
    print(f"  Critical items: {sum(1 for item in checklist['checklist'] if item['critical'])}")
    print()

    # Generate configuration
    print(f"Generating Phase {phase} configuration...")
    phase_config = create_deployment_config(phase, capital, is_paper)

    config_path = Path("config") / f"phase_{phase}_deployment.json"
    with open(config_path, "w") as f:
        json.dump(phase_config, f, indent=2)

    print(f"✓ Configuration saved to {config_path}")
    print()

    # Print deployment summary
    print("Deployment Summary:")
    print(f"  Phase: {phase}")
    print(f"  Capital: ${capital:,}")
    print(f"  Mode: {'PAPER TRADING' if is_paper else 'LIVE TRADING'}")
    print(f"  Status: {'READY' if not args.generate_only else 'CONFIG GENERATED'}")
    print()

    # Print next steps
    print("Next Steps:")
    if phase == 1:
        print("  1. Review config/phase_1_deployment.json")
        print("  2. Run: python scripts/deploy_optimized.py --phase 1 --capital 1000 --paper")
        print("  3. Monitor daily performance")
        print("  4. After 5+ trades with Sharpe >= 1.0, proceed to Phase 2")
    elif phase == 2:
        print("  1. Verify Phase 1 metrics were met")
        print("  2. Review config/phase_2_deployment.json")
        print("  3. Run: python scripts/deploy_optimized.py --phase 2 --capital 5000 --paper")
        print("  4. Monitor scaling impact on execution")
        print("  5. After 20+ trades with metrics maintained, proceed to Phase 3")
    elif phase == 3:
        print("  1. Verify Phase 2 metrics were met")
        print("  2. Review risk controls in config/phase_3_deployment.json")
        print("  3. Run compliance audit")
        print("  4. Run: python scripts/deploy_optimized.py --phase 3 --capital 100000 --live")
        print("  5. Monitor live trading with daily reviews")
    print()

    print("=" * 80)
    print("Deployment Ready!")
    print("=" * 80)


if __name__ == "__main__":
    main()
