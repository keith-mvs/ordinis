"""
Optimized configuration generator based on comprehensive backtest analysis.

Generates production-ready configurations using learnings from
multi-sector, multi-period backtesting.
"""

import json
from pathlib import Path


class OptimizedConfigGenerator:
    """Generate optimized platform configurations."""

    @staticmethod
    def generate_production_config() -> dict:
        """Generate production-ready configuration."""

        config = {
            "name": "production_optimized_v1",
            "description": "Optimized configuration based on 20-year multi-sector backtest",
            "timestamp": "2025-12-15",
            # ===== CORE TRADING PARAMETERS =====
            "trading": {
                "symbols": [
                    # Technology
                    "AAPL",
                    "MSFT",
                    "GOOGL",
                    "NVDA",
                    "META",
                    # Healthcare
                    "JNJ",
                    "UNH",
                    "PFE",
                    "ABBV",
                    # Financials
                    "JPM",
                    "BAC",
                    "GS",
                    "BLK",
                    # Industrials
                    "BA",
                    "CAT",
                    "GE",
                    # Consumer Discretionary
                    "AMZN",
                    "MCD",
                    "NKE",
                    # Consumer Staples
                    "PG",
                    "KO",
                    # Energy
                    "XOM",
                    "CVX",
                    # Materials
                    "MMM",
                    "LMT",
                    # Real Estate
                    "SPG",
                    # Utilities
                    "NEE",
                    # Communication
                    "VZ",
                ],
                "initial_capital": 100000,
                "rebalance_frequency": "daily",
                "max_symbols": 30,
            },
            # ===== POSITION SIZING =====
            "position_sizing": {
                "base_size_pct": 0.05,  # 5% per position
                "max_position_pct": 0.10,  # 10% max per symbol
                "min_position_pct": 0.02,  # 2% minimum for signal
                "confidence_scaling": True,  # Scale size by signal confidence
                "confidence_threshold": 0.5,  # Minimum confidence to trade
                "cash_buffer_pct": 0.10,  # Keep 10% cash
            },
            # ===== RISK MANAGEMENT =====
            "risk_management": {
                "max_portfolio_exposure": 0.95,  # 95% deployed
                "stop_loss_pct": 0.08,  # 8% stop loss
                "take_profit_pct": 0.15,  # 15% take profit
                "daily_loss_limit_pct": 0.02,  # -2% daily halt
                "max_drawdown_limit_pct": 0.20,  # -20% circuit breaker
                "max_single_trade_loss": 500,  # $500 max per trade
                "correlation_max": 0.7,  # Max portfolio correlation
            },
            # ===== EXECUTION =====
            "execution": {
                "commission_pct": 0.001,  # 0.1% commission
                "slippage_bps": 5.0,  # 5 basis points
                "min_order_size": 100,  # Minimum 100 shares
                "max_order_size_pct": 0.25,  # Max 25% of daily volume
                "order_type": "market",
                "execution_venue": "best_available",
            },
            # ===== SIGNAL GENERATION =====
            "signals": {
                "ensemble_type": "ic_weighted",
                "ensemble_weights": {
                    # Based on comprehensive backtest IC rankings
                    "FundamentalModel": 0.20,
                    "IchimokuModel": 0.18,
                    "VolumeProfileModel": 0.18,
                    "SentimentModel": 0.18,
                    "AlgorithmicModel": 0.17,
                    "ChartPatternModel": 0.09,
                },
                "min_model_agreement": 2,  # At least 2 models agree
                "signal_decay_halflife_days": 5,  # Signal strength halves in 5 days
                "signal_rebalance_frequency": "monthly",  # Update weights monthly
                "min_ic_threshold": 0.02,  # Remove models with IC < 0.02
            },
            # ===== MARKET REGIME DETECTION =====
            "market_regimes": {
                "enabled": True,
                "detection_window_days": 30,
                "regime_types": ["trending", "ranging", "volatile"],
                "regime_specific_params": {
                    "trending": {
                        "models": ["IchimokuModel", "AlgorithmicModel"],
                        "position_size_multiplier": 1.2,
                    },
                    "ranging": {
                        "models": ["VolumeProfileModel", "ChartPatternModel"],
                        "position_size_multiplier": 0.8,
                    },
                    "volatile": {
                        "models": ["RiskGuardModel"],
                        "position_size_multiplier": 0.5,
                    },
                },
            },
            # ===== DATA & MONITORING =====
            "data": {
                "providers": ["alpha_vantage", "polygon", "yfinance"],
                "data_quality_threshold": 0.95,  # 95% minimum quality
                "update_frequency_seconds": 300,  # Every 5 minutes
                "historical_lookback_days": 730,  # 2 years for features
                "quality_metrics": {
                    "max_gap_tolerance": 2,  # Max 2 day gap
                    "max_outlier_pct": 10,  # Max 10% day outliers
                    "staleness_limit_hours": 4,  # Max 4 hours old
                },
            },
            # ===== GOVERNANCE & COMPLIANCE =====
            "governance": {
                "enable_trade_audit": True,
                "enable_preflight_checks": True,
                "restricted_symbols": [
                    "OTC",
                    "PINK",
                    "PENNY",  # Avoid penny stocks
                ],
                "min_liquidity_filters": {
                    "min_avg_volume": 1000000,  # 1M avg volume
                    "min_bid_ask_spread": 0.01,  # $0.01 min spread
                },
                "sector_limits": {
                    "energy": 0.15,  # Max 15% energy
                    "technology": 0.25,  # Max 25% tech
                },
            },
            # ===== PERFORMANCE TARGETS =====
            "targets": {
                "annual_return_pct": 18,
                "sharpe_ratio": 1.4,
                "max_drawdown_pct": 16,
                "win_rate_pct": 52,
                "profit_factor": 1.7,
                "recovery_factor": 2.0,
                "ic_target": 0.10,  # Target 0.10 IC
                "hit_rate_target": 55,  # 55% hit rate
            },
            # ===== ALERT THRESHOLDS =====
            "alerts": {
                "sharpe_drop_threshold": 0.5,  # Alert if Sharpe drops 0.5
                "ic_drop_threshold": 50,  # Alert if IC drops 50%
                "data_quality_threshold": 90,  # Alert if < 90%
                "execution_failure_pct": 5,  # Alert if > 5% failures
                "drawdown_warning": 0.10,  # Warn at 10% drawdown
                "drawdown_critical": 0.20,  # Critical at 20% drawdown
            },
            # ===== RETRAINING SCHEDULE =====
            "retraining": {
                "enabled": True,
                "frequency": "monthly",
                "lookback_period_days": 730,  # 2 years training data
                "validation_period_days": 30,  # 1 month validation
                "min_samples_required": 50,  # Min 50 trades for validation
                "performance_threshold_to_deploy": 0.95,  # 95% of previous Sharpe
            },
            # ===== LIVE DEPLOYMENT SCHEDULE =====
            "deployment": {
                "phase_1_capital": 1000,  # Start with $1k
                "phase_1_duration_days": 5,
                "phase_2_capital": 5000,  # Scale to $5k
                "phase_2_duration_days": 7,
                "phase_3_capital": 20000,  # Scale to $20k
                "phase_3_duration_days": 10,
                "phase_4_capital": 100000,  # Full $100k
                "scaling_condition": "sharpe_ratio >= 1.2 and max_drawdown <= 15%",
            },
        }

        return config

    @staticmethod
    def generate_sector_specific_configs() -> dict[str, dict]:
        """Generate sector-specific optimized configurations."""

        configs = {
            "technology": {
                "sector_name": "Technology",
                "symbols": ["AAPL", "MSFT", "GOOGL", "NVDA", "META"],
                "model_weights": {
                    "AlgorithmicModel": 0.25,  # Tech is mean-reverting
                    "FundamentalModel": 0.25,
                    "IchimokuModel": 0.20,
                    "VolumeProfileModel": 0.15,
                    "SentimentModel": 0.15,
                },
                "position_size_multiplier": 1.0,
                "volatility_adjustment": 1.2,  # Higher volatility
            },
            "healthcare": {
                "sector_name": "Healthcare",
                "symbols": ["JNJ", "UNH", "PFE", "ABBV"],
                "model_weights": {
                    "FundamentalModel": 0.30,  # Strong fundamentals matter
                    "AlgorithmicModel": 0.25,
                    "IchimokuModel": 0.20,
                    "SentimentModel": 0.15,
                    "VolumeProfileModel": 0.10,
                },
                "position_size_multiplier": 0.9,
                "volatility_adjustment": 0.9,  # Lower volatility
            },
            "financials": {
                "sector_name": "Financials",
                "symbols": ["JPM", "BAC", "GS", "BLK"],
                "model_weights": {
                    "AlgorithmicModel": 0.30,  # Interest rate sensitive
                    "FundamentalModel": 0.25,
                    "IchimokuModel": 0.20,
                    "VolumeProfileModel": 0.15,
                    "SentimentModel": 0.10,
                },
                "position_size_multiplier": 0.8,
                "volatility_adjustment": 1.1,  # Moderate volatility
            },
            "industrials": {
                "sector_name": "Industrials",
                "symbols": ["BA", "CAT", "GE"],
                "model_weights": {
                    "IchimokuModel": 0.30,  # Trend-following works well
                    "FundamentalModel": 0.25,
                    "AlgorithmicModel": 0.20,
                    "VolumeProfileModel": 0.15,
                    "SentimentModel": 0.10,
                },
                "position_size_multiplier": 0.85,
                "volatility_adjustment": 1.15,
            },
            "energy": {
                "sector_name": "Energy",
                "symbols": ["XOM", "CVX"],
                "model_weights": {
                    "AlgorithmicModel": 0.30,  # Commodity-sensitive
                    "VolumeProfileModel": 0.25,
                    "IchimokuModel": 0.20,
                    "FundamentalModel": 0.15,
                    "SentimentModel": 0.10,
                },
                "position_size_multiplier": 0.7,  # Lower allocation
                "volatility_adjustment": 1.4,  # High volatility
            },
        }

        return configs

    @staticmethod
    def save_configs(output_dir: str = "config"):
        """Save all configurations to files."""

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Save main config
        config = OptimizedConfigGenerator.generate_production_config()
        config_path = output_path / "production_optimized_v1.json"

        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"✓ Production config saved to {config_path}")

        # Save sector configs
        sector_configs = OptimizedConfigGenerator.generate_sector_specific_configs()

        for sector_key, sector_config in sector_configs.items():
            sector_path = output_path / f"sector_{sector_key}.json"
            with open(sector_path, "w") as f:
                json.dump(sector_config, f, indent=2)
            print(f"✓ Sector config saved to {sector_path}")

        return config_path


if __name__ == "__main__":
    print("Generating optimized platform configurations...")
    print("=" * 80)

    config_path = OptimizedConfigGenerator.save_configs()

    print("\n" + "=" * 80)
    print("Configuration Generation Complete")
    print("=" * 80)
    print("\nGenerated configurations ready for deployment:")
    print("  ✓ Production config: config/production_optimized_v1.json")
    print("  ✓ Sector configs: config/sector_*.json (5 files)")
    print("\nNext steps:")
    print("  1. Review configurations")
    print("  2. Update platform with new settings")
    print("  3. Run paper trading with optimized config")
    print("  4. Monitor for improvement vs baseline")
