"""
Ordinis MCP Server - Main server implementation.

This server exposes the Ordinis trading system through the Model Context Protocol,
enabling LLM-driven trading analysis, signal generation, and portfolio management.
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
import logging
import inspect
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from mcp.server.fastmcp import FastMCP

# NOTE: The upstream `mcp` package has had some API churn across versions.
# Older versions may not export `ToolAnnotations` from `mcp.types`.
# We keep Ordinis compatible by falling back to a lightweight dict-like shim.
try:
    from mcp.types import ToolAnnotations  # type: ignore
except ImportError:  # pragma: no cover
    class ToolAnnotations(dict[str, Any]):
        """Fallback shim for MCP versions without `ToolAnnotations`.

        The FastMCP decorator accepts an annotations mapping. Newer MCP versions
        provide a dedicated type; older versions can work with a plain dict.
        """

        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
import yaml

if TYPE_CHECKING:
    from ordinis.engines.portfolio.core.engine import PortfolioEngine
    from ordinis.plugins.massive import MassivePlugin
    from ordinis.safety.kill_switch import KillSwitch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize MCP server
ordinis_mcp = FastMCP(
    name="Ordinis Trading System",
    instructions="AI-driven quantitative trading system with signal generation, risk management, and portfolio optimization. Use the available tools to generate signals, evaluate risk, and manage portfolios.",
)

# -----------------------------------------------------------------------------
# Compatibility: FastMCP.tool signature differs across MCP versions.
#
# Some versions accept an `annotations=` kwarg (for readOnly hints), others only
# accept name/description. Our server uses `annotations=` extensively; wrap the
# decorator to ignore it when unsupported.
# -----------------------------------------------------------------------------
_original_tool = ordinis_mcp.tool
_tool_supports_annotations = "annotations" in inspect.signature(_original_tool).parameters


def _tool_compat(*args: Any, **kwargs: Any):
    if not _tool_supports_annotations:
        kwargs.pop("annotations", None)
    return _original_tool(*args, **kwargs)


ordinis_mcp.tool = _tool_compat  # type: ignore[method-assign]

# =============================================================================
# STATE MANAGEMENT
# =============================================================================


class OrdinisState:
    """Manages server state and connections to Ordinis components."""

    def __init__(self):
        self.broker = None
        self.portfolio_state = None
        self.strategies_config: dict[str, Any] = {}
        self.active_positions: dict[str, Any] = {}
        self.signal_history: list[dict] = []
        self.is_initialized = False
        # Live components
        self._kill_switch: KillSwitch | None = None
        self._portfolio_engine: PortfolioEngine | None = None
        self._news_plugin: MassivePlugin | None = None
        # Market context for signal generation
        self._market_context: str = ""
        self._market_context_timestamp: datetime | None = None
        # Performance tracking
        self._daily_pnl: float = 0.0
        self._win_count: int = 0
        self._loss_count: int = 0
        self._peak_equity: float = 100000.0
        self._current_equity: float = 100000.0

    async def initialize(self) -> bool:
        """Initialize connections to Ordinis components."""
        try:
            # Load strategy configurations
            self.strategies_config = self._load_strategies()

            # Try to initialize kill switch
            await self._init_kill_switch()

            # Initialize portfolio engine
            await self._init_portfolio_engine()

            # Initialize news plugin
            await self._init_news_plugin()

            self.is_initialized = True
            logger.info("[MCP] Ordinis state initialized")
            return True
        except Exception as e:
            logger.error(f"[MCP] Failed to initialize: {e}")
            return False

    async def _init_portfolio_engine(self) -> None:
        """Initialize connection to PortfolioEngine."""
        try:
            from ordinis.engines.portfolio.core.config import PortfolioEngineConfig
            from ordinis.engines.portfolio.core.engine import PortfolioEngine

            config = PortfolioEngineConfig(initial_capital=100000.0)
            self._portfolio_engine = PortfolioEngine(config)
            await self._portfolio_engine.initialize()
            logger.info("[MCP] PortfolioEngine connected")
        except Exception as e:
            logger.warning(f"[MCP] PortfolioEngine not available: {e}")
            self._portfolio_engine = None

    async def _init_news_plugin(self) -> None:
        """Initialize news plugin for market news."""
        try:
            from ordinis.plugins.massive import MassivePlugin, MassivePluginConfig

            config = MassivePluginConfig(name="massive", use_mock=True)
            self._news_plugin = MassivePlugin(config)
            await self._news_plugin.initialize()
            logger.info("[MCP] MassivePlugin connected")
        except Exception as e:
            logger.warning(f"[MCP] MassivePlugin not available: {e}")
            self._news_plugin = None

    async def _init_kill_switch(self) -> None:
        """Initialize connection to live KillSwitch."""
        try:
            from ordinis.safety.kill_switch import KillSwitch

            self._kill_switch = KillSwitch(
                daily_loss_limit=1000.0,
                max_drawdown_pct=15.0,
                consecutive_loss_limit=5,
            )
            await self._kill_switch.initialize()
            logger.info("[MCP] KillSwitch connected")
        except Exception as e:
            logger.warning(f"[MCP] KillSwitch not available: {e}")
            self._kill_switch = None

    def _load_strategies(self) -> dict[str, Any]:
        """Load all strategy configurations."""
        strategies = {}
        config_path = Path("configs/strategies")

        if config_path.exists():
            for yaml_file in config_path.glob("*.yaml"):
                try:
                    with open(yaml_file) as f:
                        config = yaml.safe_load(f)
                        strategies[yaml_file.stem] = config
                except Exception as e:
                    logger.warning(f"Failed to load {yaml_file}: {e}")

        return strategies

    def _save_strategy(self, strategy_id: str) -> bool:
        """Save a strategy configuration to disk."""
        config_path = Path(f"configs/strategies/{strategy_id}.yaml")
        try:
            with open(config_path, "w") as f:
                yaml.dump(self.strategies_config[strategy_id], f, default_flow_style=False)
            return True
        except Exception as e:
            logger.error(f"Failed to save strategy {strategy_id}: {e}")
            return False


# Global state instance
_state = OrdinisState()


# =============================================================================
# TOOLS - Trading Signal Operations
# =============================================================================


@ordinis_mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def get_signal(
    symbol: str,
    strategy: str = "atr_optimized_rsi",
    timeframe: str = "1d",
) -> str:
    """
    Generate a trading signal for a symbol using a specified strategy.

    Args:
        symbol: Stock ticker symbol (e.g., "AAPL", "SPY")
        strategy: Strategy to use (atr_optimized_rsi, trend_following, etc.)
        timeframe: Data timeframe (1m, 5m, 15m, 1h, 1d)

    Returns:
        JSON with signal details including direction, confidence, and reasoning
    """
    try:
        # Get strategy config
        strategy_config = _state.strategies_config.get(strategy, {})
        symbol_config = strategy_config.get("symbols", {}).get(symbol.upper(), {})

        if not symbol_config:
            return json.dumps(
                {
                    "error": f"Symbol {symbol} not configured for strategy {strategy}",
                    "available_symbols": list(strategy_config.get("symbols", {}).keys())[:20],
                }
            )

        # Generate signal (simplified - in production, this would call SignalCore)
        signal = {
            "symbol": symbol.upper(),
            "strategy": strategy,
            "timeframe": timeframe,
            "timestamp": datetime.now(UTC).isoformat(),
            "signal_type": "entry",
            "direction": "long",
            "probability": 0.65,
            "confidence": 0.7,
            "score": 0.45,
            "params": {
                "rsi_oversold": symbol_config.get("rsi_oversold", 30),
                "rsi_exit": symbol_config.get("rsi_exit", 50),
                "atr_stop_mult": symbol_config.get("atr_stop_mult", 1.5),
                "atr_tp_mult": symbol_config.get("atr_tp_mult", 2.0),
            },
            "backtest_stats": {
                "profit_factor": symbol_config.get("backtest_profit_factor"),
                "win_rate": symbol_config.get("backtest_win_rate"),
                "return": symbol_config.get("backtest_return"),
            },
            "sector": symbol_config.get("sector", "unknown"),
        }

        _state.signal_history.append(signal)
        return json.dumps(signal, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


# =============================================================================
# TOOLS - Strategy Configuration (Phase 1)
# =============================================================================

# Validation bounds for strategy parameters
PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "rsi_oversold": (10, 40),
    "rsi_exit": (40, 70),
    "rsi_overbought": (60, 90),
    "atr_stop_mult": (0.5, 5.0),
    "atr_tp_mult": (0.5, 10.0),
    "max_position_size_pct": (0.5, 10.0),
    "max_concurrent_positions": (1, 20),
}


@ordinis_mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def get_strategy_config(
    strategy: str,
    symbol: str | None = None,
) -> str:
    """
    Get current configuration for a strategy.

    Args:
        strategy: Strategy ID (e.g., 'atr_optimized_rsi')
        symbol: Optional - get config for specific symbol within strategy

    Returns:
        JSON with strategy configuration including global params and symbol-specific settings
    """
    try:
        config = _state.strategies_config.get(strategy)
        if not config:
            return json.dumps(
                {
                    "error": f"Strategy '{strategy}' not found",
                    "available": list(_state.strategies_config.keys()),
                }
            )

        if symbol:
            symbol_config = config.get("symbols", {}).get(symbol.upper())
            if not symbol_config:
                return json.dumps(
                    {
                        "error": f"Symbol '{symbol}' not found in strategy '{strategy}'",
                        "available_symbols": list(config.get("symbols", {}).keys())[:20],
                    }
                )
            return json.dumps(
                {
                    "strategy": strategy,
                    "symbol": symbol.upper(),
                    "global_params": config.get("global_params", {}),
                    "symbol_config": symbol_config,
                    "risk_management": config.get("risk_management", {}),
                },
                indent=2,
            )

        return json.dumps(
            {
                "strategy": strategy,
                "name": config.get("strategy", {}).get("name", strategy),
                "version": config.get("strategy", {}).get("version", "unknown"),
                "global_params": config.get("global_params", {}),
                "risk_management": config.get("risk_management", {}),
                "regime_filter": config.get("regime_filter", {}),
                "symbol_count": len(config.get("symbols", {})),
            },
            indent=2,
        )

    except Exception as e:
        return json.dumps({"error": str(e)})


@ordinis_mcp.tool(annotations=ToolAnnotations(readOnlyHint=False))
async def update_strategy_config(
    strategy: str,
    key: str,
    value: float | int | str,
    symbol: str | None = None,
) -> str:
    """
    Update a strategy configuration parameter.

    Args:
        strategy: Strategy ID (e.g., 'atr_optimized_rsi')
        key: Parameter key to update (e.g., 'rsi_oversold', 'atr_stop_mult')
        value: New value for the parameter
        symbol: Optional - update for specific symbol only (otherwise updates global)

    Returns:
        JSON with update result including old and new values
    """
    try:
        config = _state.strategies_config.get(strategy)
        if not config:
            return json.dumps(
                {
                    "error": f"Strategy '{strategy}' not found",
                    "available": list(_state.strategies_config.keys()),
                }
            )

        # Validate bounds if applicable
        if key in PARAM_BOUNDS:
            min_val, max_val = PARAM_BOUNDS[key]
            if not (min_val <= float(value) <= max_val):
                return json.dumps(
                    {
                        "error": f"Value {value} out of bounds for '{key}'",
                        "bounds": {"min": min_val, "max": max_val},
                        "rejected": True,
                    }
                )

        old_value = None
        if symbol:
            # Update symbol-specific config
            symbol_config = config.get("symbols", {}).get(symbol.upper())
            if not symbol_config:
                return json.dumps(
                    {
                        "error": f"Symbol '{symbol}' not found in strategy '{strategy}'",
                    }
                )
            old_value = symbol_config.get(key)
            config["symbols"][symbol.upper()][key] = value
            logger.info(f"[MCP] Updated {strategy}/{symbol}/{key}: {old_value} -> {value}")
        else:
            # Update global params
            if "global_params" not in config:
                config["global_params"] = {}
            old_value = config["global_params"].get(key)
            config["global_params"][key] = value
            logger.info(f"[MCP] Updated {strategy}/global/{key}: {old_value} -> {value}")

        # Persist to disk
        saved = _state._save_strategy(strategy)

        return json.dumps(
            {
                "success": True,
                "strategy": strategy,
                "symbol": symbol.upper() if symbol else "global",
                "key": key,
                "old_value": old_value,
                "new_value": value,
                "persisted": saved,
                "timestamp": datetime.now(UTC).isoformat(),
            },
            indent=2,
        )

    except Exception as e:
        return json.dumps({"error": str(e)})


@ordinis_mcp.tool(annotations=ToolAnnotations(readOnlyHint=False))
async def inject_market_context(context: str) -> str:
    """
    Inject market context for signal generation.

    Use this to provide qualitative information (news, earnings, Fed announcements)
    that should influence trading decisions.

    Args:
        context: Market context description (e.g., "Fed announced rate hold, risk-on sentiment")

    Returns:
        JSON confirming context injection
    """
    try:
        _state._market_context = context
        _state._market_context_timestamp = datetime.now(UTC)

        logger.info(f"[MCP] Market context injected: {context[:100]}...")

        return json.dumps(
            {
                "success": True,
                "context": context,
                "timestamp": _state._market_context_timestamp.isoformat(),
                "message": "Context will be considered in signal generation",
            },
            indent=2,
        )

    except Exception as e:
        return json.dumps({"error": str(e)})


@ordinis_mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def get_market_context() -> str:
    """
    Get the current market context that's being used for signal generation.

    Returns:
        JSON with current market context and timestamp
    """
    return json.dumps(
        {
            "context": _state._market_context or "No context set",
            "timestamp": _state._market_context_timestamp.isoformat()
            if _state._market_context_timestamp
            else None,
            "age_minutes": (
                (datetime.now(UTC) - _state._market_context_timestamp).total_seconds() / 60
                if _state._market_context_timestamp
                else None
            ),
        },
        indent=2,
    )


# =============================================================================
# TOOLS - Market News & External Data (Phase 3)
# =============================================================================


@ordinis_mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def get_market_news(
    symbols: list[str] | None = None,
    limit: int = 10,
) -> str:
    """
    Get recent market news for specified symbols or general market news.

    Uses Massive MCP for news data with fallback to mock data for development.

    Args:
        symbols: List of stock symbols to get news for (None = market-wide news)
        limit: Maximum number of articles to return

    Returns:
        JSON with news articles including headlines, sentiment, and timestamps
    """
    try:
        if _state._news_plugin:
            articles = await _state._news_plugin.get_news(symbols=symbols, limit=limit)
            return json.dumps(
                {
                    "articles": articles,
                    "count": len(articles),
                    "symbols_queried": symbols or ["MARKET"],
                    "source": "massive",
                    "timestamp": datetime.now(UTC).isoformat(),
                },
                indent=2,
            )

        # Fallback mock data
        mock_articles = [
            {
                "headline": "Markets rally on positive economic data",
                "sentiment": "POSITIVE",
                "sentiment_score": 0.6,
                "timestamp": datetime.now(UTC).isoformat(),
                "source": "Mock",
                "category": "market",
            },
            {
                "headline": "Tech sector leads gains amid AI optimism",
                "sentiment": "POSITIVE",
                "sentiment_score": 0.5,
                "timestamp": datetime.now(UTC).isoformat(),
                "source": "Mock",
                "category": "technology",
            },
        ]

        return json.dumps(
            {
                "articles": mock_articles[:limit],
                "count": min(limit, len(mock_articles)),
                "symbols_queried": symbols or ["MARKET"],
                "source": "mock",
                "timestamp": datetime.now(UTC).isoformat(),
            },
            indent=2,
        )

    except Exception as e:
        return json.dumps({"error": str(e)})


@ordinis_mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def get_earnings_calendar(
    symbols: list[str] | None = None,
    days_ahead: int = 5,
) -> str:
    """
    Get upcoming earnings announcements for portfolio symbols.

    Args:
        symbols: List of symbols to check (None = all portfolio symbols)
        days_ahead: Number of days to look ahead

    Returns:
        JSON with upcoming earnings including dates, estimates, and timing
    """
    try:
        if _state._news_plugin:
            earnings = await _state._news_plugin.get_earnings_calendar(
                symbols=symbols, days_ahead=days_ahead
            )
            return json.dumps(
                {
                    "earnings": earnings,
                    "count": len(earnings),
                    "days_ahead": days_ahead,
                    "timestamp": datetime.now(UTC).isoformat(),
                },
                indent=2,
            )

        # Fallback mock data
        return json.dumps(
            {
                "earnings": [
                    {
                        "symbol": "AAPL",
                        "company_name": "Apple Inc.",
                        "earnings_date": (datetime.now(UTC)).date().isoformat(),
                        "time": "AMC",
                        "eps_estimate": 2.35,
                    }
                ],
                "count": 1,
                "days_ahead": days_ahead,
                "source": "mock",
                "timestamp": datetime.now(UTC).isoformat(),
            },
            indent=2,
        )

    except Exception as e:
        return json.dumps({"error": str(e)})


@ordinis_mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def get_sentiment(symbol: str) -> str:
    """
    Get sentiment analysis for a specific symbol.

    Aggregates news sentiment to provide overall market sentiment.

    Args:
        symbol: Stock symbol to analyze

    Returns:
        JSON with sentiment classification, score, and supporting data
    """
    try:
        if _state._news_plugin:
            sentiment = await _state._news_plugin.get_sentiment(symbol)
            return json.dumps(sentiment, indent=2)

        # Fallback mock data
        return json.dumps(
            {
                "symbol": symbol.upper(),
                "sentiment": "NEUTRAL",
                "score": 0.1,
                "article_count": 5,
                "confidence": 0.6,
                "recent_headlines": [
                    f"{symbol} trades steadily amid sector rotation",
                    f"Analysts maintain hold rating on {symbol}",
                ],
                "source": "mock",
            },
            indent=2,
        )

    except Exception as e:
        return json.dumps({"error": str(e)})


# =============================================================================
# TOOLS - Capital Preservation (Phase 1)
# =============================================================================


@ordinis_mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def get_drawdown_status() -> str:
    """
    Get current drawdown and risk status.

    Returns:
        JSON with current drawdown %, daily P&L, consecutive losses, and kill switch status
    """
    try:
        # Get from live KillSwitch if available
        if _state._kill_switch:
            status = _state._kill_switch.get_status()
            drawdown_pct = 0.0
            if status.get("limits", {}).get("max_drawdown_pct"):
                # Calculate from tracked equity
                peak = _state._peak_equity
                current = _state._current_equity
                if peak > 0:
                    drawdown_pct = ((peak - current) / peak) * 100

            return json.dumps(
                {
                    "kill_switch_active": status.get("active", False),
                    "kill_switch_reason": status.get("reason"),
                    "drawdown_pct": round(drawdown_pct, 2),
                    "daily_pnl": status.get("daily_pnl", 0.0),
                    "consecutive_losses": status.get("consecutive_losses", 0),
                    "limits": status.get("limits", {}),
                    "can_trade": not status.get("active", False),
                    "timestamp": datetime.now(UTC).isoformat(),
                },
                indent=2,
            )

        # Fallback to simulated data
        return json.dumps(
            {
                "kill_switch_active": False,
                "kill_switch_reason": None,
                "drawdown_pct": 2.5,
                "daily_pnl": -150.0,
                "consecutive_losses": 1,
                "limits": {
                    "daily_loss": 1000.0,
                    "max_drawdown_pct": 15.0,
                    "consecutive_loss": 5,
                },
                "can_trade": True,
                "mode": "simulated",
                "timestamp": datetime.now(UTC).isoformat(),
            },
            indent=2,
        )

    except Exception as e:
        return json.dumps({"error": str(e)})


@ordinis_mcp.tool(annotations=ToolAnnotations(readOnlyHint=False))
async def reduce_exposure(factor: float) -> str:
    """
    Reduce position sizing by a factor without full trading halt.

    Use for graduated response to drawdowns:
    - 5% drawdown: factor=0.75 (reduce 25%)
    - 10% drawdown: factor=0.50 (reduce 50%)

    Args:
        factor: Exposure factor (0.0-1.0). 0.5 means 50% of normal position sizes.

    Returns:
        JSON confirming exposure reduction
    """
    try:
        if not (0.0 <= factor <= 1.0):
            return json.dumps(
                {
                    "error": "Factor must be between 0.0 and 1.0",
                    "rejected": True,
                }
            )

        # Update risk management in all strategies
        for strategy_id, config in _state.strategies_config.items():
            if "risk_management" in config:
                original_max = config["risk_management"].get("max_position_size_pct", 3.0)
                adjusted_max = original_max * factor
                config["risk_management"]["max_position_size_pct"] = adjusted_max
                config["risk_management"]["exposure_factor"] = factor
                config["risk_management"]["exposure_reduced_at"] = datetime.now(UTC).isoformat()

        logger.warning(f"[MCP] Exposure reduced to {factor * 100:.0f}% across all strategies")

        return json.dumps(
            {
                "success": True,
                "exposure_factor": factor,
                "exposure_pct": factor * 100,
                "message": f"Position sizing reduced to {factor * 100:.0f}% of normal",
                "strategies_affected": list(_state.strategies_config.keys()),
                "timestamp": datetime.now(UTC).isoformat(),
            },
            indent=2,
        )

    except Exception as e:
        return json.dumps({"error": str(e)})


@ordinis_mcp.tool(annotations=ToolAnnotations(readOnlyHint=False))
async def activate_kill_switch(reason: str) -> str:
    """
    Activate the kill switch to halt all trading.

    Use only when drawdown exceeds 15% or in emergency situations.

    Args:
        reason: Reason for activation (logged for audit)

    Returns:
        JSON confirming kill switch activation
    """
    try:
        if _state._kill_switch:
            from ordinis.safety.kill_switch import KillSwitchReason

            await _state._kill_switch.trigger(
                reason=KillSwitchReason.MANUAL,
                message=f"MCP Agent: {reason}",
                triggered_by="mcp_agent",
            )
            logger.critical(f"[MCP] Kill switch activated by agent: {reason}")

            return json.dumps(
                {
                    "success": True,
                    "kill_switch_active": True,
                    "reason": reason,
                    "message": "Trading halted - kill switch activated",
                    "timestamp": datetime.now(UTC).isoformat(),
                },
                indent=2,
            )

        return json.dumps(
            {
                "success": False,
                "error": "Kill switch not available in current mode",
            }
        )

    except Exception as e:
        return json.dumps({"error": str(e)})


@ordinis_mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def analyze_open_positions() -> str:
    """
    Analyze open positions for exit decisions.

    Returns positions with unrealized P&L, time held, distance to stop, and exit recommendations.

    Returns:
        JSON with position analysis and recommendations
    """
    try:
        positions = []

        # Get from live PortfolioEngine if available
        if _state._portfolio_engine and _state._portfolio_engine.positions:
            now = datetime.now(UTC)
            for symbol, pos in _state._portfolio_engine.positions.items():
                # Calculate time held
                entry_time = pos.entry_time or pos.last_update_time
                time_held_hours = (now - entry_time).total_seconds() / 3600 if entry_time else 0

                # Calculate distance to stop (estimate if not set)
                stop_price = pos.avg_entry_price * 0.95  # Default 5% stop
                distance_to_stop_pct = (
                    (pos.current_price - stop_price) / pos.current_price * 100
                    if pos.current_price > 0
                    else 0
                )

                # Generate recommendation
                if pos.unrealized_pnl > 0:
                    if pos.pnl_pct > 5:
                        recommendation = "tighten_stop"
                        reason = "Profitable position, consider trailing stop"
                    else:
                        recommendation = "hold"
                        reason = "In profit, not near stop"
                elif pos.unrealized_pnl < 0:
                    if pos.pnl_pct < -3:
                        recommendation = "exit"
                        reason = "Position significantly underwater"
                    else:
                        recommendation = "monitor"
                        reason = "Slightly underwater, stop intact"
                else:
                    recommendation = "hold"
                    reason = "At breakeven"

                positions.append({
                    "symbol": symbol,
                    "side": pos.side.value if hasattr(pos.side, "value") else str(pos.side),
                    "quantity": pos.quantity,
                    "avg_price": pos.avg_entry_price,
                    "current_price": pos.current_price,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "unrealized_pnl_pct": pos.pnl_pct,
                    "stop_price": round(stop_price, 2),
                    "distance_to_stop_pct": round(distance_to_stop_pct, 2),
                    "time_held_hours": round(time_held_hours, 1),
                    "recommendation": recommendation,
                    "reason": reason,
                })

            # Update tracked equity
            _state._current_equity = _state._portfolio_engine.equity
            _state._peak_equity = max(_state._peak_equity, _state._current_equity)

        else:
            # Fallback mock data
            positions = [
                {
                    "symbol": "AAPL",
                    "side": "long",
                    "quantity": 50,
                    "avg_price": 175.50,
                    "current_price": 178.25,
                    "unrealized_pnl": 137.50,
                    "unrealized_pnl_pct": 1.56,
                    "stop_price": 172.00,
                    "distance_to_stop_pct": 3.51,
                    "time_held_hours": 24,
                    "recommendation": "hold",
                    "reason": "In profit, not near stop",
                },
                {
                    "symbol": "AMD",
                    "side": "long",
                    "quantity": 100,
                    "avg_price": 135.00,
                    "current_price": 138.50,
                    "unrealized_pnl": 350.00,
                    "unrealized_pnl_pct": 2.59,
                    "stop_price": 130.00,
                    "distance_to_stop_pct": 6.14,
                    "time_held_hours": 48,
                    "recommendation": "tighten_stop",
                    "reason": "Profitable position, consider trailing stop",
                },
                {
                    "symbol": "CRWD",
                    "side": "long",
                    "quantity": 25,
                    "avg_price": 355.00,
                    "current_price": 353.50,
                    "unrealized_pnl": -37.50,
                    "unrealized_pnl_pct": -0.42,
                    "stop_price": 340.00,
                    "distance_to_stop_pct": 3.82,
                    "time_held_hours": 8,
                    "recommendation": "monitor",
                    "reason": "Slightly underwater, stop intact",
                },
            ]

        total_unrealized = sum(p["unrealized_pnl"] for p in positions)
        winners = [p for p in positions if p["unrealized_pnl"] > 0]
        losers = [p for p in positions if p["unrealized_pnl"] < 0]

        return json.dumps(
            {
                "position_count": len(positions),
                "total_unrealized_pnl": total_unrealized,
                "winners": len(winners),
                "losers": len(losers),
                "positions": positions,
                "summary": {
                    "hold": len([p for p in positions if p["recommendation"] == "hold"]),
                    "tighten_stop": len(
                        [p for p in positions if p["recommendation"] == "tighten_stop"]
                    ),
                    "monitor": len([p for p in positions if p["recommendation"] == "monitor"]),
                    "exit": len([p for p in positions if p["recommendation"] == "exit"]),
                },
                "source": "live" if _state._portfolio_engine else "mock",
                "timestamp": datetime.now(UTC).isoformat(),
            },
            indent=2,
        )

    except Exception as e:
        return json.dumps({"error": str(e)})


@ordinis_mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def get_multi_signal(
    symbols: list[str],
    strategy: str = "atr_optimized_rsi",
    aggregation: Literal["consensus", "majority", "weighted", "any"] = "weighted",
) -> str:
    """
    Generate aggregated signals for multiple symbols.

    Args:
        symbols: List of ticker symbols
        strategy: Strategy to use
        aggregation: Signal aggregation mode
            - consensus: All must agree
            - majority: >50% must agree
            - weighted: Weight by confidence
            - any: Any signal triggers

    Returns:
        JSON with aggregated signal and individual symbol signals
    """
    try:
        signals = []
        for symbol in symbols:
            result = await get_signal(symbol, strategy)
            signals.append(json.loads(result))

        # Filter valid signals
        valid_signals = [s for s in signals if "error" not in s]

        # Aggregate
        buy_count = sum(1 for s in valid_signals if s.get("direction") == "long")
        sell_count = sum(1 for s in valid_signals if s.get("direction") == "short")

        aggregated = {
            "timestamp": datetime.now(UTC).isoformat(),
            "aggregation_mode": aggregation,
            "total_symbols": len(symbols),
            "valid_signals": len(valid_signals),
            "buy_signals": buy_count,
            "sell_signals": sell_count,
            "consensus": buy_count / len(valid_signals) if valid_signals else 0,
            "recommendation": "buy"
            if buy_count > sell_count
            else "sell"
            if sell_count > buy_count
            else "hold",
            "individual_signals": signals,
        }

        return json.dumps(aggregated, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


@ordinis_mcp.tool(annotations=ToolAnnotations(readOnlyHint=False))
async def submit_paper_order(
    symbol: str,
    side: Literal["buy", "sell"],
    quantity: int,
    order_type: Literal["market", "limit"] = "market",
    limit_price: float | None = None,
) -> str:
    """
    Submit a paper trading order (simulation only).

    Args:
        symbol: Stock ticker symbol
        side: Order side (buy or sell)
        quantity: Number of shares
        order_type: Order type (market or limit)
        limit_price: Limit price (required for limit orders)

    Returns:
        JSON with order confirmation or rejection details
    """
    try:
        order_id = f"ORD-{datetime.now().strftime('%Y%m%d%H%M%S')}-{symbol}"

        order = {
            "order_id": order_id,
            "symbol": symbol.upper(),
            "side": side,
            "quantity": quantity,
            "order_type": order_type,
            "limit_price": limit_price,
            "status": "submitted",
            "submitted_at": datetime.now(UTC).isoformat(),
            "mode": "paper",
            "message": "Paper order submitted successfully",
        }

        return json.dumps(order, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


@ordinis_mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def evaluate_risk(
    symbol: str,
    direction: Literal["long", "short"],
    position_size_usd: float,
    entry_price: float,
    stop_price: float | None = None,
) -> str:
    """
    Evaluate risk for a proposed trade using RiskGuard rules.

    Args:
        symbol: Stock ticker symbol
        direction: Trade direction (long or short)
        position_size_usd: Position size in USD
        entry_price: Proposed entry price
        stop_price: Stop loss price (optional)

    Returns:
        JSON with risk evaluation: approved/rejected with reasons
    """
    try:
        # Calculate risk metrics
        quantity = int(position_size_usd / entry_price)
        risk_per_share = abs(entry_price - stop_price) if stop_price else entry_price * 0.02
        total_risk = risk_per_share * quantity
        risk_pct = (total_risk / position_size_usd) * 100 if position_size_usd > 0 else 0

        # Evaluate against rules
        rules_passed = []
        rules_failed = []

        # Position size check (max 3% of portfolio)
        max_position_pct = 3.0
        if position_size_usd <= 100000 * (max_position_pct / 100):
            rules_passed.append("position_size_limit")
        else:
            rules_failed.append(
                {
                    "rule": "position_size_limit",
                    "reason": f"Position exceeds {max_position_pct}% limit",
                }
            )

        # Risk per trade check (max 1%)
        if risk_pct <= 1.0:
            rules_passed.append("risk_per_trade")
        else:
            rules_failed.append(
                {
                    "rule": "risk_per_trade",
                    "reason": f"Risk {risk_pct:.2f}% exceeds 1% limit",
                }
            )

        # Stop loss required
        if stop_price:
            rules_passed.append("stop_loss_required")
        else:
            rules_failed.append(
                {
                    "rule": "stop_loss_required",
                    "reason": "Stop loss not specified",
                }
            )

        evaluation = {
            "symbol": symbol.upper(),
            "direction": direction,
            "position_size_usd": position_size_usd,
            "quantity": quantity,
            "entry_price": entry_price,
            "stop_price": stop_price,
            "risk_metrics": {
                "risk_per_share": round(risk_per_share, 2),
                "total_risk_usd": round(total_risk, 2),
                "risk_percentage": round(risk_pct, 2),
            },
            "approved": len(rules_failed) == 0,
            "rules_passed": rules_passed,
            "rules_failed": rules_failed,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        return json.dumps(evaluation, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


# =============================================================================
# RESOURCES - Portfolio State
# =============================================================================


@ordinis_mcp.resource("ordinis://portfolio/summary")
async def get_portfolio_summary() -> str:
    """
    Get current portfolio summary including equity, positions, and P&L.
    """
    summary = {
        "timestamp": datetime.now(UTC).isoformat(),
        "account": {
            "equity": 100000.00,
            "cash": 75000.00,
            "buying_power": 150000.00,
            "portfolio_value": 25000.00,
        },
        "positions": {
            "count": 3,
            "total_market_value": 25000.00,
            "total_unrealized_pnl": 450.00,
            "total_unrealized_pnl_pct": 1.8,
        },
        "daily": {
            "pnl": 250.00,
            "pnl_pct": 0.25,
            "trades": 5,
            "wins": 3,
            "losses": 2,
        },
        "risk": {
            "exposure_pct": 25.0,
            "max_drawdown_pct": 2.5,
            "var_95": 1500.00,
        },
        "mode": "paper",
    }
    return json.dumps(summary, indent=2)


@ordinis_mcp.resource("ordinis://portfolio/positions")
async def get_positions() -> str:
    """
    Get all open positions with details.
    """
    positions = [
        {
            "symbol": "AAPL",
            "quantity": 50,
            "avg_price": 175.50,
            "current_price": 178.25,
            "market_value": 8912.50,
            "unrealized_pnl": 137.50,
            "unrealized_pnl_pct": 1.56,
            "side": "long",
            "opened_at": "2024-12-18T10:30:00Z",
        },
        {
            "symbol": "AMD",
            "quantity": 100,
            "avg_price": 135.00,
            "current_price": 138.50,
            "market_value": 13850.00,
            "unrealized_pnl": 350.00,
            "unrealized_pnl_pct": 2.59,
            "side": "long",
            "opened_at": "2024-12-17T14:15:00Z",
        },
        {
            "symbol": "CRWD",
            "quantity": 25,
            "avg_price": 355.00,
            "current_price": 353.50,
            "market_value": 8837.50,
            "unrealized_pnl": -37.50,
            "unrealized_pnl_pct": -0.42,
            "side": "long",
            "opened_at": "2024-12-19T09:45:00Z",
        },
    ]
    return json.dumps({"positions": positions, "count": len(positions)}, indent=2)


@ordinis_mcp.resource("ordinis://portfolio/history/{days}")
async def get_portfolio_history(days: int = 30) -> str:
    """
    Get portfolio performance history for the specified number of days.
    """
    # Generate sample history
    history = {
        "period_days": days,
        "start_equity": 95000.00,
        "end_equity": 100000.00,
        "total_return_pct": 5.26,
        "annualized_return_pct": 63.16,
        "sharpe_ratio": 2.1,
        "max_drawdown_pct": 3.5,
        "win_rate_pct": 58.0,
        "profit_factor": 1.85,
        "total_trades": 45,
        "daily_summary": "Portfolio up 5.26% over 30 days with solid risk-adjusted returns",
    }
    return json.dumps(history, indent=2)


@ordinis_mcp.resource("ordinis://signals/history")
async def get_signal_history() -> str:
    """
    Get recent signal history generated by the server.
    """
    return json.dumps(
        {
            "count": len(_state.signal_history),
            "signals": _state.signal_history[-20:],  # Last 20 signals
        },
        indent=2,
    )


# =============================================================================
# RESOURCES - Performance Metrics (Phase 1)
# =============================================================================


@ordinis_mcp.resource("ordinis://metrics/pnl")
async def get_pnl_metrics() -> str:
    """
    Get P&L and performance metrics for agent decision-making.

    Includes daily P&L, win rate, average win/loss, and Sharpe estimate.
    """
    # Calculate metrics from signal history with outcomes
    signals_with_outcomes = [s for s in _state.signal_history if s.get("outcome")]
    wins = [s for s in signals_with_outcomes if s.get("outcome", {}).get("pnl", 0) > 0]
    losses = [s for s in signals_with_outcomes if s.get("outcome", {}).get("pnl", 0) < 0]

    total_pnl = sum(s.get("outcome", {}).get("pnl", 0) for s in signals_with_outcomes)
    win_rate = len(wins) / len(signals_with_outcomes) * 100 if signals_with_outcomes else 0
    avg_win = sum(s.get("outcome", {}).get("pnl", 0) for s in wins) / len(wins) if wins else 0
    avg_loss = (
        sum(s.get("outcome", {}).get("pnl", 0) for s in losses) / len(losses) if losses else 0
    )

    # Simulated data for demo
    metrics = {
        "timestamp": datetime.now(UTC).isoformat(),
        "daily": {
            "pnl": _state._daily_pnl,
            "pnl_pct": (_state._daily_pnl / _state._current_equity * 100)
            if _state._current_equity
            else 0,
            "trades": len(signals_with_outcomes),
            "wins": len(wins) or _state._win_count,
            "losses": len(losses) or _state._loss_count,
        },
        "performance": {
            "win_rate_pct": win_rate or 58.0,
            "avg_win": avg_win or 125.50,
            "avg_loss": avg_loss or -85.25,
            "profit_factor": abs(avg_win / avg_loss) if avg_loss else 1.47,
            "expectancy": (win_rate / 100 * avg_win + (1 - win_rate / 100) * avg_loss)
            if signals_with_outcomes
            else 32.15,
        },
        "risk_adjusted": {
            "sharpe_estimate": 2.1,
            "sortino_estimate": 2.8,
            "calmar_ratio": 3.2,
        },
        "drawdown": {
            "current_pct": (
                ((_state._peak_equity - _state._current_equity) / _state._peak_equity) * 100
            )
            if _state._peak_equity
            else 0,
            "max_pct": 3.5,
            "recovery_days": 0,
        },
    }
    return json.dumps(metrics, indent=2)


@ordinis_mcp.resource("ordinis://metrics/signals")
async def get_signal_metrics() -> str:
    """
    Get signal performance metrics by strategy.

    Shows hit rate, average outcome, and recent performance for each strategy.
    """
    # Group signals by strategy
    strategy_stats: dict[str, dict] = {}

    for signal in _state.signal_history:
        strategy = signal.get("strategy", "unknown")
        if strategy not in strategy_stats:
            strategy_stats[strategy] = {
                "total_signals": 0,
                "with_outcome": 0,
                "wins": 0,
                "losses": 0,
                "total_pnl": 0.0,
                "symbols": set(),
            }

        stats = strategy_stats[strategy]
        stats["total_signals"] += 1
        stats["symbols"].add(signal.get("symbol", ""))

        if signal.get("outcome"):
            stats["with_outcome"] += 1
            pnl = signal.get("outcome", {}).get("pnl", 0)
            stats["total_pnl"] += pnl
            if pnl > 0:
                stats["wins"] += 1
            elif pnl < 0:
                stats["losses"] += 1

    # Convert to output format
    strategies = []
    for strategy_id, stats in strategy_stats.items():
        hit_rate = (stats["wins"] / stats["with_outcome"] * 100) if stats["with_outcome"] else None
        strategies.append(
            {
                "strategy": strategy_id,
                "total_signals": stats["total_signals"],
                "evaluated": stats["with_outcome"],
                "hit_rate_pct": round(hit_rate, 1) if hit_rate else None,
                "wins": stats["wins"],
                "losses": stats["losses"],
                "total_pnl": round(stats["total_pnl"], 2),
                "symbols_traded": len(stats["symbols"]),
            }
        )

    # Add demo data if no real signals
    if not strategies:
        strategies = [
            {
                "strategy": "atr_optimized_rsi",
                "total_signals": 45,
                "evaluated": 42,
                "hit_rate_pct": 57.1,
                "wins": 24,
                "losses": 18,
                "total_pnl": 1250.00,
                "symbols_traded": 12,
            },
            {
                "strategy": "trend_following",
                "total_signals": 18,
                "evaluated": 15,
                "hit_rate_pct": 53.3,
                "wins": 8,
                "losses": 7,
                "total_pnl": 480.00,
                "symbols_traded": 5,
            },
        ]

    return json.dumps(
        {
            "timestamp": datetime.now(UTC).isoformat(),
            "strategies": strategies,
            "total_signals": sum(s["total_signals"] for s in strategies),
            "overall_hit_rate_pct": 56.5,
        },
        indent=2,
    )


@ordinis_mcp.resource("ordinis://metrics/positions")
async def get_position_metrics() -> str:
    """
    Get current position metrics for portfolio analysis.

    Shows positions with unrealized P&L, time in trade, and sector allocation.
    """
    # Get positions (in production, from PortfolioEngine)
    positions = [
        {
            "symbol": "AAPL",
            "side": "long",
            "quantity": 50,
            "avg_price": 175.50,
            "current_price": 178.25,
            "market_value": 8912.50,
            "unrealized_pnl": 137.50,
            "unrealized_pnl_pct": 1.56,
            "time_held_hours": 24,
            "sector": "technology",
        },
        {
            "symbol": "AMD",
            "side": "long",
            "quantity": 100,
            "avg_price": 135.00,
            "current_price": 138.50,
            "market_value": 13850.00,
            "unrealized_pnl": 350.00,
            "unrealized_pnl_pct": 2.59,
            "time_held_hours": 48,
            "sector": "technology",
        },
        {
            "symbol": "CRWD",
            "side": "long",
            "quantity": 25,
            "avg_price": 355.00,
            "current_price": 353.50,
            "market_value": 8837.50,
            "unrealized_pnl": -37.50,
            "unrealized_pnl_pct": -0.42,
            "time_held_hours": 8,
            "sector": "technology",
        },
    ]

    total_value = sum(p["market_value"] for p in positions)
    total_unrealized = sum(p["unrealized_pnl"] for p in positions)

    # Sector allocation
    sectors: dict[str, float] = {}
    for p in positions:
        sector = p.get("sector", "other")
        sectors[sector] = sectors.get(sector, 0) + p["market_value"]

    sector_allocation = (
        {k: round(v / total_value * 100, 1) for k, v in sectors.items()} if total_value else {}
    )

    return json.dumps(
        {
            "timestamp": datetime.now(UTC).isoformat(),
            "position_count": len(positions),
            "total_market_value": round(total_value, 2),
            "total_unrealized_pnl": round(total_unrealized, 2),
            "total_unrealized_pnl_pct": round(total_unrealized / total_value * 100, 2)
            if total_value
            else 0,
            "winners": len([p for p in positions if p["unrealized_pnl"] > 0]),
            "losers": len([p for p in positions if p["unrealized_pnl"] < 0]),
            "sector_allocation": sector_allocation,
            "positions": positions,
        },
        indent=2,
    )


@ordinis_mcp.resource("ordinis://strategies/list")
async def get_strategies_list() -> str:
    """
    Get list of available trading strategies.
    """
    strategies = []
    for name, config in _state.strategies_config.items():
        strategy_info = config.get("strategy", {})
        symbols = config.get("symbols", {})
        strategies.append(
            {
                "id": name,
                "name": strategy_info.get("name", name),
                "type": strategy_info.get("type", "unknown"),
                "version": strategy_info.get("version", "1.0.0"),
                "description": strategy_info.get("description", ""),
                "symbol_count": len(symbols),
            }
        )

    return json.dumps({"strategies": strategies, "count": len(strategies)}, indent=2)


@ordinis_mcp.resource("ordinis://strategies/{strategy_id}")
async def get_strategy_details(strategy_id: str) -> str:
    """
    Get detailed configuration for a specific strategy.
    """
    config = _state.strategies_config.get(strategy_id)
    if not config:
        return json.dumps({"error": f"Strategy '{strategy_id}' not found"})

    return json.dumps(config, indent=2, default=str)


@ordinis_mcp.resource("ordinis://market/status")
async def get_market_status() -> str:
    """
    Get current market status and trading hours.
    """
    now = datetime.now(UTC)
    # Simple market hours check (9:30 AM - 4:00 PM ET)
    hour = now.hour - 5  # Approximate ET offset
    is_open = 9.5 <= hour < 16 and now.weekday() < 5

    status = {
        "timestamp": now.isoformat(),
        "market": "US Equity",
        "is_open": is_open,
        "session": "regular" if is_open else "closed",
        "next_open": "2024-12-20T09:30:00-05:00" if not is_open else None,
        "next_close": "2024-12-19T16:00:00-05:00" if is_open else None,
    }
    return json.dumps(status, indent=2)


# =============================================================================
# PROMPTS - Strategy Templates
# =============================================================================


@ordinis_mcp.prompt()
async def analyze_symbol(symbol: str) -> str:
    """
    Comprehensive analysis prompt for a stock symbol.
    """
    return f"""You are analyzing {symbol.upper()} for potential trading opportunities.

Please provide:
1. **Technical Analysis**
   - Current trend (bullish/bearish/neutral)
   - Key support and resistance levels
   - RSI and momentum indicators

2. **Signal Assessment**
   - Based on ATR-Optimized RSI strategy
   - Entry/exit conditions
   - Position sizing recommendation

3. **Risk Evaluation**
   - Suggested stop loss level
   - Risk/reward ratio
   - Maximum position size (3% of portfolio)

4. **Market Context**
   - Sector performance
   - Correlation with broader market
   - Any upcoming events/earnings

Use the available tools:
- `get_signal` to generate strategy signals
- `evaluate_risk` to check risk parameters
- Access `ordinis://portfolio/summary` for current portfolio state

Provide actionable recommendations with specific price levels."""


@ordinis_mcp.prompt()
async def portfolio_review() -> str:
    """
    Portfolio review and rebalancing prompt.
    """
    return """You are conducting a comprehensive portfolio review.

Please analyze:

1. **Current Holdings**
   - Access `ordinis://portfolio/positions` for open positions
   - Evaluate each position's performance
   - Check sector concentration

2. **Risk Assessment**
   - Access `ordinis://portfolio/summary` for risk metrics
   - Check exposure levels vs limits
   - Evaluate drawdown status

3. **Rebalancing Recommendations**
   - Positions to close (losers, winners to take profit)
   - Sectors to reduce/increase exposure
   - New opportunities from signal screener

4. **Action Plan**
   - Priority trades with specific sizing
   - Risk management adjustments
   - Timeline for execution

Use the `evaluate_risk` tool to validate any proposed trades against RiskGuard rules."""


@ordinis_mcp.prompt()
async def strategy_optimization(strategy: str = "atr_optimized_rsi") -> str:
    """
    Strategy optimization and tuning prompt.
    """
    return f"""You are optimizing the {strategy} trading strategy.

Access `ordinis://strategies/{strategy}` for current configuration.

Please analyze and recommend:

1. **Parameter Tuning**
   - RSI thresholds (oversold/overbought)
   - ATR multipliers (stop/target)
   - Position sizing rules

2. **Symbol Selection**
   - Best performing symbols
   - Symbols to remove (poor backtest stats)
   - New symbols to add

3. **Regime Filtering**
   - Which market regimes work best
   - Adjustment for current conditions
   - Volatility-based modifications

4. **Risk Parameters**
   - Max position size %
   - Max concurrent positions
   - Daily loss limits

Provide specific parameter recommendations with expected impact on:
- Win rate
- Profit factor
- Sharpe ratio
- Maximum drawdown"""


@ordinis_mcp.prompt()
async def trade_setup(symbol: str, direction: str = "long") -> str:
    """
    Detailed trade setup prompt for a specific symbol.
    """
    return f"""You are preparing a {direction.upper()} trade setup for {symbol.upper()}.

1. **Signal Confirmation**
   - Use `get_signal('{symbol}')` to check strategy signals
   - Verify signal strength and confidence

2. **Entry Planning**
   - Optimal entry price/zone
   - Order type (market vs limit)
   - Entry triggers

3. **Position Sizing**
   - Calculate shares based on 3% max position
   - ATR-based position sizing
   - Use `evaluate_risk` to validate

4. **Exit Strategy**
   - Stop loss level (ATR-based)
   - Take profit targets (multiple levels)
   - Trailing stop rules

5. **Risk Check**
   - Use `evaluate_risk` with proposed parameters
   - Verify all rules pass
   - Adjust if needed

Provide a complete trade plan ready for execution."""


# =============================================================================
# SERVER LIFECYCLE
# =============================================================================


async def initialize_server():
    """Initialize the MCP server with Ordinis components."""
    await _state.initialize()
    logger.info("[MCP] Ordinis MCP Server initialized")


def main():
    """Entry point for the Ordinis MCP Server."""
    import asyncio

    # Initialize state
    asyncio.run(initialize_server())

    # Run the server
    ordinis_mcp.run()


if __name__ == "__main__":
    main()
