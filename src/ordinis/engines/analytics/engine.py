from datetime import datetime
import logging
from typing import Any

import pandas as pd

from ordinis.engines.analytics.config import AnalyticsConfig
from ordinis.engines.analytics.models import Metric, TradeResult

logger = logging.getLogger(__name__)


class AnalyticsEngine:
    """
    Analytics Engine responsible for evaluating system performance and trading outcomes.
    """

    def __init__(self, config: AnalyticsConfig = AnalyticsConfig()):
        self.config = config
        self.metrics: list[Metric] = []
        self.trade_history: list[TradeResult] = []
        logger.info("AnalyticsEngine initialized")

    async def record(self, results: list[Any]) -> None:
        """
        Record execution results from the orchestration cycle.

        Args:
            results: A list of results from various engines (signals, trades, etc.)
        """
        if not self.config.enabled:
            return

        timestamp = datetime.now()

        logger.debug(f"Recording {len(results)} results at {timestamp}")

        for result in results:
            # Handle dictionary results (common from other engines)
            if isinstance(result, dict):
                # Check if it looks like a trade execution
                if "symbol" in result and "price" in result and "quantity" in result:
                    try:
                        trade = TradeResult(
                            symbol=result["symbol"],
                            side=result.get("side", "buy"),
                            quantity=float(result["quantity"]),
                            price=float(result["price"]),
                            timestamp=result.get("timestamp", timestamp),
                            order_id=result.get("order_id", f"ord_{len(self.trade_history)}"),
                            fees=float(result.get("fees", 0.0)),
                            pnl=float(result.get("pnl")) if result.get("pnl") is not None else None,
                        )
                        self.trade_history.append(trade)

                        if trade.pnl is not None:
                            self.metrics.append(
                                Metric(name="pnl", value=trade.pnl, timestamp=timestamp)
                            )
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Failed to parse trade result: {e}")

            # Handle object results (if they have attributes)
            elif (
                hasattr(result, "symbol")
                and hasattr(result, "price")
                and hasattr(result, "quantity")
            ):
                try:
                    trade = TradeResult(
                        symbol=result.symbol,
                        side=getattr(result, "side", "buy"),
                        quantity=float(result.quantity),
                        price=float(result.price),
                        timestamp=getattr(result, "timestamp", timestamp),
                        order_id=getattr(result, "order_id", f"ord_{len(self.trade_history)}"),
                        fees=float(getattr(result, "fees", 0.0)),
                        pnl=float(result.pnl) if getattr(result, "pnl", None) is not None else None,
                    )
                    self.trade_history.append(trade)
                    if trade.pnl is not None:
                        self.metrics.append(
                            Metric(name="pnl", value=trade.pnl, timestamp=timestamp)
                        )
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to parse trade object: {e}")

    def calculate_metrics(self) -> dict[str, float]:
        """Calculate performance metrics from trade history."""
        if not self.trade_history:
            return {
                "total_pnl": 0.0,
                "win_rate": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "trade_count": 0,
            }

        # Convert trade history to DataFrame for analysis
        data = [t.__dict__ for t in self.trade_history]
        df = pd.DataFrame(data)

        # Ensure pnl column exists and handle None
        if "pnl" not in df.columns:
            df["pnl"] = 0.0
        df["pnl"] = df["pnl"].fillna(0.0)

        # Total PnL
        total_pnl = df["pnl"].sum()

        # Win Rate
        winning_trades = df[df["pnl"] > 0]
        win_rate = len(winning_trades) / len(df) if len(df) > 0 else 0.0

        # Sharpe Ratio
        # Using trade PnL as returns stream (simplified)
        returns = df["pnl"]
        if len(returns) > 1 and returns.std() > 0:
            # Simple Sharpe: Mean / StdDev
            sharpe = returns.mean() / returns.std()
        else:
            sharpe = 0.0

        # Max Drawdown
        # Construct equity curve
        equity = returns.cumsum()
        running_max = equity.cummax()
        drawdown = equity - running_max
        max_drawdown = drawdown.min() if len(drawdown) > 0 else 0.0

        return {
            "total_pnl": float(total_pnl),
            "win_rate": float(win_rate),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_drawdown),
            "trade_count": len(df),
        }

    def get_performance_summary(self) -> dict[str, Any]:
        """
        Calculate and return a summary of current performance metrics.
        """
        metrics = self.calculate_metrics()
        metrics["timestamp"] = datetime.now()
        return metrics

    async def start(self) -> None:
        """Start the analytics engine."""
        logger.info("AnalyticsEngine started")

    async def stop(self) -> None:
        """Stop the analytics engine."""
        logger.info("AnalyticsEngine stopped")
