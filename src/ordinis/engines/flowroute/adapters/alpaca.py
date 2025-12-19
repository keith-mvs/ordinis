"""
Alpaca broker adapter for live and paper trading.

Connects to Alpaca's trading API for real order execution.
"""

import logging
import os
from typing import Any

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest

from ordinis.core.protocols import BrokerAdapter

from ..core.orders import Fill, Order

logger = logging.getLogger(__name__)


class AlpacaBrokerAdapter(BrokerAdapter):
    """
    Alpaca broker adapter for live and paper trading.

    Supports both paper trading (default) and live trading.
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        paper: bool = True,
    ):
        """
        Initialize Alpaca broker adapter.

        Args:
            api_key: Alpaca API key (defaults to env var ALPACA_API_KEY)
            api_secret: Alpaca API secret (defaults to env var ALPACA_API_SECRET)
            paper: Use paper trading (default True)
        """
        self.api_key = api_key or os.getenv("ALPACA_API_KEY")
        self.api_secret = api_secret or os.getenv("ALPACA_API_SECRET")
        self.paper = paper

        if not self.api_key or not self.api_secret:
            raise ValueError(
                "Alpaca API credentials required. Set ALPACA_API_KEY and "
                "ALPACA_API_SECRET environment variables."
            )

        # Initialize trading client
        self._trading_client = TradingClient(
            api_key=self.api_key,
            secret_key=self.api_secret,
            paper=self.paper,
        )

        # Initialize data client for quotes
        self._data_client = StockHistoricalDataClient(
            api_key=self.api_key,
            secret_key=self.api_secret,
        )

        self._fills: list[Fill] = []
        self._connected = False

    async def connect(self) -> bool:
        """Test connection to Alpaca."""
        try:
            account = self._trading_client.get_account()
            self._connected = account is not None
            if self._connected:
                logger.info(
                    f"Connected to Alpaca {'paper' if self.paper else 'live'} account: "
                    f"${float(account.equity):,.2f} equity"
                )
            return self._connected
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca: {e}")
            return False

    async def submit_order(self, order: Order) -> dict[str, Any]:
        """
        Submit order to Alpaca.

        Args:
            order: Order to submit

        Returns:
            Response dict with success/error and broker order ID
        """
        try:
            # Map side
            side = OrderSide.BUY if order.side.lower() == "buy" else OrderSide.SELL

            # Create order request based on type
            if order.order_type.value == "market":
                order_request = MarketOrderRequest(
                    symbol=order.symbol,
                    qty=order.quantity,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                )
            elif order.order_type.value == "limit":
                order_request = LimitOrderRequest(
                    symbol=order.symbol,
                    qty=order.quantity,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                    limit_price=order.limit_price,
                )
            else:
                return {
                    "success": False,
                    "error": f"Unsupported order type: {order.order_type.value}",
                }

            # Submit order
            alpaca_order = self._trading_client.submit_order(order_request)

            return {
                "success": True,
                "broker_order_id": alpaca_order.id,
                "status": alpaca_order.status.value,
                "submitted_at": alpaca_order.submitted_at.isoformat()
                if alpaca_order.submitted_at
                else None,
            }

        except Exception as e:
            logger.error(f"Failed to submit order: {e}")
            return {"success": False, "error": str(e)}

    async def cancel_order(self, broker_order_id: str) -> dict[str, Any]:
        """Cancel an order."""
        try:
            self._trading_client.cancel_order_by_id(broker_order_id)
            return {"success": True, "broker_order_id": broker_order_id}
        except Exception as e:
            logger.error(f"Failed to cancel order {broker_order_id}: {e}")
            return {"success": False, "error": str(e)}

    async def get_order_status(self, broker_order_id: str) -> dict[str, Any]:
        """Get order status from Alpaca."""
        try:
            order = self._trading_client.get_order_by_id(broker_order_id)
            return {
                "broker_order_id": order.id,
                "status": order.status.value,
                "filled_qty": float(order.filled_qty) if order.filled_qty else 0,
                "filled_avg_price": float(order.filled_avg_price)
                if order.filled_avg_price
                else None,
                "created_at": order.created_at.isoformat() if order.created_at else None,
                "filled_at": order.filled_at.isoformat() if order.filled_at else None,
            }
        except Exception as e:
            logger.error(f"Failed to get order status: {e}")
            return {"error": str(e)}

    async def get_account(self) -> dict[str, Any]:
        """Get account information."""
        try:
            account = self._trading_client.get_account()
            return {
                "account_id": account.id,
                "cash": float(account.cash),
                "buying_power": float(account.buying_power),
                "equity": float(account.equity),  # Add equity alias
                "total_equity": float(account.equity),
                "total_position_value": float(account.long_market_value)
                + float(account.short_market_value),
                "status": account.status.value,  # Add status
                "pattern_day_trader": account.pattern_day_trader,
                "trading_blocked": account.trading_blocked,
                "transfers_blocked": account.transfers_blocked,
                "account_blocked": account.account_blocked,
            }
        except Exception as e:
            logger.error(f"Failed to get account: {e}")
            return {"error": str(e)}

    async def get_positions(self) -> list[dict[str, Any]]:
        """Get all positions."""
        try:
            positions = self._trading_client.get_all_positions()
            return [
                {
                    "symbol": pos.symbol,
                    "quantity": int(pos.qty),
                    "avg_price": float(pos.avg_entry_price),
                    "current_price": float(pos.current_price),
                    "market_value": float(pos.market_value),
                    "unrealized_pnl": float(pos.unrealized_pl),
                    "unrealized_pnl_pct": float(pos.unrealized_plpc) * 100,
                    "side": "long" if float(pos.qty) > 0 else "short",
                }
                for pos in positions
            ]
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []

    async def get_quote(self, symbol: str) -> dict[str, Any] | None:
        """Get latest quote for a symbol."""
        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quotes = self._data_client.get_stock_latest_quote(request)

            if symbol in quotes:
                quote = quotes[symbol]
                return {
                    "symbol": symbol,
                    "bid": float(quote.bid_price),
                    "ask": float(quote.ask_price),
                    "bid_size": int(quote.bid_size),
                    "ask_size": int(quote.ask_size),
                    "timestamp": quote.timestamp.isoformat(),
                }
            return None
        except Exception as e:
            logger.error(f"Failed to get quote for {symbol}: {e}")
            return None

    def get_fills(self) -> list[Fill]:
        """Get list of fills (from orders that have been filled)."""
        return self._fills

    def reset(self, starting_cash: float = 100000.0) -> None:
        """
        Reset is not applicable for live/paper trading with a real broker.
        This method exists for interface compatibility.
        """
        logger.warning("Reset not applicable for live Alpaca trading")
