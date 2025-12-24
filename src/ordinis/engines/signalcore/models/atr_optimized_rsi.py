"""
ATR-Optimized RSI Model - Best performing configuration from optimization.

Key improvements over standard RSI:
1. ATR-based stops (adaptive to volatility)
2. ATR-based take profit (adaptive targets)
3. Optimized thresholds: RSI < 35 entry, RSI > 50 exit
4. Per-stock tuned parameters

Optimization Results (10-day intraday data):
- DKNG: +6.9% | 63% WR | 38 trades | PF 1.85
- AMD:  +5.8% | 57% WR | 69 trades | PF 1.58
- COIN: +12.3% | 59% WR | 75 trades | PF 1.79
- PORTFOLIO: +25.0% across 182 trades

Best Parameters Found:
- RSI oversold: 35 (more relaxed than standard 30)
- ATR stop multiplier: 1.5x (tighter than 2x)
- ATR take profit: varies by stock (1.5x - 3.0x)
"""

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from ..core.model import Model, ModelConfig
from ..core.signal import Direction, Signal, SignalType
from ..features.technical import TechnicalIndicators


@dataclass
class OptimizedConfig:
    """Per-stock optimized configuration."""

    rsi_oversold: int = 35
    rsi_exit: int = 50
    atr_stop_mult: float = 1.5
    atr_tp_mult: float = 2.0

    # New tunable parameters
    atr_scale: float = 1.0
    require_volume_confirmation: bool = False
    volume_mean_period: int = 10
    enforce_regime_gate: bool = False
    regime_sma_period: int = 200


# Optimized configs by symbol
OPTIMIZED_CONFIGS = {
    "DKNG": OptimizedConfig(rsi_oversold=35, atr_stop_mult=1.5, atr_tp_mult=2.0),
    "AMD": OptimizedConfig(rsi_oversold=35, atr_stop_mult=1.5, atr_tp_mult=1.5),
    "COIN": OptimizedConfig(rsi_oversold=35, atr_stop_mult=1.5, atr_tp_mult=3.0),
    # Default for unknown symbols
    "DEFAULT": OptimizedConfig(rsi_oversold=35, atr_stop_mult=1.5, atr_tp_mult=2.0),
}


class ATROptimizedRSIModel(Model):
    """
    RSI mean-reversion with ATR-based adaptive stops and targets.

    Entry: RSI < oversold threshold (default 35)
    Exit: RSI > exit threshold (default 50) OR hit stop/target

    Stop Loss: Entry - (ATR * stop_mult)
    Take Profit: Entry + (ATR * tp_mult)
    """

    def __init__(self, config: ModelConfig):
        """Initialize with config."""
        super().__init__(config)

        params = self.config.parameters

        # RSI parameters
        self.rsi_period = params.get("rsi_period", 14)
        self.rsi_oversold = params.get("rsi_oversold", 35)
        self.rsi_exit = params.get("rsi_exit", 50)

        # ATR parameters
        self.atr_period = params.get("atr_period", 14)
        self.atr_stop_mult = params.get("atr_stop_mult", 1.5)
        self.atr_tp_mult = params.get("atr_tp_mult", 2.0)

        # Use per-symbol optimization
        self.use_optimized = params.get("use_optimized", True)

        # Position tracking
        self._position = None
        self._entry_price = None
        self._stop_loss = None
        self._take_profit = None

        # Runtime tunables (can be overridden per-symbol via OptimizedConfig)
        self.atr_scale = params.get("atr_scale", 1.0)
        self.require_volume_confirmation = params.get("require_volume_confirmation", False)
        self.volume_mean_period = params.get("volume_mean_period", 10)
        self.enforce_regime_gate = params.get("enforce_regime_gate", False)
        self.regime_sma_period = params.get("regime_sma_period", 200)

    def _get_config_for_symbol(self, symbol: str) -> OptimizedConfig:
        """Get optimized config for symbol, or default."""
        if self.use_optimized and symbol in OPTIMIZED_CONFIGS:
            return OPTIMIZED_CONFIGS[symbol]
        return OPTIMIZED_CONFIGS.get("DEFAULT", OptimizedConfig())

    def _compute_atr(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Compute Average True Range."""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=self.atr_period).mean()

    def validate(self, data: pd.DataFrame) -> tuple[bool, str]:
        """Validate input data."""
        min_required = max(self.rsi_period, self.atr_period) + 10

        if len(data) < min_required:
            return False, f"Need {min_required} bars, got {len(data)}"

        required = {"open", "high", "low", "close"}
        if not required.issubset(data.columns):
            return False, f"Missing columns: {required - set(data.columns)}"

        return True, ""

    async def generate(self, symbol: str, data: pd.DataFrame, timestamp: datetime) -> Signal | None:
        """
        Generate trading signal.

        Returns Signal with ATR-based stop/target in metadata.
        """
        is_valid, _msg = self.validate(data)
        if not is_valid:
            return None

        # Get per-symbol config
        cfg = self._get_config_for_symbol(symbol)

        # Compute indicators
        close = data["close"]
        high = data["high"]
        low = data["low"]

        rsi = TechnicalIndicators.rsi(close, self.rsi_period)
        atr = self._compute_atr(high, low, close)

        current_price = close.iloc[-1]
        current_rsi = rsi.iloc[-1]
        current_atr = atr.iloc[-1]

        # Check for exit first if in position
        if self._position == "long":
            hit_stop = current_price <= self._stop_loss
            hit_tp = current_price >= self._take_profit
            # Use optimized config if enabled, otherwise model-level param
            rsi_exit_threshold = cfg.rsi_exit if self.use_optimized else self.rsi_exit
            rsi_exit = current_rsi > rsi_exit_threshold

            if hit_stop or hit_tp or rsi_exit:
                reason = "stop_loss" if hit_stop else ("take_profit" if hit_tp else "rsi_exit")
                pnl = (current_price - self._entry_price) / self._entry_price * 100

                self._position = None

                return Signal(
                    symbol=symbol,
                    direction=Direction.SHORT,
                    signal_type=SignalType.EXIT,
                    timestamp=timestamp,
                    price=current_price,
                    confidence=0.8,
                    metadata={
                        "model": "atr_optimized_rsi",
                        "reason": reason,
                        "pnl_pct": pnl,
                        "rsi": current_rsi,
                    },
                )

        # Check for entry
        rsi_oversold_threshold = cfg.rsi_oversold if self.use_optimized else self.rsi_oversold
        if self._position is None and current_rsi < rsi_oversold_threshold:
            # Enforce optional regime gate (e.g., price above long SMA)
            if (self.enforce_regime_gate or cfg.enforce_regime_gate) and "close" in data:
                sma = close.rolling(self.regime_sma_period).mean()
                if len(sma.dropna()) == 0 or current_price <= sma.iloc[-1]:
                    return None

            # Enforce optional volume confirmation (declining volume on pullback)
            if (self.require_volume_confirmation or cfg.require_volume_confirmation) and "volume" in data:
                vol_mean = data["volume"].rolling(self.volume_mean_period).mean()
                if len(vol_mean.dropna()) == 0 or data["volume"].iloc[-1] > vol_mean.iloc[-1]:
                    return None

            effective_atr = current_atr * (cfg.atr_scale if cfg.atr_scale is not None else self.atr_scale)

            self._position = "long"
            self._entry_price = current_price
            self._stop_loss = current_price - (cfg.atr_stop_mult * effective_atr)
            self._take_profit = current_price + (cfg.atr_tp_mult * effective_atr)

            return Signal(
                symbol=symbol,
                direction=Direction.LONG,
                signal_type=SignalType.ENTRY,
                timestamp=timestamp,
                price=current_price,
                confidence=0.7,
                metadata={
                    "model": "atr_optimized_rsi",
                    "rsi": current_rsi,
                    "atr": current_atr,
                    "atr_scale": cfg.atr_scale if cfg.atr_scale is not None else self.atr_scale,
                    "require_volume_confirmation": cfg.require_volume_confirmation if cfg.require_volume_confirmation is not None else self.require_volume_confirmation,
                    "enforce_regime_gate": cfg.enforce_regime_gate if cfg.enforce_regime_gate is not None else self.enforce_regime_gate,
                    "stop_loss": self._stop_loss,
                    "take_profit": self._take_profit,
                    "stop_distance_pct": (current_price - self._stop_loss) / current_price * 100,
                    "target_distance_pct": (self._take_profit - current_price)
                    / current_price
                    * 100,
                },
            )

        return None

    def describe(self) -> dict:
        """Return model description."""
        return {
            "name": "ATR-Optimized RSI",
            "type": "mean_reversion",
            "version": "1.0.0",
            "parameters": {
                "rsi_period": self.rsi_period,
                "rsi_oversold": self.rsi_oversold,
                "rsi_exit": self.rsi_exit,
                "atr_period": self.atr_period,
                "atr_stop_mult": self.atr_stop_mult,
                "atr_tp_mult": self.atr_tp_mult,
            },
            "optimized_symbols": list(OPTIMIZED_CONFIGS.keys()),
        }


def backtest(
    df: pd.DataFrame,
    rsi_os: int = 35,
    rsi_exit: int = 50,
    atr_stop_mult: float = 1.5,
    atr_tp_mult: float = 2.0,
    rsi_period: int = 14,
    atr_period: int = 14,
    atr_scale: float = 1.0,
    require_volume_confirmation: bool = False,
    volume_mean_period: int = 10,
    enforce_regime_gate: bool = False,
    regime_sma_period: int = 200,
) -> dict:
    """
    Run a simple backtest of the ATR-optimized RSI strategy.

    Returns dict with: total_return, win_rate, total_trades, profit_factor, trades
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # Compute RSI
    rsi = TechnicalIndicators.rsi(close, rsi_period)

    # Compute ATR
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=atr_period).mean()

    trades = []
    position = None
    entry_price = None
    stop_loss = None
    take_profit = None

    warmup = max(rsi_period, atr_period) + 5

    for i in range(warmup, len(df)):
        curr_rsi = rsi.iloc[i]
        curr_price = close.iloc[i]
        curr_atr = atr.iloc[i]

        if position is None:
            # Entry
            if curr_rsi < rsi_os:
                # Regime gate (optional)
                if enforce_regime_gate and "close" in df:
                    sma = close.rolling(regime_sma_period).mean()
                    if len(sma.dropna()) == 0 or curr_price <= sma.iloc[-1]:
                        continue

                # Volume confirmation (optional)
                if require_volume_confirmation and "volume" in df:
                    vol_mean = df["volume"].rolling(volume_mean_period).mean()
                    if len(vol_mean.dropna()) == 0 or df["volume"].iloc[i] > vol_mean.iloc[-1]:
                        continue

                effective_atr = curr_atr * atr_scale

                position = "long"
                entry_price = curr_price
                stop_loss = entry_price - (atr_stop_mult * effective_atr)
                take_profit = entry_price + (atr_tp_mult * effective_atr)

        elif position == "long":
            # Exit conditions
            hit_stop = curr_price <= stop_loss
            hit_tp = curr_price >= take_profit
            rsi_signal = curr_rsi > rsi_exit

            if hit_stop or hit_tp or rsi_signal:
                pnl = (curr_price - entry_price) / entry_price * 100
                trades.append(
                    {
                        "entry": entry_price,
                        "exit": curr_price,
                        "pnl": pnl,
                        "bars_held": 0,  # TODO: track
                        "reason": "stop" if hit_stop else ("tp" if hit_tp else "signal"),
                    }
                )
                position = None

    if not trades:
        return {
            "total_return": 0,
            "win_rate": 0,
            "total_trades": 0,
            "profit_factor": 0,
            "avg_trade": 0,
            "trades": [],
        }

    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_return = sum(pnls)
    win_rate = len(wins) / len(pnls) * 100 if pnls else 0
    profit_factor = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else 999

    return {
        "total_return": total_return,
        "win_rate": win_rate,
        "total_trades": len(trades),
        "profit_factor": profit_factor,
        "avg_trade": np.mean(pnls),
        "trades": trades,
    }
