#!/usr/bin/env python3
"""
Technical Indicators Calculator

Comprehensive implementation of twelve core technical indicators for
quantitative trading and market analysis.

Author: Ordinis-1 Project
License: MIT
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class IndicatorConfig:
    """Configuration for indicator calculations."""

    # Moving Averages
    ma_fast: int = 20
    ma_slow: int = 50
    ma_long: int = 200

    # Momentum
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0

    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    stoch_k: int = 14
    stoch_d: int = 3
    stoch_overbought: float = 80.0
    stoch_oversold: float = 20.0

    cci_period: int = 20

    # Trend
    adx_period: int = 14
    adx_threshold: float = 25.0

    ichimoku_tenkan: int = 9
    ichimoku_kijun: int = 26
    ichimoku_senkou: int = 52

    psar_af_start: float = 0.02
    psar_af_increment: float = 0.02
    psar_af_max: float = 0.20

    # Volatility
    atr_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0

    # Volume
    obv_ma_period: int = 20


class TechnicalIndicators:
    """
    Comprehensive technical indicators calculator.

    Implements twelve core indicators across trend, momentum, volatility,
    and volume categories. All calculations follow established methodologies
    from academic and industry sources.
    """

    def __init__(self, data: pd.DataFrame, config: IndicatorConfig | None = None):
        """
        Initialize calculator with price data.

        Parameters:
        -----------
        data : pd.DataFrame
            Must contain: Open, High, Low, Close, Volume columns
        config : IndicatorConfig, optional
            Configuration for indicator parameters
        """
        self.data = data.copy()
        self.config = config or IndicatorConfig()

        # Validate required columns
        required = ["Open", "High", "Low", "Close", "Volume"]
        missing = [col for col in required if col not in self.data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        self.indicators = {}

    # ========================================================================
    # TREND INDICATORS
    # ========================================================================

    def calculate_adx(self) -> dict[str, pd.Series]:
        """Calculate Average Directional Index."""
        high = self.data["High"]
        low = self.data["Low"]
        close = self.data["Close"]
        period = self.config.adx_period

        # True Range
        high_low = high - low
        high_close = abs(high - close.shift(1))
        low_close = abs(low - close.shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # Directional Movement
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        # Smooth with Wilder's smoothing
        atr = tr.ewm(alpha=1 / period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr)

        # DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(alpha=1 / period, adjust=False).mean()

        result = {"ADX": adx, "+DI": plus_di, "-DI": minus_di}
        self.indicators["adx"] = result
        return result

    def calculate_ichimoku(self) -> dict[str, pd.Series]:
        """Calculate Ichimoku Cloud components."""
        high = self.data["High"]
        low = self.data["Low"]
        close = self.data["Close"]

        # Tenkan-sen (Conversion Line)
        tenkan_sen = (
            high.rolling(self.config.ichimoku_tenkan).max()
            + low.rolling(self.config.ichimoku_tenkan).min()
        ) / 2

        # Kijun-sen (Base Line)
        kijun_sen = (
            high.rolling(self.config.ichimoku_kijun).max()
            + low.rolling(self.config.ichimoku_kijun).min()
        ) / 2

        # Senkou Span A (Leading Span A)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(self.config.ichimoku_kijun)

        # Senkou Span B (Leading Span B)
        senkou_span_b = (
            (
                high.rolling(self.config.ichimoku_senkou).max()
                + low.rolling(self.config.ichimoku_senkou).min()
            )
            / 2
        ).shift(self.config.ichimoku_kijun)

        # Chikou Span (Lagging Span)
        chikou_span = close.shift(-self.config.ichimoku_kijun)

        result = {
            "tenkan_sen": tenkan_sen,
            "kijun_sen": kijun_sen,
            "senkou_span_a": senkou_span_a,
            "senkou_span_b": senkou_span_b,
            "chikou_span": chikou_span,
        }
        self.indicators["ichimoku"] = result
        return result

    def calculate_moving_averages(self) -> dict[str, pd.Series]:
        """Calculate multiple moving average types."""
        close = self.data["Close"]
        volume = self.data["Volume"]

        result = {
            "SMA_fast": close.rolling(self.config.ma_fast).mean(),
            "SMA_slow": close.rolling(self.config.ma_slow).mean(),
            "SMA_long": close.rolling(self.config.ma_long).mean(),
            "EMA_fast": close.ewm(span=self.config.ma_fast, adjust=False).mean(),
            "EMA_slow": close.ewm(span=self.config.ma_slow, adjust=False).mean(),
            "EMA_long": close.ewm(span=self.config.ma_long, adjust=False).mean(),
        }

        # VWAP (intraday, simplified)
        typical_price = (self.data["High"] + self.data["Low"] + close) / 3
        result["VWAP"] = (typical_price * volume).cumsum() / volume.cumsum()

        self.indicators["ma"] = result
        return result

    def calculate_parabolic_sar(self) -> pd.Series:
        """Calculate Parabolic SAR."""
        high = self.data["High"].values
        low = self.data["Low"].values
        close = self.data["Close"].values

        length = len(close)
        sar = np.zeros(length)
        trend = np.ones(length)
        ep = np.zeros(length)
        af = np.zeros(length)

        # Initialize
        sar[0] = low[0]
        trend[0] = 1
        ep[0] = high[0]
        af[0] = self.config.psar_af_start

        for i in range(1, length):
            # Calculate new SAR
            sar[i] = sar[i - 1] + af[i - 1] * (ep[i - 1] - sar[i - 1])

            # Check for reversal
            if trend[i - 1] == 1:  # Uptrend
                if low[i] < sar[i]:
                    # Reversal to downtrend
                    trend[i] = -1
                    sar[i] = ep[i - 1]
                    ep[i] = low[i]
                    af[i] = self.config.psar_af_start
                else:
                    trend[i] = 1
                    if high[i] > ep[i - 1]:
                        ep[i] = high[i]
                        af[i] = min(
                            af[i - 1] + self.config.psar_af_increment, self.config.psar_af_max
                        )
                    else:
                        ep[i] = ep[i - 1]
                        af[i] = af[i - 1]
            elif high[i] > sar[i]:
                # Reversal to uptrend
                trend[i] = 1
                sar[i] = ep[i - 1]
                ep[i] = high[i]
                af[i] = self.config.psar_af_start
            else:
                trend[i] = -1
                if low[i] < ep[i - 1]:
                    ep[i] = low[i]
                    af[i] = min(af[i - 1] + self.config.psar_af_increment, self.config.psar_af_max)
                else:
                    ep[i] = ep[i - 1]
                    af[i] = af[i - 1]

        result = pd.Series(sar, index=self.data.index, name="PSAR")
        self.indicators["psar"] = result
        return result

    # ========================================================================
    # MOMENTUM INDICATORS
    # ========================================================================

    def calculate_macd(self) -> dict[str, pd.Series]:
        """Calculate MACD indicator."""
        close = self.data["Close"]

        # EMAs
        ema_fast = close.ewm(span=self.config.macd_fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.config.macd_slow, adjust=False).mean()

        # MACD Line
        macd_line = ema_fast - ema_slow

        # Signal Line
        signal_line = macd_line.ewm(span=self.config.macd_signal, adjust=False).mean()

        # Histogram
        histogram = macd_line - signal_line

        result = {"MACD": macd_line, "Signal": signal_line, "Histogram": histogram}
        self.indicators["macd"] = result
        return result

    def calculate_rsi(self) -> pd.Series:
        """Calculate Relative Strength Index."""
        close = self.data["Close"]
        period = self.config.rsi_period

        # Price changes
        delta = close.diff()

        # Gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)

        # Average gain and loss
        avg_gain = gains.ewm(alpha=1 / period, adjust=False).mean()
        avg_loss = losses.ewm(alpha=1 / period, adjust=False).mean()

        # RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        self.indicators["rsi"] = rsi
        return rsi

    def calculate_stochastic(self) -> dict[str, pd.Series]:
        """Calculate Stochastic Oscillator."""
        high = self.data["High"]
        low = self.data["Low"]
        close = self.data["Close"]

        k_period = self.config.stoch_k
        d_period = self.config.stoch_d

        # %K
        lowest_low = low.rolling(k_period).min()
        highest_high = high.rolling(k_period).max()
        k = 100 * (close - lowest_low) / (highest_high - lowest_low)

        # %D (smoothed %K)
        d = k.rolling(d_period).mean()

        result = {"%K": k, "%D": d}
        self.indicators["stochastic"] = result
        return result

    def calculate_cci(self) -> pd.Series:
        """Calculate Commodity Channel Index."""
        high = self.data["High"]
        low = self.data["Low"]
        close = self.data["Close"]
        period = self.config.cci_period

        # Typical Price
        tp = (high + low + close) / 3

        # SMA of Typical Price
        sma_tp = tp.rolling(period).mean()

        # Mean Deviation
        mean_deviation = (tp - sma_tp).abs().rolling(period).mean()

        # CCI
        cci = (tp - sma_tp) / (0.015 * mean_deviation)

        self.indicators["cci"] = cci
        return cci

    # ========================================================================
    # VOLATILITY INDICATORS
    # ========================================================================

    def calculate_atr(self) -> pd.Series:
        """Calculate Average True Range."""
        high = self.data["High"]
        low = self.data["Low"]
        close = self.data["Close"]
        period = self.config.atr_period

        # True Range components
        high_low = high - low
        high_close = abs(high - close.shift(1))
        low_close = abs(low - close.shift(1))

        # True Range
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # ATR
        atr = tr.ewm(alpha=1 / period, adjust=False).mean()

        self.indicators["atr"] = atr
        return atr

    def calculate_bollinger_bands(self) -> dict[str, pd.Series]:
        """Calculate Bollinger Bands."""
        close = self.data["Close"]
        period = self.config.bb_period
        std_dev = self.config.bb_std

        # Middle band (SMA)
        middle = close.rolling(period).mean()

        # Standard deviation
        std = close.rolling(period).std()

        # Upper and lower bands
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)

        # Bandwidth
        bandwidth = (upper - lower) / middle

        # %B
        percent_b = (close - lower) / (upper - lower)

        result = {
            "upper": upper,
            "middle": middle,
            "lower": lower,
            "bandwidth": bandwidth,
            "%B": percent_b,
        }
        self.indicators["bb"] = result
        return result

    # ========================================================================
    # VOLUME INDICATORS
    # ========================================================================

    def calculate_obv(self) -> pd.Series:
        """Calculate On-Balance Volume."""
        close = self.data["Close"]
        volume = self.data["Volume"]

        # Direction
        direction = np.sign(close.diff())

        # OBV
        obv = (direction * volume).cumsum()

        self.indicators["obv"] = obv
        return obv

    # ========================================================================
    # STATIC PRICE LEVELS
    # ========================================================================

    def calculate_fibonacci_levels(self, lookback: int = 100) -> dict:
        """Calculate Fibonacci retracement levels."""
        high = self.data["High"].rolling(lookback).max().iloc[-1]
        low = self.data["Low"].rolling(lookback).min().iloc[-1]

        # Determine direction
        high_idx = (
            self.data["High"].rolling(lookback).apply(lambda x: x.idxmax(), raw=False).iloc[-1]
        )
        low_idx = self.data["Low"].rolling(lookback).apply(lambda x: x.idxmin(), raw=False).iloc[-1]

        direction = "up" if high_idx > low_idx else "down"
        diff = high - low

        if direction == "up":
            levels = {
                "0.0%": high,
                "23.6%": high - diff * 0.236,
                "38.2%": high - diff * 0.382,
                "50.0%": high - diff * 0.500,
                "61.8%": high - diff * 0.618,
                "78.6%": high - diff * 0.786,
                "100.0%": low,
            }
        else:
            levels = {
                "0.0%": low,
                "23.6%": low + diff * 0.236,
                "38.2%": low + diff * 0.382,
                "50.0%": low + diff * 0.500,
                "61.8%": low + diff * 0.618,
                "78.6%": low + diff * 0.786,
                "100.0%": high,
            }

        result = {"levels": levels, "swing_high": high, "swing_low": low, "direction": direction}
        self.indicators["fibonacci"] = result
        return result

    # ========================================================================
    # COMPREHENSIVE ANALYSIS
    # ========================================================================

    def calculate_all_indicators(self) -> dict:
        """Calculate all indicators at once."""
        results = {
            "trend": {
                "adx": self.calculate_adx(),
                "ichimoku": self.calculate_ichimoku(),
                "ma": self.calculate_moving_averages(),
                "psar": self.calculate_parabolic_sar(),
            },
            "momentum": {
                "macd": self.calculate_macd(),
                "rsi": self.calculate_rsi(),
                "stochastic": self.calculate_stochastic(),
                "cci": self.calculate_cci(),
            },
            "volatility": {"atr": self.calculate_atr(), "bb": self.calculate_bollinger_bands()},
            "volume": {"obv": self.calculate_obv()},
            "fibonacci": self.calculate_fibonacci_levels(),
        }

        return results

    def generate_signals(self, indicators: dict | None = None) -> dict[str, str]:
        """
        Generate trading signals from indicators.

        Returns:
        --------
        dict with signal for each indicator category:
        BULLISH, BEARISH, NEUTRAL, or EXTREME
        """
        if indicators is None:
            indicators = self.calculate_all_indicators()

        signals = {}

        # Trend signals
        if "adx" in self.indicators:
            adx = self.indicators["adx"]
            if adx["ADX"].iloc[-1] > self.config.adx_threshold:
                if adx["+DI"].iloc[-1] > adx["-DI"].iloc[-1]:
                    signals["adx"] = "BULLISH"
                else:
                    signals["adx"] = "BEARISH"
            else:
                signals["adx"] = "NEUTRAL"

        # Momentum signals
        if "rsi" in self.indicators:
            rsi = self.indicators["rsi"].iloc[-1]
            if rsi > self.config.rsi_overbought:
                signals["rsi"] = "EXTREME_OVERBOUGHT"
            elif rsi < self.config.rsi_oversold:
                signals["rsi"] = "EXTREME_OVERSOLD"
            elif rsi > 50:
                signals["rsi"] = "BULLISH"
            else:
                signals["rsi"] = "BEARISH"

        if "macd" in self.indicators:
            macd = self.indicators["macd"]
            if macd["MACD"].iloc[-1] > macd["Signal"].iloc[-1]:
                signals["macd"] = "BULLISH"
            else:
                signals["macd"] = "BEARISH"

        return signals


if __name__ == "__main__":
    # Example usage
    import yfinance as yf

    # Download sample data
    ticker = yf.Ticker("SPY")
    data = ticker.history(period="1y")

    # Calculate indicators
    ti = TechnicalIndicators(data)
    results = ti.calculate_all_indicators()

    # Generate signals
    signals = ti.generate_signals(results)

    print("Current Signals:")
    for indicator, signal in signals.items():
        print(f"{indicator.upper()}: {signal}")
