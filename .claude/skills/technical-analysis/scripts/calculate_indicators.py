#!/usr/bin/env python3
"""
Technical Indicators Calculation Script

Production-ready implementation of twelve core technical indicators for market analysis.
All calculations follow established methodologies from CMT curriculum and authoritative sources.

Usage:
    python calculate_indicators.py --symbol SPY --indicator RSI --period 14
    python calculate_indicators.py --symbol QQQ --indicator MACD
    python calculate_indicators.py --symbol AAPL --indicator BOLLINGER --period 20 --std-dev 2.0

Author: ordinis-1 Technical Analysis Framework
"""

import argparse
import sys
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Attempt to import optional libraries
try:
    from ta import trend, momentum, volatility, volume
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    print("Warning: 'ta' library not available. Using manual calculations.")


class TechnicalIndicators:
    """
    Comprehensive technical indicator calculations with production-grade error handling.
    
    All methods follow established calculation methodologies and include extensive
    validation and type hints for enterprise deployment.
    """
    
    @staticmethod
    def validate_series(data: pd.Series, min_length: int = 2) -> None:
        """Validate input series for calculations."""
        if data is None or len(data) < min_length:
            raise ValueError(f"Data series must contain at least {min_length} values")
        if data.isnull().any():
            raise ValueError("Data series contains null values")
    
    @staticmethod
    def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            close: Closing prices
            period: Lookback period (default 14)
            
        Returns:
            RSI values (0-100 scale)
        """
        TechnicalIndicators.validate_series(close, min_length=period + 1)
        
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def calculate_macd(close: pd.Series, fast: int = 12, slow: int = 26, 
                      signal: int = 9) -> pd.DataFrame:
        """
        Calculate Moving Average Convergence Divergence (MACD).
        
        Args:
            close: Closing prices
            fast: Fast EMA period (default 12)
            slow: Slow EMA period (default 26)
            signal: Signal line period (default 9)
            
        Returns:
            DataFrame with MACD, Signal, and Histogram
        """
        TechnicalIndicators.validate_series(close, min_length=slow + signal)
        
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return pd.DataFrame({
            'MACD': macd_line,
            'Signal': signal_line,
            'Histogram': histogram
        })
    
    @staticmethod
    def calculate_bollinger_bands(close: pd.Series, period: int = 20, 
                                 std_dev: float = 2.0) -> pd.DataFrame:
        """
        Calculate Bollinger Bands.
        
        Args:
            close: Closing prices
            period: Moving average period (default 20)
            std_dev: Standard deviation multiplier (default 2.0)
            
        Returns:
            DataFrame with Upper, Middle, Lower bands and %B
        """
        TechnicalIndicators.validate_series(close, min_length=period)
        
        middle_band = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        upper_band = middle_band + (std_dev * std)
        lower_band = middle_band - (std_dev * std)
        
        # %B indicator (position within bands)
        percent_b = (close - lower_band) / (upper_band - lower_band)
        
        # Bandwidth
        bandwidth = (upper_band - lower_band) / middle_band
        
        return pd.DataFrame({
            'Upper': upper_band,
            'Middle': middle_band,
            'Lower': lower_band,
            '%B': percent_b,
            'Bandwidth': bandwidth
        })
    
    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, 
                     period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR).
        
        Args:
            high: High prices
            low: Low prices
            close: Closing prices
            period: Lookback period (default 14)
            
        Returns:
            ATR values
        """
        TechnicalIndicators.validate_series(high, min_length=period + 1)
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    @staticmethod
    def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, 
                     period: int = 14) -> pd.DataFrame:
        """
        Calculate Average Directional Index (ADX) with +DI and -DI.
        
        Args:
            high: High prices
            low: Low prices
            close: Closing prices
            period: Lookback period (default 14)
            
        Returns:
            DataFrame with ADX, +DI, -DI
        """
        TechnicalIndicators.validate_series(high, min_length=period * 2)
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        # Smooth components
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        # ADX calculation
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return pd.DataFrame({
            'ADX': adx,
            '+DI': plus_di,
            '-DI': minus_di
        })
    
    @staticmethod
    def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                           k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """
        Calculate Stochastic Oscillator.
        
        Args:
            high: High prices
            low: Low prices
            close: Closing prices
            k_period: %K period (default 14)
            d_period: %D period (default 3)
            
        Returns:
            DataFrame with %K and %D
        """
        TechnicalIndicators.validate_series(high, min_length=k_period + d_period)
        
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        percent_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        percent_d = percent_k.rolling(window=d_period).mean()
        
        return pd.DataFrame({
            '%K': percent_k,
            '%D': percent_d
        })
    
    @staticmethod
    def calculate_cci(high: pd.Series, low: pd.Series, close: pd.Series, 
                     period: int = 20) -> pd.Series:
        """
        Calculate Commodity Channel Index (CCI).
        
        Args:
            high: High prices
            low: Low prices
            close: Closing prices
            period: Lookback period (default 20)
            
        Returns:
            CCI values
        """
        TechnicalIndicators.validate_series(high, min_length=period)
        
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mean_deviation = (typical_price - sma_tp).abs().rolling(window=period).mean()
        cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
        
        return cci
    
    @staticmethod
    def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Calculate On-Balance Volume (OBV).
        
        Args:
            close: Closing prices
            volume: Volume
            
        Returns:
            OBV values
        """
        TechnicalIndicators.validate_series(close, min_length=2)
        TechnicalIndicators.validate_series(volume, min_length=2)
        
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]
        
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    @staticmethod
    def calculate_moving_averages(close: pd.Series, 
                                 periods: list = [20, 50, 200]) -> pd.DataFrame:
        """
        Calculate Simple and Exponential Moving Averages.
        
        Args:
            close: Closing prices
            periods: List of periods to calculate (default [20, 50, 200])
            
        Returns:
            DataFrame with SMA and EMA for each period
        """
        TechnicalIndicators.validate_series(close, min_length=max(periods))
        
        result = pd.DataFrame(index=close.index)
        
        for period in periods:
            result[f'SMA_{period}'] = close.rolling(window=period).mean()
            result[f'EMA_{period}'] = close.ewm(span=period, adjust=False).mean()
        
        return result
    
    @staticmethod
    def calculate_parabolic_sar(high: pd.Series, low: pd.Series, 
                               af_start: float = 0.02, af_increment: float = 0.02,
                               af_max: float = 0.20) -> pd.Series:
        """
        Calculate Parabolic SAR.
        
        Args:
            high: High prices
            low: Low prices
            af_start: Initial acceleration factor (default 0.02)
            af_increment: AF increment (default 0.02)
            af_max: Maximum AF (default 0.20)
            
        Returns:
            SAR values
        """
        TechnicalIndicators.validate_series(high, min_length=2)
        
        length = len(high)
        sar = np.zeros(length)
        ep = np.zeros(length)
        af = np.zeros(length)
        trend = np.zeros(length, dtype=int)
        
        # Initialize
        sar[0] = low.iloc[0]
        ep[0] = high.iloc[0]
        af[0] = af_start
        trend[0] = 1  # 1 for uptrend, -1 for downtrend
        
        for i in range(1, length):
            # Calculate SAR
            sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
            
            # Check for reversal
            if trend[i-1] == 1:  # Uptrend
                if low.iloc[i] < sar[i]:
                    # Reverse to downtrend
                    trend[i] = -1
                    sar[i] = ep[i-1]
                    ep[i] = low.iloc[i]
                    af[i] = af_start
                else:
                    trend[i] = 1
                    if high.iloc[i] > ep[i-1]:
                        ep[i] = high.iloc[i]
                        af[i] = min(af[i-1] + af_increment, af_max)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
            else:  # Downtrend
                if high.iloc[i] > sar[i]:
                    # Reverse to uptrend
                    trend[i] = 1
                    sar[i] = ep[i-1]
                    ep[i] = high.iloc[i]
                    af[i] = af_start
                else:
                    trend[i] = -1
                    if low.iloc[i] < ep[i-1]:
                        ep[i] = low.iloc[i]
                        af[i] = min(af[i-1] + af_increment, af_max)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
        
        return pd.Series(sar, index=high.index)
    
    @staticmethod
    def calculate_fibonacci_retracement(high: float, low: float, 
                                       trend_direction: str = 'up') -> Dict[str, float]:
        """
        Calculate Fibonacci retracement levels.
        
        Args:
            high: Swing high price
            low: Swing low price
            trend_direction: 'up' for uptrend, 'down' for downtrend
            
        Returns:
            Dictionary of retracement levels
        """
        diff = high - low
        
        if trend_direction == 'up':
            levels = {
                '0.0%': high,
                '23.6%': high - (diff * 0.236),
                '38.2%': high - (diff * 0.382),
                '50.0%': high - (diff * 0.500),
                '61.8%': high - (diff * 0.618),
                '78.6%': high - (diff * 0.786),
                '100.0%': low
            }
        else:
            levels = {
                '0.0%': low,
                '23.6%': low + (diff * 0.236),
                '38.2%': low + (diff * 0.382),
                '50.0%': low + (diff * 0.500),
                '61.8%': low + (diff * 0.618),
                '78.6%': low + (diff * 0.786),
                '100.0%': high
            }
        
        return levels


def fetch_data(symbol: str, days: int = 365) -> pd.DataFrame:
    """
    Fetch historical OHLCV data for symbol.
    
    Args:
        symbol: Stock symbol
        days: Number of days of historical data
        
    Returns:
        DataFrame with OHLCV data
    """
    try:
        import yfinance as yf
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        data = yf.download(symbol, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            raise ValueError(f"No data retrieved for symbol {symbol}")
        
        return data
    except ImportError:
        print("Error: yfinance library not installed. Install with: pip install yfinance")
        sys.exit(1)
    except Exception as e:
        print(f"Error fetching data: {e}")
        sys.exit(1)


def interpret_indicator(indicator_name: str, values: pd.DataFrame, 
                       latest: pd.Series) -> str:
    """Generate interpretation text for indicator values."""
    interpretations = {
        'RSI': f"""
RSI Interpretation:
Current: {latest['RSI']:.2f}
- Overbought (>70): {'Yes' if latest['RSI'] > 70 else 'No'}
- Oversold (<30): {'Yes' if latest['RSI'] < 30 else 'No'}
- Trend: {'Bullish' if latest['RSI'] > 50 else 'Bearish'}
""",
        'MACD': f"""
MACD Interpretation:
MACD: {latest['MACD']:.2f}
Signal: {latest['Signal']:.2f}
Histogram: {latest['Histogram']:.2f}
- Crossover: {'Bullish' if latest['MACD'] > latest['Signal'] else 'Bearish'}
- Zero Line: {'Above (Bullish)' if latest['MACD'] > 0 else 'Below (Bearish)'}
- Momentum: {'Increasing' if latest['Histogram'] > 0 else 'Decreasing'}
""",
        'BOLLINGER': f"""
Bollinger Bands Interpretation:
Price: {latest['Close']:.2f}
Upper: {latest['Upper']:.2f}
Middle: {latest['Middle']:.2f}
Lower: {latest['Lower']:.2f}
%B: {latest['%B']:.2f}
- Position: {'Above upper band' if latest['%B'] > 1 else 'Below lower band' if latest['%B'] < 0 else 'Within bands'}
- Volatility: {'High' if latest['Bandwidth'] > values['Bandwidth'].quantile(0.75) else 'Normal'}
"""
    }
    
    return interpretations.get(indicator_name, "Interpretation not available")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Calculate technical indicators for market analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--symbol', type=str, required=True, 
                       help='Stock symbol (e.g., SPY, QQQ, AAPL)')
    parser.add_argument('--indicator', type=str, required=True,
                       choices=['RSI', 'MACD', 'BOLLINGER', 'ATR', 'ADX', 
                               'STOCHASTIC', 'CCI', 'OBV', 'MA', 'PSAR', 'FIBONACCI'],
                       help='Indicator to calculate')
    parser.add_argument('--period', type=int, default=14,
                       help='Period for calculation (default: 14)')
    parser.add_argument('--std-dev', type=float, default=2.0,
                       help='Standard deviation multiplier for Bollinger Bands (default: 2.0)')
    parser.add_argument('--days', type=int, default=365,
                       help='Days of historical data to fetch (default: 365)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path for results (CSV format)')
    
    args = parser.parse_args()
    
    print(f"\nFetching data for {args.symbol}...")
    data = fetch_data(args.symbol, args.days)
    
    calc = TechnicalIndicators()
    
    print(f"Calculating {args.indicator}...")
    
    # Calculate requested indicator
    if args.indicator == 'RSI':
        result = calc.calculate_rsi(data['Close'], args.period)
        data['RSI'] = result
        print(interpret_indicator('RSI', data, data.iloc[-1]))
    
    elif args.indicator == 'MACD':
        result = calc.calculate_macd(data['Close'])
        for col in result.columns:
            data[col] = result[col]
        print(interpret_indicator('MACD', data, data.iloc[-1]))
    
    elif args.indicator == 'BOLLINGER':
        result = calc.calculate_bollinger_bands(data['Close'], args.period, args.std_dev)
        for col in result.columns:
            data[col] = result[col]
        print(interpret_indicator('BOLLINGER', data, data.iloc[-1]))
    
    elif args.indicator == 'ATR':
        result = calc.calculate_atr(data['High'], data['Low'], data['Close'], args.period)
        data['ATR'] = result
        print(f"\nATR: {data['ATR'].iloc[-1]:.2f}")
        print(f"ATR%: {(data['ATR'].iloc[-1] / data['Close'].iloc[-1] * 100):.2f}%")
    
    elif args.indicator == 'ADX':
        result = calc.calculate_adx(data['High'], data['Low'], data['Close'], args.period)
        for col in result.columns:
            data[col] = result[col]
        print(f"\nADX: {data['ADX'].iloc[-1]:.2f}")
        print(f"Trend Strength: {'Strong' if data['ADX'].iloc[-1] > 25 else 'Weak'}")
    
    elif args.indicator == 'STOCHASTIC':
        result = calc.calculate_stochastic(data['High'], data['Low'], data['Close'], args.period)
        for col in result.columns:
            data[col] = result[col]
        print(f"\n%K: {data['%K'].iloc[-1]:.2f}")
        print(f"%D: {data['%D'].iloc[-1]:.2f}")
        print(f"Overbought (>80): {'Yes' if data['%K'].iloc[-1] > 80 else 'No'}")
        print(f"Oversold (<20): {'Yes' if data['%K'].iloc[-1] < 20 else 'No'}")
    
    elif args.indicator == 'CCI':
        result = calc.calculate_cci(data['High'], data['Low'], data['Close'], args.period)
        data['CCI'] = result
        print(f"\nCCI: {data['CCI'].iloc[-1]:.2f}")
        print(f"Overbought (>100): {'Yes' if data['CCI'].iloc[-1] > 100 else 'No'}")
        print(f"Oversold (<-100): {'Yes' if data['CCI'].iloc[-1] < -100 else 'No'}")
    
    elif args.indicator == 'OBV':
        result = calc.calculate_obv(data['Close'], data['Volume'])
        data['OBV'] = result
        print(f"\nOBV: {data['OBV'].iloc[-1]:,.0f}")
    
    elif args.indicator == 'MA':
        result = calc.calculate_moving_averages(data['Close'])
        for col in result.columns:
            data[col] = result[col]
        print("\nMoving Averages:")
        for col in result.columns:
            print(f"{col}: {data[col].iloc[-1]:.2f}")
    
    elif args.indicator == 'PSAR':
        result = calc.calculate_parabolic_sar(data['High'], data['Low'])
        data['PSAR'] = result
        print(f"\nParabolic SAR: {data['PSAR'].iloc[-1]:.2f}")
        print(f"Position: {'Below price (Bullish)' if data['PSAR'].iloc[-1] < data['Close'].iloc[-1] else 'Above price (Bearish)'}")
    
    elif args.indicator == 'FIBONACCI':
        # Calculate from recent swing high/low
        recent_high = data['High'].tail(args.period).max()
        recent_low = data['Low'].tail(args.period).min()
        levels = calc.calculate_fibonacci_retracement(recent_high, recent_low, 'up')
        print("\nFibonacci Retracement Levels:")
        for level, price in levels.items():
            print(f"{level}: ${price:.2f}")
    
    # Save output if requested
    if args.output:
        data.to_csv(args.output)
        print(f"\nResults saved to {args.output}")
    
    print("\nCalculation complete.")


if __name__ == '__main__':
    main()
