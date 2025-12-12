#!/usr/bin/env python3
"""
Technical Indicators Visualization

Creates comprehensive charts with multiple technical indicators.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.dates import DateFormatter
import sys
from pathlib import Path

try:
    from indicator_calculator import TechnicalIndicators, IndicatorConfig
except ImportError:
    # If run as script, adjust import
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from indicator_calculator import TechnicalIndicators, IndicatorConfig


def plot_comprehensive_analysis(data: pd.DataFrame, 
                                save_path: str = None,
                                title: str = "Technical Analysis"):
    """
    Create comprehensive multi-panel chart with indicators.
    
    Parameters:
    -----------
    data : pd.DataFrame
        OHLCV data
    save_path : str, optional
        Path to save figure
    title : str
        Chart title
    """
    # Calculate indicators
    ti = TechnicalIndicators(data)
    indicators = ti.calculate_all_indicators()
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(6, 1, height_ratios=[3, 1, 1, 1, 1, 1], hspace=0.3)
    
    # Format dates
    date_format = DateFormatter("%Y-%m-%d")
    
    # 1. Price with Bollinger Bands and Moving Averages
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(data.index, data['Close'], label='Close', linewidth=1.5, color='black')
    
    # Bollinger Bands
    bb = indicators['volatility']['bb']
    ax1.plot(data.index, bb['upper'], label='BB Upper', linestyle='--', 
             color='gray', alpha=0.7)
    ax1.plot(data.index, bb['middle'], label='BB Middle', linestyle='--', 
             color='blue', alpha=0.7)
    ax1.plot(data.index, bb['lower'], label='BB Lower', linestyle='--', 
             color='gray', alpha=0.7)
    ax1.fill_between(data.index, bb['upper'], bb['lower'], alpha=0.1, color='gray')
    
    # Moving Averages
    ma = indicators['trend']['ma']
    ax1.plot(data.index, ma['SMA_fast'], label=f'SMA {ti.config.ma_fast}', 
             color='orange', alpha=0.7)
    ax1.plot(data.index, ma['SMA_slow'], label=f'SMA {ti.config.ma_slow}', 
             color='red', alpha=0.7)
    
    # Parabolic SAR
    psar = indicators['trend']['psar']
    # Plot SAR as dots
    uptrend = psar < data['Close']
    downtrend = ~uptrend
    ax1.scatter(data.index[uptrend], psar[uptrend], color='green', 
                s=20, alpha=0.5, label='PSAR Up')
    ax1.scatter(data.index[downtrend], psar[downtrend], color='red', 
                s=20, alpha=0.5, label='PSAR Down')
    
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price', fontsize=10)
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(date_format)
    
    # 2. Volume with OBV
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.bar(data.index, data['Volume'], alpha=0.5, color='blue', label='Volume')
    ax2.set_ylabel('Volume', fontsize=10, color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    
    # OBV on secondary axis
    ax2b = ax2.twinx()
    obv = indicators['volume']['obv']
    ax2b.plot(data.index, obv, color='orange', label='OBV', linewidth=1.5)
    ax2b.set_ylabel('OBV', fontsize=10, color='orange')
    ax2b.tick_params(axis='y', labelcolor='orange')
    ax2.legend(loc='upper left', fontsize=8)
    ax2b.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. MACD
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    macd = indicators['momentum']['macd']
    ax3.plot(data.index, macd['MACD'], label='MACD', color='blue', linewidth=1.5)
    ax3.plot(data.index, macd['Signal'], label='Signal', color='red', linewidth=1.5)
    ax3.bar(data.index, macd['Histogram'], label='Histogram', alpha=0.3, color='gray')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_ylabel('MACD', fontsize=10)
    ax3.legend(loc='upper left', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 4. RSI
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    rsi = ti.calculate_rsi()
    ax4.plot(data.index, rsi, label='RSI', color='purple', linewidth=1.5)
    ax4.axhline(y=70, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax4.axhline(y=50, color='gray', linestyle='-', linewidth=0.5)
    ax4.axhline(y=30, color='green', linestyle='--', linewidth=1, alpha=0.7)
    ax4.fill_between(data.index, 30, 70, alpha=0.1, color='gray')
    ax4.set_ylabel('RSI', fontsize=10)
    ax4.set_ylim(0, 100)
    ax4.legend(loc='upper left', fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # 5. Stochastic
    ax5 = fig.add_subplot(gs[4], sharex=ax1)
    stoch = indicators['momentum']['stochastic']
    ax5.plot(data.index, stoch['%K'], label='%K', color='blue', linewidth=1.5)
    ax5.plot(data.index, stoch['%D'], label='%D', color='red', linewidth=1.5)
    ax5.axhline(y=80, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax5.axhline(y=50, color='gray', linestyle='-', linewidth=0.5)
    ax5.axhline(y=20, color='green', linestyle='--', linewidth=1, alpha=0.7)
    ax5.fill_between(data.index, 20, 80, alpha=0.1, color='gray')
    ax5.set_ylabel('Stochastic', fontsize=10)
    ax5.set_ylim(0, 100)
    ax5.legend(loc='upper left', fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    # 6. ADX
    ax6 = fig.add_subplot(gs[5], sharex=ax1)
    adx = indicators['trend']['adx']
    ax6.plot(data.index, adx['ADX'], label='ADX', color='black', linewidth=2)
    ax6.plot(data.index, adx['+DI'], label='+DI', color='green', linewidth=1.5)
    ax6.plot(data.index, adx['-DI'], label='-DI', color='red', linewidth=1.5)
    ax6.axhline(y=25, color='blue', linestyle='--', linewidth=1, alpha=0.7, 
                label='Trend Threshold')
    ax6.set_ylabel('ADX', fontsize=10)
    ax6.set_xlabel('Date', fontsize=10)
    ax6.legend(loc='upper left', fontsize=8)
    ax6.grid(True, alpha=0.3)
    ax6.xaxis.set_major_formatter(date_format)
    
    # Rotate x-axis labels
    plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Chart saved to: {save_path}")
    else:
        plt.show()
    
    return fig


def main():
    """Command-line interface for visualization."""
    if len(sys.argv) < 2:
        print("Usage: python visualize_indicators.py <data_file.csv> [output.png]")
        print("\nCreates comprehensive technical indicator chart.")
        sys.exit(1)
    
    input_file = Path(sys.argv[1])
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not input_file.exists():
        print(f"Error: File not found: {input_file}")
        sys.exit(1)
    
    print(f"Loading data from: {input_file.name}")
    
    # Load data
    try:
        data = pd.read_csv(input_file, index_col=0, parse_dates=True)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)
    
    print(f"Data loaded: {len(data)} rows")
    print("Calculating indicators and creating chart...")
    
    # Create chart
    plot_comprehensive_analysis(
        data, 
        save_path=output_file,
        title=f"Technical Analysis - {input_file.stem}"
    )
    
    if output_file:
        print("Done!")
    else:
        print("Displaying chart...")


if __name__ == "__main__":
    main()
