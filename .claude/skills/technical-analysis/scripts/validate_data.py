#!/usr/bin/env python3
"""
Data Validation Utility

Validates price data quality and completeness for technical indicator calculation.
"""

import pandas as pd
import numpy as np
import sys
from typing import Tuple, List, Dict
from pathlib import Path


def validate_data(data: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate OHLCV data for indicator calculations.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Price data to validate
    
    Returns:
    --------
    tuple of (is_valid, list of error messages)
    """
    errors = []
    
    # Check required columns
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")
        return False, errors
    
    # Check data types
    for col in required_columns:
        if not pd.api.types.is_numeric_dtype(data[col]):
            errors.append(f"Column {col} is not numeric")
    
    # Check for null values
    null_counts = data[required_columns].isnull().sum()
    if null_counts.any():
        for col, count in null_counts.items():
            if count > 0:
                errors.append(f"Column {col} has {count} null values")
    
    # Check for negative values (except potentially Volume)
    price_cols = ['Open', 'High', 'Low', 'Close']
    for col in price_cols:
        if (data[col] < 0).any():
            errors.append(f"Column {col} contains negative values")
    
    # Check High/Low relationship
    if (data['High'] < data['Low']).any():
        count = (data['High'] < data['Low']).sum()
        errors.append(f"Found {count} rows where High < Low")
    
    # Check OHLC relationships
    if (data['High'] < data['Close']).any():
        count = (data['High'] < data['Close']).sum()
        errors.append(f"Found {count} rows where High < Close")
    
    if (data['Low'] > data['Close']).any():
        count = (data['Low'] > data['Close']).sum()
        errors.append(f"Found {count} rows where Low > Close")
    
    # Check for zero volume
    zero_volume = (data['Volume'] == 0).sum()
    if zero_volume > 0:
        errors.append(f"Warning: {zero_volume} rows with zero volume")
    
    # Check data length
    min_length = 200  # Minimum for most long-term indicators
    if len(data) < min_length:
        errors.append(f"Warning: Only {len(data)} rows. "
                     f"Recommended minimum: {min_length}")
    
    # Check for gaps in index (if datetime)
    if isinstance(data.index, pd.DatetimeIndex):
        gaps = data.index.to_series().diff()[1:]
        median_gap = gaps.median()
        large_gaps = gaps[gaps > median_gap * 5]
        if len(large_gaps) > 0:
            errors.append(f"Warning: Found {len(large_gaps)} large time gaps")
    
    # Check for outliers (price spikes)
    returns = data['Close'].pct_change()
    outliers = returns[abs(returns) > 0.5]  # 50% daily move
    if len(outliers) > 0:
        errors.append(f"Warning: Found {len(outliers)} potential outliers "
                     f"(>50% daily moves)")
    
    is_valid = len([e for e in errors if not e.startswith('Warning')]) == 0
    
    return is_valid, errors


def check_minimum_periods(data: pd.DataFrame) -> Dict[str, bool]:
    """
    Check if data has sufficient periods for each indicator.
    
    Returns:
    --------
    dict with indicator names and whether minimum period is met
    """
    length = len(data)
    
    requirements = {
        'RSI (14)': 14,
        'MACD (26)': 26,
        'Bollinger Bands (20)': 20,
        'ADX (14)': 28,  # Needs smoothing period
        'Ichimoku (52)': 52,
        'Stochastic (14)': 14,
        'CCI (20)': 20,
        'ATR (14)': 14,
        'OBV': 1,
        'Parabolic SAR': 10,
        '50-day MA': 50,
        '200-day MA': 200
    }
    
    results = {}
    for name, required in requirements.items():
        results[name] = length >= required
    
    return results


def main():
    """Command-line interface for data validation."""
    if len(sys.argv) < 2:
        print("Usage: python validate_data.py <data_file.csv>")
        print("\nValidates OHLCV data for technical indicator calculation.")
        sys.exit(1)
    
    file_path = Path(sys.argv[1])
    
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    
    print(f"Validating: {file_path.name}")
    print("=" * 60)
    
    # Load data
    try:
        data = pd.read_csv(file_path, index_col=0, parse_dates=True)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)
    
    print(f"\nData shape: {data.shape[0]} rows, {data.shape[1]} columns")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    
    # Validate
    is_valid, errors = validate_data(data)
    
    if is_valid:
        print("\n✓ Data validation PASSED")
    else:
        print("\n✗ Data validation FAILED")
    
    if errors:
        print("\nIssues found:")
        for i, error in enumerate(errors, 1):
            symbol = "⚠" if error.startswith("Warning") else "✗"
            print(f"  {symbol} {error}")
    
    # Check indicator requirements
    print("\nIndicator Period Requirements:")
    print("-" * 60)
    requirements = check_minimum_periods(data)
    
    for name, met in requirements.items():
        symbol = "✓" if met else "✗"
        print(f"  {symbol} {name}")
    
    # Summary statistics
    print("\nData Summary:")
    print("-" * 60)
    print(data[['Open', 'High', 'Low', 'Close', 'Volume']].describe())
    
    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()
