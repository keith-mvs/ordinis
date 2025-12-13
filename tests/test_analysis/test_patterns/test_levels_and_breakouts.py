"""Tests for support/resistance levels and breakout detection."""

import pandas as pd

from ordinis.analysis.technical.patterns.breakout import BreakoutDetector
from ordinis.analysis.technical.patterns.support_resistance import SupportResistanceLocator


def test_support_resistance_merges_close_levels():
    high = pd.Series([10, 11, 10.5, 11.2, 11.0, 10.9, 10.8])
    low = pd.Series([9, 9.5, 9.2, 9.4, 9.3, 9.35, 9.4])
    levels = SupportResistanceLocator.find_levels(high, low, window=1, tolerance=0.02)
    assert levels.resistance is not None
    assert levels.support is not None
    assert levels.resistance_touches >= 1
    assert levels.support_touches >= 1


def test_breakout_detection_above_resistance():
    close = pd.Series([10, 10.1, 10.2, 10.25, 10.6])
    signal = BreakoutDetector.detect(close, support=9.8, resistance=10.3, tolerance=0.01)
    assert signal.direction == "bullish"
    assert signal.level == 10.3
    assert signal.confirmed is True


def test_breakdown_detection_below_support():
    close = pd.Series([10, 9.95, 9.9, 9.4])
    signal = BreakoutDetector.detect(close, support=9.8, resistance=None, tolerance=0.01)
    assert signal.direction == "bearish"
    assert signal.level == 9.8
    assert signal.confirmed is True
