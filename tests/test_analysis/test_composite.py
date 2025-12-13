"""Tests for composite indicator aggregation."""

from ordinis.analysis.technical.composite import CompositeIndicator


def test_weighted_sum_normalizes():
    values = {"a": 1.0, "b": 3.0}
    weights = {"a": 1.0, "b": 3.0}
    result = CompositeIndicator.weighted_sum(values, weights=weights, normalize=True)
    assert result.method == "weighted_sum"
    assert abs(result.value - 2.5) < 1e-6  # (1*1 + 3*3) / (1+3)


def test_vote_tie_returns_neutral():
    signals = ["buy", "sell"]
    result = CompositeIndicator.vote(signals)
    assert result.value == "neutral"


def test_min_max():
    vals = [3.0, 1.0, 5.0]
    assert CompositeIndicator.min_value(vals).value == 1.0
    assert CompositeIndicator.max_value(vals).value == 5.0
