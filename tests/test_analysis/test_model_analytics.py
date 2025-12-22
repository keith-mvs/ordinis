"""Tests for model_analytics module.

Tests cover:
- ModelPerformanceRecord dataclass
- ModelMetrics dataclass
- ModelPerformanceAnalyzer class
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
import numpy as np
import pandas as pd

from ordinis.analysis.model_analytics import (
    ModelPerformanceRecord,
    ModelMetrics,
    ModelPerformanceAnalyzer,
)


class TestModelPerformanceRecord:
    """Tests for ModelPerformanceRecord dataclass."""

    @pytest.mark.unit
    def test_create_record_basic(self):
        """Test creating a basic performance record."""
        now = datetime.now()
        record = ModelPerformanceRecord(
            model_id="model_a",
            signal_timestamp=now,
            signal_score=0.8,
            signal_direction="long",
            entry_price=100.0,
            exit_price=110.0,
            exit_timestamp=now + timedelta(days=5),
            pnl=10.0,
            pnl_pct=0.10,
            holding_days=5,
        )

        assert record.model_id == "model_a"
        assert record.signal_score == 0.8
        assert record.signal_direction == "long"
        assert record.pnl == 10.0

    @pytest.mark.unit
    def test_record_default_ic(self):
        """Test default information_coefficient is 0."""
        now = datetime.now()
        record = ModelPerformanceRecord(
            model_id="model_a",
            signal_timestamp=now,
            signal_score=0.8,
            signal_direction="long",
            entry_price=100.0,
            exit_price=110.0,
            exit_timestamp=now + timedelta(days=5),
            pnl=10.0,
            pnl_pct=0.10,
            holding_days=5,
        )

        assert record.information_coefficient == 0.0

    @pytest.mark.unit
    def test_record_with_custom_ic(self):
        """Test record with custom information_coefficient."""
        now = datetime.now()
        record = ModelPerformanceRecord(
            model_id="model_a",
            signal_timestamp=now,
            signal_score=0.8,
            signal_direction="short",
            entry_price=100.0,
            exit_price=90.0,
            exit_timestamp=now + timedelta(days=3),
            pnl=10.0,
            pnl_pct=0.10,
            holding_days=3,
            information_coefficient=0.15,
        )

        assert record.information_coefficient == 0.15


class TestModelMetrics:
    """Tests for ModelMetrics dataclass."""

    @pytest.mark.unit
    def test_create_metrics_basic(self):
        """Test creating basic metrics."""
        now = datetime.now()
        metrics = ModelMetrics(
            model_id="model_a",
            period_start=now - timedelta(days=30),
            period_end=now,
            total_return=0.15,
            avg_return=0.02,
            sharpe_ratio=1.5,
            hit_rate=0.65,
            win_count=13,
            loss_count=7,
            ic_score=0.10,
            ic_decay_halflife=5.0,
            win_streak=3,
            consistency_score=0.75,
            signals_generated=20,
            avg_signal_age=2.5,
        )

        assert metrics.model_id == "model_a"
        assert metrics.total_return == 0.15
        assert metrics.sharpe_ratio == 1.5

    @pytest.mark.unit
    def test_metrics_hit_rate(self):
        """Test hit rate calculation."""
        now = datetime.now()
        metrics = ModelMetrics(
            model_id="model_a",
            period_start=now - timedelta(days=30),
            period_end=now,
            total_return=0.15,
            avg_return=0.02,
            sharpe_ratio=1.5,
            hit_rate=0.65,
            win_count=65,
            loss_count=35,
            ic_score=0.10,
            ic_decay_halflife=5.0,
            win_streak=3,
            consistency_score=0.75,
            signals_generated=100,
            avg_signal_age=2.5,
        )

        # Hit rate = win_count / (win_count + loss_count)
        calculated_hit_rate = metrics.win_count / (metrics.win_count + metrics.loss_count)
        assert abs(calculated_hit_rate - metrics.hit_rate) < 0.001


class TestModelPerformanceAnalyzer:
    """Tests for ModelPerformanceAnalyzer class."""

    @pytest.mark.unit
    def test_init_empty_records(self):
        """Test initialization with empty records."""
        analyzer = ModelPerformanceAnalyzer()
        assert analyzer.records == []

    @pytest.mark.unit
    def test_add_record(self):
        """Test adding a record."""
        analyzer = ModelPerformanceAnalyzer()
        now = datetime.now()

        record = ModelPerformanceRecord(
            model_id="model_a",
            signal_timestamp=now,
            signal_score=0.8,
            signal_direction="long",
            entry_price=100.0,
            exit_price=110.0,
            exit_timestamp=now + timedelta(days=5),
            pnl=10.0,
            pnl_pct=0.10,
            holding_days=5,
        )

        analyzer.add_record(record)

        assert len(analyzer.records) == 1
        assert analyzer.records[0] == record

    @pytest.mark.unit
    def test_add_multiple_records(self):
        """Test adding multiple records."""
        analyzer = ModelPerformanceAnalyzer()
        now = datetime.now()

        for i in range(5):
            record = ModelPerformanceRecord(
                model_id=f"model_{i}",
                signal_timestamp=now + timedelta(days=i),
                signal_score=0.5 + i * 0.1,
                signal_direction="long" if i % 2 == 0 else "short",
                entry_price=100.0,
                exit_price=100.0 + i * 2,
                exit_timestamp=now + timedelta(days=i + 3),
                pnl=i * 2,
                pnl_pct=i * 0.02,
                holding_days=3,
            )
            analyzer.add_record(record)

        assert len(analyzer.records) == 5

    @pytest.mark.unit
    def test_compute_hit_rate_empty(self):
        """Test hit rate with no records returns zeros."""
        analyzer = ModelPerformanceAnalyzer()

        hit_rate, wins, losses = analyzer.compute_hit_rate()
        assert hit_rate == 0.0
        assert wins == 0
        assert losses == 0

    @pytest.mark.unit
    def test_compute_hit_rate_with_records(self):
        """Test hit rate with records."""
        analyzer = ModelPerformanceAnalyzer()
        now = datetime.now()

        # Add winning and losing trades
        for i, pnl in enumerate([10.0, -5.0, 15.0, -3.0, 8.0]):
            record = ModelPerformanceRecord(
                model_id="model_a",
                signal_timestamp=now + timedelta(days=i),
                signal_score=0.8,
                signal_direction="long",
                entry_price=100.0,
                exit_price=100.0 + pnl,
                exit_timestamp=now + timedelta(days=i + 5),
                pnl=pnl,
                pnl_pct=pnl / 100.0,
                holding_days=5,
            )
            analyzer.add_record(record)

        hit_rate, wins, losses = analyzer.compute_hit_rate()
        assert wins == 3  # 10.0, 15.0, 8.0
        assert losses == 2  # -5.0, -3.0
        assert abs(hit_rate - 0.6) < 0.001  # 3 / 5

    @pytest.mark.unit
    def test_compute_hit_rate_by_model(self):
        """Test hit rate filtered by model."""
        analyzer = ModelPerformanceAnalyzer()
        now = datetime.now()

        # Add records for different models
        for i, (model, pnl) in enumerate([
            ("model_a", 10.0),
            ("model_a", -5.0),
            ("model_b", 15.0),
            ("model_b", 20.0),
        ]):
            record = ModelPerformanceRecord(
                model_id=model,
                signal_timestamp=now + timedelta(days=i),
                signal_score=0.8,
                signal_direction="long",
                entry_price=100.0,
                exit_price=100.0 + pnl,
                exit_timestamp=now + timedelta(days=i + 5),
                pnl=pnl,
                pnl_pct=pnl / 100.0,
                holding_days=5,
            )
            analyzer.add_record(record)

        # Model A: 1 win, 1 loss
        hit_rate_a, wins_a, losses_a = analyzer.compute_hit_rate("model_a")
        assert wins_a == 1
        assert losses_a == 1

        # Model B: 2 wins, 0 losses
        hit_rate_b, wins_b, losses_b = analyzer.compute_hit_rate("model_b")
        assert wins_b == 2
        assert losses_b == 0

    @pytest.mark.unit
    def test_compute_ic_empty(self):
        """Test IC with no records returns zeros."""
        analyzer = ModelPerformanceAnalyzer()

        ic_mean, ic_std, halflife = analyzer.compute_ic()
        assert ic_mean == 0.0
        assert ic_std == 0.0
        assert halflife == 0.0

    @pytest.mark.unit
    def test_compute_ic_insufficient_records(self):
        """Test IC with too few records returns zeros."""
        analyzer = ModelPerformanceAnalyzer()
        now = datetime.now()

        record = ModelPerformanceRecord(
            model_id="model_a",
            signal_timestamp=now,
            signal_score=0.8,
            signal_direction="long",
            entry_price=100.0,
            exit_price=110.0,
            exit_timestamp=now + timedelta(days=5),
            pnl=10.0,
            pnl_pct=0.10,
            holding_days=5,
        )
        analyzer.add_record(record)

        ic_mean, ic_std, halflife = analyzer.compute_ic()
        assert ic_mean == 0.0
