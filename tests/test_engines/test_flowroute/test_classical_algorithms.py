from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from ordinis.engines.flowroute.execution.classical_algorithms import (
    AlmgrenChrissExecutor,
    TWAPExecutor,
    VWAPExecutor,
)


@pytest.mark.unit
def test_vwap_default_profile_and_plan_sums_quantity() -> None:
    vwap = VWAPExecutor(volume_profile=None, max_participation=0.15)

    assert vwap.volume_profile is not None
    assert len(vwap.volume_profile) == 78
    assert abs(float(vwap.volume_profile.sum()) - 1.0) < 1e-9

    start = datetime(2024, 1, 1, 9, 30)
    end = start + timedelta(minutes=30)

    plan = vwap.create_plan(total_quantity=1000, start_time=start, end_time=end)

    assert len(plan) == 6  # 30 minutes / 5 minute slices
    assert sum(s.quantity for s in plan) == 1000
    assert all(s.target_participation == 0.15 for s in plan)
    assert plan[0].notes.startswith("VWAP slice")


@pytest.mark.unit
def test_twap_plan_deterministic_without_randomize() -> None:
    twap = TWAPExecutor(interval_minutes=10, randomize=False)

    start = datetime(2024, 1, 1, 9, 30)
    end = start + timedelta(minutes=50)

    plan = twap.create_plan(total_quantity=100, start_time=start, end_time=end)

    # 50 min / 10 min => 5 slices
    assert len(plan) == 5
    assert sum(s.quantity for s in plan) == 100
    # Base quantity 20 each when no randomize.
    assert [s.quantity for s in plan] == [20, 20, 20, 20, 20]

    costs = twap.estimate_cost(
        total_quantity=100,
        avg_daily_volume=1_000_000,
        volatility=0.02,
        spread=0.0002,
        duration_hours=1.0,
    )
    assert costs["total_cost_bps"] > 0


@pytest.mark.unit
def test_almgren_chriss_trajectory_branches_and_plan_sums() -> None:
    ac = AlmgrenChrissExecutor(risk_aversion=1e-8, temporary_impact=0.1, permanent_impact=0.01)

    # Force small-kappa branch: kappa * total_time ~ 0.
    traj_linear = ac._compute_trajectory(total_quantity=1000, intervals=10, kappa=1e-12, tau=1.0)
    assert len(traj_linear) == 10
    assert traj_linear.sum() == pytest.approx(1000, rel=0.05)

    # Force AC formula branch with larger kappa.
    traj_ac = ac._compute_trajectory(total_quantity=1000, intervals=10, kappa=0.05, tau=1.0)
    assert len(traj_ac) == 10
    assert traj_ac.sum() == pytest.approx(1000, rel=0.05)

    start = datetime(2024, 1, 1, 9, 30)
    end = start + timedelta(minutes=100)

    plan = ac.create_plan(
        total_quantity=1000,
        start_time=start,
        end_time=end,
        volatility=0.2,
        avg_daily_volume=5_000_000,
        intervals=10,
    )

    assert len(plan) == 10
    assert sum(s.quantity for s in plan) == 1000
    assert all(s.quantity >= 0 for s in plan)

    costs = ac.estimate_cost(
        total_quantity=1000,
        avg_daily_volume=5_000_000,
        volatility=0.2,
        spread=0.0005,
        duration_hours=1.0,
    )
    assert costs["total_cost_bps"] > 0
