"""
Classical trade execution algorithms.

- VWAP: Volume-Weighted Average Price
- TWAP: Time-Weighted Average Price
- Almgren-Chriss: Optimal execution with market impact
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ExecutionSlice:
    """Single slice of an execution plan."""

    time: datetime
    quantity: int
    target_participation: float  # Fraction of volume to target
    urgency: float
    notes: str


class VWAPExecutor:
    """
    Volume-Weighted Average Price execution algorithm.

    Executes order in proportion to historical volume profile
    to minimize market impact and track VWAP benchmark.
    """

    def __init__(
        self,
        volume_profile: pd.Series | None = None,
        max_participation: float = 0.10,  # Max 10% of volume
    ):
        """
        Initialize VWAP executor.

        Args:
            volume_profile: Historical intraday volume profile (optional)
            max_participation: Maximum participation rate
        """
        self.volume_profile = volume_profile
        self.max_participation = max_participation

        # Default volume profile if none provided (typical U-shape)
        if volume_profile is None:
            self._create_default_profile()

    def _create_default_profile(self) -> None:
        """Create typical U-shaped intraday volume profile."""
        # 78 five-minute intervals in trading day (9:30-4:00)
        intervals = 78
        x = np.linspace(0, 1, intervals)

        # U-shape: high at open/close, low midday
        profile = 0.3 + 0.7 * (4 * (x - 0.5) ** 2)
        profile = profile / profile.sum()  # Normalize

        times = pd.date_range("09:30", "16:00", periods=intervals)
        self.volume_profile = pd.Series(profile, index=times.time)

    def create_plan(
        self,
        total_quantity: int,
        start_time: datetime,
        end_time: datetime,
        side: str = "buy",
    ) -> list[ExecutionSlice]:
        """
        Create VWAP execution plan.

        Args:
            total_quantity: Total shares to execute
            start_time: Execution start
            end_time: Execution end
            side: 'buy' or 'sell'

        Returns:
            List of ExecutionSlice objects
        """
        plan = []
        duration = (end_time - start_time).total_seconds() / 60  # Minutes
        intervals = max(1, int(duration / 5))  # 5-minute intervals

        for i in range(intervals):
            slice_time = start_time + timedelta(minutes=i * 5)
            time_key = slice_time.time()

            # Get volume weight for this interval
            if self.volume_profile is not None and time_key in self.volume_profile.index:
                weight = float(self.volume_profile.loc[time_key])  # type: ignore[call-overload]
            else:
                weight = 1.0 / intervals

            # Calculate quantity for this slice
            slice_quantity = int(
                total_quantity
                * weight
                / sum(
                    float(self.volume_profile.iloc[j])
                    for j in range(
                        min(i, len(self.volume_profile)), min(intervals, len(self.volume_profile))
                    )
                )
                if self.volume_profile is not None
                else total_quantity // intervals
            )

            plan.append(
                ExecutionSlice(
                    time=slice_time,
                    quantity=max(1, slice_quantity),
                    target_participation=self.max_participation,
                    urgency=i / intervals,
                    notes=f"VWAP slice {i + 1}/{intervals}",
                )
            )

        # Adjust to ensure total matches
        total_planned = sum(s.quantity for s in plan)
        if total_planned != total_quantity and plan:
            plan[-1].quantity += total_quantity - total_planned

        return plan

    def estimate_cost(
        self,
        total_quantity: int,
        avg_daily_volume: int,
        volatility: float,
        spread: float,
    ) -> dict[str, float]:
        """
        Estimate execution cost for VWAP strategy.

        Args:
            total_quantity: Shares to execute
            avg_daily_volume: Average daily volume
            volatility: Daily volatility
            spread: Bid-ask spread

        Returns:
            Cost estimates in basis points
        """
        participation = total_quantity / avg_daily_volume

        # Simple cost model
        spread_cost = spread * 0.5  # Half spread
        impact_cost = 0.1 * volatility * np.sqrt(participation)  # Market impact
        timing_risk = 0.5 * volatility  # Timing risk

        return {
            "spread_cost_bps": spread_cost * 10000,
            "impact_cost_bps": impact_cost * 10000,
            "timing_risk_bps": timing_risk * 10000,
            "total_cost_bps": (spread_cost + impact_cost + timing_risk) * 10000,
            "participation_rate": participation,
        }

    def describe(self) -> dict[str, Any]:
        """Get executor description."""
        return {
            "algorithm": "VWAP",
            "max_participation": self.max_participation,
            "has_volume_profile": self.volume_profile is not None,
        }


class TWAPExecutor:
    """
    Time-Weighted Average Price execution algorithm.

    Executes order in equal slices over time.
    Simpler than VWAP, useful when volume profile unknown.
    """

    def __init__(
        self,
        interval_minutes: int = 5,
        randomize: bool = True,
        randomize_pct: float = 0.20,
    ):
        """
        Initialize TWAP executor.

        Args:
            interval_minutes: Time between slices
            randomize: Add randomization to avoid detection
            randomize_pct: Randomization percentage
        """
        self.interval_minutes = interval_minutes
        self.randomize = randomize
        self.randomize_pct = randomize_pct

    def create_plan(
        self,
        total_quantity: int,
        start_time: datetime,
        end_time: datetime,
        side: str = "buy",
    ) -> list[ExecutionSlice]:
        """
        Create TWAP execution plan.

        Args:
            total_quantity: Total shares to execute
            start_time: Execution start
            end_time: Execution end
            side: 'buy' or 'sell'

        Returns:
            List of ExecutionSlice objects
        """
        duration_minutes = (end_time - start_time).total_seconds() / 60
        intervals = max(1, int(duration_minutes / self.interval_minutes))
        base_quantity = total_quantity // intervals

        plan = []
        remaining = total_quantity

        for i in range(intervals):
            slice_time = start_time + timedelta(minutes=i * self.interval_minutes)

            # Add randomization
            if self.randomize and i < intervals - 1:
                variation = int(base_quantity * self.randomize_pct * (2 * np.random.random() - 1))
                slice_quantity = max(1, base_quantity + variation)
            else:
                slice_quantity = remaining if i == intervals - 1 else base_quantity

            slice_quantity = min(slice_quantity, remaining)
            remaining -= slice_quantity

            plan.append(
                ExecutionSlice(
                    time=slice_time,
                    quantity=slice_quantity,
                    target_participation=0.0,  # TWAP doesn't target participation
                    urgency=i / intervals,
                    notes=f"TWAP slice {i + 1}/{intervals}",
                )
            )

        return plan

    def estimate_cost(
        self,
        total_quantity: int,
        avg_daily_volume: int,
        volatility: float,
        spread: float,
        duration_hours: float,
    ) -> dict[str, float]:
        """
        Estimate execution cost for TWAP strategy.

        Args:
            total_quantity: Shares to execute
            avg_daily_volume: Average daily volume
            volatility: Daily volatility
            spread: Bid-ask spread
            duration_hours: Execution duration in hours

        Returns:
            Cost estimates
        """
        participation = total_quantity / avg_daily_volume

        # TWAP cost model
        spread_cost = spread * 0.5
        impact_cost = 0.1 * volatility * np.sqrt(participation)
        # TWAP has higher timing risk than VWAP
        timing_risk = volatility * np.sqrt(duration_hours / 6.5)  # 6.5 hour trading day

        return {
            "spread_cost_bps": spread_cost * 10000,
            "impact_cost_bps": impact_cost * 10000,
            "timing_risk_bps": timing_risk * 10000,
            "total_cost_bps": (spread_cost + impact_cost + timing_risk) * 10000,
        }

    def describe(self) -> dict[str, Any]:
        """Get executor description."""
        return {
            "algorithm": "TWAP",
            "interval_minutes": self.interval_minutes,
            "randomize": self.randomize,
            "randomize_pct": self.randomize_pct,
        }


class AlmgrenChrissExecutor:
    """
    Almgren-Chriss optimal execution algorithm.

    Balances market impact cost against timing risk
    using mean-variance optimization.

    Reference: Almgren & Chriss (2001), "Optimal Execution of Portfolio Transactions"
    """

    def __init__(
        self,
        risk_aversion: float = 1e-6,
        temporary_impact: float = 0.1,
        permanent_impact: float = 0.01,
    ):
        """
        Initialize Almgren-Chriss executor.

        Args:
            risk_aversion: Risk aversion parameter (lambda)
            temporary_impact: Temporary market impact coefficient
            permanent_impact: Permanent market impact coefficient
        """
        self.risk_aversion = risk_aversion
        self.temporary_impact = temporary_impact
        self.permanent_impact = permanent_impact

    def create_plan(
        self,
        total_quantity: int,
        start_time: datetime,
        end_time: datetime,
        volatility: float,
        avg_daily_volume: int,
        side: str = "buy",
        intervals: int = 20,
    ) -> list[ExecutionSlice]:
        """
        Create optimal execution plan using Almgren-Chriss model.

        Args:
            total_quantity: Total shares to execute
            start_time: Execution start
            end_time: Execution end
            volatility: Daily volatility
            avg_daily_volume: Average daily volume
            side: 'buy' or 'sell'
            intervals: Number of execution intervals

        Returns:
            List of ExecutionSlice objects
        """
        duration_minutes = (end_time - start_time).total_seconds() / 60
        tau = duration_minutes / intervals  # Time per interval

        # Almgren-Chriss parameters
        sigma = volatility / np.sqrt(252)  # Per-minute volatility (approx)
        eta = self.temporary_impact
        gamma = self.permanent_impact

        # Optimal trajectory parameter
        kappa_sq = self.risk_aversion * sigma**2 / eta
        kappa = np.sqrt(kappa_sq) if kappa_sq > 0 else 0.01

        # Generate optimal trajectory
        trajectory = self._compute_trajectory(total_quantity, intervals, kappa, tau)

        plan = []
        for i in range(intervals):
            slice_time = start_time + timedelta(minutes=i * tau)
            slice_quantity = int(trajectory[i])

            plan.append(
                ExecutionSlice(
                    time=slice_time,
                    quantity=max(0, slice_quantity),
                    target_participation=slice_quantity / (avg_daily_volume / (6.5 * 60 / tau)),
                    urgency=i / intervals,
                    notes=f"AC optimal slice {i + 1}/{intervals}",
                )
            )

        # Ensure total matches
        total_planned = sum(s.quantity for s in plan)
        if total_planned != total_quantity and plan:
            plan[-1].quantity += total_quantity - total_planned

        return plan

    def _compute_trajectory(
        self,
        total_quantity: int,
        intervals: int,
        kappa: float,
        tau: float,
    ) -> np.ndarray:
        """
        Compute optimal execution trajectory.

        Uses the Almgren-Chriss closed-form solution.
        """
        # Trajectory: x_j = X * sinh(kappa * (T - t_j)) / sinh(kappa * T)
        total_time = intervals * tau
        time_points = np.arange(intervals) * tau

        if kappa * total_time > 1e-6:
            # Use AC formula
            holdings = (
                total_quantity
                * np.sinh(kappa * (total_time - time_points))
                / np.sinh(kappa * total_time)
            )
        else:
            # Linear (TWAP) when kappa is small
            holdings = total_quantity * (1 - time_points / total_time)

        # Convert holdings to trades
        trades = np.diff(np.append(total_quantity, holdings))

        return np.abs(trades)  # Take absolute value

    def estimate_cost(
        self,
        total_quantity: int,
        avg_daily_volume: int,
        volatility: float,
        spread: float,
        duration_hours: float,
    ) -> dict[str, float]:
        """
        Estimate execution cost for Almgren-Chriss strategy.

        Returns optimal cost achievable under the model.
        """
        qty = total_quantity
        duration = duration_hours
        sigma = volatility / np.sqrt(252 * 6.5)  # Hourly vol
        eta = self.temporary_impact
        gamma = self.permanent_impact

        # Optimal cost components
        kappa_sq = self.risk_aversion * sigma**2 / eta
        kappa = np.sqrt(max(0, kappa_sq))

        if kappa * duration > 1e-6:
            # Expected cost (simplified)
            impact_cost = 0.5 * gamma * qty**2 / avg_daily_volume
            timing_cost = eta * qty**2 * kappa / (2 * np.tanh(kappa * duration / 2))
            variance_cost = sigma**2 * qty**2 * duration / 3  # Approximate variance
        else:
            # Linear (TWAP) cost
            impact_cost = 0.5 * gamma * qty**2 / avg_daily_volume
            timing_cost = eta * qty**2 / duration
            variance_cost = sigma**2 * qty**2 * duration / 3

        spread_cost = spread * 0.5

        return {
            "spread_cost_bps": spread_cost * 10000,
            "impact_cost_bps": impact_cost * 10000,
            "timing_cost_bps": timing_cost * 10000,
            "variance_cost_bps": variance_cost * 10000,
            "total_cost_bps": (spread_cost + impact_cost + timing_cost) * 10000,
            "optimal_kappa": kappa,
        }

    def describe(self) -> dict[str, Any]:
        """Get executor description."""
        return {
            "algorithm": "Almgren-Chriss",
            "risk_aversion": self.risk_aversion,
            "temporary_impact": self.temporary_impact,
            "permanent_impact": self.permanent_impact,
        }
