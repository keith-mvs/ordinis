"""
Order Execution Quality Tracking and Analytics.

Implements production-grade execution analysis:
- Fill quality metrics (slippage, timing, price improvement)
- Execution cost analysis (spread, impact, fees)
- Venue performance tracking
- Smart order routing analytics
- TCA (Transaction Cost Analysis) reports

Step 7 of Trade Enhancement Roadmap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum, auto
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class FillQuality(Enum):
    """Fill quality classification."""
    
    EXCELLENT = auto()  # Price improvement > 50% of spread
    GOOD = auto()  # Minimal slippage (< 25% of spread)
    ACCEPTABLE = auto()  # Normal market conditions
    POOR = auto()  # Significant slippage
    VERY_POOR = auto()  # Excessive slippage or failed


class OrderTiming(Enum):
    """Order timing relative to signal."""
    
    IMMEDIATE = auto()  # < 1 second
    FAST = auto()  # < 5 seconds
    NORMAL = auto()  # < 30 seconds
    DELAYED = auto()  # < 5 minutes
    STALE = auto()  # > 5 minutes


class CostCategory(Enum):
    """Execution cost categories."""
    
    SPREAD = auto()  # Bid-ask spread cost
    SLIPPAGE = auto()  # Price movement during execution
    MARKET_IMPACT = auto()  # Our order moving the market
    TIMING_COST = auto()  # Delay cost
    COMMISSION = auto()  # Broker fees
    OPPORTUNITY = auto()  # Missed trades


@dataclass
class ExecutionRecord:
    """Single execution record for tracking."""
    
    order_id: str
    symbol: str
    side: str  # "buy" or "sell"
    
    # Order details
    order_time: datetime
    order_price: Decimal  # Limit price or market reference
    order_quantity: int
    order_type: str
    
    # Fill details
    fill_time: datetime
    fill_price: Decimal
    fill_quantity: int
    
    # Market context at order time
    bid_at_order: Decimal
    ask_at_order: Decimal
    mid_at_order: Decimal
    
    # Market context at fill time
    bid_at_fill: Decimal
    ask_at_fill: Decimal
    mid_at_fill: Decimal
    
    # Costs
    commission: Decimal = Decimal("0")
    fees: Decimal = Decimal("0")
    
    # Venue info
    venue: str = "unknown"
    
    # Metadata
    signal_time: datetime | None = None
    strategy_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def spread_at_order(self) -> Decimal:
        """Bid-ask spread at order time."""
        return self.ask_at_order - self.bid_at_order
        
    @property
    def spread_at_fill(self) -> Decimal:
        """Bid-ask spread at fill time."""
        return self.ask_at_fill - self.bid_at_fill
        
    @property
    def execution_time(self) -> timedelta:
        """Time from order to fill."""
        return self.fill_time - self.order_time
        
    @property
    def signal_to_fill_time(self) -> timedelta | None:
        """Time from signal to fill."""
        if self.signal_time:
            return self.fill_time - self.signal_time
        return None


@dataclass
class FillAnalysis:
    """Analysis of a single fill."""
    
    execution_record: ExecutionRecord
    
    # Slippage metrics
    slippage_bps: float  # Basis points
    slippage_dollars: Decimal
    slippage_pct_of_spread: float
    
    # Price improvement
    price_improvement_bps: float
    price_improvement_dollars: Decimal
    
    # Costs
    total_cost_bps: float
    cost_breakdown: dict[CostCategory, Decimal]
    
    # Quality assessment
    fill_quality: FillQuality
    timing: OrderTiming
    
    # Benchmark comparisons
    vs_vwap: float  # Bps vs VWAP (if available)
    vs_twap: float  # Bps vs TWAP (if available)
    vs_arrival: float  # Bps vs arrival price
    
    # Context
    market_volatility: float  # At execution time
    order_size_pct_adv: float  # % of average daily volume


@dataclass
class VenueStats:
    """Statistics for a trading venue."""
    
    venue: str
    total_orders: int
    total_fills: int
    fill_rate: float
    
    # Quality metrics
    avg_slippage_bps: float
    avg_price_improvement_bps: float
    avg_execution_time_ms: float
    
    # Reliability
    rejection_rate: float
    timeout_rate: float
    
    # Volume
    total_volume: Decimal
    avg_order_size: Decimal


@dataclass
class TCAReport:
    """Transaction Cost Analysis Report."""
    
    period_start: datetime
    period_end: datetime
    
    # Summary
    total_orders: int
    total_fills: int
    total_volume: Decimal
    
    # Costs
    total_cost_bps: float
    cost_by_category: dict[CostCategory, Decimal]
    
    # Quality distribution
    quality_distribution: dict[FillQuality, int]
    
    # Venue analysis
    venue_stats: dict[str, VenueStats]
    
    # Best/worst executions
    best_executions: list[FillAnalysis]
    worst_executions: list[FillAnalysis]
    
    # Trends
    slippage_trend: list[tuple[datetime, float]]  # Daily slippage
    cost_trend: list[tuple[datetime, float]]  # Daily cost
    
    # Recommendations
    recommendations: list[str]
    
    timestamp: datetime = field(default_factory=datetime.utcnow)


class ExecutionTracker:
    """
    Order Execution Quality Tracker.
    
    Tracks and analyzes execution quality metrics.
    
    Example:
        >>> tracker = ExecutionTracker()
        >>> tracker.record_execution(execution_record)
        >>> analysis = tracker.analyze_execution(order_id)
        >>> report = tracker.generate_tca_report(start, end)
    """
    
    def __init__(self) -> None:
        """Initialize tracker."""
        self._executions: dict[str, ExecutionRecord] = {}
        self._analyses: dict[str, FillAnalysis] = {}
        self._vwap_cache: dict[str, dict[datetime, Decimal]] = {}
        
    def record_execution(self, record: ExecutionRecord) -> None:
        """Record an execution for tracking."""
        self._executions[record.order_id] = record
        
        # Immediately analyze
        analysis = self._analyze_fill(record)
        self._analyses[record.order_id] = analysis
        
        logger.info(
            f"Recorded execution {record.order_id}: "
            f"{record.side} {record.fill_quantity} {record.symbol} @ {record.fill_price} "
            f"(slippage: {analysis.slippage_bps:.1f}bps, quality: {analysis.fill_quality.name})"
        )
        
    def get_execution(self, order_id: str) -> ExecutionRecord | None:
        """Get execution record by order ID."""
        return self._executions.get(order_id)
        
    def get_analysis(self, order_id: str) -> FillAnalysis | None:
        """Get fill analysis by order ID."""
        return self._analyses.get(order_id)
        
    def _analyze_fill(self, record: ExecutionRecord) -> FillAnalysis:
        """Analyze a single fill."""
        is_buy = record.side.lower() == "buy"
        
        # Reference price (mid at order time)
        reference = record.mid_at_order
        
        # Slippage calculation
        if is_buy:
            # For buys, slippage is positive if we paid more than mid
            slippage_decimal = record.fill_price - reference
        else:
            # For sells, slippage is positive if we received less than mid
            slippage_decimal = reference - record.fill_price
            
        slippage_dollars = slippage_decimal * record.fill_quantity
        slippage_bps = float(slippage_decimal / reference * 10000) if reference else 0
        
        # Slippage as % of spread
        spread = record.spread_at_order
        slippage_pct_spread = float(slippage_decimal / spread * 100) if spread else 0
        
        # Price improvement (vs aggressive side)
        if is_buy:
            aggressive_price = record.ask_at_order
            improvement = aggressive_price - record.fill_price
        else:
            aggressive_price = record.bid_at_order
            improvement = record.fill_price - aggressive_price
            
        pi_bps = float(improvement / reference * 10000) if reference else 0
        pi_dollars = improvement * record.fill_quantity
        
        # Cost breakdown
        half_spread = spread / 2
        spread_cost = half_spread * record.fill_quantity
        
        market_impact = Decimal("0")
        if is_buy and record.mid_at_fill > record.mid_at_order:
            market_impact = (record.mid_at_fill - record.mid_at_order) * record.fill_quantity
        elif not is_buy and record.mid_at_fill < record.mid_at_order:
            market_impact = (record.mid_at_order - record.mid_at_fill) * record.fill_quantity
            
        timing_cost = Decimal("0")
        exec_seconds = record.execution_time.total_seconds()
        if exec_seconds > 60:
            # Estimate timing cost based on typical volatility
            timing_cost = reference * Decimal("0.0001") * record.fill_quantity
            
        cost_breakdown = {
            CostCategory.SPREAD: spread_cost,
            CostCategory.SLIPPAGE: abs(slippage_dollars),
            CostCategory.MARKET_IMPACT: market_impact,
            CostCategory.TIMING_COST: timing_cost,
            CostCategory.COMMISSION: record.commission,
        }
        
        total_cost = sum(cost_breakdown.values())
        total_cost_bps = float(total_cost / (reference * record.fill_quantity) * 10000) if reference else 0
        
        # Classify fill quality
        fill_quality = self._classify_quality(slippage_bps, slippage_pct_spread)
        
        # Classify timing
        timing = self._classify_timing(record.execution_time)
        
        # Benchmark comparisons
        vs_arrival = slippage_bps  # Arrival = mid at order
        
        return FillAnalysis(
            execution_record=record,
            slippage_bps=slippage_bps,
            slippage_dollars=slippage_dollars,
            slippage_pct_of_spread=slippage_pct_spread,
            price_improvement_bps=pi_bps,
            price_improvement_dollars=pi_dollars,
            total_cost_bps=total_cost_bps,
            cost_breakdown=cost_breakdown,
            fill_quality=fill_quality,
            timing=timing,
            vs_vwap=0.0,  # Would need VWAP data
            vs_twap=0.0,  # Would need TWAP data
            vs_arrival=vs_arrival,
            market_volatility=0.0,  # Would need volatility data
            order_size_pct_adv=0.0,  # Would need ADV data
        )
        
    def _classify_quality(
        self,
        slippage_bps: float,
        slippage_pct_spread: float,
    ) -> FillQuality:
        """Classify fill quality based on slippage."""
        if slippage_bps < -5:  # Price improvement
            return FillQuality.EXCELLENT
        elif slippage_bps < 2 or slippage_pct_spread < 25:
            return FillQuality.GOOD
        elif slippage_bps < 5 or slippage_pct_spread < 50:
            return FillQuality.ACCEPTABLE
        elif slippage_bps < 15 or slippage_pct_spread < 100:
            return FillQuality.POOR
        else:
            return FillQuality.VERY_POOR
            
    def _classify_timing(self, execution_time: timedelta) -> OrderTiming:
        """Classify order timing."""
        seconds = execution_time.total_seconds()
        
        if seconds < 1:
            return OrderTiming.IMMEDIATE
        elif seconds < 5:
            return OrderTiming.FAST
        elif seconds < 30:
            return OrderTiming.NORMAL
        elif seconds < 300:
            return OrderTiming.DELAYED
        else:
            return OrderTiming.STALE
            
    def generate_venue_stats(self, venue: str) -> VenueStats:
        """Generate statistics for a specific venue."""
        venue_executions = [
            e for e in self._executions.values()
            if e.venue == venue
        ]
        venue_analyses = [
            self._analyses[e.order_id] for e in venue_executions
            if e.order_id in self._analyses
        ]
        
        if not venue_executions:
            return VenueStats(
                venue=venue,
                total_orders=0,
                total_fills=0,
                fill_rate=0,
                avg_slippage_bps=0,
                avg_price_improvement_bps=0,
                avg_execution_time_ms=0,
                rejection_rate=0,
                timeout_rate=0,
                total_volume=Decimal("0"),
                avg_order_size=Decimal("0"),
            )
            
        total_orders = len(venue_executions)
        total_fills = sum(1 for e in venue_executions if e.fill_quantity > 0)
        
        slippages = [a.slippage_bps for a in venue_analyses]
        pis = [a.price_improvement_bps for a in venue_analyses]
        exec_times = [e.execution_time.total_seconds() * 1000 for e in venue_executions]
        
        total_volume = sum(
            Decimal(str(e.fill_price * e.fill_quantity))
            for e in venue_executions
        )
        
        return VenueStats(
            venue=venue,
            total_orders=total_orders,
            total_fills=total_fills,
            fill_rate=total_fills / total_orders if total_orders else 0,
            avg_slippage_bps=float(np.mean(slippages)) if slippages else 0,
            avg_price_improvement_bps=float(np.mean(pis)) if pis else 0,
            avg_execution_time_ms=float(np.mean(exec_times)) if exec_times else 0,
            rejection_rate=0,  # Would need rejection tracking
            timeout_rate=0,  # Would need timeout tracking
            total_volume=total_volume,
            avg_order_size=total_volume / total_orders if total_orders else Decimal("0"),
        )
        
    def generate_tca_report(
        self,
        start: datetime,
        end: datetime,
        top_n: int = 5,
    ) -> TCAReport:
        """
        Generate Transaction Cost Analysis report.
        
        Args:
            start: Period start
            end: Period end
            top_n: Number of best/worst executions to include
            
        Returns:
            TCAReport
        """
        # Filter executions in period
        period_executions = [
            e for e in self._executions.values()
            if start <= e.fill_time <= end
        ]
        period_analyses = [
            self._analyses[e.order_id] for e in period_executions
            if e.order_id in self._analyses
        ]
        
        if not period_executions:
            return self._empty_report(start, end)
            
        # Summary
        total_orders = len(period_executions)
        total_fills = sum(1 for e in period_executions if e.fill_quantity > 0)
        total_volume = sum(
            Decimal(str(e.fill_price * e.fill_quantity))
            for e in period_executions
        )
        
        # Aggregate costs
        total_cost_bps = float(np.mean([a.total_cost_bps for a in period_analyses]))
        
        cost_by_category: dict[CostCategory, Decimal] = {}
        for cat in CostCategory:
            cost_by_category[cat] = sum(
                a.cost_breakdown.get(cat, Decimal("0"))
                for a in period_analyses
            )
            
        # Quality distribution
        quality_distribution = {q: 0 for q in FillQuality}
        for a in period_analyses:
            quality_distribution[a.fill_quality] += 1
            
        # Venue stats
        venues = set(e.venue for e in period_executions)
        venue_stats = {v: self.generate_venue_stats(v) for v in venues}
        
        # Best/worst executions
        sorted_analyses = sorted(period_analyses, key=lambda a: a.slippage_bps)
        best_executions = sorted_analyses[:top_n]
        worst_executions = sorted_analyses[-top_n:][::-1]
        
        # Daily trends
        slippage_trend = self._calculate_daily_trend(period_analyses, "slippage_bps")
        cost_trend = self._calculate_daily_trend(period_analyses, "total_cost_bps")
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            period_analyses, venue_stats, quality_distribution
        )
        
        return TCAReport(
            period_start=start,
            period_end=end,
            total_orders=total_orders,
            total_fills=total_fills,
            total_volume=total_volume,
            total_cost_bps=total_cost_bps,
            cost_by_category=cost_by_category,
            quality_distribution=quality_distribution,
            venue_stats=venue_stats,
            best_executions=best_executions,
            worst_executions=worst_executions,
            slippage_trend=slippage_trend,
            cost_trend=cost_trend,
            recommendations=recommendations,
        )
        
    def _empty_report(self, start: datetime, end: datetime) -> TCAReport:
        """Create empty TCA report."""
        return TCAReport(
            period_start=start,
            period_end=end,
            total_orders=0,
            total_fills=0,
            total_volume=Decimal("0"),
            total_cost_bps=0,
            cost_by_category={c: Decimal("0") for c in CostCategory},
            quality_distribution={q: 0 for q in FillQuality},
            venue_stats={},
            best_executions=[],
            worst_executions=[],
            slippage_trend=[],
            cost_trend=[],
            recommendations=["No executions in period"],
        )
        
    def _calculate_daily_trend(
        self,
        analyses: list[FillAnalysis],
        metric: str,
    ) -> list[tuple[datetime, float]]:
        """Calculate daily average for a metric."""
        from collections import defaultdict
        
        daily_values: dict[datetime, list[float]] = defaultdict(list)
        
        for a in analyses:
            day = a.execution_record.fill_time.replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            value = getattr(a, metric, 0)
            daily_values[day].append(value)
            
        trend = [
            (day, float(np.mean(values)))
            for day, values in sorted(daily_values.items())
        ]
        
        return trend
        
    def _generate_recommendations(
        self,
        analyses: list[FillAnalysis],
        venue_stats: dict[str, VenueStats],
        quality_dist: dict[FillQuality, int],
    ) -> list[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Check overall quality
        total = sum(quality_dist.values())
        if total > 0:
            poor_pct = (quality_dist[FillQuality.POOR] + quality_dist[FillQuality.VERY_POOR]) / total
            if poor_pct > 0.2:
                recommendations.append(
                    f"High poor fill rate ({poor_pct:.0%}). Consider using limit orders "
                    "or breaking up large orders."
                )
                
        # Check venue performance
        if venue_stats:
            best_venue = min(venue_stats.values(), key=lambda v: v.avg_slippage_bps)
            worst_venue = max(venue_stats.values(), key=lambda v: v.avg_slippage_bps)
            
            if best_venue.avg_slippage_bps < worst_venue.avg_slippage_bps - 5:
                recommendations.append(
                    f"Route more orders to {best_venue.venue} "
                    f"(avg slippage {best_venue.avg_slippage_bps:.1f}bps vs "
                    f"{worst_venue.venue} at {worst_venue.avg_slippage_bps:.1f}bps)"
                )
                
        # Check timing
        delayed_count = sum(
            1 for a in analyses
            if a.timing in (OrderTiming.DELAYED, OrderTiming.STALE)
        )
        if total > 0 and delayed_count / total > 0.1:
            recommendations.append(
                f"{delayed_count}/{total} orders delayed. Review order routing latency."
            )
            
        if not recommendations:
            recommendations.append("Execution quality is within acceptable parameters.")
            
        return recommendations


class SmartOrderRouter:
    """
    Smart Order Routing with execution analytics.
    
    Selects optimal venue based on historical performance.
    """
    
    def __init__(self, tracker: ExecutionTracker) -> None:
        """Initialize router with tracker."""
        self.tracker = tracker
        self._venue_priorities: dict[str, float] = {}
        
    def update_priorities(self) -> None:
        """Update venue priorities based on recent performance."""
        # Get all venues
        venues = set(e.venue for e in self.tracker._executions.values())
        
        for venue in venues:
            stats = self.tracker.generate_venue_stats(venue)
            
            if stats.total_fills == 0:
                self._venue_priorities[venue] = 0
                continue
                
            # Score based on multiple factors
            slippage_score = max(0, 10 - stats.avg_slippage_bps)  # Lower is better
            fill_rate_score = stats.fill_rate * 10
            speed_score = max(0, 10 - stats.avg_execution_time_ms / 100)  # Faster is better
            
            # Weighted combination
            score = 0.5 * slippage_score + 0.3 * fill_rate_score + 0.2 * speed_score
            self._venue_priorities[venue] = score
            
    def select_venue(
        self,
        symbol: str,
        side: str,
        quantity: int,
        order_type: str = "market",
    ) -> str:
        """
        Select optimal venue for order.
        
        Args:
            symbol: Trading symbol
            side: "buy" or "sell"
            quantity: Order quantity
            order_type: Order type
            
        Returns:
            Recommended venue
        """
        if not self._venue_priorities:
            return "default"
            
        # Select highest priority venue
        best_venue = max(self._venue_priorities.items(), key=lambda x: x[1])
        
        logger.info(
            f"Selected venue {best_venue[0]} (score: {best_venue[1]:.1f}) "
            f"for {side} {quantity} {symbol}"
        )
        
        return best_venue[0]
        
    def get_venue_rankings(self) -> list[tuple[str, float]]:
        """Get current venue rankings."""
        return sorted(
            self._venue_priorities.items(),
            key=lambda x: x[1],
            reverse=True,
        )
