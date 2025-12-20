"""
Enhanced Governance Rules - Advanced portfolio risk controls.

Provides additional governance rules beyond basic position limits:
- Sector concentration limits
- Correlation clustering constraints
- Liquidity-adjusted position limits
- Drawdown-based position scaling
- Volatility regime constraints

Gap Addressed: Limited governance hooks (missing sector concentration,
liquidity-based limits, correlation awareness).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
import logging

from ordinis.engines.base import PreflightContext
from ordinis.engines.portfolio.hooks.governance import PortfolioRule

logger = logging.getLogger(__name__)


# ============================================================================
# Sector Concentration Rules
# ============================================================================


@dataclass
class SectorMapping:
    """Mapping of symbols to sectors."""

    symbol_to_sector: dict[str, str] = field(default_factory=dict)
    sector_weights: dict[str, float] = field(default_factory=dict)

    def get_sector(self, symbol: str) -> str:
        """Get sector for a symbol."""
        return self.symbol_to_sector.get(symbol, "UNKNOWN")

    def get_sector_exposure(
        self,
        positions: dict[str, float],
        prices: dict[str, float],
    ) -> dict[str, float]:
        """Calculate sector exposure percentages.

        Args:
            positions: Position quantities
            prices: Current prices

        Returns:
            Sector exposure as percentage of total
        """
        total_value = sum(positions.get(s, 0) * prices.get(s, 0) for s in positions)

        if total_value <= 0:
            return {}

        sector_values: dict[str, float] = {}
        for symbol, qty in positions.items():
            sector = self.get_sector(symbol)
            value = qty * prices.get(symbol, 0)
            sector_values[sector] = sector_values.get(sector, 0) + value

        return {s: v / total_value * 100 for s, v in sector_values.items()}


@dataclass
class SectorConcentrationRule(PortfolioRule):
    """Rule enforcing sector concentration limits.

    Prevents over-concentration in any single sector.

    Attributes:
        max_sector_pct: Maximum exposure to any single sector
        sector_mapping: Mapping of symbols to sectors
        exclude_sectors: Sectors to exclude from limits (e.g., cash)
    """

    max_sector_pct: float = 30.0  # Max 30% in any sector
    sector_mapping: SectorMapping = field(default_factory=SectorMapping)
    exclude_sectors: list[str] = field(default_factory=lambda: ["CASH", "UNKNOWN"])

    def check(self, context: PreflightContext) -> tuple[bool, str]:
        """Check sector concentration limits.

        Args:
            context: Preflight context with positions and prices

        Returns:
            Tuple of (passed, reason)
        """
        params = context.parameters
        positions = params.get("positions", {})
        prices = params.get("prices", {})

        if not positions:
            return True, "No positions to check"

        exposures = self.sector_mapping.get_sector_exposure(positions, prices)

        violations = []
        for sector, pct in exposures.items():
            if sector in self.exclude_sectors:
                continue
            if pct > self.max_sector_pct:
                violations.append(f"{sector}: {pct:.1f}% > {self.max_sector_pct}%")

        if violations:
            return False, f"Sector concentration exceeded: {'; '.join(violations)}"

        return True, f"All sectors within {self.max_sector_pct}% limit"


# ============================================================================
# Correlation Clustering Rules
# ============================================================================


@dataclass
class CorrelationClusterRule(PortfolioRule):
    """Rule limiting exposure to correlated asset clusters.

    Prevents portfolio from being too concentrated in highly
    correlated assets, which increases tail risk.

    Attributes:
        max_cluster_pct: Maximum exposure to correlated cluster
        correlation_threshold: Correlation above which assets cluster
        correlation_matrix: Pre-computed correlation matrix
    """

    max_cluster_pct: float = 40.0
    correlation_threshold: float = 0.7
    correlation_matrix: dict[tuple[str, str], float] = field(default_factory=dict)

    def set_correlation(self, sym1: str, sym2: str, corr: float) -> None:
        """Set correlation between two symbols."""
        self.correlation_matrix[(sym1, sym2)] = corr
        self.correlation_matrix[(sym2, sym1)] = corr

    def get_correlation(self, sym1: str, sym2: str) -> float:
        """Get correlation between two symbols."""
        if sym1 == sym2:
            return 1.0
        return self.correlation_matrix.get((sym1, sym2), 0.0)

    def find_clusters(self, symbols: list[str]) -> list[set[str]]:
        """Find clusters of correlated symbols.

        Args:
            symbols: List of symbols to cluster

        Returns:
            List of correlated symbol clusters
        """
        # Simple single-linkage clustering
        clusters: list[set[str]] = []
        assigned = set()

        for sym in symbols:
            if sym in assigned:
                continue

            cluster = {sym}
            for other in symbols:
                if other == sym or other in assigned:
                    continue
                if self.get_correlation(sym, other) >= self.correlation_threshold:
                    cluster.add(other)

            for s in cluster:
                assigned.add(s)

            if len(cluster) > 1:
                clusters.append(cluster)

        return clusters

    def check(self, context: PreflightContext) -> tuple[bool, str]:
        """Check correlation cluster exposure.

        Args:
            context: Preflight context with positions and prices

        Returns:
            Tuple of (passed, reason)
        """
        params = context.parameters
        positions = params.get("positions", {})
        prices = params.get("prices", {})

        if not positions or not self.correlation_matrix:
            return True, "No correlations configured"

        symbols = list(positions.keys())
        clusters = self.find_clusters(symbols)

        if not clusters:
            return True, "No correlated clusters found"

        # Calculate total portfolio value
        total_value = sum(positions.get(s, 0) * prices.get(s, 0) for s in positions)

        if total_value <= 0:
            return True, "Portfolio value is zero"

        violations = []
        for cluster in clusters:
            cluster_value = sum(positions.get(s, 0) * prices.get(s, 0) for s in cluster)
            cluster_pct = (cluster_value / total_value) * 100

            if cluster_pct > self.max_cluster_pct:
                symbols_str = ", ".join(sorted(cluster))
                violations.append(
                    f"Cluster [{symbols_str}]: {cluster_pct:.1f}% > {self.max_cluster_pct}%"
                )

        if violations:
            return False, f"Correlated cluster exposure exceeded: {'; '.join(violations)}"

        return True, "All correlation clusters within limits"


# ============================================================================
# Liquidity-Adjusted Rules
# ============================================================================


@dataclass
class LiquidityMetrics:
    """Liquidity metrics for position sizing."""

    symbol: str
    avg_daily_volume: float
    avg_spread_bps: float = 10.0
    market_cap: float = 0.0


@dataclass
class LiquidityAdjustedRule(PortfolioRule):
    """Rule adjusting position limits based on liquidity.

    Larger positions allowed in more liquid assets.
    Prevents large positions in illiquid assets that could
    cause significant market impact.

    Attributes:
        max_participation_rate: Max % of daily volume for position
        max_illiquid_position_pct: Max position % for illiquid assets
        liquidity_threshold: ADV threshold for liquid vs illiquid
        liquidity_data: Pre-loaded liquidity metrics
    """

    max_participation_rate: float = 5.0  # Max 5% of daily volume
    max_illiquid_position_pct: float = 2.0  # Max 2% in illiquid
    liquidity_threshold: float = 500_000  # $500k ADV = liquid
    liquidity_data: dict[str, LiquidityMetrics] = field(default_factory=dict)

    def set_liquidity(
        self,
        symbol: str,
        avg_daily_volume: float,
        avg_spread_bps: float = 10.0,
    ) -> None:
        """Set liquidity metrics for a symbol."""
        self.liquidity_data[symbol] = LiquidityMetrics(
            symbol=symbol,
            avg_daily_volume=avg_daily_volume,
            avg_spread_bps=avg_spread_bps,
        )

    def get_liquidity(self, symbol: str) -> LiquidityMetrics | None:
        """Get liquidity metrics for a symbol."""
        return self.liquidity_data.get(symbol)

    def is_liquid(self, symbol: str, price: float) -> bool:
        """Check if an asset is considered liquid.

        Args:
            symbol: Asset symbol
            price: Current price

        Returns:
            True if liquid based on ADV threshold
        """
        metrics = self.get_liquidity(symbol)
        if not metrics:
            return True  # Assume liquid if no data

        dollar_volume = metrics.avg_daily_volume * price
        return dollar_volume >= self.liquidity_threshold

    def check(self, context: PreflightContext) -> tuple[bool, str]:
        """Check liquidity-adjusted position limits.

        Args:
            context: Preflight context with positions and prices

        Returns:
            Tuple of (passed, reason)
        """
        params = context.parameters
        positions = params.get("positions", {})
        prices = params.get("prices", {})

        if not positions or not self.liquidity_data:
            return True, "No liquidity data configured"

        total_value = sum(positions.get(s, 0) * prices.get(s, 0) for s in positions)

        if total_value <= 0:
            return True, "Portfolio value is zero"

        violations = []
        warnings = []

        for symbol, qty in positions.items():
            price = prices.get(symbol, 0)
            if price <= 0:
                continue

            position_value = qty * price
            position_pct = (position_value / total_value) * 100

            metrics = self.get_liquidity(symbol)
            if not metrics:
                continue

            # Check participation rate
            if metrics.avg_daily_volume > 0:
                participation = (qty / metrics.avg_daily_volume) * 100
                if participation > self.max_participation_rate:
                    violations.append(
                        f"{symbol}: {participation:.1f}% participation > "
                        f"{self.max_participation_rate}%"
                    )

            # Check illiquid position limit
            if not self.is_liquid(symbol, price):
                if position_pct > self.max_illiquid_position_pct:
                    violations.append(
                        f"{symbol}: {position_pct:.1f}% in illiquid asset > "
                        f"{self.max_illiquid_position_pct}%"
                    )
                elif position_pct > self.max_illiquid_position_pct * 0.8:
                    warnings.append(f"{symbol}: {position_pct:.1f}% approaching illiquid limit")

        if violations:
            return False, f"Liquidity limits exceeded: {'; '.join(violations)}"

        if warnings:
            return True, f"Passed with warnings: {'; '.join(warnings)}"

        return True, "All positions within liquidity limits"


# ============================================================================
# Drawdown-Based Rules
# ============================================================================


@dataclass
class DrawdownRule(PortfolioRule):
    """Rule scaling position sizes based on recent drawdown.

    Reduces exposure when portfolio is in drawdown to preserve capital.
    Gradually increases exposure as drawdown recovers.

    Attributes:
        max_drawdown_for_full_size: Drawdown at which full sizing allowed
        min_size_at_max_drawdown: Minimum sizing at maximum drawdown
        max_drawdown_limit: Maximum drawdown before blocking new trades
        current_drawdown_pct: Current portfolio drawdown
        peak_equity: Peak portfolio equity
    """

    max_drawdown_for_full_size: float = 5.0  # Full size if DD < 5%
    min_size_at_max_drawdown: float = 0.25  # 25% size at max DD
    max_drawdown_limit: float = 20.0  # Block trades if DD > 20%
    current_drawdown_pct: float = 0.0
    peak_equity: float = 0.0

    def update_drawdown(self, current_equity: float) -> float:
        """Update drawdown calculation.

        Args:
            current_equity: Current portfolio equity

        Returns:
            Current drawdown percentage
        """
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
            self.current_drawdown_pct = 0.0
        elif self.peak_equity > 0:
            self.current_drawdown_pct = (self.peak_equity - current_equity) / self.peak_equity * 100
        return self.current_drawdown_pct

    def get_sizing_multiplier(self) -> float:
        """Calculate sizing multiplier based on drawdown.

        Returns:
            Multiplier between min_size_at_max_drawdown and 1.0
        """
        if self.current_drawdown_pct <= self.max_drawdown_for_full_size:
            return 1.0

        if self.current_drawdown_pct >= self.max_drawdown_limit:
            return self.min_size_at_max_drawdown

        # Linear interpolation between thresholds
        range_dd = self.max_drawdown_limit - self.max_drawdown_for_full_size
        range_size = 1.0 - self.min_size_at_max_drawdown
        position_in_range = (self.current_drawdown_pct - self.max_drawdown_for_full_size) / range_dd

        return 1.0 - (position_in_range * range_size)

    def check(self, context: PreflightContext) -> tuple[bool, str]:
        """Check drawdown-based trading limits.

        Args:
            context: Preflight context

        Returns:
            Tuple of (passed, reason)
        """
        if self.current_drawdown_pct >= self.max_drawdown_limit:
            return False, (
                f"Maximum drawdown exceeded: {self.current_drawdown_pct:.1f}% >= "
                f"{self.max_drawdown_limit}%"
            )

        multiplier = self.get_sizing_multiplier()
        if multiplier < 1.0:
            return True, (
                f"Drawdown at {self.current_drawdown_pct:.1f}% - "
                f"sizing reduced to {multiplier:.0%}"
            )

        return True, f"Drawdown within limits ({self.current_drawdown_pct:.1f}%)"


# ============================================================================
# Volatility Regime Rules
# ============================================================================


@dataclass
class VolatilityRegimeRule(PortfolioRule):
    """Rule adjusting exposure based on market volatility regime.

    Reduces exposure in high volatility environments to manage risk.

    Attributes:
        low_vol_threshold: Annualized vol below which full size allowed
        high_vol_threshold: Vol above which minimum size applied
        min_size_high_vol: Minimum sizing in high vol regime
        current_vix: Current VIX or volatility proxy
        lookback_vol: Recent realized volatility
    """

    low_vol_threshold: float = 15.0  # VIX < 15 = low vol
    high_vol_threshold: float = 30.0  # VIX > 30 = high vol
    min_size_high_vol: float = 0.5  # 50% size in high vol
    current_vix: float = 20.0  # Default moderate
    lookback_vol: float = 0.0

    def update_volatility(
        self,
        vix: float | None = None,
        realized_vol: float | None = None,
    ) -> None:
        """Update volatility metrics.

        Args:
            vix: Current VIX level
            realized_vol: Recent realized volatility (annualized)
        """
        if vix is not None:
            self.current_vix = vix
        if realized_vol is not None:
            self.lookback_vol = realized_vol

    def get_sizing_multiplier(self) -> float:
        """Calculate sizing multiplier based on volatility regime.

        Returns:
            Multiplier between min_size_high_vol and 1.0
        """
        vol_metric = max(self.current_vix, self.lookback_vol * 100)

        if vol_metric <= self.low_vol_threshold:
            return 1.0

        if vol_metric >= self.high_vol_threshold:
            return self.min_size_high_vol

        # Linear interpolation
        range_vol = self.high_vol_threshold - self.low_vol_threshold
        range_size = 1.0 - self.min_size_high_vol
        position_in_range = (vol_metric - self.low_vol_threshold) / range_vol

        return 1.0 - (position_in_range * range_size)

    def check(self, context: PreflightContext) -> tuple[bool, str]:
        """Check volatility-based trading limits.

        Args:
            context: Preflight context

        Returns:
            Tuple of (passed, reason)
        """
        multiplier = self.get_sizing_multiplier()
        vol_metric = max(self.current_vix, self.lookback_vol * 100)

        if vol_metric >= self.high_vol_threshold:
            return True, (
                f"High volatility regime (VIX={self.current_vix:.1f}) - "
                f"sizing reduced to {multiplier:.0%}"
            )
        if vol_metric >= self.low_vol_threshold:
            return True, (
                f"Moderate volatility (VIX={self.current_vix:.1f}) - " f"sizing at {multiplier:.0%}"
            )

        return True, f"Low volatility regime (VIX={self.current_vix:.1f}) - full sizing"


# ============================================================================
# Market Hours Rule
# ============================================================================


@dataclass
class MarketHoursRule(PortfolioRule):
    """Rule restricting trading to market hours.

    Prevents inadvertent extended hours trading.

    Attributes:
        market_open_hour: Market open hour (ET)
        market_close_hour: Market close hour (ET)
        allow_extended_hours: Whether to allow extended hours
        extended_open_hour: Pre-market open
        extended_close_hour: After-hours close
    """

    market_open_hour: int = 9  # 9:30 AM ET
    market_open_minute: int = 30
    market_close_hour: int = 16  # 4:00 PM ET
    allow_extended_hours: bool = False
    extended_open_hour: int = 4  # 4:00 AM ET
    extended_close_hour: int = 20  # 8:00 PM ET

    def check(self, context: PreflightContext) -> tuple[bool, str]:
        """Check if trading is allowed based on time.

        Args:
            context: Preflight context

        Returns:
            Tuple of (passed, reason)
        """
        now = context.timestamp or datetime.now(UTC)
        # Convert to ET (simplified - real implementation would use pytz)
        et_hour = (now.hour - 5) % 24  # Rough UTC to ET conversion
        et_minute = now.minute

        is_regular_hours = (
            et_hour > self.market_open_hour
            or (et_hour == self.market_open_hour and et_minute >= self.market_open_minute)
        ) and et_hour < self.market_close_hour

        if is_regular_hours:
            return True, "Within regular market hours"

        if self.allow_extended_hours:
            is_extended = et_hour >= self.extended_open_hour and et_hour < self.extended_close_hour
            if is_extended:
                return True, "Within extended hours (allowed)"

        return False, f"Outside trading hours ({et_hour}:{et_minute:02d} ET)"


# ============================================================================
# Factory Functions
# ============================================================================


def create_standard_governance_rules(
    max_position_pct: float = 25.0,
    max_sector_pct: float = 30.0,
    max_cluster_pct: float = 40.0,
    max_drawdown: float = 20.0,
) -> list[PortfolioRule]:
    """Create a standard set of governance rules.

    Args:
        max_position_pct: Maximum single position percentage
        max_sector_pct: Maximum sector concentration
        max_cluster_pct: Maximum correlated cluster exposure
        max_drawdown: Maximum drawdown before blocking trades

    Returns:
        List of configured governance rules
    """
    from ordinis.engines.portfolio.hooks.governance import (
        PositionLimitRule,
        TradeValueRule,
    )

    return [
        PositionLimitRule(max_position_pct=max_position_pct / 100),
        TradeValueRule(min_trade_value=1.0),  # Alpaca $1 minimum
        SectorConcentrationRule(max_sector_pct=max_sector_pct),
        CorrelationClusterRule(max_cluster_pct=max_cluster_pct),
        DrawdownRule(max_drawdown_limit=max_drawdown),
        VolatilityRegimeRule(),
    ]


def create_conservative_governance_rules() -> list[PortfolioRule]:
    """Create conservative governance rules for risk-averse portfolios."""
    return create_standard_governance_rules(
        max_position_pct=15.0,
        max_sector_pct=20.0,
        max_cluster_pct=30.0,
        max_drawdown=10.0,
    )


def create_aggressive_governance_rules() -> list[PortfolioRule]:
    """Create aggressive governance rules for higher risk tolerance."""
    return create_standard_governance_rules(
        max_position_pct=40.0,
        max_sector_pct=50.0,
        max_cluster_pct=60.0,
        max_drawdown=30.0,
    )
