# Earnings Calendar Integration

## Overview

Systematic earnings event trading requires robust calendar infrastructure integrating multiple data sources, handling timing uncertainty, and maintaining real-time updates. This document covers production-grade calendar management for earnings-based strategies.

---

## Data Sources

### Primary Earnings Calendar Providers

```python
from dataclasses import dataclass
from datetime import datetime, date, time
from typing import Optional, List, Dict
from enum import Enum
import requests
import pandas as pd


class EarningsDataProvider(Enum):
    """Available earnings calendar data sources."""
    POLYGON = "polygon"
    ALPHA_VANTAGE = "alpha_vantage"
    FINNHUB = "finnhub"
    YAHOO_FINANCE = "yahoo"
    NASDAQ = "nasdaq"
    ZACKS = "zacks"
    EARNINGS_WHISPERS = "earnings_whispers"


@dataclass
class EarningsCalendarConfig:
    """Configuration for earnings calendar data."""
    primary_source: EarningsDataProvider
    backup_sources: List[EarningsDataProvider]

    # Timing
    refresh_interval_minutes: int = 60
    pre_market_refresh: time = time(6, 0)  # 6 AM ET

    # Data quality
    min_sources_for_confirmation: int = 2
    timing_confidence_threshold: float = 0.80

    # Historical
    lookback_quarters: int = 12

    # API keys (load from environment)
    api_keys: Dict[str, str] = None


class EarningsCalendarAPI:
    """
    Multi-source earnings calendar aggregator.
    """

    def __init__(self, config: EarningsCalendarConfig):
        self.config = config
        self._cache = {}
        self._last_refresh = None

    def get_upcoming_earnings(
        self,
        start_date: date,
        end_date: date,
        symbols: List[str] = None
    ) -> pd.DataFrame:
        """
        Get earnings announcements in date range.
        """
        results = []

        # Query primary source
        primary_data = self._fetch_from_source(
            self.config.primary_source,
            start_date,
            end_date,
            symbols
        )
        results.append(primary_data)

        # Query backup sources for confirmation
        for backup in self.config.backup_sources[:2]:  # Max 2 backups
            try:
                backup_data = self._fetch_from_source(
                    backup, start_date, end_date, symbols
                )
                results.append(backup_data)
            except Exception:
                continue

        # Merge and reconcile
        return self._reconcile_sources(results)

    def _fetch_from_source(
        self,
        source: EarningsDataProvider,
        start_date: date,
        end_date: date,
        symbols: List[str]
    ) -> pd.DataFrame:
        """Fetch from specific data source."""

        if source == EarningsDataProvider.POLYGON:
            return self._fetch_polygon(start_date, end_date, symbols)
        elif source == EarningsDataProvider.FINNHUB:
            return self._fetch_finnhub(start_date, end_date, symbols)
        elif source == EarningsDataProvider.ALPHA_VANTAGE:
            return self._fetch_alpha_vantage(symbols)
        else:
            raise ValueError(f"Unsupported source: {source}")

    def _fetch_polygon(
        self,
        start_date: date,
        end_date: date,
        symbols: List[str]
    ) -> pd.DataFrame:
        """
        Fetch from Polygon.io earnings calendar.
        """
        api_key = self.config.api_keys.get('polygon')
        base_url = "https://api.polygon.io/vX/reference/financials"

        records = []
        for symbol in symbols or []:
            url = f"{base_url}?ticker={symbol}&apiKey={api_key}"
            resp = requests.get(url)
            if resp.status_code == 200:
                data = resp.json()
                # Parse earnings dates from financials
                for result in data.get('results', []):
                    records.append({
                        'symbol': symbol,
                        'report_date': result.get('filing_date'),
                        'fiscal_quarter': result.get('fiscal_period'),
                        'source': 'polygon'
                    })

        return pd.DataFrame(records)

    def _fetch_finnhub(
        self,
        start_date: date,
        end_date: date,
        symbols: List[str]
    ) -> pd.DataFrame:
        """
        Fetch from Finnhub earnings calendar.
        """
        api_key = self.config.api_keys.get('finnhub')
        url = "https://finnhub.io/api/v1/calendar/earnings"

        params = {
            'from': start_date.isoformat(),
            'to': end_date.isoformat(),
            'token': api_key
        }

        resp = requests.get(url, params=params)
        if resp.status_code != 200:
            return pd.DataFrame()

        data = resp.json()
        records = []

        for item in data.get('earningsCalendar', []):
            if symbols is None or item['symbol'] in symbols:
                records.append({
                    'symbol': item['symbol'],
                    'report_date': item['date'],
                    'timing': item.get('hour', 'unknown'),  # bmo, amc, dmh
                    'eps_estimate': item.get('epsEstimate'),
                    'eps_actual': item.get('epsActual'),
                    'revenue_estimate': item.get('revenueEstimate'),
                    'source': 'finnhub'
                })

        return pd.DataFrame(records)

    def _reconcile_sources(
        self,
        source_dfs: List[pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Reconcile earnings data from multiple sources.
        """
        if not source_dfs:
            return pd.DataFrame()

        # Merge on symbol and approximate date
        merged = source_dfs[0].copy()

        for df in source_dfs[1:]:
            if df.empty:
                continue

            # Check for date conflicts
            merged = pd.merge(
                merged,
                df[['symbol', 'report_date', 'timing']],
                on='symbol',
                how='outer',
                suffixes=('', '_backup')
            )

        # Calculate confidence based on source agreement
        def calc_confidence(row):
            dates = [row.get('report_date'), row.get('report_date_backup')]
            dates = [d for d in dates if pd.notna(d)]

            if len(dates) == 0:
                return 0.0
            elif len(dates) == 1:
                return 0.5
            elif dates[0] == dates[1]:
                return 1.0
            else:
                return 0.3  # Conflict

        merged['date_confidence'] = merged.apply(calc_confidence, axis=1)

        return merged
```

---

## Timing Resolution

### BMO/AMC Classification

```python
class EarningsTiming(Enum):
    """Earnings announcement timing categories."""
    BMO = "before_market_open"      # 5:00-9:30 AM ET
    AMC = "after_market_close"      # 4:00-8:00 PM ET
    DMH = "during_market_hours"     # 9:30 AM - 4:00 PM ET
    UNKNOWN = "unknown"


@dataclass
class EarningsEvent:
    """Structured earnings event with timing details."""
    symbol: str
    report_date: date
    timing: EarningsTiming
    timing_confidence: float

    # Consensus estimates
    eps_estimate: Optional[float] = None
    revenue_estimate: Optional[float] = None

    # Historical context
    historical_timing: Optional[EarningsTiming] = None  # What they usually do
    timing_consistency: float = 0.0  # % of time they stick to pattern

    # Actual results (populated after release)
    eps_actual: Optional[float] = None
    revenue_actual: Optional[float] = None
    actual_release_time: Optional[datetime] = None


class TimingResolver:
    """
    Resolve earnings timing with historical patterns.
    """

    def __init__(self, historical_data: pd.DataFrame):
        """
        Args:
            historical_data: DataFrame with columns:
                symbol, report_date, actual_timing, release_time
        """
        self.historical = historical_data
        self._timing_patterns = self._build_patterns()

    def _build_patterns(self) -> Dict[str, Dict]:
        """Build timing patterns per symbol."""
        patterns = {}

        for symbol in self.historical['symbol'].unique():
            symbol_data = self.historical[
                self.historical['symbol'] == symbol
            ].sort_values('report_date', ascending=False)

            # Last 8 quarters
            recent = symbol_data.head(8)

            if len(recent) < 4:
                patterns[symbol] = {
                    'typical_timing': EarningsTiming.UNKNOWN,
                    'consistency': 0.0,
                    'sample_size': len(recent)
                }
                continue

            # Calculate mode timing
            timing_counts = recent['actual_timing'].value_counts()
            typical = timing_counts.index[0]
            consistency = timing_counts.iloc[0] / len(recent)

            patterns[symbol] = {
                'typical_timing': EarningsTiming(typical),
                'consistency': consistency,
                'sample_size': len(recent),
                'recent_timings': recent['actual_timing'].tolist()
            }

        return patterns

    def resolve_timing(
        self,
        symbol: str,
        announced_timing: Optional[EarningsTiming],
        announced_confidence: float
    ) -> Dict:
        """
        Resolve timing using announced + historical patterns.
        """
        pattern = self._timing_patterns.get(symbol, {})

        if announced_timing and announced_confidence > 0.8:
            # High confidence announcement
            return {
                'timing': announced_timing,
                'confidence': announced_confidence,
                'source': 'announced'
            }

        if pattern.get('consistency', 0) > 0.9:
            # Very consistent historical pattern
            return {
                'timing': pattern['typical_timing'],
                'confidence': pattern['consistency'] * 0.9,  # Slight discount
                'source': 'historical_pattern'
            }

        if announced_timing:
            # Use announced with adjustment
            return {
                'timing': announced_timing,
                'confidence': max(announced_confidence, 0.5),
                'source': 'announced_low_confidence'
            }

        # Fall back to historical
        if pattern:
            return {
                'timing': pattern.get('typical_timing', EarningsTiming.UNKNOWN),
                'confidence': pattern.get('consistency', 0) * 0.7,
                'source': 'historical_fallback'
            }

        return {
            'timing': EarningsTiming.UNKNOWN,
            'confidence': 0.0,
            'source': 'unknown'
        }


def get_trading_windows(event: EarningsEvent) -> Dict:
    """
    Calculate trading windows based on timing.
    """
    if event.timing == EarningsTiming.BMO:
        return {
            'blackout_start': datetime.combine(
                event.report_date - pd.Timedelta(days=1),
                time(16, 0)  # Prior close
            ),
            'blackout_end': datetime.combine(
                event.report_date,
                time(10, 0)  # 30 min after open
            ),
            'volatility_window_hours': 16,
            'recommended_entry_delay_minutes': 30
        }

    elif event.timing == EarningsTiming.AMC:
        return {
            'blackout_start': datetime.combine(
                event.report_date,
                time(15, 30)  # 30 min before close
            ),
            'blackout_end': datetime.combine(
                event.report_date + pd.Timedelta(days=1),
                time(10, 0)
            ),
            'volatility_window_hours': 18,
            'recommended_entry_delay_minutes': 30
        }

    elif event.timing == EarningsTiming.DMH:
        return {
            'blackout_start': datetime.combine(
                event.report_date,
                time(9, 30)
            ),
            'blackout_end': datetime.combine(
                event.report_date,
                time(16, 0)
            ),
            'volatility_window_hours': 6.5,
            'recommended_entry_delay_minutes': 15,
            'note': 'High uncertainty - avoid if possible'
        }

    return {'note': 'Unknown timing - treat as high risk'}
```

---

## Calendar Synchronization

### Real-Time Updates

```python
import asyncio
from datetime import timedelta
import json


class EarningsCalendarSync:
    """
    Real-time earnings calendar synchronization.
    """

    def __init__(
        self,
        api: EarningsCalendarAPI,
        db_connection,
        refresh_interval_minutes: int = 60
    ):
        self.api = api
        self.db = db_connection
        self.refresh_interval = refresh_interval_minutes
        self._running = False

    async def start_sync(self):
        """Start continuous calendar synchronization."""
        self._running = True

        while self._running:
            try:
                await self._sync_cycle()
            except Exception as e:
                print(f"Sync error: {e}")

            await asyncio.sleep(self.refresh_interval * 60)

    async def _sync_cycle(self):
        """Single synchronization cycle."""
        # Get next 30 days
        start = date.today()
        end = start + timedelta(days=30)

        # Fetch from all sources
        calendar = self.api.get_upcoming_earnings(start, end)

        # Compare with database
        changes = await self._detect_changes(calendar)

        if changes['added'] or changes['modified'] or changes['removed']:
            await self._apply_changes(changes)
            await self._notify_changes(changes)

    async def _detect_changes(self, new_data: pd.DataFrame) -> Dict:
        """Detect changes from current database state."""
        # Load current from DB
        current = await self._load_current_calendar()

        changes = {
            'added': [],
            'modified': [],
            'removed': []
        }

        current_keys = set(
            (r['symbol'], r['report_date'])
            for _, r in current.iterrows()
        )
        new_keys = set(
            (r['symbol'], r['report_date'])
            for _, r in new_data.iterrows()
        )

        # New entries
        for key in new_keys - current_keys:
            row = new_data[
                (new_data['symbol'] == key[0]) &
                (new_data['report_date'] == key[1])
            ].iloc[0]
            changes['added'].append(row.to_dict())

        # Removed entries
        for key in current_keys - new_keys:
            changes['removed'].append({
                'symbol': key[0],
                'report_date': key[1]
            })

        # Modified entries
        for key in current_keys & new_keys:
            curr_row = current[
                (current['symbol'] == key[0]) &
                (current['report_date'] == key[1])
            ].iloc[0]
            new_row = new_data[
                (new_data['symbol'] == key[0]) &
                (new_data['report_date'] == key[1])
            ].iloc[0]

            # Check for timing changes
            if curr_row.get('timing') != new_row.get('timing'):
                changes['modified'].append({
                    'symbol': key[0],
                    'report_date': key[1],
                    'field': 'timing',
                    'old_value': curr_row.get('timing'),
                    'new_value': new_row.get('timing')
                })

        return changes

    async def _notify_changes(self, changes: Dict):
        """Notify trading system of calendar changes."""
        # Critical: Date changes affect position management
        for mod in changes['modified']:
            if mod['field'] == 'timing':
                await self._send_alert(
                    level='WARNING',
                    message=f"Earnings timing change: {mod['symbol']} "
                            f"{mod['old_value']} -> {mod['new_value']}"
                )

        for removed in changes['removed']:
            await self._send_alert(
                level='INFO',
                message=f"Earnings removed from calendar: {removed['symbol']} "
                        f"on {removed['report_date']}"
            )


class EarningsDateChangeHandler:
    """
    Handle earnings date changes for active positions.
    """

    def __init__(self, position_manager, risk_manager):
        self.positions = position_manager
        self.risk = risk_manager

    def handle_date_change(
        self,
        symbol: str,
        old_date: date,
        new_date: date
    ) -> Dict:
        """
        Handle earnings date change for symbol.
        """
        actions = []

        # Check if we have positions
        position = self.positions.get_position(symbol)
        if not position:
            return {'actions': [], 'note': 'No position affected'}

        today = date.today()

        # Date moved earlier
        if new_date < old_date:
            if new_date <= today + timedelta(days=3):
                # Urgently approaching - need to decide
                actions.append({
                    'action': 'REDUCE_OR_CLOSE',
                    'urgency': 'HIGH',
                    'reason': 'earnings_date_moved_earlier',
                    'deadline': new_date
                })
            else:
                actions.append({
                    'action': 'UPDATE_BLACKOUT',
                    'urgency': 'MEDIUM',
                    'new_blackout_start': new_date - timedelta(days=3)
                })

        # Date moved later
        elif new_date > old_date:
            if old_date <= today + timedelta(days=3):
                # Was imminent, now delayed - opportunity
                actions.append({
                    'action': 'REVIEW_POSITION',
                    'urgency': 'LOW',
                    'reason': 'earnings_delayed_reduces_urgency',
                    'note': 'May resume normal position sizing'
                })

        return {'actions': actions}
```

---

## Consensus Estimates Integration

### Estimate Tracking

```python
@dataclass
class ConsensusEstimate:
    """Consensus estimate data."""
    symbol: str
    fiscal_quarter: str

    # EPS estimates
    eps_mean: float
    eps_high: float
    eps_low: float
    eps_std: float
    num_analysts: int

    # Revenue estimates
    revenue_mean: float
    revenue_high: float
    revenue_low: float
    num_revenue_analysts: int

    # Whisper number (if available)
    whisper_eps: Optional[float] = None

    # Estimate revision tracking
    eps_30d_ago: Optional[float] = None
    eps_90d_ago: Optional[float] = None
    revision_trend: Optional[str] = None  # 'up', 'down', 'stable'

    # Timestamp
    as_of_date: date = None


class EstimateTracker:
    """
    Track and analyze consensus estimate revisions.
    """

    def __init__(self, estimate_history: pd.DataFrame):
        """
        Args:
            estimate_history: Historical estimates with columns:
                symbol, as_of_date, eps_mean, revenue_mean, etc.
        """
        self.history = estimate_history

    def get_revision_momentum(
        self,
        symbol: str,
        days: int = 30
    ) -> Dict:
        """
        Calculate estimate revision momentum.
        """
        symbol_data = self.history[
            self.history['symbol'] == symbol
        ].sort_values('as_of_date')

        if len(symbol_data) < 2:
            return {'momentum': 0, 'confidence': 0}

        # Get estimates from N days ago
        cutoff = date.today() - timedelta(days=days)
        recent = symbol_data[symbol_data['as_of_date'] >= cutoff]

        if len(recent) < 2:
            return {'momentum': 0, 'confidence': 0}

        # Calculate revision
        earliest = recent.iloc[0]['eps_mean']
        latest = recent.iloc[-1]['eps_mean']

        if earliest == 0:
            return {'momentum': 0, 'confidence': 0}

        revision_pct = (latest - earliest) / abs(earliest)

        # Momentum classification
        if revision_pct > 0.05:
            momentum = 'STRONG_UP'
        elif revision_pct > 0.02:
            momentum = 'UP'
        elif revision_pct < -0.05:
            momentum = 'STRONG_DOWN'
        elif revision_pct < -0.02:
            momentum = 'DOWN'
        else:
            momentum = 'STABLE'

        return {
            'momentum': momentum,
            'revision_pct': revision_pct,
            'earliest_estimate': earliest,
            'latest_estimate': latest,
            'sample_days': days,
            'confidence': min(len(recent) / 10, 1.0)
        }

    def get_estimate_dispersion(self, symbol: str) -> Dict:
        """
        Calculate analyst estimate dispersion (uncertainty).
        """
        latest = self.history[
            self.history['symbol'] == symbol
        ].sort_values('as_of_date').iloc[-1]

        if latest['num_analysts'] < 3:
            return {
                'dispersion': None,
                'note': 'Insufficient analyst coverage'
            }

        # Coefficient of variation
        cv = latest['eps_std'] / abs(latest['eps_mean']) if latest['eps_mean'] != 0 else 0

        # High dispersion = more uncertainty = larger potential move
        if cv > 0.20:
            uncertainty = 'HIGH'
        elif cv > 0.10:
            uncertainty = 'MODERATE'
        else:
            uncertainty = 'LOW'

        return {
            'dispersion': cv,
            'uncertainty_level': uncertainty,
            'eps_range': (latest['eps_low'], latest['eps_high']),
            'num_analysts': latest['num_analysts'],
            'implication': 'Higher dispersion often means larger post-earnings move'
        }
```

---

## Production Calendar Manager

### Complete Implementation

```python
class ProductionEarningsCalendar:
    """
    Production-grade earnings calendar manager.
    """

    def __init__(
        self,
        config: EarningsCalendarConfig,
        db_connection,
        alert_system
    ):
        self.config = config
        self.db = db_connection
        self.alerts = alert_system

        self.api = EarningsCalendarAPI(config)
        self.sync = EarningsCalendarSync(
            self.api, db_connection, config.refresh_interval_minutes
        )
        self.timing_resolver = None  # Initialized after historical load
        self.estimate_tracker = None

    async def initialize(self):
        """Initialize with historical data."""
        # Load historical timing patterns
        historical = await self._load_historical_earnings()
        self.timing_resolver = TimingResolver(historical)

        # Load estimate history
        estimates = await self._load_estimate_history()
        self.estimate_tracker = EstimateTracker(estimates)

        # Start sync
        asyncio.create_task(self.sync.start_sync())

    def get_events_for_portfolio(
        self,
        symbols: List[str],
        days_ahead: int = 30
    ) -> List[EarningsEvent]:
        """
        Get upcoming earnings for portfolio symbols.
        """
        start = date.today()
        end = start + timedelta(days=days_ahead)

        calendar = self.api.get_upcoming_earnings(start, end, symbols)

        events = []
        for _, row in calendar.iterrows():
            # Resolve timing
            timing_info = self.timing_resolver.resolve_timing(
                row['symbol'],
                EarningsTiming(row.get('timing', 'unknown')),
                row.get('timing_confidence', 0.5)
            )

            # Get estimate info
            revision_info = self.estimate_tracker.get_revision_momentum(
                row['symbol']
            )

            event = EarningsEvent(
                symbol=row['symbol'],
                report_date=pd.to_datetime(row['report_date']).date(),
                timing=timing_info['timing'],
                timing_confidence=timing_info['confidence'],
                eps_estimate=row.get('eps_estimate'),
                revenue_estimate=row.get('revenue_estimate')
            )

            events.append(event)

        # Sort by date
        events.sort(key=lambda e: e.report_date)

        return events

    def check_blackout_conflicts(
        self,
        symbol: str,
        proposed_entry_date: date
    ) -> Dict:
        """
        Check if proposed trade conflicts with earnings blackout.
        """
        # Get next earnings
        events = self.get_events_for_portfolio([symbol], days_ahead=60)

        if not events:
            return {
                'conflict': False,
                'note': 'No upcoming earnings found'
            }

        next_event = events[0]
        windows = get_trading_windows(next_event)

        blackout_start = windows.get('blackout_start')
        if not blackout_start:
            return {'conflict': False, 'note': 'Unknown timing'}

        if proposed_entry_date >= blackout_start.date():
            return {
                'conflict': True,
                'reason': 'inside_blackout_window',
                'earnings_date': next_event.report_date,
                'earnings_timing': next_event.timing.value,
                'recommendation': 'Do not enter new position'
            }

        days_until_blackout = (blackout_start.date() - proposed_entry_date).days

        if days_until_blackout < 5:
            return {
                'conflict': False,
                'warning': True,
                'days_until_blackout': days_until_blackout,
                'recommendation': 'Short holding period only'
            }

        return {
            'conflict': False,
            'days_until_blackout': days_until_blackout,
            'earnings_date': next_event.report_date
        }
```

---

## Data Quality Monitoring

### Validation Framework

```python
class CalendarDataQuality:
    """
    Monitor and validate earnings calendar data quality.
    """

    def __init__(self, calendar_manager: ProductionEarningsCalendar):
        self.calendar = calendar_manager
        self.quality_metrics = {}

    def run_quality_checks(self) -> Dict:
        """
        Run comprehensive data quality checks.
        """
        results = {
            'timestamp': datetime.now(),
            'checks': {}
        }

        # Check 1: Source agreement
        results['checks']['source_agreement'] = self._check_source_agreement()

        # Check 2: Timing consistency
        results['checks']['timing_consistency'] = self._check_timing_patterns()

        # Check 3: Date reasonableness
        results['checks']['date_validation'] = self._check_date_validity()

        # Check 4: Coverage gaps
        results['checks']['coverage'] = self._check_coverage()

        # Overall score
        scores = [
            c['score'] for c in results['checks'].values()
            if 'score' in c
        ]
        results['overall_score'] = sum(scores) / len(scores) if scores else 0

        return results

    def _check_source_agreement(self) -> Dict:
        """Check agreement between data sources."""
        # Compare dates from multiple sources
        disagreements = []

        # Query each source separately
        for symbol in self._get_active_symbols():
            dates_by_source = {}

            for source in [
                EarningsDataProvider.POLYGON,
                EarningsDataProvider.FINNHUB
            ]:
                try:
                    data = self.calendar.api._fetch_from_source(
                        source,
                        date.today(),
                        date.today() + timedelta(days=30),
                        [symbol]
                    )
                    if not data.empty:
                        dates_by_source[source.value] = data.iloc[0]['report_date']
                except Exception:
                    continue

            if len(dates_by_source) > 1:
                dates = list(dates_by_source.values())
                if len(set(dates)) > 1:
                    disagreements.append({
                        'symbol': symbol,
                        'dates': dates_by_source
                    })

        agreement_rate = 1 - (len(disagreements) / max(len(self._get_active_symbols()), 1))

        return {
            'score': agreement_rate,
            'disagreements': disagreements,
            'interpretation': 'HIGH' if agreement_rate > 0.95 else
                            'MEDIUM' if agreement_rate > 0.85 else 'LOW'
        }

    def _check_timing_patterns(self) -> Dict:
        """Check if timing follows historical patterns."""
        deviations = []

        for symbol in self._get_active_symbols():
            pattern = self.calendar.timing_resolver._timing_patterns.get(symbol, {})

            if pattern.get('consistency', 0) > 0.9:
                # Symbol has strong pattern
                # Check if upcoming timing matches
                events = self.calendar.get_events_for_portfolio([symbol])

                if events:
                    announced = events[0].timing
                    typical = pattern['typical_timing']

                    if announced != typical and announced != EarningsTiming.UNKNOWN:
                        deviations.append({
                            'symbol': symbol,
                            'typical': typical.value,
                            'announced': announced.value,
                            'note': 'Verify - unusual timing'
                        })

        return {
            'score': 1 - (len(deviations) / max(len(self._get_active_symbols()), 1)),
            'deviations': deviations
        }
```

---

## Academic References

1. **DellaVigna, S., & Pollet, J. M. (2009)**. "Investor Inattention and Friday Earnings Announcements." *Journal of Finance*.

2. **Hirshleifer, D., Lim, S. S., & Teoh, S. H. (2009)**. "Driven to Distraction: Extraneous Events and Underreaction to Earnings News." *Journal of Finance*.

3. **deHaan, E., Shevlin, T., & Thornock, J. (2015)**. "Market (In)Attention and the Strategic Scheduling and Timing of Earnings Announcements." *Journal of Accounting and Economics*.

4. **Johnson, T. L., & So, E. C. (2018)**. "Time Will Tell: Information in the Timing of Scheduled Earnings News." *Journal of Financial and Quantitative Analysis*.

---

## Document Metadata

```yaml
version: "1.0.0"
created: "2025-12-12"
status: "published"
tags: ["earnings", "calendar", "data-integration", "event-timing", "api"]
code_lines: 850
```

---

**END OF DOCUMENT**
