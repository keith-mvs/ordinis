# Geopolitical Events Trading

## Overview

Geopolitical events - elections, wars, policy changes, sanctions, and international tensions - create market dislocations with unique risk/reward profiles. Unlike scheduled economic releases, geopolitical events are often unpredictable in timing but follow identifiable patterns in market impact.

---

## Event Classification

### Geopolitical Event Taxonomy

```python
from dataclasses import dataclass
from datetime import datetime, date
from typing import Optional, List, Dict
from enum import Enum
import numpy as np
import pandas as pd


class GeopoliticalEventType(Enum):
    """Categories of geopolitical events."""
    ELECTION = "election"
    MILITARY_CONFLICT = "military_conflict"
    TRADE_POLICY = "trade_policy"
    SANCTIONS = "sanctions"
    CENTRAL_BANK_POLICY = "central_bank"
    REGULATORY_CHANGE = "regulatory"
    POLITICAL_CRISIS = "political_crisis"
    TERRORISM = "terrorism"
    NATURAL_DISASTER = "natural_disaster"
    PANDEMIC = "pandemic"


class EventPredictability(Enum):
    """How predictable is the event timing?"""
    SCHEDULED = "scheduled"          # Elections, policy votes
    SEMI_SCHEDULED = "semi_scheduled"  # Summit outcomes
    SURPRISE = "surprise"            # Wars, attacks, crises


class MarketImpactScope(Enum):
    """Scope of market impact."""
    GLOBAL = "global"
    REGIONAL = "regional"
    SECTOR = "sector"
    SINGLE_ASSET = "single_asset"


@dataclass
class GeopoliticalEvent:
    """Structured geopolitical event."""
    event_type: GeopoliticalEventType
    predictability: EventPredictability
    impact_scope: MarketImpactScope

    # Event details
    description: str
    date_occurred: datetime
    countries_involved: List[str]

    # Market impact
    expected_impact: str  # 'bullish', 'bearish', 'uncertain'
    impact_duration: str  # 'short', 'medium', 'long'

    # Affected assets
    primary_affected: List[str]  # Tickers, sectors, or asset classes
    secondary_affected: List[str]

    # Historical analog (if any)
    historical_precedent: Optional[str] = None


class GeopoliticalEventClassifier:
    """
    Classify and analyze geopolitical events.
    """

    # Historical impact patterns
    EVENT_IMPACT_PATTERNS = {
        GeopoliticalEventType.ELECTION: {
            'typical_vol_spike': 0.15,
            'duration_days': 30,
            'uncertainty_premium': 0.05,
            'affected_assets': ['equities', 'fx', 'rates']
        },
        GeopoliticalEventType.MILITARY_CONFLICT: {
            'typical_vol_spike': 0.30,
            'duration_days': 90,
            'flight_to_safety': True,
            'affected_assets': ['oil', 'gold', 'defense', 'equities', 'fx']
        },
        GeopoliticalEventType.TRADE_POLICY: {
            'typical_vol_spike': 0.10,
            'duration_days': 60,
            'affected_assets': ['equities', 'fx', 'affected_sectors']
        },
        GeopoliticalEventType.SANCTIONS: {
            'typical_vol_spike': 0.20,
            'duration_days': 180,
            'affected_assets': ['fx', 'commodities', 'targeted_companies']
        }
    }

    def classify_event(
        self,
        event_description: str,
        countries: List[str],
        is_scheduled: bool
    ) -> GeopoliticalEvent:
        """
        Classify a geopolitical event based on description.
        """
        # Keyword-based classification
        event_type = self._detect_event_type(event_description)

        # Determine predictability
        if is_scheduled:
            predictability = EventPredictability.SCHEDULED
        elif 'summit' in event_description.lower():
            predictability = EventPredictability.SEMI_SCHEDULED
        else:
            predictability = EventPredictability.SURPRISE

        # Determine scope
        scope = self._determine_scope(event_type, countries)

        return GeopoliticalEvent(
            event_type=event_type,
            predictability=predictability,
            impact_scope=scope,
            description=event_description,
            date_occurred=datetime.now(),
            countries_involved=countries,
            expected_impact='uncertain',
            impact_duration=self._estimate_duration(event_type),
            primary_affected=self._get_primary_affected(event_type, countries),
            secondary_affected=[]
        )

    def _detect_event_type(self, description: str) -> GeopoliticalEventType:
        """Detect event type from description."""
        desc_lower = description.lower()

        if any(w in desc_lower for w in ['election', 'vote', 'referendum']):
            return GeopoliticalEventType.ELECTION
        elif any(w in desc_lower for w in ['war', 'military', 'invasion', 'attack']):
            return GeopoliticalEventType.MILITARY_CONFLICT
        elif any(w in desc_lower for w in ['tariff', 'trade', 'import', 'export']):
            return GeopoliticalEventType.TRADE_POLICY
        elif any(w in desc_lower for w in ['sanction', 'embargo', 'freeze']):
            return GeopoliticalEventType.SANCTIONS
        else:
            return GeopoliticalEventType.POLITICAL_CRISIS

    def _determine_scope(
        self,
        event_type: GeopoliticalEventType,
        countries: List[str]
    ) -> MarketImpactScope:
        """Determine market impact scope."""
        major_economies = ['US', 'CN', 'EU', 'JP', 'UK', 'DE']

        if event_type == GeopoliticalEventType.MILITARY_CONFLICT:
            return MarketImpactScope.GLOBAL

        if len(countries) > 3:
            return MarketImpactScope.GLOBAL

        if any(c in major_economies for c in countries):
            return MarketImpactScope.GLOBAL

        return MarketImpactScope.REGIONAL
```

---

## Election Trading

### Election Cycle Patterns

```python
class ElectionTradingStrategy:
    """
    Trading strategies around elections.
    """

    # Historical US election patterns
    US_ELECTION_PATTERNS = {
        'pre_election_uncertainty': {
            'start_months_before': 3,
            'vix_elevation': 0.15,  # 15% above normal
            'sector_rotation': ['defense', 'healthcare', 'financials']
        },
        'election_week': {
            'expected_vol_spike': 0.25,
            'resolution_rally_probability': 0.70,  # Historical
            'uncertain_outcome_risk': 0.50
        },
        'post_election': {
            'relief_rally_days': 30,
            'policy_positioning_months': 3,
            'historical_avg_return': 0.03  # First 3 months
        }
    }

    def analyze_election_positioning(
        self,
        election_date: date,
        current_date: date,
        polling_margin: float,  # Leading candidate's margin
        incumbent_party: str,
        challenging_party: str
    ) -> Dict:
        """
        Generate positioning recommendations for election.
        """
        days_to_election = (election_date - current_date).days

        # Pre-election phase
        if days_to_election > 90:
            return {
                'phase': 'early',
                'action': 'MONITOR',
                'vol_position': 'NEUTRAL',
                'note': 'Too early for election-specific positioning'
            }

        elif 30 < days_to_election <= 90:
            # Build election hedges
            return {
                'phase': 'pre_election_hedging',
                'action': 'BUILD_HEDGES',
                'vol_position': 'LONG_VIX',
                'equity_action': 'REDUCE_BETA',
                'size': 'GRADUAL',
                'rationale': 'Uncertainty premium building'
            }

        elif 7 < days_to_election <= 30:
            # Active positioning based on outcome scenarios
            scenarios = self._build_election_scenarios(
                incumbent_party, challenging_party, polling_margin
            )
            return {
                'phase': 'active_positioning',
                'action': 'SCENARIO_BASED',
                'scenarios': scenarios,
                'vol_position': 'LONG_VIX',
                'note': 'Position for specific outcomes'
            }

        elif 0 < days_to_election <= 7:
            # Final week - reduce risk
            return {
                'phase': 'election_week',
                'action': 'REDUCE_ALL_RISK',
                'vol_position': 'MAINTAIN_HEDGES',
                'equity_action': 'FLATTEN',
                'rationale': 'Binary event risk'
            }

        else:  # Post-election
            return self._post_election_signal(
                abs(days_to_election),  # Days since election
                polling_margin
            )

    def _build_election_scenarios(
        self,
        incumbent: str,
        challenger: str,
        margin: float
    ) -> List[Dict]:
        """
        Build election outcome scenarios.
        """
        scenarios = []

        # Scenario 1: Incumbent wins (as expected if leading)
        scenarios.append({
            'outcome': f'{incumbent}_wins_expected',
            'probability': 0.5 + margin/2 if margin > 0 else 0.3,
            'market_reaction': 'RALLY',
            'magnitude': 'SMALL',
            'sectors': {
                'winners': ['status_quo_beneficiaries'],
                'losers': ['policy_change_plays']
            }
        })

        # Scenario 2: Challenger wins (upset if trailing)
        scenarios.append({
            'outcome': f'{challenger}_wins',
            'probability': 0.5 - margin/2 if margin > 0 else 0.7,
            'market_reaction': 'VOLATILE',
            'magnitude': 'LARGE',
            'sectors': {
                'winners': ['challenger_policy_beneficiaries'],
                'losers': ['incumbent_policy_beneficiaries']
            }
        })

        # Scenario 3: Contested/Delayed result
        scenarios.append({
            'outcome': 'contested_result',
            'probability': 0.10 if abs(margin) > 0.05 else 0.25,
            'market_reaction': 'SELL_OFF',
            'magnitude': 'LARGE',
            'sectors': {
                'winners': ['gold', 'utilities', 'staples'],
                'losers': ['high_beta', 'small_caps']
            }
        })

        return scenarios

    def _post_election_signal(
        self,
        days_since: int,
        final_margin: float
    ) -> Dict:
        """
        Post-election trading signal.
        """
        if days_since < 5:
            return {
                'phase': 'immediate_post',
                'action': 'WATCH_FOR_RESOLUTION_RALLY',
                'vol_position': 'REDUCE_HEDGES',
                'equity_action': 'ADD_ON_DIPS'
            }

        elif days_since < 30:
            return {
                'phase': 'policy_positioning',
                'action': 'SECTOR_ROTATION',
                'vol_position': 'NEUTRAL',
                'focus': 'Position for winner\'s policy agenda'
            }

        else:
            return {
                'phase': 'normal',
                'action': 'STANDARD_STRATEGY',
                'note': 'Election effect dissipated'
            }
```

---

## Military Conflict Analysis

### Conflict Market Impact

```python
class ConflictTradingStrategy:
    """
    Trading around military conflicts and geopolitical tensions.
    """

    # Historical conflict patterns
    CONFLICT_PLAYBOOK = {
        'initial_shock': {
            'duration_days': 3,
            'spx_avg_drop': -0.05,
            'vix_spike': 0.50,
            'gold_rally': 0.03,
            'oil_spike': 0.10,
            'usd_move': 0.02  # Generally stronger
        },
        'uncertainty_phase': {
            'duration_days': 30,
            'vol_remains_elevated': True,
            'risk_premium': 0.10
        },
        'resolution_or_normalization': {
            'duration_days': 90,
            'recovery_probability': 0.80,
            'full_recovery_months': 6
        }
    }

    SAFE_HAVEN_ASSETS = ['GLD', 'TLT', 'UUP', 'VXX', 'CHF', 'JPY']
    RISK_ASSETS = ['SPY', 'QQQ', 'EEM', 'HYG', 'JNK']
    CONFLICT_BENEFICIARIES = ['XLE', 'LMT', 'RTX', 'NOC', 'GD']  # Energy, defense

    def analyze_conflict_outbreak(
        self,
        conflict_type: str,
        regions_involved: List[str],
        severity: str,  # 'low', 'medium', 'high', 'extreme'
        oil_producing_region: bool
    ) -> Dict:
        """
        Analyze market implications of conflict outbreak.
        """
        severity_multiplier = {
            'low': 0.5,
            'medium': 1.0,
            'high': 1.5,
            'extreme': 2.0
        }.get(severity, 1.0)

        base_impact = self.CONFLICT_PLAYBOOK['initial_shock']

        expected_impact = {
            'equities': base_impact['spx_avg_drop'] * severity_multiplier,
            'vix': base_impact['vix_spike'] * severity_multiplier,
            'gold': base_impact['gold_rally'] * severity_multiplier,
            'oil': base_impact['oil_spike'] * severity_multiplier * (1.5 if oil_producing_region else 1.0),
            'usd': base_impact['usd_move'] * severity_multiplier
        }

        # Trading recommendations
        recommendations = {
            'immediate': {
                'reduce_risk_assets': True,
                'add_safe_havens': self.SAFE_HAVEN_ASSETS,
                'sector_plays': self.CONFLICT_BENEFICIARIES if severity in ['high', 'extreme'] else [],
                'timeline': '0-3 days'
            },
            'short_term': {
                'wait_for_clarity': True,
                'vol_position': 'stay_long_if_uncertainty_persists',
                'timeline': '3-30 days'
            },
            'medium_term': {
                'look_for_normalization_trades': True,
                'recovery_plays': self.RISK_ASSETS,
                'timeline': '30-90 days'
            }
        }

        return {
            'expected_impact': expected_impact,
            'recommendations': recommendations,
            'key_monitors': self._get_escalation_indicators(conflict_type)
        }

    def _get_escalation_indicators(self, conflict_type: str) -> List[str]:
        """
        Key indicators to monitor for escalation/de-escalation.
        """
        return [
            'UN Security Council activity',
            'NATO/alliance statements',
            'Commodity supply disruption news',
            'Sanctions announcements',
            'Diplomatic channels status',
            'Military movement reports',
            'Credit spreads on affected sovereigns',
            'Currency moves of involved countries'
        ]

    def flight_to_safety_signal(
        self,
        gold_move_1d: float,
        vix_move_1d: float,
        treasury_move_1d: float,
        jpy_move_1d: float
    ) -> Dict:
        """
        Detect flight-to-safety behavior.
        """
        # Count safe haven rallies
        safe_haven_count = 0
        if gold_move_1d > 0.01:
            safe_haven_count += 1
        if vix_move_1d > 0.10:
            safe_haven_count += 1
        if treasury_move_1d > 0.005:  # Price up = yield down
            safe_haven_count += 1
        if jpy_move_1d > 0.005:  # JPY strengthening
            safe_haven_count += 1

        if safe_haven_count >= 3:
            return {
                'signal': 'STRONG_FLIGHT_TO_SAFETY',
                'action': 'REDUCE_RISK_IMMEDIATELY',
                'add_hedges': True
            }
        elif safe_haven_count >= 2:
            return {
                'signal': 'MODERATE_RISK_OFF',
                'action': 'REDUCE_BETA',
                'monitor': True
            }
        else:
            return {
                'signal': 'NORMAL',
                'action': 'STANDARD_RISK'
            }
```

---

## Trade Policy & Tariffs

### Trade War Trading

```python
class TradePolicyStrategy:
    """
    Trading around trade policy changes and tariff announcements.
    """

    def analyze_tariff_announcement(
        self,
        announcing_country: str,
        target_country: str,
        affected_goods: List[str],
        tariff_rate: float,
        effective_date: date
    ) -> Dict:
        """
        Analyze market implications of tariff announcement.
        """
        # Identify affected sectors
        affected_sectors = self._map_goods_to_sectors(affected_goods)

        # Estimate impact
        impact = {
            'importing_country': {
                'consumer_impact': 'negative',  # Higher prices
                'affected_retailers': 'negative',
                'domestic_competitors': 'positive'
            },
            'exporting_country': {
                'exporters': 'negative',
                'currency': 'weaker',
                'domestic_pivot': 'positive'
            }
        }

        # Retaliation probability
        retaliation_prob = self._estimate_retaliation(
            announcing_country, target_country, tariff_rate
        )

        # Trading opportunities
        opportunities = []

        # Direct impact plays
        for sector in affected_sectors:
            opportunities.append({
                'trade': f'Short {target_country} {sector} exporters',
                'rationale': 'Direct tariff impact',
                'risk': 'Retaliation could shift focus'
            })

        # Beneficiary plays
        opportunities.append({
            'trade': f'Long domestic alternatives in {announcing_country}',
            'rationale': 'Domestic substitution',
            'risk': 'Supply chain adjustments take time'
        })

        # Currency play
        opportunities.append({
            'trade': f'Short {target_country} currency',
            'rationale': 'Export weakness, capital flight',
            'risk': 'Policy intervention'
        })

        return {
            'impact_assessment': impact,
            'retaliation_probability': retaliation_prob,
            'opportunities': opportunities,
            'key_dates': {
                'effective': effective_date,
                'watch_for_retaliation': effective_date - pd.Timedelta(days=7)
            }
        }

    def _map_goods_to_sectors(self, goods: List[str]) -> List[str]:
        """Map tariffed goods to market sectors."""
        goods_sector_map = {
            'steel': ['materials', 'industrials'],
            'aluminum': ['materials', 'industrials'],
            'automobiles': ['consumer_discretionary', 'industrials'],
            'electronics': ['technology', 'consumer_discretionary'],
            'agriculture': ['consumer_staples', 'materials'],
            'semiconductors': ['technology'],
            'pharmaceuticals': ['healthcare'],
            'textiles': ['consumer_discretionary']
        }

        sectors = set()
        for good in goods:
            good_lower = good.lower()
            for key, sector_list in goods_sector_map.items():
                if key in good_lower:
                    sectors.update(sector_list)

        return list(sectors) if sectors else ['general_market']

    def _estimate_retaliation(
        self,
        announcer: str,
        target: str,
        rate: float
    ) -> float:
        """Estimate probability of retaliation."""
        # Major economies likely to retaliate
        major_retaliators = ['CN', 'EU', 'US', 'JP']

        base_prob = 0.5

        if target in major_retaliators:
            base_prob += 0.3

        if rate > 0.25:  # High tariff rate
            base_prob += 0.15

        return min(base_prob, 0.95)

    def trade_escalation_deescalation(
        self,
        current_stage: str,  # 'threat', 'implemented', 'negotiating', 'resolved'
        direction: str  # 'escalating', 'stable', 'deescalating'
    ) -> Dict:
        """
        Position for trade war escalation/de-escalation.
        """
        if current_stage == 'threat' and direction == 'escalating':
            return {
                'action': 'REDUCE_EXPOSED_SECTORS',
                'vol_position': 'LONG',
                'safe_havens': 'ADD',
                'timeline': 'Until implementation or resolution'
            }

        elif current_stage == 'implemented' and direction == 'stable':
            return {
                'action': 'FIND_WINNERS_LOSERS',
                'vol_position': 'NEUTRAL',
                'focus': 'Sector rotation to beneficiaries'
            }

        elif direction == 'deescalating':
            return {
                'action': 'ADD_RISK',
                'vol_position': 'SHORT',
                'focus': 'Previously punished names',
                'timeline': 'Resolution rally opportunity'
            }

        return {'action': 'MONITOR'}
```

---

## Sanctions Analysis

### Sanctions Trading Framework

```python
class SanctionsTradingStrategy:
    """
    Trading around economic sanctions.
    """

    def analyze_sanctions_announcement(
        self,
        sanctioning_entity: str,
        target_entity: str,
        sanction_type: str,  # 'trade', 'financial', 'travel', 'comprehensive'
        targeted_individuals: List[str],
        targeted_companies: List[str]
    ) -> Dict:
        """
        Analyze sanctions implications.
        """
        # Impact assessment
        if sanction_type == 'comprehensive':
            severity = 'extreme'
            duration = 'long_term'
        elif sanction_type == 'financial':
            severity = 'high'
            duration = 'medium_term'
        elif sanction_type == 'trade':
            severity = 'medium'
            duration = 'medium_term'
        else:
            severity = 'low'
            duration = 'short_term'

        # Direct impacts
        direct_impacts = {
            'targeted_companies': {
                'action': 'EXIT_POSITIONS',
                'reason': 'Compliance risk, liquidity risk',
                'urgency': 'IMMEDIATE'
            },
            'target_country_assets': {
                'currency': 'SELL',
                'equities': 'SELL',
                'bonds': 'SELL',
                'reason': 'Capital flight, economic damage'
            }
        }

        # Indirect impacts
        indirect_impacts = self._analyze_indirect_sanctions_impact(
            target_entity, sanction_type
        )

        # Beneficiaries
        beneficiaries = self._find_sanctions_beneficiaries(
            target_entity, sanction_type
        )

        return {
            'severity': severity,
            'duration': duration,
            'direct_impacts': direct_impacts,
            'indirect_impacts': indirect_impacts,
            'beneficiaries': beneficiaries,
            'compliance_note': 'Verify all trades for sanctions compliance'
        }

    def _analyze_indirect_sanctions_impact(
        self,
        target: str,
        sanction_type: str
    ) -> Dict:
        """
        Analyze knock-on effects of sanctions.
        """
        impacts = {}

        if sanction_type in ['comprehensive', 'trade']:
            impacts['supply_chain'] = {
                'companies_with_exposure': 'REVIEW',
                'action': 'Identify supply chain dependencies'
            }

        if sanction_type in ['comprehensive', 'financial']:
            impacts['financial_system'] = {
                'banks_with_exposure': 'REVIEW',
                'correspondent_banking': 'May be disrupted'
            }

        # Commodity impacts
        impacts['commodities'] = self._commodity_sanctions_impact(target)

        return impacts

    def _commodity_sanctions_impact(self, target: str) -> Dict:
        """
        Commodity market impact from sanctions.
        """
        commodity_producers = {
            'RU': ['oil', 'natural_gas', 'wheat', 'palladium', 'nickel'],
            'IR': ['oil'],
            'VE': ['oil'],
            'CN': ['rare_earths', 'manufacturing_inputs']
        }

        commodities = commodity_producers.get(target, [])

        if commodities:
            return {
                'affected_commodities': commodities,
                'expected_direction': 'UP',
                'rationale': 'Supply disruption',
                'trade': f'Long {", ".join(commodities)}'
            }

        return {'impact': 'minimal'}
```

---

## Risk Management

### Geopolitical Risk Management

```python
class GeopoliticalRiskManager:
    """
    Risk management for geopolitical events.
    """

    def __init__(
        self,
        max_geo_risk_pct: float = 0.10,
        hedge_trigger_events: List[GeopoliticalEventType] = None
    ):
        self.max_geo_risk = max_geo_risk_pct
        self.hedge_triggers = hedge_trigger_events or [
            GeopoliticalEventType.MILITARY_CONFLICT,
            GeopoliticalEventType.ELECTION
        ]

    def assess_portfolio_geo_exposure(
        self,
        positions: List[Dict],
        country_exposures: Dict[str, float]
    ) -> Dict:
        """
        Assess portfolio exposure to geopolitical risk.
        """
        # Identify concentrated exposures
        concentrated = [
            (country, exp) for country, exp in country_exposures.items()
            if exp > 0.20
        ]

        # Risk hotspots (regions with elevated tensions)
        risk_hotspots = self._get_current_hotspots()

        # Overlap
        exposed_hotspots = [
            (country, exp) for country, exp in concentrated
            if country in risk_hotspots
        ]

        risk_level = 'LOW'
        if len(exposed_hotspots) > 0:
            total_hotspot_exposure = sum(exp for _, exp in exposed_hotspots)
            if total_hotspot_exposure > 0.30:
                risk_level = 'HIGH'
            elif total_hotspot_exposure > 0.15:
                risk_level = 'MEDIUM'

        return {
            'concentrated_exposures': concentrated,
            'hotspot_exposures': exposed_hotspots,
            'risk_level': risk_level,
            'recommendation': 'Hedge or reduce' if risk_level == 'HIGH' else 'Monitor'
        }

    def calculate_geo_var(
        self,
        position_value: float,
        geo_event: GeopoliticalEvent,
        historical_shocks: List[float]
    ) -> Dict:
        """
        Calculate VaR for geopolitical event.
        """
        # Use historical shock distribution
        if not historical_shocks:
            # Use defaults based on event type
            if geo_event.event_type == GeopoliticalEventType.MILITARY_CONFLICT:
                shock_estimate = 0.10
            elif geo_event.event_type == GeopoliticalEventType.ELECTION:
                shock_estimate = 0.05
            else:
                shock_estimate = 0.03
        else:
            shock_estimate = np.percentile(np.abs(historical_shocks), 95)

        return {
            'var_95': position_value * shock_estimate,
            'var_99': position_value * shock_estimate * 1.5,
            'worst_case': position_value * shock_estimate * 2.0,
            'note': 'Geopolitical VaR is highly uncertain'
        }
```

---

## Academic References

1. **Berkman, H., Jacobsen, B., & Lee, J. B. (2011)**. "Time-varying Rare Disaster Risk and Stock Returns." *Journal of Financial Economics*.

2. **Glick, R., & Taylor, A. M. (2010)**. "Collateral Damage: Trade Disruption and the Economic Impact of War." *Review of Economics and Statistics*.

3. **Baker, S. R., Bloom, N., & Davis, S. J. (2016)**. "Measuring Economic Policy Uncertainty." *Quarterly Journal of Economics*.

4. **Brogaard, J., & Detzel, A. (2015)**. "The Asset-Pricing Implications of Government Economic Policy Uncertainty." *Management Science*.

---

## Document Metadata

```yaml
version: "1.0.0"
created: "2025-12-12"
status: "published"
tags: ["geopolitical", "elections", "conflicts", "sanctions", "trade-policy"]
code_lines: 650
```

---

**END OF DOCUMENT**
