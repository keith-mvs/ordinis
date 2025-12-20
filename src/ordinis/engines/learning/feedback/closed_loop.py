"""
LearningEngine Closed-Loop Feedback Integration.

Implements production-grade learning feedback:
- Trade outcome collection and analysis
- Strategy performance feedback loop
- Model retraining triggers
- A/B testing framework
- Continuous improvement pipeline

Step 9 of Trade Enhancement Roadmap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum, auto
import hashlib
import logging
from typing import Any, Callable
import random

import numpy as np

logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """Types of feedback events."""
    
    TRADE_OUTCOME = auto()  # Trade completed
    SIGNAL_ACCURACY = auto()  # Signal prediction vs reality
    EXECUTION_QUALITY = auto()  # Fill quality
    RISK_EVENT = auto()  # Risk limit triggered
    MODEL_PREDICTION = auto()  # Model prediction vs actual
    USER_FEEDBACK = auto()  # Explicit user feedback
    SYSTEM_ERROR = auto()  # System errors


class ModelType(Enum):
    """Types of models in the system."""
    
    SIGNAL_GENERATOR = auto()
    RISK_PREDICTOR = auto()
    POSITION_SIZER = auto()
    MARKET_REGIME = auto()
    EXECUTION_TIMING = auto()


class ExperimentStatus(Enum):
    """A/B experiment status."""
    
    DRAFT = auto()
    RUNNING = auto()
    PAUSED = auto()
    COMPLETED = auto()
    CANCELLED = auto()


@dataclass
class FeedbackEvent:
    """Single feedback event."""
    
    event_id: str
    feedback_type: FeedbackType
    timestamp: datetime
    
    # Source
    source_engine: str
    source_model: ModelType | None = None
    
    # Context
    symbol: str | None = None
    strategy_id: str | None = None
    
    # Prediction vs Reality
    predicted_value: float | None = None
    actual_value: float | None = None
    prediction_error: float | None = None
    
    # Trade outcome
    pnl: Decimal | None = None
    holding_period: timedelta | None = None
    
    # Quality metrics
    quality_score: float | None = None
    
    # Metadata
    features: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_positive(self) -> bool:
        """Check if feedback is positive."""
        if self.pnl is not None:
            return self.pnl > 0
        if self.quality_score is not None:
            return self.quality_score > 0.5
        return False


@dataclass
class PerformanceSnapshot:
    """Strategy/model performance snapshot."""
    
    snapshot_id: str
    timestamp: datetime
    
    # Identity
    strategy_id: str | None = None
    model_type: ModelType | None = None
    model_version: str | None = None
    
    # Performance metrics
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    total_pnl: Decimal = field(default_factory=lambda: Decimal("0"))
    
    # Prediction metrics (for ML models)
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1_score: float | None = None
    
    # Period
    period_start: datetime | None = None
    period_end: datetime | None = None
    
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrainingTrigger:
    """Trigger for model retraining."""
    
    trigger_id: str
    triggered_at: datetime
    model_type: ModelType
    model_version: str
    
    # Reason
    reason: str
    severity: str  # "low", "medium", "high", "critical"
    
    # Metrics that triggered
    triggering_metrics: dict[str, float]
    thresholds: dict[str, float]
    
    # Recommendation
    recommended_action: str
    auto_retrain: bool = False


@dataclass
class Experiment:
    """A/B experiment definition."""
    
    experiment_id: str
    name: str
    description: str
    
    # Variants
    control_config: dict[str, Any]
    treatment_config: dict[str, Any]
    
    # Traffic split
    treatment_percentage: float = 0.5
    
    # Status
    status: ExperimentStatus = ExperimentStatus.DRAFT
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    ended_at: datetime | None = None
    
    # Results
    control_metrics: dict[str, float] = field(default_factory=dict)
    treatment_metrics: dict[str, float] = field(default_factory=dict)
    control_count: int = 0
    treatment_count: int = 0
    
    # Statistical significance
    p_value: float | None = None
    is_significant: bool = False
    winner: str | None = None  # "control", "treatment", or None
    
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningConfig:
    """Configuration for learning engine."""
    
    # Feedback collection
    min_samples_for_snapshot: int = 100
    snapshot_interval_hours: int = 24
    
    # Retraining triggers
    min_accuracy_threshold: float = 0.55
    max_drawdown_threshold: float = 0.15
    min_sharpe_threshold: float = 0.5
    degradation_threshold: float = 0.20  # 20% degradation
    
    # A/B testing
    min_samples_per_variant: int = 50
    significance_level: float = 0.05
    
    # Auto-retraining (G-LE-2: enabled by default with safeguards)
    auto_retrain_enabled: bool = True
    min_retrain_samples: int = 1000


class FeedbackCollector:
    """
    Collects and aggregates feedback events.
    
    Central hub for all learning feedback.
    """
    
    def __init__(self, config: LearningConfig | None = None) -> None:
        """Initialize collector."""
        self.config = config or LearningConfig()
        self._events: list[FeedbackEvent] = []
        self._snapshots: list[PerformanceSnapshot] = []
        self._triggers: list[RetrainingTrigger] = []
        self._last_snapshot: dict[str, datetime] = {}
        
        # Callbacks
        self._on_trigger: list[Callable[[RetrainingTrigger], None]] = []
        
    def record_event(self, event: FeedbackEvent) -> None:
        """Record a feedback event."""
        self._events.append(event)
        
        logger.debug(
            f"Recorded feedback event {event.event_id}: "
            f"{event.feedback_type.name} from {event.source_engine}"
        )
        
        # Check if snapshot needed
        self._maybe_create_snapshot(event)
        
    def record_trade_outcome(
        self,
        strategy_id: str,
        symbol: str,
        pnl: Decimal,
        holding_period: timedelta,
        predicted_direction: float | None = None,
        actual_direction: float | None = None,
        features: dict[str, float] | None = None,
    ) -> FeedbackEvent:
        """Record a trade outcome."""
        event = FeedbackEvent(
            event_id=self._generate_id(),
            feedback_type=FeedbackType.TRADE_OUTCOME,
            timestamp=datetime.utcnow(),
            source_engine="trading",
            symbol=symbol,
            strategy_id=strategy_id,
            predicted_value=predicted_direction,
            actual_value=actual_direction,
            prediction_error=abs(predicted_direction - actual_direction) if predicted_direction and actual_direction else None,
            pnl=pnl,
            holding_period=holding_period,
            features=features or {},
        )
        
        self.record_event(event)
        return event
        
    def record_signal_accuracy(
        self,
        strategy_id: str,
        symbol: str,
        predicted_direction: float,
        actual_direction: float,
        confidence: float,
    ) -> FeedbackEvent:
        """Record signal prediction accuracy."""
        correct = (predicted_direction > 0) == (actual_direction > 0)
        
        event = FeedbackEvent(
            event_id=self._generate_id(),
            feedback_type=FeedbackType.SIGNAL_ACCURACY,
            timestamp=datetime.utcnow(),
            source_engine="signalcore",
            source_model=ModelType.SIGNAL_GENERATOR,
            symbol=symbol,
            strategy_id=strategy_id,
            predicted_value=predicted_direction,
            actual_value=actual_direction,
            prediction_error=abs(predicted_direction - actual_direction),
            quality_score=1.0 if correct else 0.0,
            metadata={"confidence": confidence},
        )
        
        self.record_event(event)
        return event
        
    def record_execution_quality(
        self,
        symbol: str,
        slippage_bps: float,
        execution_time_ms: float,
        fill_quality: str,
    ) -> FeedbackEvent:
        """Record execution quality feedback."""
        # Normalize quality score
        quality_map = {
            "EXCELLENT": 1.0,
            "GOOD": 0.8,
            "ACCEPTABLE": 0.6,
            "POOR": 0.3,
            "VERY_POOR": 0.0,
        }
        quality_score = quality_map.get(fill_quality, 0.5)
        
        event = FeedbackEvent(
            event_id=self._generate_id(),
            feedback_type=FeedbackType.EXECUTION_QUALITY,
            timestamp=datetime.utcnow(),
            source_engine="flowroute",
            symbol=symbol,
            quality_score=quality_score,
            metadata={
                "slippage_bps": slippage_bps,
                "execution_time_ms": execution_time_ms,
                "fill_quality": fill_quality,
            },
        )
        
        self.record_event(event)
        return event
        
    def get_events(
        self,
        feedback_type: FeedbackType | None = None,
        strategy_id: str | None = None,
        since: datetime | None = None,
        limit: int | None = None,
    ) -> list[FeedbackEvent]:
        """Get filtered feedback events."""
        events = self._events
        
        if feedback_type:
            events = [e for e in events if e.feedback_type == feedback_type]
        if strategy_id:
            events = [e for e in events if e.strategy_id == strategy_id]
        if since:
            events = [e for e in events if e.timestamp >= since]
            
        # Sort by timestamp descending
        events = sorted(events, key=lambda e: e.timestamp, reverse=True)
        
        if limit:
            events = events[:limit]
            
        return events
        
    def _generate_id(self) -> str:
        """Generate unique ID."""
        timestamp = datetime.utcnow().isoformat()
        hash_input = f"{timestamp}-{len(self._events)}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
        
    def _maybe_create_snapshot(self, event: FeedbackEvent) -> None:
        """Create snapshot if conditions met."""
        key = event.strategy_id or event.source_engine
        
        last = self._last_snapshot.get(key)
        if last:
            elapsed = (datetime.utcnow() - last).total_seconds() / 3600
            if elapsed < self.config.snapshot_interval_hours:
                return
                
        # Check sample count
        recent_events = self.get_events(
            strategy_id=event.strategy_id,
            since=datetime.utcnow() - timedelta(hours=self.config.snapshot_interval_hours),
        )
        
        if len(recent_events) >= self.config.min_samples_for_snapshot:
            self._create_snapshot(key, recent_events)
            
    def _create_snapshot(self, key: str, events: list[FeedbackEvent]) -> None:
        """Create performance snapshot from events."""
        trade_events = [e for e in events if e.feedback_type == FeedbackType.TRADE_OUTCOME]
        signal_events = [e for e in events if e.feedback_type == FeedbackType.SIGNAL_ACCURACY]
        
        # Calculate metrics
        total_pnl = sum(e.pnl for e in trade_events if e.pnl) or Decimal("0")
        wins = sum(1 for e in trade_events if e.pnl and e.pnl > 0)
        win_rate = wins / len(trade_events) if trade_events else 0
        
        # Signal accuracy
        if signal_events:
            accuracy = sum(e.quality_score or 0 for e in signal_events) / len(signal_events)
        else:
            accuracy = None
            
        snapshot = PerformanceSnapshot(
            snapshot_id=self._generate_id(),
            timestamp=datetime.utcnow(),
            strategy_id=key if events[0].strategy_id else None,
            total_trades=len(trade_events),
            win_rate=win_rate,
            total_pnl=total_pnl,
            accuracy=accuracy,
            period_start=min(e.timestamp for e in events),
            period_end=max(e.timestamp for e in events),
        )
        
        self._snapshots.append(snapshot)
        self._last_snapshot[key] = datetime.utcnow()
        
        # Check for retraining trigger
        self._check_retraining_trigger(snapshot)
        
        logger.info(
            f"Created performance snapshot for {key}: "
            f"{snapshot.total_trades} trades, {snapshot.win_rate:.0%} win rate"
        )
        
    def _check_retraining_trigger(self, snapshot: PerformanceSnapshot) -> None:
        """Check if retraining should be triggered."""
        reasons = []
        triggering_metrics = {}
        thresholds = {}
        
        # Check accuracy
        if snapshot.accuracy is not None:
            if snapshot.accuracy < self.config.min_accuracy_threshold:
                reasons.append(f"Accuracy below threshold ({snapshot.accuracy:.0%})")
                triggering_metrics["accuracy"] = snapshot.accuracy
                thresholds["accuracy"] = self.config.min_accuracy_threshold
                
        # Check win rate
        if snapshot.win_rate < 0.4:
            reasons.append(f"Win rate critically low ({snapshot.win_rate:.0%})")
            triggering_metrics["win_rate"] = snapshot.win_rate
            thresholds["win_rate"] = 0.4
            
        # Check for degradation vs previous
        if len(self._snapshots) >= 2:
            prev = self._snapshots[-2]
            if prev.accuracy and snapshot.accuracy:
                degradation = (prev.accuracy - snapshot.accuracy) / prev.accuracy
                if degradation > self.config.degradation_threshold:
                    reasons.append(f"Performance degradation ({degradation:.0%})")
                    triggering_metrics["degradation"] = degradation
                    thresholds["degradation"] = self.config.degradation_threshold
                    
        if reasons:
            severity = "critical" if len(reasons) > 1 else "medium"
            
            trigger = RetrainingTrigger(
                trigger_id=self._generate_id(),
                triggered_at=datetime.utcnow(),
                model_type=ModelType.SIGNAL_GENERATOR,
                model_version=snapshot.model_version or "unknown",
                reason="; ".join(reasons),
                severity=severity,
                triggering_metrics=triggering_metrics,
                thresholds=thresholds,
                recommended_action="retrain_model",
                auto_retrain=self.config.auto_retrain_enabled and severity == "critical",
            )
            
            self._triggers.append(trigger)
            
            # Notify callbacks
            for callback in self._on_trigger:
                try:
                    callback(trigger)
                except Exception as e:
                    logger.error(f"Trigger callback failed: {e}")
                    
            logger.warning(f"Retraining trigger: {trigger.reason}")
            
    def on_retraining_trigger(
        self,
        callback: Callable[[RetrainingTrigger], None],
    ) -> None:
        """Register callback for retraining triggers."""
        self._on_trigger.append(callback)


class ABTestingFramework:
    """
    A/B Testing Framework for strategy/model experiments.
    
    Enables controlled testing of changes.
    """
    
    def __init__(self, feedback_collector: FeedbackCollector) -> None:
        """Initialize framework."""
        self.feedback_collector = feedback_collector
        self._experiments: dict[str, Experiment] = {}
        self._assignments: dict[str, str] = {}  # entity -> variant
        
    def create_experiment(
        self,
        name: str,
        description: str,
        control_config: dict[str, Any],
        treatment_config: dict[str, Any],
        treatment_percentage: float = 0.5,
    ) -> Experiment:
        """Create a new A/B experiment."""
        experiment = Experiment(
            experiment_id=self._generate_id(),
            name=name,
            description=description,
            control_config=control_config,
            treatment_config=treatment_config,
            treatment_percentage=treatment_percentage,
        )
        
        self._experiments[experiment.experiment_id] = experiment
        
        logger.info(f"Created experiment '{name}' ({experiment.experiment_id})")
        return experiment
        
    def start_experiment(self, experiment_id: str) -> None:
        """Start an experiment."""
        exp = self._experiments.get(experiment_id)
        if not exp:
            raise ValueError(f"Experiment {experiment_id} not found")
            
        exp.status = ExperimentStatus.RUNNING
        exp.started_at = datetime.utcnow()
        
        logger.info(f"Started experiment '{exp.name}'")
        
    def get_variant(self, experiment_id: str, entity_id: str) -> str:
        """
        Get variant assignment for an entity.
        
        Returns "control" or "treatment"
        """
        exp = self._experiments.get(experiment_id)
        if not exp or exp.status != ExperimentStatus.RUNNING:
            return "control"
            
        # Check existing assignment
        assignment_key = f"{experiment_id}:{entity_id}"
        if assignment_key in self._assignments:
            return self._assignments[assignment_key]
            
        # Make new assignment
        if random.random() < exp.treatment_percentage:
            variant = "treatment"
        else:
            variant = "control"
            
        self._assignments[assignment_key] = variant
        
        return variant
        
    def record_result(
        self,
        experiment_id: str,
        entity_id: str,
        metrics: dict[str, float],
    ) -> None:
        """Record experiment result for an entity."""
        exp = self._experiments.get(experiment_id)
        if not exp:
            return
            
        variant = self.get_variant(experiment_id, entity_id)
        
        if variant == "control":
            exp.control_count += 1
            for key, value in metrics.items():
                exp.control_metrics[key] = (
                    exp.control_metrics.get(key, 0) * (exp.control_count - 1) + value
                ) / exp.control_count
        else:
            exp.treatment_count += 1
            for key, value in metrics.items():
                exp.treatment_metrics[key] = (
                    exp.treatment_metrics.get(key, 0) * (exp.treatment_count - 1) + value
                ) / exp.treatment_count
                
        # Check for significance
        self._check_significance(exp)
        
    def _check_significance(self, exp: Experiment) -> None:
        """Check if experiment has reached statistical significance."""
        min_samples = self.feedback_collector.config.min_samples_per_variant
        
        if exp.control_count < min_samples or exp.treatment_count < min_samples:
            return
            
        # Simple significance check (would use proper statistical test in production)
        # Using approximate z-test for proportions
        
        primary_metric = "win_rate"  # Or configurable
        
        if primary_metric not in exp.control_metrics or primary_metric not in exp.treatment_metrics:
            return
            
        p1 = exp.control_metrics[primary_metric]
        p2 = exp.treatment_metrics[primary_metric]
        n1 = exp.control_count
        n2 = exp.treatment_count
        
        # Pooled proportion
        p = (p1 * n1 + p2 * n2) / (n1 + n2)
        
        # Standard error
        se = np.sqrt(p * (1 - p) * (1/n1 + 1/n2))
        
        if se == 0:
            return
            
        # Z-score
        z = (p2 - p1) / se
        
        # Two-tailed p-value (approximate)
        from scipy import stats  # Would need scipy
        try:
            p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        except Exception:
            p_value = 1.0  # Fallback
            
        exp.p_value = p_value
        exp.is_significant = p_value < self.feedback_collector.config.significance_level
        
        if exp.is_significant:
            if p2 > p1:
                exp.winner = "treatment"
            else:
                exp.winner = "control"
                
            logger.info(
                f"Experiment '{exp.name}' significant! "
                f"Winner: {exp.winner} (p={p_value:.4f})"
            )
            
    def get_experiment_results(self, experiment_id: str) -> dict[str, Any]:
        """Get experiment results summary."""
        exp = self._experiments.get(experiment_id)
        if not exp:
            return {}
            
        return {
            "experiment_id": experiment_id,
            "name": exp.name,
            "status": exp.status.name,
            "control_count": exp.control_count,
            "treatment_count": exp.treatment_count,
            "control_metrics": exp.control_metrics,
            "treatment_metrics": exp.treatment_metrics,
            "p_value": exp.p_value,
            "is_significant": exp.is_significant,
            "winner": exp.winner,
            "treatment_lift": {
                k: (exp.treatment_metrics.get(k, 0) - exp.control_metrics.get(k, 0)) 
                   / exp.control_metrics.get(k, 1) 
                for k in exp.control_metrics
            } if exp.control_metrics else {},
        }
        
    def _generate_id(self) -> str:
        """Generate unique ID."""
        timestamp = datetime.utcnow().isoformat()
        hash_input = f"{timestamp}-{len(self._experiments)}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]


class ContinuousImprovementPipeline:
    """
    Continuous improvement pipeline.
    
    Orchestrates the feedback -> analysis -> improvement cycle.
    """
    
    def __init__(
        self,
        feedback_collector: FeedbackCollector,
        ab_testing: ABTestingFramework,
    ) -> None:
        """Initialize pipeline."""
        self.feedback_collector = feedback_collector
        self.ab_testing = ab_testing
        self._improvement_queue: list[dict[str, Any]] = []
        
        # Register for retraining triggers
        feedback_collector.on_retraining_trigger(self._handle_trigger)
        
    def _handle_trigger(self, trigger: RetrainingTrigger) -> None:
        """Handle retraining trigger."""
        improvement = {
            "trigger_id": trigger.trigger_id,
            "model_type": trigger.model_type.name,
            "reason": trigger.reason,
            "severity": trigger.severity,
            "recommended_action": trigger.recommended_action,
            "auto_retrain": trigger.auto_retrain,
            "status": "queued",
            "queued_at": datetime.utcnow(),
        }
        
        self._improvement_queue.append(improvement)
        
        if trigger.auto_retrain:
            logger.info(f"Auto-retraining queued for {trigger.model_type.name}")
            
    def get_pending_improvements(self) -> list[dict[str, Any]]:
        """Get pending improvement actions."""
        return [i for i in self._improvement_queue if i["status"] == "queued"]
        
    def generate_improvement_report(self) -> dict[str, Any]:
        """Generate overall improvement report."""
        # Get recent snapshots
        recent_snapshots = self.feedback_collector._snapshots[-10:]
        
        # Analyze trends
        if len(recent_snapshots) >= 2:
            first = recent_snapshots[0]
            last = recent_snapshots[-1]
            
            win_rate_trend = (last.win_rate - first.win_rate) if first.win_rate else 0
            accuracy_trend = (
                (last.accuracy - first.accuracy) 
                if last.accuracy and first.accuracy else 0
            )
        else:
            win_rate_trend = 0
            accuracy_trend = 0
            
        # Get experiment summaries
        experiments = [
            self.ab_testing.get_experiment_results(exp_id)
            for exp_id in self.ab_testing._experiments.keys()
        ]
        
        return {
            "report_time": datetime.utcnow().isoformat(),
            "snapshots_analyzed": len(recent_snapshots),
            "win_rate_trend": win_rate_trend,
            "accuracy_trend": accuracy_trend,
            "pending_improvements": len(self.get_pending_improvements()),
            "active_experiments": len([e for e in experiments if e.get("status") == "RUNNING"]),
            "significant_experiments": [e for e in experiments if e.get("is_significant")],
            "retraining_triggers": len(self.feedback_collector._triggers),
            "recommendations": self._generate_recommendations(),
        }
        
    def _generate_recommendations(self) -> list[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        # Check for pending improvements
        pending = self.get_pending_improvements()
        if pending:
            critical = [p for p in pending if p["severity"] == "critical"]
            if critical:
                recommendations.append(
                    f"{len(critical)} critical improvements pending - prioritize model retraining"
                )
                
        # Check recent performance
        snapshots = self.feedback_collector._snapshots[-5:]
        if snapshots:
            avg_win_rate = np.mean([s.win_rate for s in snapshots])
            if avg_win_rate < 0.45:
                recommendations.append(
                    f"Win rate trending low ({avg_win_rate:.0%}) - review strategy parameters"
                )
                
        # Check experiments
        for exp_id, exp in self.ab_testing._experiments.items():
            if exp.is_significant and exp.winner == "treatment":
                recommendations.append(
                    f"Experiment '{exp.name}' shows treatment wins - consider rollout"
                )
                
        if not recommendations:
            recommendations.append("System performing within expected parameters")
            
        return recommendations
