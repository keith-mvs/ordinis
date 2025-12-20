# ordinis.optimizations.confidence_filter

Confidence-based signal filtering to improve win rate.

Key Finding: Signals with 80%+ confidence have 51.3% win rate vs. 44.7% baseline
Implementation: Only execute trades when confidence >= 80%
Expected Improvement: +6.5% win rate

Usage:
    filter = ConfidenceFilter(min_confidence=0.80)
    if filter.should_execute(signal):
        execute_trade(signal)

## AdaptiveConfidenceFilter

Adaptive confidence filter that adjusts thresholds based on market conditions.

Features:
- Lower threshold in trending markets (easier to reach confidence)
- Higher threshold in choppy markets (require more certainty)
- Per-sector confidence adjustments

### Methods

#### `__init__(self, base_confidence: float = 0.8, min_agreeing_models: int = 4)`

Initialize adaptive filter.

#### `calculate_confidence(self, model_signals: Dict[str, float], market_volatility: float = 0) -> ordinis.optimizations.confidence_filter.ConfidenceMetrics`

Calculate overall confidence from individual model signals.

Args:
    model_signals: Dict of {model_name: signal_strength (-1 to 1)}
    market_volatility: Current market volatility (0-1)

Returns:
    ConfidenceMetrics with detailed breakdown

#### `get_filter_stats(self) -> Dict`

Get filtering statistics.

#### `get_position_size_multiplier(self, confidence: float) -> float`

Get position size multiplier based on confidence.

High-confidence signals get larger positions.
Low-confidence signals get smaller positions or skipped.

Args:
    confidence: Confidence score (0-1)

Returns:
    Position size multiplier (0.0 to 1.5)

#### `get_stop_loss_adjustment(self, confidence: float) -> float`

Get stop loss adjustment based on confidence.

High-confidence signals can use tighter stops.
Low-confidence signals use wider stops.

Args:
    confidence: Confidence score (0-1)

Returns:
    Stop loss multiplier (0.5 to 2.0)

#### `reset_stats(self)`

Reset filter statistics.

#### `should_execute(self, signal: Dict) -> bool`

Adaptive execution decision based on market conditions.

Args:
    signal: Signal dict

Returns:
    True if confidence meets adaptive threshold


---

## ConfidenceFilter

Filter trades by confidence threshold.

### Methods

#### `__init__(self, min_confidence: float = 0.8, min_agreeing_models: int = 4, apply_volatility_adjustment: bool = True)`

Initialize confidence filter.

Args:
    min_confidence: Minimum confidence score (0-1, default 0.80)
    min_agreeing_models: Minimum models that must agree (default 4 of 6)
    apply_volatility_adjustment: Reduce confidence in high-vol markets

#### `calculate_confidence(self, model_signals: Dict[str, float], market_volatility: float = 0) -> ordinis.optimizations.confidence_filter.ConfidenceMetrics`

Calculate overall confidence from individual model signals.

Args:
    model_signals: Dict of {model_name: signal_strength (-1 to 1)}
    market_volatility: Current market volatility (0-1)

Returns:
    ConfidenceMetrics with detailed breakdown

#### `get_filter_stats(self) -> Dict`

Get filtering statistics.

#### `get_position_size_multiplier(self, confidence: float) -> float`

Get position size multiplier based on confidence.

High-confidence signals get larger positions.
Low-confidence signals get smaller positions or skipped.

Args:
    confidence: Confidence score (0-1)

Returns:
    Position size multiplier (0.0 to 1.5)

#### `get_stop_loss_adjustment(self, confidence: float) -> float`

Get stop loss adjustment based on confidence.

High-confidence signals can use tighter stops.
Low-confidence signals use wider stops.

Args:
    confidence: Confidence score (0-1)

Returns:
    Stop loss multiplier (0.5 to 2.0)

#### `reset_stats(self)`

Reset filter statistics.

#### `should_execute(self, signal: Dict) -> bool`

Determine if signal should be executed based on confidence.

Args:
    signal: Signal dict with confidence_score and related metrics

Returns:
    True if confidence meets threshold, False otherwise


---

## ConfidenceMetrics

Confidence score breakdown.

### Methods

#### `__init__(self, model_agreement: float, confidence_score: float, num_agreeing_models: int, signal_strength: float, volatility_adjusted: float, regime_adjusted: float) -> None`


---
