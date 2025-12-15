# ordinis.optimizations.confidence_calibrator

Machine learning-based confidence calibration and threshold recommendation.

This module fits a logistic model to map raw confidence signals to calibrated
probabilities of winning trades. It also provides risk-tolerance aware
threshold suggestions and evaluation helpers.

Dependencies:
- Uses scikit-learn if available. Falls back to a lightweight numpy-based
  logistic regression implementation when sklearn is not installed.

## CalibrationMetrics

Summary of calibration quality.

### Methods

#### `__init__(self, brier_score: 'float', log_loss: 'float', accuracy: 'float', average_probability: 'float', feature_importance: 'Dict[str, float]') -> None`


---

## ConfidenceCalibrator

Calibrate confidence scores to win probabilities.

### Methods

#### `__init__(self, learning_rate: 'float' = 0.05, max_iter: 'int' = 4000)`

#### `calibrate_trades(self, trades: 'List[Dict]') -> 'List[Dict]'`

Attach calibrated probabilities to each trade.

#### `evaluate_thresholds(self, trades: 'List[Dict]', probabilities: 'List[float]', thresholds: 'List[float]') -> 'List[Dict]'`

Evaluate performance across probability thresholds.

#### `fit(self, trades: 'List[Dict]') -> 'CalibrationMetrics'`

Fit calibrator to historical trade outcomes.

Args:
    trades: Trade dictionaries with outcome fields.

Returns:
    CalibrationMetrics describing model quality.

#### `predict_probability(self, trade: 'Dict') -> 'float'`

Predict win probability for a single trade.

#### `threshold_for_risk_tolerance(self, calibrated_probs: 'List[float]', risk_tolerance: 'float', base_threshold: 'float' = 0.8, min_trades: 'int' = 150) -> 'float'`

Suggest a probability threshold based on risk tolerance.

Low risk tolerance => higher threshold (fewer trades).
High risk tolerance => lower threshold (more trades).


---
