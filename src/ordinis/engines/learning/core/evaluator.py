"""
Model Evaluator for Learning Engine (G-LE-1).

Provides a structured evaluation framework that computes metrics on held-out
datasets and enforces promotion gates before models can move to production.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


class EvaluationGate(Enum):
    """Promotion gate decisions."""

    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"


@dataclass
class EvaluationThresholds:
    """Thresholds for model promotion gates."""

    # Accuracy metrics
    min_accuracy: float = 0.52  # Slightly better than random
    min_precision: float = 0.50
    min_recall: float = 0.45
    min_f1_score: float = 0.48

    # Financial metrics
    min_sharpe_ratio: float = 0.5
    max_max_drawdown: float = 0.25  # 25% max drawdown
    min_profit_factor: float = 1.1
    min_win_rate: float = 0.45

    # Stability metrics
    max_prediction_variance: float = 0.3
    min_sample_size: int = 100


@dataclass
class EvaluationResult:
    """Result of model evaluation with gate decisions."""

    model_id: str
    model_version: str
    evaluated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Classification metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0

    # Financial metrics
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    profit_factor: float = 0.0
    win_rate: float = 0.0
    total_return: float = 0.0

    # Stability metrics
    prediction_variance: float = 0.0
    sample_size: int = 0

    # Gate decisions
    gate_decision: EvaluationGate = EvaluationGate.FAIL
    gate_reasons: list[str] = field(default_factory=list)

    # Raw data for debugging
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "evaluated_at": self.evaluated_at.isoformat(),
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "profit_factor": self.profit_factor,
            "win_rate": self.win_rate,
            "total_return": self.total_return,
            "prediction_variance": self.prediction_variance,
            "sample_size": self.sample_size,
            "gate_decision": self.gate_decision.value,
            "gate_reasons": self.gate_reasons,
            "metadata": self.metadata,
        }


class ModelEvaluator:
    """
    Evaluates model performance on held-out datasets and enforces promotion gates.

    This class addresses G-LE-1: "No ModelEvaluator / evaluation gates" by providing:
    - Holdout dataset evaluation
    - Classification metrics (accuracy, precision, recall, F1)
    - Financial metrics (Sharpe, drawdown, profit factor)
    - Configurable promotion thresholds
    - Clear pass/fail gate decisions with reasons

    Example:
        evaluator = ModelEvaluator(thresholds=EvaluationThresholds(min_accuracy=0.55))
        result = evaluator.evaluate(
            model_id="lstm_v1",
            model_version="1.0.0",
            predictions=predictions,
            actuals=actuals,
            returns=returns,
        )
        if result.gate_decision == EvaluationGate.PASS:
            promote_model(model_id)
    """

    def __init__(self, thresholds: EvaluationThresholds | None = None):
        """
        Initialize evaluator with thresholds.

        Args:
            thresholds: Evaluation thresholds for promotion gates.
                       Defaults to EvaluationThresholds() if not provided.
        """
        self.thresholds = thresholds or EvaluationThresholds()

    def evaluate(
        self,
        model_id: str,
        model_version: str,
        predictions: np.ndarray | pd.Series,
        actuals: np.ndarray | pd.Series,
        returns: np.ndarray | pd.Series | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        """
        Evaluate model predictions against actuals and apply promotion gates.

        Args:
            model_id: Identifier of the model being evaluated.
            model_version: Version string of the model.
            predictions: Model predictions (0/1 for direction, or continuous).
            actuals: Actual outcomes (0/1 for direction, or continuous).
            returns: Optional strategy returns for financial metrics.
            metadata: Optional metadata to include in result.

        Returns:
            EvaluationResult with metrics and gate decision.
        """
        predictions = np.asarray(predictions)
        actuals = np.asarray(actuals)

        result = EvaluationResult(
            model_id=model_id,
            model_version=model_version,
            sample_size=len(predictions),
            metadata=metadata or {},
        )

        # Check minimum sample size
        if len(predictions) < self.thresholds.min_sample_size:
            result.gate_decision = EvaluationGate.FAIL
            result.gate_reasons.append(
                f"Sample size {len(predictions)} < min {self.thresholds.min_sample_size}"
            )
            return result

        # Compute classification metrics
        result.accuracy = self._compute_accuracy(predictions, actuals)
        result.precision = self._compute_precision(predictions, actuals)
        result.recall = self._compute_recall(predictions, actuals)
        result.f1_score = self._compute_f1(result.precision, result.recall)
        result.prediction_variance = float(np.var(predictions))

        # Compute financial metrics if returns provided
        if returns is not None:
            returns = np.asarray(returns)
            result.sharpe_ratio = self._compute_sharpe(returns)
            result.max_drawdown = self._compute_max_drawdown(returns)
            result.profit_factor = self._compute_profit_factor(returns)
            result.win_rate = self._compute_win_rate(returns)
            result.total_return = float(np.sum(returns))

        # Apply promotion gates
        result.gate_decision, result.gate_reasons = self._apply_gates(result)

        return result

    def _compute_accuracy(self, predictions: np.ndarray, actuals: np.ndarray) -> float:
        """Compute accuracy score."""
        # Binarize if continuous
        pred_binary = (predictions > 0.5).astype(int)
        actual_binary = (actuals > 0.5).astype(int)
        return float(np.mean(pred_binary == actual_binary))

    def _compute_precision(self, predictions: np.ndarray, actuals: np.ndarray) -> float:
        """Compute precision (positive predictive value)."""
        pred_binary = (predictions > 0.5).astype(int)
        actual_binary = (actuals > 0.5).astype(int)
        true_positives = np.sum((pred_binary == 1) & (actual_binary == 1))
        predicted_positives = np.sum(pred_binary == 1)
        if predicted_positives == 0:
            return 0.0
        return float(true_positives / predicted_positives)

    def _compute_recall(self, predictions: np.ndarray, actuals: np.ndarray) -> float:
        """Compute recall (sensitivity)."""
        pred_binary = (predictions > 0.5).astype(int)
        actual_binary = (actuals > 0.5).astype(int)
        true_positives = np.sum((pred_binary == 1) & (actual_binary == 1))
        actual_positives = np.sum(actual_binary == 1)
        if actual_positives == 0:
            return 0.0
        return float(true_positives / actual_positives)

    def _compute_f1(self, precision: float, recall: float) -> float:
        """Compute F1 score."""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def _compute_sharpe(self, returns: np.ndarray, risk_free: float = 0.0) -> float:
        """Compute annualized Sharpe ratio."""
        excess_returns = returns - risk_free / 252  # Daily risk-free rate
        if np.std(excess_returns) == 0:
            return 0.0
        return float(np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252))

    def _compute_max_drawdown(self, returns: np.ndarray) -> float:
        """Compute maximum drawdown."""
        cumulative = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak
        return float(np.abs(np.min(drawdown)))

    def _compute_profit_factor(self, returns: np.ndarray) -> float:
        """Compute profit factor (gross profit / gross loss)."""
        gross_profit = np.sum(returns[returns > 0])
        gross_loss = np.abs(np.sum(returns[returns < 0]))
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return float(gross_profit / gross_loss)

    def _compute_win_rate(self, returns: np.ndarray) -> float:
        """Compute win rate (percentage of positive returns)."""
        if len(returns) == 0:
            return 0.0
        return float(np.sum(returns > 0) / len(returns))

    def _apply_gates(self, result: EvaluationResult) -> tuple[EvaluationGate, list[str]]:
        """
        Apply promotion gates based on thresholds.

        Returns:
            Tuple of (gate_decision, list of failure reasons).
        """
        reasons: list[str] = []
        t = self.thresholds

        # Classification gates
        if result.accuracy < t.min_accuracy:
            reasons.append(f"Accuracy {result.accuracy:.3f} < min {t.min_accuracy}")
        if result.precision < t.min_precision:
            reasons.append(f"Precision {result.precision:.3f} < min {t.min_precision}")
        if result.recall < t.min_recall:
            reasons.append(f"Recall {result.recall:.3f} < min {t.min_recall}")
        if result.f1_score < t.min_f1_score:
            reasons.append(f"F1 {result.f1_score:.3f} < min {t.min_f1_score}")

        # Stability gates
        if result.prediction_variance > t.max_prediction_variance:
            reasons.append(
                f"Prediction variance {result.prediction_variance:.3f} > max {t.max_prediction_variance}"
            )

        # Financial gates (only if we have returns data - use small epsilon for float comparison)
        has_financial_data = abs(result.sharpe_ratio) > 1e-9 or abs(result.max_drawdown) > 1e-9
        if has_financial_data:
            if result.sharpe_ratio < t.min_sharpe_ratio:
                reasons.append(f"Sharpe {result.sharpe_ratio:.3f} < min {t.min_sharpe_ratio}")
            if result.max_drawdown > t.max_max_drawdown:
                reasons.append(f"Max drawdown {result.max_drawdown:.3f} > max {t.max_max_drawdown}")
            if result.profit_factor < t.min_profit_factor:
                reasons.append(
                    f"Profit factor {result.profit_factor:.3f} < min {t.min_profit_factor}"
                )
            if result.win_rate < t.min_win_rate:
                reasons.append(f"Win rate {result.win_rate:.3f} < min {t.min_win_rate}")

        if reasons:
            return EvaluationGate.FAIL, reasons
        return EvaluationGate.PASS, []

    def evaluate_holdout(
        self,
        model: Any,
        holdout_data: pd.DataFrame,
        target_column: str = "target",
        feature_columns: list[str] | None = None,
    ) -> EvaluationResult:
        """
        Evaluate a model on a held-out dataset.

        This is a convenience method that handles the prediction step.

        Args:
            model: Model with a predict() method.
            holdout_data: DataFrame with features and target.
            target_column: Name of target column.
            feature_columns: List of feature column names. If None, uses all non-target columns.

        Returns:
            EvaluationResult with metrics and gate decision.
        """
        if feature_columns is None:
            feature_columns = [c for c in holdout_data.columns if c != target_column]

        features = holdout_data[feature_columns]
        actuals = holdout_data[target_column].values

        # Get predictions
        if hasattr(model, "predict_proba"):
            predictions = model.predict_proba(features)[:, 1]
        elif hasattr(model, "predict"):
            predictions = model.predict(features)
        else:
            raise ValueError("Model must have predict() or predict_proba() method")

        return self.evaluate(
            model_id=getattr(model, "model_id", "unknown"),
            model_version=getattr(model, "version", "1.0.0"),
            predictions=predictions,
            actuals=actuals,
            metadata={"holdout_size": len(holdout_data)},
        )
