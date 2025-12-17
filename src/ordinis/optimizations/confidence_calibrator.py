"""
Machine learning-based confidence calibration and threshold recommendation.

This module fits a logistic model to map raw confidence signals to calibrated
probabilities of winning trades. It also provides risk-tolerance aware
threshold suggestions and evaluation helpers.

Dependencies:
- Uses scikit-learn if available. Falls back to a lightweight numpy-based
  logistic regression implementation when sklearn is not installed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CalibrationMetrics:
    """Summary of calibration quality."""

    brier_score: float
    log_loss: float
    accuracy: float
    average_probability: float
    feature_importance: dict[str, float]


class ConfidenceCalibrator:
    """Calibrate confidence scores to win probabilities."""

    def __init__(self, learning_rate: float = 0.05, max_iter: int = 4000):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self._use_sklearn = False
        self._model = None
        self._coef: np.ndarray | None = None
        self._bias: float = 0.0
        self._feature_mean: np.ndarray | None = None
        self._feature_std: np.ndarray | None = None
        self.feature_names: list[str] = [
            "confidence_score",
            "num_agreeing_models",
            "market_volatility",
            "signal_strength",
            "holding_days",
        ]

        try:
            from sklearn.linear_model import LogisticRegression  # type: ignore

            self._model = LogisticRegression(max_iter=2000)
            self._use_sklearn = True
        except Exception:
            # Optional dependency; fall back to manual training.
            self._model = None
            self._use_sklearn = False

    def fit(self, trades: list[dict]) -> CalibrationMetrics:
        """Fit calibrator to historical trade outcomes.

        Args:
            trades: Trade dictionaries with outcome fields.

        Returns:
            CalibrationMetrics describing model quality.
        """
        features, labels = self._prepare_features(trades)
        scaled, mean, std = self._standardize(features)
        self._feature_mean = mean
        self._feature_std = std

        if self._use_sklearn and self._model is not None:
            self._model.fit(scaled, labels)
            self._coef = self._model.coef_.ravel()
            self._bias = float(self._model.intercept_[0])
        else:
            self._fit_manual(scaled, labels)

        preds = self._predict_from_matrix(scaled)
        metrics = self._compute_metrics(labels, preds)
        metrics.feature_importance = self._estimate_importance()
        return metrics

    def calibrate_trades(self, trades: list[dict]) -> list[dict]:
        """Attach calibrated probabilities to each trade."""
        if self._feature_mean is None or self._feature_std is None:
            raise RuntimeError("Calibrator must be fit before calibration.")

        calibrated = []
        for trade in trades:
            prob = self.predict_probability(trade)
            enriched = trade.copy()
            enriched["calibrated_probability"] = prob
            calibrated.append(enriched)
        return calibrated

    def predict_probability(self, trade: dict) -> float:
        """Predict win probability for a single trade."""
        if self._feature_mean is None or self._feature_std is None:
            raise RuntimeError("Calibrator must be fit before prediction.")

        vector = np.array(
            [
                float(trade.get("confidence_score", 0.0)),
                float(trade.get("num_agreeing_models", 0.0)),
                float(trade.get("market_volatility", 0.0)),
                float(trade.get("signal_strength", 0.0)),
                float(trade.get("holding_days", 0.0)),
            ]
        )
        scaled = (vector - self._feature_mean) / self._feature_std
        prob = float(self._sigmoid(np.dot(scaled, self._coef) + self._bias))
        return float(np.clip(prob, 0.001, 0.999))

    def threshold_for_risk_tolerance(
        self,
        calibrated_probs: list[float],
        risk_tolerance: float,
        base_threshold: float = 0.80,
        min_trades: int = 150,
    ) -> float:
        """Suggest a probability threshold based on risk tolerance.

        Low risk tolerance => higher threshold (fewer trades).
        High risk tolerance => lower threshold (more trades).
        """
        if not calibrated_probs:
            return base_threshold

        tolerance = float(np.clip(risk_tolerance, 0.0, 1.0))
        # Map tolerance into quantile space (conservative 0.85 -> aggressive 0.45).
        quantile = 0.85 - 0.40 * tolerance
        threshold = float(np.quantile(calibrated_probs, quantile))

        if min_trades:
            sorted_probs = sorted(calibrated_probs, reverse=True)
            target_idx = min(len(sorted_probs) - 1, max(0, min_trades - 1))
            coverage_threshold = sorted_probs[target_idx]
            threshold = min(threshold, coverage_threshold)
        return max(0.05, min(0.95, threshold))

    def evaluate_thresholds(
        self,
        trades: list[dict],
        probabilities: list[float],
        thresholds: list[float],
    ) -> list[dict]:
        """Evaluate performance across probability thresholds."""
        df = pd.DataFrame(trades).copy()
        df["calibrated_probability"] = probabilities

        results = []
        for threshold in thresholds:
            selected = df[df["calibrated_probability"] >= threshold]
            if selected.empty:
                results.append(
                    {
                        "threshold": threshold,
                        "trades": 0,
                        "win_rate": 0.0,
                        "avg_return": 0.0,
                    }
                )
                continue

            win_rate = float(selected["win"].mean())
            avg_return = float(selected["return_pct"].mean())
            results.append(
                {
                    "threshold": threshold,
                    "trades": len(selected),
                    "win_rate": win_rate,
                    "avg_return": avg_return,
                }
            )
        return results

    def _prepare_features(self, trades: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        df = pd.DataFrame(trades)
        if df.empty:
            raise ValueError("Trades are required for calibration.")

        missing = [f for f in self.feature_names if f not in df.columns]
        if missing:
            for name in missing:
                df[name] = 0.0

        features = df[self.feature_names].fillna(0.0).astype(float).values
        labels = df.get("win")
        if labels is None:
            raise ValueError("Trades must include 'win' outcomes.")

        labels_array = labels.astype(int).values
        return features, labels_array

    def _standardize(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mean = features.mean(axis=0)
        std = features.std(axis=0)
        std = np.where(std == 0, 1.0, std)
        scaled = (features - mean) / std
        return scaled, mean, std

    def _fit_manual(self, features: np.ndarray, labels: np.ndarray) -> None:
        self._coef = np.zeros(features.shape[1])
        self._bias = 0.0

        for _ in range(self.max_iter):
            logits = np.dot(features, self._coef) + self._bias
            preds = self._sigmoid(logits)
            error = preds - labels
            grad_coef = np.dot(features.T, error) / len(labels)
            grad_bias = float(np.mean(error))
            self._coef -= self.learning_rate * grad_coef
            self._bias -= self.learning_rate * grad_bias

    def _predict_from_matrix(self, features: np.ndarray) -> np.ndarray:
        if self._use_sklearn and self._model is not None:
            proba = self._model.predict_proba(features)[:, 1]
            return np.clip(proba, 0.001, 0.999)

        logits = np.dot(features, self._coef) + self._bias
        return np.clip(self._sigmoid(logits), 0.001, 0.999)

    def _compute_metrics(self, labels: np.ndarray, preds: np.ndarray) -> CalibrationMetrics:
        brier = float(np.mean((preds - labels) ** 2))
        eps = 1e-9
        log_loss = float(
            -np.mean(labels * np.log(preds + eps) + (1 - labels) * np.log(1 - preds + eps))
        )
        accuracy = float(np.mean((preds >= 0.5) == labels))
        avg_prob = float(np.mean(preds))
        return CalibrationMetrics(
            brier_score=brier,
            log_loss=log_loss,
            accuracy=accuracy,
            average_probability=avg_prob,
            feature_importance={},
        )

    def _estimate_importance(self) -> dict[str, float]:
        if self._coef is None:
            return dict.fromkeys(self.feature_names, 0.0)

        magnitudes = np.abs(self._coef)
        total = magnitudes.sum() or 1.0
        return {
            name: float(weight / total)
            for name, weight in zip(self.feature_names, magnitudes, strict=False)
        }

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))
