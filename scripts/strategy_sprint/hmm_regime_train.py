"""
HMM Regime Training on SPY.

Trains Hidden Markov Model on SPY to detect market-wide regimes
that can be used as a conditioning signal for other strategies.
"""

import asyncio
from datetime import datetime
import logging
from pathlib import Path
import sys
from typing import Dict, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

logger = logging.getLogger(__name__)


def load_historical_data(symbol: str = "SPY", days: int = 1000) -> pd.DataFrame:
    """Load historical price data."""
    cache_path = Path(f"data/historical/{symbol}_daily.parquet")

    if cache_path.exists():
        return pd.read_parquet(cache_path)

    csv_path = Path(f"data/historical/{symbol}_daily.csv")
    if csv_path.exists():
        return pd.read_csv(csv_path, parse_dates=["date"], index_col="date")

    return generate_synthetic_regime_data(symbol, days)


def generate_synthetic_regime_data(symbol: str, days: int = 1000) -> pd.DataFrame:
    """Generate synthetic data with clear regime structure."""
    np.random.seed(42 if symbol == "SPY" else hash(symbol) % 2**32)

    # Define regimes
    regimes = {
        "bull": {"drift": 0.0006, "vol": 0.008},
        "bear": {"drift": -0.0008, "vol": 0.02},
        "neutral": {"drift": 0.0001, "vol": 0.012},
    }

    # Generate regime sequence with persistence
    regime_names = list(regimes.keys())
    current_regime = np.random.choice(regime_names)
    regime_sequence = []

    for i in range(days):
        # Regime persistence
        if np.random.random() > 0.98:  # 2% chance of regime switch
            current_regime = np.random.choice(regime_names)
        regime_sequence.append(current_regime)

    # Generate returns based on regime
    returns = []
    for regime in regime_sequence:
        params = regimes[regime]
        ret = np.random.normal(params["drift"], params["vol"])
        returns.append(ret)

    returns = np.array(returns)
    base_price = 400.0
    prices = base_price * np.exp(np.cumsum(returns))

    dates = pd.date_range(end=datetime.now(), periods=days, freq="D")

    df = pd.DataFrame(index=dates)
    df["close"] = prices
    df["high"] = prices * (1 + np.abs(np.random.normal(0, 0.008, days)))
    df["low"] = prices * (1 - np.abs(np.random.normal(0, 0.008, days)))
    df["open"] = df["close"].shift(1).fillna(df["close"].iloc[0])
    df["volume"] = np.random.randint(50_000_000, 200_000_000, days).astype(float)

    df["high"] = df[["high", "close", "open"]].max(axis=1)
    df["low"] = df[["low", "close", "open"]].min(axis=1)

    return df


def calculate_features(df: pd.DataFrame) -> np.ndarray:
    """Calculate features for HMM training."""
    features = pd.DataFrame(index=df.index)

    # Returns
    features["return"] = df["close"].pct_change()

    # Realized volatility (5-day)
    features["vol_5d"] = features["return"].rolling(5).std()

    # Realized volatility (20-day)
    features["vol_20d"] = features["return"].rolling(20).std()

    # Trend (5-day vs 20-day SMA)
    sma5 = df["close"].rolling(5).mean()
    sma20 = df["close"].rolling(20).mean()
    features["trend"] = (sma5 - sma20) / sma20

    # Volume relative to average
    vol_sma = df["volume"].rolling(20).mean()
    features["vol_ratio"] = df["volume"] / (vol_sma + 1e-10)

    # Drop NaN
    features = features.dropna()

    return features.values, features.index


class GaussianHMM:
    """Simple Gaussian Hidden Markov Model implementation."""

    def __init__(self, n_states: int = 3, n_iter: int = 100):
        self.n_states = n_states
        self.n_iter = n_iter

        # Parameters
        self.means = None  # State means
        self.covars = None  # State covariances
        self.transmat = None  # Transition matrix
        self.startprob = None  # Initial state probabilities

    def _init_params(self, X: np.ndarray):
        """Initialize parameters."""
        n_samples, n_features = X.shape

        # Initialize with K-means style clustering
        indices = np.random.choice(n_samples, self.n_states, replace=False)
        self.means = X[indices]

        # Initialize covariances
        self.covars = np.array(
            [np.eye(n_features) * np.var(X, axis=0) for _ in range(self.n_states)]
        )

        # Initialize transition matrix (uniform with persistence)
        self.transmat = np.ones((self.n_states, self.n_states)) * 0.1
        np.fill_diagonal(self.transmat, 0.8)
        self.transmat /= self.transmat.sum(axis=1, keepdims=True)

        # Initial probabilities
        self.startprob = np.ones(self.n_states) / self.n_states

    def _gaussian_pdf(self, x: np.ndarray, mean: np.ndarray, covar: np.ndarray) -> float:
        """Multivariate Gaussian PDF."""
        n = len(mean)
        det = np.linalg.det(covar)

        if det <= 0:
            det = 1e-10

        diff = x - mean
        try:
            inv_covar = np.linalg.inv(covar)
            exponent = -0.5 * diff @ inv_covar @ diff
        except:
            exponent = -0.5 * np.sum(diff**2) / (np.mean(np.diag(covar)) + 1e-10)

        return np.exp(exponent) / (np.sqrt((2 * np.pi) ** n * det) + 1e-10)

    def _emission_probs(self, X: np.ndarray) -> np.ndarray:
        """Calculate emission probabilities for all observations."""
        n_samples = len(X)
        probs = np.zeros((n_samples, self.n_states))

        for i in range(n_samples):
            for j in range(self.n_states):
                probs[i, j] = self._gaussian_pdf(X[i], self.means[j], self.covars[j])

        # Avoid zeros
        probs = np.maximum(probs, 1e-10)

        return probs

    def _forward(self, emission_probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward algorithm."""
        n_samples = len(emission_probs)
        alpha = np.zeros((n_samples, self.n_states))
        scale = np.zeros(n_samples)

        # Initialize
        alpha[0] = self.startprob * emission_probs[0]
        scale[0] = alpha[0].sum()
        alpha[0] /= scale[0]

        # Forward pass
        for t in range(1, n_samples):
            alpha[t] = emission_probs[t] * (alpha[t - 1] @ self.transmat)
            scale[t] = alpha[t].sum()
            if scale[t] > 0:
                alpha[t] /= scale[t]

        return alpha, scale

    def _backward(self, emission_probs: np.ndarray, scale: np.ndarray) -> np.ndarray:
        """Backward algorithm."""
        n_samples = len(emission_probs)
        beta = np.zeros((n_samples, self.n_states))

        # Initialize
        beta[-1] = 1.0

        # Backward pass
        for t in range(n_samples - 2, -1, -1):
            beta[t] = self.transmat @ (emission_probs[t + 1] * beta[t + 1])
            if scale[t + 1] > 0:
                beta[t] /= scale[t + 1]

        return beta

    def fit(self, X: np.ndarray) -> "GaussianHMM":
        """Fit HMM using Baum-Welch algorithm."""
        self._init_params(X)
        n_samples, n_features = X.shape

        for iteration in range(self.n_iter):
            # E-step
            emission_probs = self._emission_probs(X)
            alpha, scale = self._forward(emission_probs)
            beta = self._backward(emission_probs, scale)

            # Compute responsibilities
            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True) + 1e-10

            # Compute xi (transition responsibilities)
            xi = np.zeros((n_samples - 1, self.n_states, self.n_states))
            for t in range(n_samples - 1):
                for i in range(self.n_states):
                    for j in range(self.n_states):
                        xi[t, i, j] = (
                            alpha[t, i]
                            * self.transmat[i, j]
                            * emission_probs[t + 1, j]
                            * beta[t + 1, j]
                        )
                xi[t] /= xi[t].sum() + 1e-10

            # M-step
            # Update start probabilities
            self.startprob = gamma[0]

            # Update transition matrix
            self.transmat = xi.sum(axis=0) / (gamma[:-1].sum(axis=0, keepdims=True).T + 1e-10)

            # Update means
            for j in range(self.n_states):
                self.means[j] = np.sum(gamma[:, j : j + 1] * X, axis=0) / (
                    gamma[:, j].sum() + 1e-10
                )

            # Update covariances
            for j in range(self.n_states):
                diff = X - self.means[j]
                weighted_cov = np.zeros((n_features, n_features))
                for t in range(n_samples):
                    weighted_cov += gamma[t, j] * np.outer(diff[t], diff[t])
                self.covars[j] = weighted_cov / (gamma[:, j].sum() + 1e-10)
                # Add regularization
                self.covars[j] += np.eye(n_features) * 1e-4

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict most likely state sequence (Viterbi)."""
        emission_probs = self._emission_probs(X)
        n_samples = len(X)

        # Viterbi algorithm
        viterbi = np.zeros((n_samples, self.n_states))
        backpointer = np.zeros((n_samples, self.n_states), dtype=int)

        # Initialize
        viterbi[0] = np.log(self.startprob + 1e-10) + np.log(emission_probs[0] + 1e-10)

        # Forward pass
        for t in range(1, n_samples):
            for j in range(self.n_states):
                trans_scores = viterbi[t - 1] + np.log(self.transmat[:, j] + 1e-10)
                backpointer[t, j] = np.argmax(trans_scores)
                viterbi[t, j] = trans_scores[backpointer[t, j]] + np.log(
                    emission_probs[t, j] + 1e-10
                )

        # Backtrack
        states = np.zeros(n_samples, dtype=int)
        states[-1] = np.argmax(viterbi[-1])
        for t in range(n_samples - 2, -1, -1):
            states[t] = backpointer[t + 1, states[t + 1]]

        return states

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict state probabilities."""
        emission_probs = self._emission_probs(X)
        alpha, _ = self._forward(emission_probs)
        return alpha


def characterize_regimes(
    df: pd.DataFrame,
    states: np.ndarray,
    feature_index: pd.DatetimeIndex,
) -> dict[int, dict]:
    """Characterize each regime by its statistics."""
    df_aligned = df.loc[feature_index]
    returns = df_aligned["close"].pct_change()

    regime_stats = {}

    for state in range(max(states) + 1):
        mask = states == state
        state_returns = returns[mask]
        state_vol = df_aligned.loc[mask, "close"].pct_change().std()

        regime_stats[state] = {
            "count": mask.sum(),
            "pct": mask.sum() / len(mask) * 100,
            "avg_return": state_returns.mean() * 252 * 100,  # Annualized %
            "volatility": state_vol * np.sqrt(252) * 100,  # Annualized %
            "sharpe": (state_returns.mean() * 252) / (state_vol * np.sqrt(252))
            if state_vol > 0
            else 0,
        }

    return regime_stats


def backtest_regime_strategy(
    df: pd.DataFrame,
    states: np.ndarray,
    feature_index: pd.DatetimeIndex,
    bull_state: int,
) -> dict:
    """Backtest simple regime-following strategy."""
    df_aligned = df.loc[feature_index].copy()
    returns = df_aligned["close"].pct_change()

    # Strategy: Long only in bull regime
    position = (states == bull_state).astype(float)

    strat_returns = position * returns
    strat_returns = strat_returns.dropna()

    # Buy and hold comparison
    bh_returns = returns.dropna()

    # Metrics
    total_strat = (1 + strat_returns).prod() - 1
    total_bh = (1 + bh_returns).prod() - 1

    vol_strat = strat_returns.std() * np.sqrt(252)
    sharpe_strat = (strat_returns.mean() * 252) / vol_strat if vol_strat > 0 else 0

    vol_bh = bh_returns.std() * np.sqrt(252)
    sharpe_bh = (bh_returns.mean() * 252) / vol_bh if vol_bh > 0 else 0

    return {
        "strategy_return": total_strat * 100,
        "bh_return": total_bh * 100,
        "strategy_sharpe": sharpe_strat,
        "bh_sharpe": sharpe_bh,
        "time_in_market": position.mean() * 100,
    }


async def run() -> dict:
    """Run HMM regime training."""
    logger.info("=" * 50)
    logger.info("HMM REGIME TRAINING ON SPY")
    logger.info("=" * 50)

    # Load SPY data
    df = load_historical_data("SPY", days=1000)
    logger.info(f"Loaded {len(df)} days of SPY data")

    # Calculate features
    features, feature_index = calculate_features(df)
    logger.info(f"Calculated features: {features.shape}")

    # Train HMM with different state counts
    best_model = None
    best_bic = np.inf
    best_n_states = 3

    for n_states in [2, 3, 4]:
        logger.info(f"\n--- Training HMM with {n_states} states ---")

        model = GaussianHMM(n_states=n_states, n_iter=50)
        model.fit(features)

        # Predict states
        states = model.predict(features)

        # Characterize regimes
        regime_stats = characterize_regimes(df, states, feature_index)

        for state, stats in regime_stats.items():
            logger.info(
                f"  State {state}: {stats['pct']:.1f}% of time, "
                f"Return: {stats['avg_return']:+.1f}%, "
                f"Vol: {stats['volatility']:.1f}%, "
                f"Sharpe: {stats['sharpe']:.2f}"
            )

        # Simple BIC approximation
        n_params = n_states * (n_states + 2 * features.shape[1])
        log_lik = np.log(
            model._emission_probs(features)[np.arange(len(states)), states] + 1e-10
        ).sum()
        bic = n_params * np.log(len(features)) - 2 * log_lik

        logger.info(f"  BIC: {bic:.0f}")

        if bic < best_bic:
            best_bic = bic
            best_model = model
            best_n_states = n_states

    logger.info(f"\nBest model: {best_n_states} states")

    # Final predictions with best model
    final_states = best_model.predict(features)
    final_probs = best_model.predict_proba(features)

    # Identify bull regime (highest return)
    regime_stats = characterize_regimes(df, final_states, feature_index)
    bull_state = max(regime_stats.keys(), key=lambda k: regime_stats[k]["avg_return"])
    bear_state = min(regime_stats.keys(), key=lambda k: regime_stats[k]["avg_return"])

    logger.info(f"\nIdentified Bull regime: State {bull_state}")
    logger.info(f"Identified Bear regime: State {bear_state}")

    # Current regime
    current_state = final_states[-1]
    current_probs = final_probs[-1]

    logger.info(f"\n--- Current Market Regime ---")
    logger.info(f"  State: {current_state}")
    logger.info(f"  Probabilities: {current_probs}")

    regime_name = (
        "BULL"
        if current_state == bull_state
        else "BEAR"
        if current_state == bear_state
        else "NEUTRAL"
    )
    logger.info(f"  Regime: {regime_name}")

    # Backtest
    logger.info(f"\n--- Regime Strategy Backtest ---")
    bt = backtest_regime_strategy(df, final_states, feature_index, bull_state)

    logger.info(f"  Strategy Return: {bt['strategy_return']:.1f}%")
    logger.info(f"  Buy-Hold Return: {bt['bh_return']:.1f}%")
    logger.info(f"  Strategy Sharpe: {bt['strategy_sharpe']:.2f}")
    logger.info(f"  Buy-Hold Sharpe: {bt['bh_sharpe']:.2f}")
    logger.info(f"  Time in Market: {bt['time_in_market']:.1f}%")

    # Save results
    output_dir = Path("artifacts/reports/strategy_sprint")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Regime history
    regime_df = pd.DataFrame(
        {
            "date": feature_index,
            "state": final_states,
            "prob_0": final_probs[:, 0],
            "prob_1": final_probs[:, 1] if best_n_states > 1 else 0,
            "prob_2": final_probs[:, 2] if best_n_states > 2 else 0,
        }
    )
    regime_df.to_csv(output_dir / "hmm_regime_history.csv", index=False)

    # Regime stats
    stats_df = pd.DataFrame([{"state": k, **v} for k, v in regime_stats.items()])
    stats_df.to_csv(output_dir / "hmm_regime_stats.csv", index=False)

    # Model parameters
    model_params = {
        "n_states": best_n_states,
        "means": best_model.means.tolist(),
        "transmat": best_model.transmat.tolist(),
        "bull_state": int(bull_state),
        "bear_state": int(bear_state),
    }

    import json

    with open(output_dir / "hmm_model_params.json", "w") as f:
        json.dump(model_params, f, indent=2)

    return {
        "success": True,
        "summary": f"Trained {best_n_states}-state HMM on SPY",
        "current_regime": regime_name,
        "current_state": int(current_state),
        "bull_state": int(bull_state),
        "bear_state": int(bear_state),
        "regime_stats": regime_stats,
    }


if __name__ == "__main__":
    asyncio.run(run())
