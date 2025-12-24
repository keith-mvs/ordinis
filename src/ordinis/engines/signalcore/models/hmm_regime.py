"""
Hidden Markov Model Regime-Switching Strategy.

Uses HMM to identify latent market regimes (bull/bear/neutral).
Strategy adapts behavior based on detected regime.

Theory:
- Markets exhibit regime-switching behavior (trending vs mean-reverting)
- HMM models hidden states with observable emissions
- Viterbi algorithm finds most likely regime sequence
- Each regime has different mean return and volatility
- Trade with trend in trending regimes, mean-revert in choppy regimes
"""

from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum
import logging

import numpy as np
import pandas as pd

from ordinis.engines.signalcore.core.model import Model, ModelConfig
from ordinis.engines.signalcore.core.signal import Direction, Signal, SignalType

logger = logging.getLogger(__name__)


class MarketRegime(IntEnum):
    """Market regime states."""

    BEAR = 0  # Declining prices, high volatility
    NEUTRAL = 1  # Sideways, low volatility
    BULL = 2  # Rising prices, moderate volatility


@dataclass
class HMMConfig:
    """HMM Regime configuration."""

    # Model parameters
    n_regimes: int = 3  # Number of hidden states
    lookback: int = 252  # Lookback for training
    retrain_frequency: int = 21  # Retrain every N bars

    # Observation parameters
    return_period: int = 5  # Returns period for observations
    vol_window: int = 20  # Volatility calculation window

    # Signal parameters
    regime_momentum_long: bool = True  # Use momentum in trending regimes
    regime_mr_neutral: bool = True  # Use mean reversion in neutral regime
    rsi_period: int = 14
    rsi_oversold: float = 35
    rsi_overbought: float = 65

    # Risk parameters
    atr_period: int = 14
    atr_stop_mult: float = 1.5
    atr_tp_mult: float = 2.0
    bear_position_mult: float = 0.5  # Reduce size in bear markets


@dataclass
class RegimeState:
    """Current regime state."""

    regime: MarketRegime
    probability: float
    regime_duration: int  # Bars in current regime
    transition_prob_bull: float
    transition_prob_bear: float


class SimpleHMM:
    """
    Simple Gaussian HMM implementation.

    Uses EM algorithm for training and Viterbi for decoding.
    Falls back to this when hmmlearn not available.
    """

    def __init__(self, n_states: int = 3, n_iter: int = 100):
        """Initialize HMM."""
        self.n_states = n_states
        self.n_iter = n_iter

        # Parameters (will be estimated)
        self.means: np.ndarray = np.zeros(n_states)
        self.variances: np.ndarray = np.ones(n_states)
        self.transition: np.ndarray = np.eye(n_states) * 0.9 + 0.1 / n_states
        self.initial: np.ndarray = np.ones(n_states) / n_states

        self.fitted = False

    def _gaussian_pdf(self, x: float, mean: float, var: float) -> float:
        """Gaussian probability density."""
        return np.exp(-0.5 * (x - mean) ** 2 / var) / np.sqrt(2 * np.pi * var)

    def _emission_probs(self, obs: np.ndarray) -> np.ndarray:
        """Calculate emission probabilities for all observations."""
        T = len(obs)
        probs = np.zeros((T, self.n_states))

        for j in range(self.n_states):
            probs[:, j] = self._gaussian_pdf(obs, self.means[j], self.variances[j])

        # Normalize to prevent underflow
        probs = np.clip(probs, 1e-300, None)
        return probs

    def fit(self, observations: np.ndarray) -> "SimpleHMM":
        """
        Fit HMM using EM algorithm.

        Args:
            observations: 1D array of observations

        Returns:
            Self for chaining
        """
        obs = np.asarray(observations).flatten()
        T = len(obs)

        if T < 50:
            logger.warning("Not enough observations for HMM training")
            return self

        # Initialize parameters using k-means-like approach
        sorted_obs = np.sort(obs)
        for i in range(self.n_states):
            start = int(i * T / self.n_states)
            end = int((i + 1) * T / self.n_states)
            self.means[i] = sorted_obs[start:end].mean()
            self.variances[i] = max(sorted_obs[start:end].var(), 1e-6)

        # EM iterations
        for iteration in range(self.n_iter):
            # E-step: Forward-backward algorithm
            emission = self._emission_probs(obs)

            # Forward pass
            alpha = np.zeros((T, self.n_states))
            alpha[0] = self.initial * emission[0]
            alpha[0] /= alpha[0].sum() + 1e-300

            for t in range(1, T):
                for j in range(self.n_states):
                    alpha[t, j] = emission[t, j] * np.sum(alpha[t - 1] * self.transition[:, j])
                alpha[t] /= alpha[t].sum() + 1e-300

            # Backward pass
            beta = np.zeros((T, self.n_states))
            beta[-1] = 1.0

            for t in range(T - 2, -1, -1):
                for i in range(self.n_states):
                    beta[t, i] = np.sum(self.transition[i, :] * emission[t + 1] * beta[t + 1])
                beta[t] /= beta[t].sum() + 1e-300

            # Gamma (state occupancy probabilities)
            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300

            # Xi (transition probabilities)
            xi = np.zeros((T - 1, self.n_states, self.n_states))
            for t in range(T - 1):
                for i in range(self.n_states):
                    for j in range(self.n_states):
                        xi[t, i, j] = (
                            alpha[t, i]
                            * self.transition[i, j]
                            * emission[t + 1, j]
                            * beta[t + 1, j]
                        )
                xi[t] /= xi[t].sum() + 1e-300

            # M-step: Update parameters
            # Initial distribution
            self.initial = gamma[0]

            # Transition matrix
            for i in range(self.n_states):
                for j in range(self.n_states):
                    self.transition[i, j] = xi[:, i, j].sum() / (gamma[:-1, i].sum() + 1e-300)

            # Emission parameters
            for j in range(self.n_states):
                weight = gamma[:, j]
                self.means[j] = np.average(obs, weights=weight)
                self.variances[j] = max(
                    np.average((obs - self.means[j]) ** 2, weights=weight), 1e-6
                )

        # Sort states by mean (bear < neutral < bull)
        order = np.argsort(self.means)
        self.means = self.means[order]
        self.variances = self.variances[order]
        self.transition = self.transition[order][:, order]
        self.initial = self.initial[order]

        self.fitted = True
        return self

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predict most likely states using Viterbi algorithm.

        Args:
            observations: 1D array of observations

        Returns:
            Array of predicted state indices
        """
        if not self.fitted:
            raise ValueError("Model not fitted")

        obs = np.asarray(observations).flatten()
        T = len(obs)
        emission = self._emission_probs(obs)

        # Viterbi algorithm
        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)

        # Initialize
        delta[0] = np.log(self.initial + 1e-300) + np.log(emission[0] + 1e-300)

        # Forward pass
        for t in range(1, T):
            for j in range(self.n_states):
                temp = delta[t - 1] + np.log(self.transition[:, j] + 1e-300)
                psi[t, j] = np.argmax(temp)
                delta[t, j] = temp[psi[t, j]] + np.log(emission[t, j] + 1e-300)

        # Backtrack
        states = np.zeros(T, dtype=int)
        states[-1] = np.argmax(delta[-1])

        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        return states

    def predict_proba(self, observations: np.ndarray) -> np.ndarray:
        """
        Predict state probabilities using forward algorithm.

        Args:
            observations: 1D array of observations

        Returns:
            Array of state probabilities (T x n_states)
        """
        if not self.fitted:
            raise ValueError("Model not fitted")

        obs = np.asarray(observations).flatten()
        T = len(obs)
        emission = self._emission_probs(obs)

        # Forward pass
        alpha = np.zeros((T, self.n_states))
        alpha[0] = self.initial * emission[0]
        alpha[0] /= alpha[0].sum() + 1e-300

        for t in range(1, T):
            for j in range(self.n_states):
                alpha[t, j] = emission[t, j] * np.sum(alpha[t - 1] * self.transition[:, j])
            alpha[t] /= alpha[t].sum() + 1e-300

        return alpha


def get_hmm_model(n_states: int = 3) -> SimpleHMM:
    """Get HMM model, preferring hmmlearn if available."""
    try:
        from hmmlearn.hmm import GaussianHMM

        class HMMLearnWrapper:
            """Wrapper for hmmlearn GaussianHMM."""

            def __init__(self, n_states):
                self.model = GaussianHMM(
                    n_components=n_states,
                    covariance_type="diag",
                    n_iter=100,
                    random_state=42,
                )
                self.n_states = n_states
                self.fitted = False

            def fit(self, observations):
                obs = np.asarray(observations).reshape(-1, 1)
                self.model.fit(obs)
                self.fitted = True
                return self

            def predict(self, observations):
                obs = np.asarray(observations).reshape(-1, 1)
                return self.model.predict(obs)

            def predict_proba(self, observations):
                obs = np.asarray(observations).reshape(-1, 1)
                return self.model.predict_proba(obs)

            @property
            def means(self):
                return self.model.means_.flatten()

            @property
            def transition(self):
                return self.model.transmat_

        logger.info("Using hmmlearn for HMM")
        return HMMLearnWrapper(n_states)

    except ImportError:
        logger.info("hmmlearn not available, using SimpleHMM fallback")
        return SimpleHMM(n_states)


class HMMRegimeModel(Model):
    """
    HMM Regime-Switching Strategy.

    Uses Hidden Markov Model to identify market regimes,
    then applies appropriate sub-strategy per regime.

    Signal Logic:
    1. Train HMM on returns to identify regimes
    2. Predict current regime (bull/bear/neutral)
    3. In BULL: momentum long (RSI > 50 + uptrend)
    4. In BEAR: momentum short or stay flat
    5. In NEUTRAL: mean reversion (RSI extremes)

    Example:
        config = ModelConfig(
            model_id="hmm_regime",
            model_type="regime",
            parameters={"n_regimes": 3}
        )
        model = HMMRegimeModel(config)
        signal = await model.generate("AAPL", df, timestamp)
    """

    def __init__(self, config: ModelConfig):
        """Initialize HMM Regime model."""
        super().__init__(config)
        params = config.parameters or {}

        self.hmm_config = HMMConfig(
            n_regimes=params.get("n_regimes", 3),
            lookback=params.get("lookback", 252),
            retrain_frequency=params.get("retrain_frequency", 21),
            return_period=params.get("return_period", 5),
            vol_window=params.get("vol_window", 20),
            rsi_period=params.get("rsi_period", 14),
            rsi_oversold=params.get("rsi_oversold", 35),
            rsi_overbought=params.get("rsi_overbought", 65),
            atr_period=params.get("atr_period", 14),
            atr_stop_mult=params.get("atr_stop_mult", 1.5),
            atr_tp_mult=params.get("atr_tp_mult", 2.0),
            bear_position_mult=params.get("bear_position_mult", 0.5),
        )

        # Model cache per symbol
        self._models: dict[str, SimpleHMM] = {}
        self._last_train: dict[str, int] = {}
        self._regime_history: dict[str, list[int]] = {}

    def _calculate_observations(self, df: pd.DataFrame) -> pd.Series:
        """Calculate observation sequence for HMM."""
        # Use returns as observations
        returns = df["close"].pct_change(self.hmm_config.return_period)
        return returns.dropna()

    def _calculate_rsi(self, df: pd.DataFrame) -> float:
        """Calculate current RSI."""
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(self.hmm_config.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.hmm_config.rsi_period).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]

    def _calculate_atr(self, df: pd.DataFrame) -> float:
        """Calculate Average True Range."""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(self.hmm_config.atr_period).mean().iloc[-1]

    def _get_trend(self, df: pd.DataFrame) -> int:
        """Get trend direction: 1=up, -1=down, 0=flat."""
        sma_short = df["close"].rolling(20).mean().iloc[-1]
        sma_long = df["close"].rolling(50).mean().iloc[-1]
        current = df["close"].iloc[-1]

        if current > sma_short > sma_long:
            return 1
        if current < sma_short < sma_long:
            return -1
        return 0

    def detect_regime(
        self,
        df: pd.DataFrame,
        symbol: str,
    ) -> RegimeState:
        """
        Detect current market regime.

        Args:
            df: OHLCV DataFrame
            symbol: Symbol for model caching

        Returns:
            RegimeState with current regime info
        """
        current_bar = len(df)

        # Retrain if needed
        if (
            symbol not in self._models
            or current_bar - self._last_train.get(symbol, 0) >= self.hmm_config.retrain_frequency
        ):
            observations = self._calculate_observations(df)
            train_obs = observations.iloc[-self.hmm_config.lookback :]

            model = get_hmm_model(self.hmm_config.n_regimes)
            model.fit(train_obs.values)

            self._models[symbol] = model
            self._last_train[symbol] = current_bar

        model = self._models[symbol]

        # Predict current state
        recent_obs = self._calculate_observations(df).iloc[-60:]

        if len(recent_obs) < 10:
            return RegimeState(
                regime=MarketRegime.NEUTRAL,
                probability=0.5,
                regime_duration=0,
                transition_prob_bull=0.33,
                transition_prob_bear=0.33,
            )

        states = model.predict(recent_obs.values)
        probs = model.predict_proba(recent_obs.values)

        current_state = states[-1]
        current_prob = probs[-1, current_state]

        # Map to regime enum
        if self.hmm_config.n_regimes == 3:
            regime = MarketRegime(current_state)
        else:
            # Binary: 0=bear, 1=bull
            regime = MarketRegime.BEAR if current_state == 0 else MarketRegime.BULL

        # Calculate regime duration
        duration = 1
        for i in range(len(states) - 2, -1, -1):
            if states[i] == current_state:
                duration += 1
            else:
                break

        # Track history
        if symbol not in self._regime_history:
            self._regime_history[symbol] = []
        self._regime_history[symbol].append(current_state)
        if len(self._regime_history[symbol]) > 100:
            self._regime_history[symbol] = self._regime_history[symbol][-100:]

        # Transition probabilities
        trans = model.transition
        trans_to_bull = (
            trans[current_state, MarketRegime.BULL] if hasattr(trans, "__getitem__") else 0.33
        )
        trans_to_bear = (
            trans[current_state, MarketRegime.BEAR] if hasattr(trans, "__getitem__") else 0.33
        )

        return RegimeState(
            regime=regime,
            probability=current_prob,
            regime_duration=duration,
            transition_prob_bull=trans_to_bull,
            transition_prob_bear=trans_to_bear,
        )

    async def generate(
        self,
        symbol: str,
        data: pd.DataFrame,
        timestamp: datetime,
    ) -> Signal | None:
        """
        Generate regime-aware signal.

        Args:
            symbol: Stock symbol
            data: OHLCV DataFrame
            timestamp: Current timestamp

        Returns:
            Signal if conditions met
        """
        min_bars = self.hmm_config.lookback + 50
        if len(data) < min_bars:
            return None

        # Detect regime
        regime_state = self.detect_regime(data, symbol)
        regime = regime_state.regime

        # Calculate indicators
        rsi = self._calculate_rsi(data)
        trend = self._get_trend(data)
        current_price = data["close"].iloc[-1]

        # Determine signal based on regime
        signal_type = SignalType.HOLD
        direction = Direction.NEUTRAL
        position_mult = 1.0
        score = 0.0

        if regime == MarketRegime.BULL:
            # Momentum long in uptrend
            if trend >= 0 and rsi > 50:
                signal_type = SignalType.ENTRY
                direction = Direction.LONG
                score = 0.6

        elif regime == MarketRegime.BEAR:
            # Conservative: reduce exposure or short on weakness
            position_mult = self.hmm_config.bear_position_mult
            if trend < 0 and rsi < 45:
                signal_type = SignalType.ENTRY
                direction = Direction.SHORT
                score = -0.6

        elif regime == MarketRegime.NEUTRAL:
            # Mean reversion in neutral regime
            if self.hmm_config.regime_mr_neutral:
                if rsi < self.hmm_config.rsi_oversold:
                    signal_type = SignalType.ENTRY
                    direction = Direction.LONG
                    score = 0.5
                elif rsi > self.hmm_config.rsi_overbought:
                    signal_type = SignalType.ENTRY
                    direction = Direction.SHORT
                    score = -0.5

        if signal_type == SignalType.HOLD:
            return Signal(
                symbol=symbol,
                timestamp=timestamp,
                signal_type=SignalType.HOLD,
                direction=Direction.NEUTRAL,
                probability=0.0,
                score=0.0,
                model_id=self.config.model_id,
                model_version=self.config.version,
                confidence=0.0,
                metadata={
                    "regime": regime.name,
                    "regime_probability": regime_state.probability,
                    "regime_duration": regime_state.regime_duration,
                    "rsi": rsi,
                    "trend": trend,
                },
            )

        # Calculate stops
        atr = self._calculate_atr(data)

        if direction == Direction.LONG:
            stop_loss = current_price - (atr * self.hmm_config.atr_stop_mult)
            take_profit = current_price + (atr * self.hmm_config.atr_tp_mult)
        else:
            stop_loss = current_price + (atr * self.hmm_config.atr_stop_mult)
            take_profit = current_price - (atr * self.hmm_config.atr_tp_mult)

        # Confidence based on regime probability and duration
        regime_confidence = regime_state.probability
        duration_confidence = min(1.0, regime_state.regime_duration / 10)
        confidence = 0.6 * regime_confidence + 0.4 * duration_confidence

        return Signal(
            symbol=symbol,
            timestamp=timestamp,
            signal_type=signal_type,
            direction=direction,
            probability=confidence,
            score=float(score),
            model_id=self.config.model_id,
            model_version=self.config.version,
            confidence=confidence,
            metadata={
                "strategy": "hmm_regime",
                "regime": regime.name,
                "regime_probability": regime_state.probability,
                "regime_duration": regime_state.regime_duration,
                "transition_prob_bull": regime_state.transition_prob_bull,
                "transition_prob_bear": regime_state.transition_prob_bear,
                "direction": 1 if direction == Direction.LONG else (-1 if direction == Direction.SHORT else 0),
                "position_mult": position_mult,
                "rsi": rsi,
                "trend": trend,
                "entry_price": current_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "atr": atr,
            },
        )


def analyze_regimes(
    df: pd.DataFrame,
    n_regimes: int = 3,
) -> pd.DataFrame:
    """
    Analyze regime characteristics over time.

    Returns DataFrame with regime assignments and statistics.
    """
    returns = df["close"].pct_change(5).dropna()

    model = get_hmm_model(n_regimes)
    model.fit(returns.values)

    states = model.predict(returns.values)
    probs = model.predict_proba(returns.values)

    result = pd.DataFrame(index=returns.index)
    result["returns"] = returns
    result["regime"] = states
    result["regime_prob"] = probs.max(axis=1)

    # Regime statistics
    for i in range(n_regimes):
        mask = states == i
        regime_returns = returns.values[mask]
        logger.info(
            f"Regime {i}: mean={regime_returns.mean():.4f}, "
            f"std={regime_returns.std():.4f}, "
            f"count={mask.sum()}"
        )

    return result


def backtest(
    df: pd.DataFrame,
    symbol: str,
    config: HMMConfig | None = None,
) -> dict:
    """Backtest HMM Regime strategy."""
    if config is None:
        config = HMMConfig()

    model_config = ModelConfig(
        model_id=f"hmm_regime_backtest_{symbol}",
        model_type="regime",
        parameters={
            "n_regimes": config.n_regimes,
            "lookback": config.lookback,
        },
    )

    model = HMMRegimeModel(model_config)

    trades = []
    position = None

    start_idx = config.lookback + 50

    for i in range(start_idx, len(df)):
        window = df.iloc[: i + 1]
        timestamp = df.index[i]

        import asyncio

        signal = asyncio.get_event_loop().run_until_complete(
            model.generate(symbol, window, timestamp)
        )

        if signal is None:
            continue

        current_price = df["close"].iloc[i]

        # Check exits
        if position is not None:
            exit_reason = None

            if position["direction"] == 1:
                if current_price <= position["stop_loss"]:
                    exit_reason = "stop_loss"
                elif current_price >= position["take_profit"]:
                    exit_reason = "take_profit"
            elif current_price >= position["stop_loss"]:
                exit_reason = "stop_loss"
            elif current_price <= position["take_profit"]:
                exit_reason = "take_profit"

            if exit_reason:
                pnl = (current_price - position["entry"]) * position["direction"]
                pnl_pct = pnl / position["entry"] * 100
                trades.append(
                    {
                        "entry_time": position["entry_time"],
                        "exit_time": timestamp,
                        "entry_price": position["entry"],
                        "exit_price": current_price,
                        "direction": position["direction"],
                        "regime": position["regime"],
                        "pnl_pct": pnl_pct,
                        "exit_reason": exit_reason,
                    }
                )
                position = None

        # New entry
        if position is None and signal.signal_type != SignalType.HOLD:
            direction = signal.metadata.get("direction", 0)
            if direction != 0:
                position = {
                    "entry": current_price,
                    "entry_time": timestamp,
                    "direction": direction,
                    "regime": signal.metadata["regime"],
                    "stop_loss": signal.metadata["stop_loss"],
                    "take_profit": signal.metadata["take_profit"],
                }

    if not trades:
        return {"trades": 0, "total_return": 0, "win_rate": 0}

    trades_df = pd.DataFrame(trades)
    winners = trades_df[trades_df["pnl_pct"] > 0]

    # Per-regime stats
    regime_stats = trades_df.groupby("regime")["pnl_pct"].agg(["mean", "count", "sum"])

    return {
        "trades": len(trades),
        "total_return": trades_df["pnl_pct"].sum(),
        "win_rate": len(winners) / len(trades) * 100,
        "avg_win": winners["pnl_pct"].mean() if len(winners) > 0 else 0,
        "avg_loss": trades_df[trades_df["pnl_pct"] <= 0]["pnl_pct"].mean(),
        "regime_stats": regime_stats.to_dict(),
        "trades_df": trades_df,
    }
