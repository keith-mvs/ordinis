"""Monte Carlo simulation helpers for ProofBench."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MonteCarloResult:
    simulations: int
    var_95: float
    cvar_95: float
    prob_loss: float
    mean_return: float


class MonteCarloAnalyzer:
    """Basic Monte Carlo utilities for returns."""

    def __init__(self, simulations: int = 1000, seed: int | None = None):
        self.simulations = simulations
        self.rng = np.random.default_rng(seed)

    def return_bootstrap(self, returns: pd.Series | np.ndarray) -> MonteCarloResult:
        ret = pd.Series(returns).dropna().to_numpy()
        if len(ret) == 0:
            raise ValueError("No returns provided")

        sims = []
        for _ in range(self.simulations):
            sampled = self.rng.choice(ret, size=len(ret), replace=True)
            sims.append(sampled.mean())

        return self._summarize(np.array(sims))

    def trade_shuffle(self, returns: pd.Series | np.ndarray) -> MonteCarloResult:
        ret = pd.Series(returns).dropna().to_numpy()
        if len(ret) == 0:
            raise ValueError("No returns provided")

        sims = []
        for _ in range(self.simulations):
            shuffled = self.rng.permutation(ret)
            sims.append(shuffled.mean())

        return self._summarize(np.array(sims))

    def _summarize(self, sims: np.ndarray) -> MonteCarloResult:
        var_95 = float(np.percentile(sims, 5))
        cvar_95 = float(sims[sims <= var_95].mean()) if np.any(sims <= var_95) else var_95
        prob_loss = float(np.mean(sims < 0))
        mean_ret = float(sims.mean())
        return MonteCarloResult(
            simulations=len(sims),
            var_95=var_95,
            cvar_95=cvar_95,
            prob_loss=prob_loss,
            mean_return=mean_ret,
        )
