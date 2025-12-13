"""Walk-forward analysis for ProofBench."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class WalkForwardResult:
    in_sample_returns: list[float]
    out_sample_returns: list[float]
    robustness_ratio: float
    num_windows: int


class WalkForwardAnalyzer:
    """Simple walk-forward analyzer that alternates train/test windows."""

    def __init__(self, train_size: int = 60, test_size: int = 30):
        if train_size <= 0 or test_size <= 0:
            raise ValueError("train_size and test_size must be positive")
        self.train_size = train_size
        self.test_size = test_size

    def analyze(self, returns: pd.Series | np.ndarray) -> WalkForwardResult:
        """
        Perform walk-forward using fixed-size rolling windows.

        Args:
            returns: Series or array of periodic returns.
        """
        ret = pd.Series(returns).dropna().reset_index(drop=True)
        in_sample = []
        out_sample = []
        idx = 0
        while idx + self.train_size + self.test_size <= len(ret):
            train_slice = ret.iloc[idx : idx + self.train_size]
            test_slice = ret.iloc[idx + self.train_size : idx + self.train_size + self.test_size]
            in_sample.append(float(train_slice.mean()))
            out_sample.append(float(test_slice.mean()))
            idx += self.test_size  # advance by test window

        robustness = self._robustness_ratio(in_sample, out_sample)
        return WalkForwardResult(in_sample, out_sample, robustness_ratio=robustness, num_windows=len(in_sample))

    @staticmethod
    def _robustness_ratio(in_sample: list[float], out_sample: list[float]) -> float:
        if not in_sample or not out_sample:
            return 0.0
        in_avg = np.mean(in_sample)
        out_avg = np.mean(out_sample)
        if in_avg == 0:
            return 0.0
        return float(out_avg / in_avg)
