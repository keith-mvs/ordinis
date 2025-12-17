### 1.3 Conditional Probability and Bayes' Theorem

**Bayesian Updating for Trading**:

```python
def bayesian_update(prior: float, likelihood: float, evidence: float) -> float:
    """
    Bayes' theorem: P(H|E) = P(E|H) × P(H) / P(E)

    Applications:
    - Updating probability of regime given new data
    - Signal confidence adjustment
    - Parameter estimation
    """
    return (likelihood * prior) / evidence

# Example: Updating trend probability
class BayesianTrendDetector:
    def __init__(self, prior_trend: float = 0.5):
        self.p_trend = prior_trend

    def update(self, observation: float, historical_stats: dict) -> float:
        """
        Update trend probability given new price observation.
        """
        # Likelihood of observation given trend
        p_obs_given_trend = self._likelihood(observation, historical_stats['trend'])
        # Likelihood of observation given no trend
        p_obs_given_no_trend = self._likelihood(observation, historical_stats['no_trend'])

        # Evidence (marginal likelihood)
        evidence = (p_obs_given_trend * self.p_trend +
                   p_obs_given_no_trend * (1 - self.p_trend))

        # Posterior
        self.p_trend = (p_obs_given_trend * self.p_trend) / evidence
        return self.p_trend
```

---
