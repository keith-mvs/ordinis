### 5.2 Regime Detection

**Hidden Markov Models**:

```python
from hmmlearn.hmm import GaussianHMM

def fit_regime_model(
    returns: np.array,
    n_regimes: int = 2
) -> dict:
    """
    Fit Hidden Markov Model for regime detection.

    Typical regimes: Bull/Bear, Low/High volatility
    """
    model = GaussianHMM(
        n_components=n_regimes,
        covariance_type='full',
        n_iter=1000,
        random_state=42
    )

    returns_2d = returns.reshape(-1, 1)
    model.fit(returns_2d)

    hidden_states = model.predict(returns_2d)
    state_probs = model.predict_proba(returns_2d)

    return {
        'model': model,
        'states': hidden_states,
        'state_probabilities': state_probs,
        'means': model.means_.flatten(),
        'variances': np.array([c[0,0] for c in model.covars_]),
        'transition_matrix': model.transmat_
    }

def regime_switching_forecast(
    hmm_model,
    current_state_probs: np.array,
    horizon: int = 5
) -> np.array:
    """
    Forecast expected returns under regime-switching model.
    """
    forecasts = []
    probs = current_state_probs.copy()

    for _ in range(horizon):
        # Update state probabilities
        probs = probs @ hmm_model.transmat_
        # Expected return
        expected_return = probs @ hmm_model.means_.flatten()
        forecasts.append(expected_return)

    return np.array(forecasts)
```

---
