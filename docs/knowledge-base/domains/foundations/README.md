# Mathematical & Algorithmic Foundations for Systematic Trading

## Purpose

This section provides the mathematical and algorithmic foundations that underpin systematic trading strategies. Understanding these concepts is essential for building rigorous, quantitatively-driven trading systems.

---

## 1. Probability Theory Foundations

### 1.1 Probability Spaces and Random Variables

**Formal Framework**:

A probability space is a triple (Ω, ℱ, ℙ) where:
- Ω = sample space (all possible outcomes)
- ℱ = σ-algebra (collection of events)
- ℙ = probability measure

```python
# Conceptual representation
@dataclass
class ProbabilitySpace:
    sample_space: Set       # Ω
    events: SigmaAlgebra    # ℱ
    measure: Probability    # ℙ

# Random variable X: Ω → ℝ
# Maps outcomes to real numbers (e.g., stock returns)
```

**Key Distributions in Finance**:

| Distribution | Use Case | Parameters |
|--------------|----------|------------|
| Normal | Log returns (approx) | μ, σ² |
| Log-normal | Asset prices | μ, σ² |
| Student-t | Fat-tailed returns | ν (degrees of freedom) |
| Poisson | Jump events | λ (intensity) |
| Exponential | Time between events | λ |

---

### 1.2 Moments and Tail Risk

**Standard Moments**:

```python
# Moment calculations
def moments(returns: np.array) -> dict:
    return {
        'mean': np.mean(returns),                    # 1st moment: E[X]
        'variance': np.var(returns),                 # 2nd central moment
        'skewness': scipy.stats.skew(returns),       # 3rd standardized moment
        'kurtosis': scipy.stats.kurtosis(returns)    # 4th standardized moment (excess)
    }

# Interpretation for trading
SKEWNESS_INTERPRETATION = {
    'negative': 'Left tail risk (crash risk)',
    'zero': 'Symmetric distribution',
    'positive': 'Right tail (windfall potential)'
}

KURTOSIS_INTERPRETATION = {
    'leptokurtic': 'kurtosis > 0: Fat tails, more extreme events',
    'mesokurtic': 'kurtosis ≈ 0: Normal-like tails',
    'platykurtic': 'kurtosis < 0: Thin tails'
}
```

**Tail Risk Measures**:

```python
def value_at_risk(returns: np.array, confidence: float = 0.95) -> float:
    """
    VaR: Maximum loss at given confidence level.
    VaR_α = -inf{x : P(X ≤ x) ≥ 1 - α}
    """
    return -np.percentile(returns, (1 - confidence) * 100)

def expected_shortfall(returns: np.array, confidence: float = 0.95) -> float:
    """
    ES (CVaR): Expected loss beyond VaR.
    ES_α = E[X | X ≤ -VaR_α]
    """
    var = value_at_risk(returns, confidence)
    return -returns[returns <= -var].mean()

def tail_index_hill(returns: np.array, k: int) -> float:
    """
    Hill estimator for tail index (power law exponent).
    Used for extreme value analysis.
    """
    sorted_returns = np.sort(np.abs(returns))[::-1]
    return k / np.sum(np.log(sorted_returns[:k] / sorted_returns[k]))
```

---

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

## 2. Stochastic Processes

### 2.1 Brownian Motion (Wiener Process)

**Definition**: A stochastic process W(t) is standard Brownian motion if:
1. W(0) = 0
2. W(t) has independent increments
3. W(t) - W(s) ~ N(0, t-s) for t > s
4. W(t) has continuous paths

**Geometric Brownian Motion (GBM)**:

The standard model for asset prices:

```
dS(t) = μS(t)dt + σS(t)dW(t)
```

Solution:
```
S(t) = S(0) × exp((μ - σ²/2)t + σW(t))
```

```python
def simulate_gbm(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    N: int,
    n_paths: int = 1
) -> np.array:
    """
    Simulate Geometric Brownian Motion paths.

    Parameters:
        S0: Initial price
        mu: Drift (expected return)
        sigma: Volatility
        T: Time horizon
        N: Number of time steps
        n_paths: Number of simulation paths

    Returns:
        Array of shape (n_paths, N+1) with simulated prices
    """
    dt = T / N
    paths = np.zeros((n_paths, N + 1))
    paths[:, 0] = S0

    # Generate random increments
    dW = np.random.normal(0, np.sqrt(dt), (n_paths, N))

    # Simulate paths using exact solution
    for i in range(N):
        paths[:, i+1] = paths[:, i] * np.exp(
            (mu - 0.5 * sigma**2) * dt + sigma * dW[:, i]
        )

    return paths
```

---

### 2.2 Martingales

**Definition**: A process M(t) is a martingale if:
```
E[M(t) | ℱ(s)] = M(s)  for all s ≤ t
```

**Key Properties for Trading**:
- Under risk-neutral measure, discounted asset prices are martingales
- No free lunch with vanishing risk (NFLVR) theorem
- Optional stopping theorem (cannot profit from perfect timing alone)

```python
def is_martingale_test(prices: np.array, lags: int = 10) -> dict:
    """
    Statistical tests for martingale property.

    Uses variance ratio test (Lo-MacKinlay).
    Under martingale: Var(r_k) = k × Var(r_1)
    """
    returns = np.diff(np.log(prices))

    results = {}
    for k in range(2, lags + 1):
        # k-period returns
        returns_k = np.diff(np.log(prices[::k]))

        # Variance ratio
        vr = np.var(returns_k) / (k * np.var(returns))

        # Test statistic (under null, VR = 1)
        results[k] = {
            'variance_ratio': vr,
            'deviation_from_1': abs(vr - 1)
        }

    return results
```

---

### 2.3 Jump-Diffusion Processes

**Merton Jump-Diffusion Model**:

```
dS(t) = μS(t)dt + σS(t)dW(t) + S(t)dJ(t)
```

Where J(t) is a compound Poisson process with jump size Y.

```python
def simulate_merton_jump_diffusion(
    S0: float,
    mu: float,
    sigma: float,
    lambda_: float,  # Jump intensity
    mu_j: float,     # Mean jump size (log)
    sigma_j: float,  # Jump size volatility
    T: float,
    N: int,
    n_paths: int = 1
) -> np.array:
    """
    Simulate Merton jump-diffusion model.
    """
    dt = T / N
    paths = np.zeros((n_paths, N + 1))
    paths[:, 0] = S0

    for i in range(N):
        # Diffusion component
        dW = np.random.normal(0, np.sqrt(dt), n_paths)

        # Jump component (Poisson arrivals)
        n_jumps = np.random.poisson(lambda_ * dt, n_paths)
        jump_sizes = np.zeros(n_paths)
        for j in range(n_paths):
            if n_jumps[j] > 0:
                jumps = np.random.normal(mu_j, sigma_j, n_jumps[j])
                jump_sizes[j] = np.sum(np.exp(jumps) - 1)

        # Update prices
        paths[:, i+1] = paths[:, i] * np.exp(
            (mu - 0.5 * sigma**2) * dt + sigma * dW
        ) * (1 + jump_sizes)

    return paths
```

---

### 2.4 Stochastic Calculus (Itô Calculus)

**Itô's Lemma**: For f(S,t) where S follows dS = μdt + σdW:

```
df = (∂f/∂t + μ∂f/∂S + ½σ²∂²f/∂S²)dt + σ(∂f/∂S)dW
```

**Applications**:

```python
# Example: Deriving Black-Scholes using Itô's Lemma
"""
Let V(S,t) be option value. Under risk-neutral measure:

dV = (∂V/∂t + rS∂V/∂S + ½σ²S²∂²V/∂S²)dt + σS(∂V/∂S)dW

For replicating portfolio (Δ shares + bond):
dΠ = Δ×dS + r(V - ΔS)dt

Matching terms and eliminating dW gives Black-Scholes PDE:
∂V/∂t + rS∂V/∂S + ½σ²S²∂²V/∂S² - rV = 0
"""

def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Black-Scholes call option price.

    Derived from solving the BS PDE with boundary condition max(S-K, 0).
    """
    from scipy.stats import norm

    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
```

---

### 2.5 Mean-Reverting Processes

**Ornstein-Uhlenbeck Process**:

```
dX(t) = θ(μ - X(t))dt + σdW(t)
```

Where θ = speed of mean reversion, μ = long-term mean.

```python
def simulate_ornstein_uhlenbeck(
    X0: float,
    theta: float,  # Mean reversion speed
    mu: float,     # Long-term mean
    sigma: float,  # Volatility
    T: float,
    N: int,
    n_paths: int = 1
) -> np.array:
    """
    Simulate Ornstein-Uhlenbeck (OU) process.

    Used for:
    - Interest rate modeling
    - Pairs trading spread
    - Volatility modeling
    """
    dt = T / N
    paths = np.zeros((n_paths, N + 1))
    paths[:, 0] = X0

    # Exact simulation (not Euler discretization)
    exp_theta = np.exp(-theta * dt)
    std = sigma * np.sqrt((1 - exp_theta**2) / (2 * theta))

    for i in range(N):
        paths[:, i+1] = (
            mu + (paths[:, i] - mu) * exp_theta +
            std * np.random.normal(0, 1, n_paths)
        )

    return paths

def estimate_ou_parameters(spread: np.array, dt: float = 1/252) -> dict:
    """
    Estimate OU parameters from time series (for pairs trading).

    Uses OLS regression: X(t+1) - X(t) = θ(μ - X(t))dt + noise
    """
    X = spread[:-1]
    dX = np.diff(spread)

    # Regression: dX = a + b*X
    b, a = np.polyfit(X, dX, 1)

    theta = -b / dt
    mu = a / (theta * dt)
    residuals = dX - (a + b * X)
    sigma = np.std(residuals) / np.sqrt(dt)

    # Half-life of mean reversion
    half_life = np.log(2) / theta

    return {
        'theta': theta,
        'mu': mu,
        'sigma': sigma,
        'half_life_days': half_life
    }
```

---

## 3. Time Series Analysis

### 3.1 Stationarity and Unit Roots

**Stationarity Conditions**:
- Strict stationarity: Joint distribution invariant under time shifts
- Weak stationarity: Constant mean, variance, and autocovariance

```python
from statsmodels.tsa.stattools import adfuller, kpss

def stationarity_tests(series: np.array) -> dict:
    """
    Test for stationarity using ADF and KPSS tests.

    ADF: H0 = unit root (non-stationary)
    KPSS: H0 = stationary

    Ideal: Reject ADF, fail to reject KPSS
    """
    # Augmented Dickey-Fuller test
    adf_result = adfuller(series, autolag='AIC')

    # KPSS test
    kpss_result = kpss(series, regression='c')

    return {
        'adf': {
            'statistic': adf_result[0],
            'p_value': adf_result[1],
            'critical_values': adf_result[4],
            'is_stationary': adf_result[1] < 0.05
        },
        'kpss': {
            'statistic': kpss_result[0],
            'p_value': kpss_result[1],
            'critical_values': kpss_result[3],
            'is_stationary': kpss_result[1] > 0.05
        }
    }

def make_stationary(series: pd.Series) -> Tuple[pd.Series, int]:
    """
    Difference series until stationary.
    Returns (stationary_series, order_of_differencing).
    """
    d = 0
    diff_series = series.copy()

    while not stationarity_tests(diff_series.dropna())['adf']['is_stationary']:
        diff_series = diff_series.diff()
        d += 1
        if d > 2:
            break

    return diff_series.dropna(), d
```

---

### 3.2 ARIMA Models

**ARIMA(p, d, q)**: AutoRegressive Integrated Moving Average

```
(1 - Σφ_i L^i)(1-L)^d X_t = (1 + Σθ_j L^j)ε_t
```

Where L is the lag operator.

```python
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf

def identify_arima_order(series: np.array, max_order: int = 5) -> dict:
    """
    Identify ARIMA order using ACF/PACF analysis.

    AR(p): PACF cuts off after lag p, ACF decays
    MA(q): ACF cuts off after lag q, PACF decays
    """
    acf_values = acf(series, nlags=max_order)
    pacf_values = pacf(series, nlags=max_order)

    # Significance threshold (approximate)
    threshold = 1.96 / np.sqrt(len(series))

    # Find where ACF/PACF become insignificant
    significant_acf = np.where(np.abs(acf_values[1:]) > threshold)[0]
    significant_pacf = np.where(np.abs(pacf_values[1:]) > threshold)[0]

    return {
        'acf': acf_values,
        'pacf': pacf_values,
        'suggested_p': len(significant_pacf) if len(significant_pacf) > 0 else 0,
        'suggested_q': len(significant_acf) if len(significant_acf) > 0 else 0,
        'threshold': threshold
    }

def fit_arima(series: np.array, order: Tuple[int, int, int]) -> dict:
    """
    Fit ARIMA model and return diagnostics.
    """
    model = ARIMA(series, order=order)
    fitted = model.fit()

    return {
        'params': fitted.params,
        'aic': fitted.aic,
        'bic': fitted.bic,
        'residuals': fitted.resid,
        'forecast': fitted.forecast,
        'summary': fitted.summary()
    }

def auto_arima(series: np.array, max_p: int = 5, max_q: int = 5) -> dict:
    """
    Automatic ARIMA order selection using AIC.
    """
    best_aic = np.inf
    best_order = (0, 0, 0)

    _, d = make_stationary(pd.Series(series))

    for p in range(max_p + 1):
        for q in range(max_q + 1):
            try:
                model = ARIMA(series, order=(p, d, q))
                fitted = model.fit()
                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_order = (p, d, q)
            except:
                continue

    return {
        'best_order': best_order,
        'best_aic': best_aic
    }
```

---

### 3.3 GARCH Models (Volatility Modeling)

**GARCH(p, q)**: Generalized Autoregressive Conditional Heteroskedasticity

```
r_t = μ + ε_t,  ε_t = σ_t × z_t,  z_t ~ N(0,1)
σ²_t = ω + Σα_i ε²_{t-i} + Σβ_j σ²_{t-j}
```

```python
from arch import arch_model

def fit_garch(returns: np.array, p: int = 1, q: int = 1) -> dict:
    """
    Fit GARCH(p,q) model for volatility forecasting.
    """
    model = arch_model(returns, vol='Garch', p=p, q=q, rescale=False)
    fitted = model.fit(disp='off')

    return {
        'params': {
            'omega': fitted.params['omega'],
            'alpha': [fitted.params[f'alpha[{i}]'] for i in range(1, p+1)],
            'beta': [fitted.params[f'beta[{i}]'] for i in range(1, q+1)]
        },
        'conditional_volatility': fitted.conditional_volatility,
        'standardized_residuals': fitted.std_resid,
        'forecast': fitted.forecast,
        'aic': fitted.aic,
        'bic': fitted.bic
    }

def forecast_volatility(
    returns: np.array,
    horizon: int = 5,
    model_type: str = 'GARCH'
) -> np.array:
    """
    Forecast volatility using GARCH-family models.
    """
    if model_type == 'GARCH':
        model = arch_model(returns, vol='Garch', p=1, q=1)
    elif model_type == 'EGARCH':
        model = arch_model(returns, vol='EGARCH', p=1, q=1)
    elif model_type == 'GJR-GARCH':
        model = arch_model(returns, vol='Garch', p=1, o=1, q=1)

    fitted = model.fit(disp='off')
    forecast = fitted.forecast(horizon=horizon)

    return np.sqrt(forecast.variance.values[-1])
```

**GARCH Variants**:

| Model | Feature | Use Case |
|-------|---------|----------|
| GARCH | Symmetric shocks | General volatility |
| EGARCH | Asymmetric, no positivity constraint | Leverage effect |
| GJR-GARCH | Asymmetric threshold | Equity volatility |
| TGARCH | Threshold model | Regime-dependent vol |

---

### 3.4 Cointegration

**Engle-Granger Two-Step Procedure**:

```python
from statsmodels.tsa.stattools import coint

def test_cointegration(y1: np.array, y2: np.array) -> dict:
    """
    Test for cointegration between two time series.

    Two series are cointegrated if:
    1. Both are I(1) (integrated of order 1)
    2. A linear combination is I(0) (stationary)
    """
    # Engle-Granger test
    coint_stat, p_value, crit_values = coint(y1, y2)

    # Estimate hedge ratio via OLS
    hedge_ratio = np.polyfit(y2, y1, 1)[0]

    # Spread
    spread = y1 - hedge_ratio * y2

    # Test spread for stationarity
    spread_test = stationarity_tests(spread)

    return {
        'coint_statistic': coint_stat,
        'p_value': p_value,
        'critical_values': crit_values,
        'hedge_ratio': hedge_ratio,
        'spread': spread,
        'spread_stationary': spread_test['adf']['is_stationary'],
        'is_cointegrated': p_value < 0.05
    }

def johansen_cointegration(data: np.array, det_order: int = 0) -> dict:
    """
    Johansen cointegration test for multiple time series.
    """
    from statsmodels.tsa.vector_ar.vecm import coint_johansen

    result = coint_johansen(data, det_order, 1)

    return {
        'eigenvalues': result.eig,
        'trace_statistic': result.lr1,
        'max_eigen_statistic': result.lr2,
        'critical_values_trace': result.cvt,
        'critical_values_max_eigen': result.cvm,
        'cointegrating_vectors': result.evec
    }
```

---

### 3.5 Signal Processing for Finance

**Fourier Analysis**:

```python
def spectral_analysis(series: np.array, sampling_freq: float = 252) -> dict:
    """
    Spectral analysis of time series.

    Identifies dominant frequencies/cycles in data.
    """
    from scipy.fft import fft, fftfreq

    n = len(series)
    fft_values = fft(series - np.mean(series))
    frequencies = fftfreq(n, 1/sampling_freq)

    # Power spectrum (positive frequencies only)
    positive_freq_idx = frequencies > 0
    power = np.abs(fft_values[positive_freq_idx])**2
    freqs = frequencies[positive_freq_idx]

    # Find dominant frequencies
    peak_indices = np.argsort(power)[-5:][::-1]  # Top 5 peaks
    dominant_frequencies = freqs[peak_indices]
    dominant_periods = 1 / dominant_frequencies  # In trading days

    return {
        'frequencies': freqs,
        'power_spectrum': power,
        'dominant_frequencies': dominant_frequencies,
        'dominant_periods_days': dominant_periods
    }

def wavelet_decomposition(series: np.array, wavelet: str = 'db4', level: int = 4) -> dict:
    """
    Wavelet decomposition for multi-scale analysis.

    Useful for identifying trends at different time scales.
    """
    import pywt

    coeffs = pywt.wavedec(series, wavelet, level=level)

    # Reconstruct components at each level
    components = []
    for i, coeff in enumerate(coeffs):
        # Zero out other coefficients
        temp_coeffs = [np.zeros_like(c) for c in coeffs]
        temp_coeffs[i] = coeff
        component = pywt.waverec(temp_coeffs, wavelet)[:len(series)]
        components.append(component)

    return {
        'coefficients': coeffs,
        'components': components,
        'approximation': components[0],  # Low-frequency trend
        'details': components[1:]         # High-frequency noise
    }
```

**Kalman Filter**:

```python
from filterpy.kalman import KalmanFilter

def kalman_trend_filter(prices: np.array) -> dict:
    """
    Kalman filter for trend extraction.

    State: [level, velocity]
    Observation: price
    """
    kf = KalmanFilter(dim_x=2, dim_z=1)

    # State transition matrix (constant velocity model)
    kf.F = np.array([[1, 1],
                     [0, 1]])

    # Measurement matrix
    kf.H = np.array([[1, 0]])

    # Covariance matrices
    kf.Q = np.array([[0.01, 0],
                     [0, 0.001]])  # Process noise
    kf.R = np.array([[1.0]])       # Measurement noise

    # Initial state
    kf.x = np.array([[prices[0]], [0]])
    kf.P *= 100

    # Run filter
    filtered_state = np.zeros((len(prices), 2))
    for i, price in enumerate(prices):
        kf.predict()
        kf.update(price)
        filtered_state[i] = kf.x.flatten()

    return {
        'level': filtered_state[:, 0],      # Filtered price level
        'velocity': filtered_state[:, 1],    # Trend velocity
        'trend_direction': np.sign(filtered_state[:, 1])
    }
```

---

## 4. Optimization and Control

### 4.1 Mean-Variance Optimization (Markowitz)

**Classic Formulation**:

```
min  w'Σw           (minimize variance)
s.t. w'μ ≥ r_target (return constraint)
     w'1 = 1        (weights sum to 1)
     w ≥ 0          (no short selling, optional)
```

```python
from scipy.optimize import minimize

def mean_variance_optimization(
    expected_returns: np.array,
    cov_matrix: np.array,
    target_return: Optional[float] = None,
    risk_free_rate: float = 0.0
) -> dict:
    """
    Mean-variance portfolio optimization.
    """
    n_assets = len(expected_returns)

    def portfolio_variance(weights):
        return weights @ cov_matrix @ weights

    def portfolio_return(weights):
        return weights @ expected_returns

    # Constraints
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

    if target_return is not None:
        constraints.append({
            'type': 'eq',
            'fun': lambda w: portfolio_return(w) - target_return
        })

    # Bounds (long-only)
    bounds = [(0, 1) for _ in range(n_assets)]

    # Initial guess
    w0 = np.ones(n_assets) / n_assets

    result = minimize(
        portfolio_variance,
        w0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    optimal_weights = result.x
    optimal_return = portfolio_return(optimal_weights)
    optimal_volatility = np.sqrt(portfolio_variance(optimal_weights))
    sharpe_ratio = (optimal_return - risk_free_rate) / optimal_volatility

    return {
        'weights': optimal_weights,
        'expected_return': optimal_return,
        'volatility': optimal_volatility,
        'sharpe_ratio': sharpe_ratio
    }

def efficient_frontier(
    expected_returns: np.array,
    cov_matrix: np.array,
    n_points: int = 50
) -> Tuple[np.array, np.array, np.array]:
    """
    Compute the efficient frontier.
    """
    min_ret = expected_returns.min()
    max_ret = expected_returns.max()
    target_returns = np.linspace(min_ret, max_ret, n_points)

    frontier_volatilities = []
    frontier_weights = []

    for target in target_returns:
        try:
            result = mean_variance_optimization(
                expected_returns, cov_matrix, target_return=target
            )
            frontier_volatilities.append(result['volatility'])
            frontier_weights.append(result['weights'])
        except:
            frontier_volatilities.append(np.nan)
            frontier_weights.append(None)

    return target_returns, np.array(frontier_volatilities), frontier_weights
```

---

### 4.2 Black-Litterman Model

**Combines market equilibrium with investor views**:

```python
def black_litterman(
    cov_matrix: np.array,
    market_caps: np.array,
    views: np.array,          # P × μ = Q + ε
    view_matrix: np.array,    # P (picking matrix)
    view_confidence: np.array, # Ω (uncertainty)
    risk_aversion: float = 2.5,
    tau: float = 0.05
) -> dict:
    """
    Black-Litterman model for portfolio optimization.
    """
    n_assets = len(market_caps)

    # Market-implied equilibrium returns
    market_weights = market_caps / market_caps.sum()
    pi = risk_aversion * cov_matrix @ market_weights

    # View precision (inverse of view covariance)
    omega = np.diag(view_confidence)
    omega_inv = np.linalg.inv(omega)

    # Posterior expected returns
    tau_sigma = tau * cov_matrix
    tau_sigma_inv = np.linalg.inv(tau_sigma)

    # Combined estimate
    M = np.linalg.inv(tau_sigma_inv + view_matrix.T @ omega_inv @ view_matrix)
    posterior_mean = M @ (tau_sigma_inv @ pi + view_matrix.T @ omega_inv @ views)

    # Posterior covariance
    posterior_cov = M + cov_matrix

    return {
        'equilibrium_returns': pi,
        'posterior_returns': posterior_mean,
        'posterior_covariance': posterior_cov,
        'market_weights': market_weights
    }
```

---

### 4.3 Convex Optimization

**Risk Parity Portfolio**:

```python
def risk_parity_portfolio(cov_matrix: np.array) -> np.array:
    """
    Equal risk contribution portfolio.

    Each asset contributes equally to total portfolio risk.
    """
    n_assets = cov_matrix.shape[0]

    def risk_contribution(weights):
        portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
        marginal_contrib = cov_matrix @ weights
        risk_contrib = weights * marginal_contrib / portfolio_vol
        return risk_contrib

    def objective(weights):
        rc = risk_contribution(weights)
        # Minimize difference from equal contribution
        return np.sum((rc - rc.mean())**2)

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0.01, 1) for _ in range(n_assets)]
    w0 = np.ones(n_assets) / n_assets

    result = minimize(objective, w0, method='SLSQP',
                     bounds=bounds, constraints=constraints)

    return result.x
```

**Robust Optimization**:

```python
def robust_portfolio_optimization(
    expected_returns: np.array,
    cov_matrix: np.array,
    uncertainty_set: float = 0.1  # ε for ||μ - μ̂|| ≤ ε
) -> dict:
    """
    Robust portfolio optimization accounting for estimation error.

    Uses ellipsoidal uncertainty set around expected returns.
    """
    n_assets = len(expected_returns)

    def worst_case_return(weights):
        # Worst-case return within uncertainty set
        nominal_return = weights @ expected_returns
        uncertainty_penalty = uncertainty_set * np.sqrt(weights @ cov_matrix @ weights)
        return nominal_return - uncertainty_penalty

    def neg_worst_case_return(weights):
        return -worst_case_return(weights)

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0, 1) for _ in range(n_assets)]
    w0 = np.ones(n_assets) / n_assets

    result = minimize(neg_worst_case_return, w0, method='SLSQP',
                     bounds=bounds, constraints=constraints)

    return {
        'weights': result.x,
        'worst_case_return': -result.fun
    }
```

---

### 4.4 Dynamic Programming for Trading

**Optimal Execution (Almgren-Chriss)**:

```python
def almgren_chriss_optimal_execution(
    total_shares: int,
    time_steps: int,
    volatility: float,
    temporary_impact: float,  # η
    permanent_impact: float,  # γ
    risk_aversion: float      # λ
) -> Tuple[np.array, float]:
    """
    Optimal trade schedule to minimize execution cost + risk.

    Objective: min E[cost] + λ × Var[cost]
    """
    # Optimal trading rate (closed-form solution)
    kappa = np.sqrt(risk_aversion * volatility**2 / temporary_impact)

    # Trade schedule
    trade_times = np.arange(time_steps + 1)
    remaining_shares = total_shares * np.sinh(kappa * (time_steps - trade_times)) / np.sinh(kappa * time_steps)

    # Trades at each step
    trades = -np.diff(remaining_shares)

    # Expected cost
    expected_cost = (
        permanent_impact * total_shares**2 / 2 +
        temporary_impact * np.sum(trades**2)
    )

    return trades, expected_cost
```

---

## 5. Statistical Learning for Trading

### 5.1 Factor Models

**Fama-French Type Models**:

```python
def estimate_factor_model(
    returns: np.array,        # Asset returns (T × N)
    factors: np.array,        # Factor returns (T × K)
    fit_intercept: bool = True
) -> dict:
    """
    Estimate multi-factor model: R = α + β × F + ε
    """
    from sklearn.linear_model import LinearRegression

    n_assets = returns.shape[1]
    results = {
        'alphas': [],
        'betas': [],
        'r_squared': [],
        'residuals': []
    }

    for i in range(n_assets):
        model = LinearRegression(fit_intercept=fit_intercept)
        model.fit(factors, returns[:, i])

        results['alphas'].append(model.intercept_ if fit_intercept else 0)
        results['betas'].append(model.coef_)
        results['r_squared'].append(model.score(factors, returns[:, i]))
        results['residuals'].append(returns[:, i] - model.predict(factors))

    return {
        'alphas': np.array(results['alphas']),
        'betas': np.array(results['betas']),
        'r_squared': np.array(results['r_squared']),
        'residual_matrix': np.array(results['residuals']).T
    }

def pca_factor_extraction(
    returns: np.array,
    n_factors: int = 5
) -> dict:
    """
    Extract statistical factors using PCA.
    """
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_factors)
    factors = pca.fit_transform(returns)

    return {
        'factors': factors,
        'loadings': pca.components_.T,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'cumulative_variance': np.cumsum(pca.explained_variance_ratio_)
    }
```

---

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

### 5.3 Machine Learning for Alpha

**Cross-Sectional Prediction**:

```python
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

def cross_sectional_ml_model(
    features: np.array,       # (T, N, K) - time, assets, features
    returns: np.array,        # (T, N) - forward returns
    model_type: str = 'rf'
) -> dict:
    """
    Train ML model for cross-sectional return prediction.

    Features might include: momentum, value, size, quality, etc.
    """
    T, N, K = features.shape

    # Flatten for training
    X = features.reshape(-1, K)
    y = returns.flatten()

    # Remove NaN
    valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[valid_mask]
    y = y[valid_mask]

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Model selection
    if model_type == 'rf':
        model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    elif model_type == 'gbm':
        model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)

    # Time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []

    for train_idx, val_idx in tscv.split(X_scaled):
        model.fit(X_scaled[train_idx], y[train_idx])
        score = model.score(X_scaled[val_idx], y[val_idx])
        cv_scores.append(score)

    # Final model on all data
    model.fit(X_scaled, y)

    return {
        'model': model,
        'scaler': scaler,
        'cv_scores': cv_scores,
        'feature_importance': model.feature_importances_ if hasattr(model, 'feature_importances_') else None
    }
```

---

### 5.4 Regularization Methods

```python
from sklearn.linear_model import Lasso, Ridge, ElasticNet

def regularized_factor_selection(
    returns: np.array,
    candidate_factors: np.array,
    alpha_range: np.array = np.logspace(-4, 0, 50)
) -> dict:
    """
    Use LASSO for sparse factor selection.
    """
    from sklearn.model_selection import cross_val_score

    best_alpha = None
    best_score = -np.inf

    for alpha in alpha_range:
        model = Lasso(alpha=alpha)
        scores = cross_val_score(model, candidate_factors, returns, cv=5)
        mean_score = scores.mean()

        if mean_score > best_score:
            best_score = mean_score
            best_alpha = alpha

    # Fit final model
    final_model = Lasso(alpha=best_alpha)
    final_model.fit(candidate_factors, returns)

    selected_factors = np.where(np.abs(final_model.coef_) > 1e-6)[0]

    return {
        'selected_factors': selected_factors,
        'coefficients': final_model.coef_,
        'best_alpha': best_alpha,
        'best_cv_score': best_score
    }
```

---

## 6. Numerical Methods and Simulation

### 6.1 Monte Carlo Methods

**Variance Reduction Techniques**:

```python
def monte_carlo_option_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_simulations: int = 100000,
    antithetic: bool = True,
    control_variate: bool = True
) -> dict:
    """
    Monte Carlo option pricing with variance reduction.
    """
    dt = T
    discount = np.exp(-r * T)

    # Generate random numbers
    Z = np.random.normal(0, 1, n_simulations)

    # Antithetic variates
    if antithetic:
        Z = np.concatenate([Z, -Z])

    # Simulate terminal prices
    S_T = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    # Option payoffs
    payoffs = np.maximum(S_T - K, 0)

    # Control variate using forward price
    if control_variate:
        forward = S0 * np.exp(r * T)
        # Covariance between payoff and control
        cov = np.cov(payoffs, S_T)[0, 1]
        var_control = np.var(S_T)
        beta = cov / var_control

        # Adjusted payoffs
        payoffs = payoffs - beta * (S_T - forward)

    # Price estimate
    price = discount * np.mean(payoffs)
    std_error = discount * np.std(payoffs) / np.sqrt(len(payoffs))

    return {
        'price': price,
        'std_error': std_error,
        '95_ci': (price - 1.96 * std_error, price + 1.96 * std_error)
    }

def importance_sampling_var(
    returns: np.array,
    confidence: float = 0.99,
    n_simulations: int = 100000
) -> dict:
    """
    Importance sampling for rare event (VaR) estimation.
    """
    mu = np.mean(returns)
    sigma = np.std(returns)

    # Shift distribution to sample more tail events
    shift = -2 * sigma  # Shift mean to focus on left tail

    # Generate samples from shifted distribution
    samples = np.random.normal(mu + shift, sigma, n_simulations)

    # Importance weights (likelihood ratio)
    weights = np.exp(
        -0.5 * ((samples - mu)**2 - (samples - (mu + shift))**2) / sigma**2
    )

    # Weighted quantile estimation
    sorted_indices = np.argsort(samples)
    sorted_samples = samples[sorted_indices]
    sorted_weights = weights[sorted_indices]
    cumulative_weights = np.cumsum(sorted_weights) / np.sum(sorted_weights)

    var_index = np.searchsorted(cumulative_weights, 1 - confidence)
    var_estimate = -sorted_samples[var_index]

    return {
        'var': var_estimate,
        'effective_sample_size': np.sum(weights)**2 / np.sum(weights**2)
    }
```

---

### 6.2 Finite Difference Methods

**Solving Black-Scholes PDE**:

```python
def crank_nicolson_option(
    S_max: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    M: int = 100,  # Price steps
    N: int = 100,  # Time steps
    option_type: str = 'call'
) -> np.array:
    """
    Crank-Nicolson scheme for American/European options.
    """
    dS = S_max / M
    dt = T / N
    S = np.linspace(0, S_max, M + 1)

    # Initialize option values at maturity
    if option_type == 'call':
        V = np.maximum(S - K, 0)
    else:
        V = np.maximum(K - S, 0)

    # Coefficients
    alpha = 0.25 * dt * (sigma**2 * np.arange(M+1)**2 - r * np.arange(M+1))
    beta = -0.5 * dt * (sigma**2 * np.arange(M+1)**2 + r)
    gamma = 0.25 * dt * (sigma**2 * np.arange(M+1)**2 + r * np.arange(M+1))

    # Tridiagonal matrices
    A = np.diag(1 - beta[1:M]) + np.diag(-alpha[2:M], -1) + np.diag(-gamma[1:M-1], 1)
    B = np.diag(1 + beta[1:M]) + np.diag(alpha[2:M], -1) + np.diag(gamma[1:M-1], 1)

    # Time stepping
    for n in range(N):
        rhs = B @ V[1:M]

        # Boundary conditions
        rhs[0] += alpha[1] * (V[0] + V[0])  # At S=0
        rhs[-1] += gamma[M-1] * (V[M] + V[M])  # At S=S_max

        V[1:M] = np.linalg.solve(A, rhs)

    return S, V
```

---

### 6.3 Discretization Schemes for SDEs

**Euler-Maruyama vs Milstein**:

```python
def euler_maruyama(
    drift: Callable,
    diffusion: Callable,
    X0: float,
    T: float,
    N: int,
    n_paths: int = 1
) -> np.array:
    """
    Euler-Maruyama discretization.

    dX = μ(X,t)dt + σ(X,t)dW

    Weak order 1, strong order 0.5
    """
    dt = T / N
    paths = np.zeros((n_paths, N + 1))
    paths[:, 0] = X0

    for i in range(N):
        t = i * dt
        X = paths[:, i]
        dW = np.random.normal(0, np.sqrt(dt), n_paths)

        paths[:, i+1] = X + drift(X, t) * dt + diffusion(X, t) * dW

    return paths

def milstein(
    drift: Callable,
    diffusion: Callable,
    diffusion_derivative: Callable,
    X0: float,
    T: float,
    N: int,
    n_paths: int = 1
) -> np.array:
    """
    Milstein scheme with higher-order correction.

    Strong order 1.0
    """
    dt = T / N
    paths = np.zeros((n_paths, N + 1))
    paths[:, 0] = X0

    for i in range(N):
        t = i * dt
        X = paths[:, i]
        dW = np.random.normal(0, np.sqrt(dt), n_paths)

        # Milstein correction term
        correction = 0.5 * diffusion(X, t) * diffusion_derivative(X, t) * (dW**2 - dt)

        paths[:, i+1] = (
            X +
            drift(X, t) * dt +
            diffusion(X, t) * dW +
            correction
        )

    return paths
```

---

## 7. Key Academic References

### Probability & Stochastic Processes

| Author | Title | Year | Topic |
|--------|-------|------|-------|
| Shreve, S. | Stochastic Calculus for Finance I & II | 2004 | Foundation |
| Øksendal, B. | Stochastic Differential Equations | 2003 | SDEs |
| Karatzas, I. & Shreve, S. | Brownian Motion and Stochastic Calculus | 1991 | Theory |
| Protter, P. | Stochastic Integration and Differential Equations | 2005 | Advanced |

### Time Series

| Author | Title | Year | Topic |
|--------|-------|------|-------|
| Hamilton, J.D. | Time Series Analysis | 1994 | Comprehensive |
| Tsay, R.S. | Analysis of Financial Time Series | 2010 | Finance-specific |
| Engle, R.F. | GARCH 101 | 2001 | Volatility |
| Johansen, S. | Likelihood-Based Inference in Cointegration | 1995 | Cointegration |

### Optimization

| Author | Title | Year | Topic |
|--------|-------|------|-------|
| Boyd, S. & Vandenberghe, L. | Convex Optimization | 2004 | Theory |
| Nocedal, J. & Wright, S. | Numerical Optimization | 2006 | Methods |
| Bertsekas, D. | Dynamic Programming and Optimal Control | 2012 | Control |
| Markowitz, H. | Portfolio Selection | 1952 | Foundation |

### Statistical Learning

| Author | Title | Year | Topic |
|--------|-------|------|-------|
| Hastie, T. et al. | Elements of Statistical Learning | 2009 | ML theory |
| De Prado, M.L. | Advances in Financial Machine Learning | 2018 | Finance ML |
| Murphy, K. | Machine Learning: A Probabilistic Perspective | 2012 | Bayesian |
| Bishop, C. | Pattern Recognition and Machine Learning | 2006 | Foundation |

### Numerical Methods

| Author | Title | Year | Topic |
|--------|-------|------|-------|
| Glasserman, P. | Monte Carlo Methods in Financial Engineering | 2003 | MC methods |
| Kloeden, P. & Platen, E. | Numerical Solution of SDEs | 1992 | SDE numerics |
| Wilmott, P. | Paul Wilmott on Quantitative Finance | 2006 | Comprehensive |

---

## 8. Implementation Notes

### Best Practices

1. **Numerical stability**: Use log prices for long simulations
2. **Random seeds**: Set seeds for reproducibility
3. **Validation**: Compare MC results to closed-form solutions when available
4. **Convergence**: Check convergence with increasing samples/steps
5. **Units**: Keep time units consistent (annual vs daily)

### Common Pitfalls

```python
COMMON_ERRORS = {
    'look_ahead_bias': 'Using future data in estimation',
    'survivorship_bias': 'Only including surviving assets',
    'overfitting': 'Too many parameters for data',
    'non_stationarity': 'Applying stationary methods to non-stationary data',
    'correlation_vs_causation': 'Assuming correlation implies predictability',
    'transaction_costs': 'Ignoring trading costs in optimization'
}
```

---

## Key Takeaways

1. **Probability foundation**: Understand distributions, moments, and tail risk
2. **Stochastic processes**: GBM is a starting point, not reality
3. **Time series**: Test for stationarity before modeling
4. **Optimization**: Consider estimation error and transaction costs
5. **ML with caution**: Beware of overfitting and non-stationarity
6. **Simulation validation**: Always validate numerical methods
7. **Fat tails matter**: Normal distribution underestimates extreme events
