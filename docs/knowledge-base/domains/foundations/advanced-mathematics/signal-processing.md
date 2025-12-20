# Signal Processing for Financial Time Series

Signal processing techniques extract meaningful patterns from noisy financial data through filtering, decomposition, and adaptive estimation methods.

---

## Overview

Financial time series contain noise, trends, cycles, and structural breaks. Signal processing enables:

1. **Wavelet Analysis**: Multi-scale decomposition
2. **Empirical Mode Decomposition**: Adaptive decomposition
3. **Kalman Filtering**: Optimal state estimation
4. **Singular Spectrum Analysis**: Trend extraction
5. **Adaptive Filtering**: Real-time signal tracking

---

## 1. Wavelet Analysis

### 1.1 Discrete Wavelet Transform

DWT decomposes signals into approximation (low-frequency) and detail (high-frequency) coefficients:

$$x(t) = \sum_k a_{J,k} \phi_{J,k}(t) + \sum_{j=1}^{J} \sum_k d_{j,k} \psi_{j,k}(t)$$

where:
- $\phi$ is the scaling function (father wavelet)
- $\psi$ is the wavelet function (mother wavelet)
- $a_{J,k}$ are approximation coefficients
- $d_{j,k}$ are detail coefficients at scale $j$

### 1.2 Python Implementation

```python
import numpy as np
import pandas as pd
import pywt
from typing import Tuple, List, Dict


class WaveletAnalysis:
    """
    Wavelet analysis for financial time series.
    """

    def __init__(self, wavelet: str = 'db4', max_level: int = None):
        """
        Initialize wavelet analyzer.

        Args:
            wavelet: Wavelet name ('db4', 'haar', 'sym5', etc.)
            max_level: Maximum decomposition level
        """
        self.wavelet = wavelet
        self.max_level = max_level

    def decompose(self, signal: np.ndarray) -> Dict:
        """
        Perform wavelet decomposition.

        Returns:
            Dictionary with 'approx' and 'details' at each level
        """
        max_level = self.max_level or pywt.dwt_max_level(len(signal), self.wavelet)
        coeffs = pywt.wavedec(signal, self.wavelet, level=max_level)

        return {
            'approximation': coeffs[0],
            'details': coeffs[1:],
            'levels': max_level
        }

    def denoise(
        self,
        signal: np.ndarray,
        threshold_method: str = 'soft',
        threshold_type: str = 'universal'
    ) -> np.ndarray:
        """
        Wavelet denoising using thresholding.

        Args:
            signal: Input signal
            threshold_method: 'soft' or 'hard'
            threshold_type: 'universal' or 'sure'

        Returns:
            Denoised signal
        """
        coeffs = pywt.wavedec(signal, self.wavelet, level=self.max_level)

        # Estimate noise from finest detail coefficients
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745

        # Universal threshold
        n = len(signal)
        threshold = sigma * np.sqrt(2 * np.log(n))

        # Apply thresholding to detail coefficients
        denoised_coeffs = [coeffs[0]]  # Keep approximation
        for detail in coeffs[1:]:
            if threshold_method == 'soft':
                denoised = pywt.threshold(detail, threshold, mode='soft')
            else:
                denoised = pywt.threshold(detail, threshold, mode='hard')
            denoised_coeffs.append(denoised)

        return pywt.waverec(denoised_coeffs, self.wavelet)[:len(signal)]

    def multi_resolution_analysis(self, prices: pd.Series) -> pd.DataFrame:
        """
        Multi-resolution analysis of price series.

        Decomposes into trend + cycles at different frequencies.
        """
        log_prices = np.log(prices.values)
        decomp = self.decompose(log_prices)

        result = pd.DataFrame(index=prices.index)
        result['price'] = prices

        # Reconstruct trend (approximation only)
        trend_coeffs = [decomp['approximation']] + [np.zeros_like(d) for d in decomp['details']]
        trend = pywt.waverec(trend_coeffs, self.wavelet)[:len(prices)]
        result['trend'] = np.exp(trend)

        # Reconstruct each detail level
        for i, detail in enumerate(decomp['details']):
            detail_coeffs = [np.zeros_like(decomp['approximation'])]
            for j, d in enumerate(decomp['details']):
                if j == i:
                    detail_coeffs.append(d)
                else:
                    detail_coeffs.append(np.zeros_like(d))
            component = pywt.waverec(detail_coeffs, self.wavelet)[:len(prices)]
            result[f'detail_{i+1}'] = component

        return result

    def detect_regime_changes(
        self,
        signal: np.ndarray,
        level: int = 3
    ) -> np.ndarray:
        """
        Detect regime changes using wavelet energy.

        Returns:
            Array of regime change indicators
        """
        coeffs = pywt.wavedec(signal, self.wavelet, level=level)

        # Compute local wavelet energy
        energy = np.zeros(len(signal))

        for detail in coeffs[1:]:
            # Upsample detail to signal length
            upsampled = np.repeat(detail ** 2, len(signal) // len(detail) + 1)[:len(signal)]
            energy += upsampled

        # Smooth and detect peaks
        window = min(50, len(signal) // 10)
        smoothed = pd.Series(energy).rolling(window, center=True).mean().values

        # Z-score for regime detection
        z_score = (smoothed - np.nanmean(smoothed)) / np.nanstd(smoothed)

        return z_score


# Example
if __name__ == "__main__":
    np.random.seed(42)
    t = np.linspace(0, 10, 1000)
    signal = np.sin(2 * np.pi * 0.5 * t) + 0.5 * np.sin(2 * np.pi * 2 * t) + 0.3 * np.random.randn(len(t))

    wa = WaveletAnalysis(wavelet='db4')

    # Denoise
    denoised = wa.denoise(signal)
    print(f"Noise reduction: {np.std(signal - denoised):.4f}")

    # Regime detection
    regime_scores = wa.detect_regime_changes(signal)
    print(f"Max regime score: {np.max(np.abs(regime_scores)):.2f}")
```

---

## 2. Empirical Mode Decomposition (EMD)

### 2.1 Theory

EMD decomposes signals into Intrinsic Mode Functions (IMFs) adaptively:

1. Identify local extrema
2. Interpolate upper/lower envelopes
3. Compute mean envelope
4. Subtract mean to get IMF candidate
5. Iterate until IMF criteria satisfied
6. Subtract IMF from signal, repeat

### 2.2 Python Implementation

```python
try:
    from PyEMD import EMD, EEMD
    HAS_EMD = True
except ImportError:
    HAS_EMD = False


class EMDAnalysis:
    """
    Empirical Mode Decomposition for adaptive signal analysis.
    """

    def __init__(self):
        if not HAS_EMD:
            raise ImportError("PyEMD required: pip install EMD-signal")
        self.emd = EMD()
        self.eemd = EEMD()

    def decompose(self, signal: np.ndarray) -> np.ndarray:
        """
        Standard EMD decomposition.

        Returns:
            Array of IMFs (n_imfs x signal_length)
        """
        imfs = self.emd(signal)
        return imfs

    def ensemble_decompose(
        self,
        signal: np.ndarray,
        n_trials: int = 100,
        noise_width: float = 0.2
    ) -> np.ndarray:
        """
        Ensemble EMD for more robust decomposition.
        """
        self.eemd.noise_seed(42)
        self.eemd.trials = n_trials
        imfs = self.eemd(signal, max_imf=10)
        return imfs

    def hilbert_huang_transform(self, signal: np.ndarray) -> Dict:
        """
        Hilbert-Huang Transform for time-frequency analysis.

        Returns instantaneous frequency and amplitude for each IMF.
        """
        from scipy.signal import hilbert

        imfs = self.decompose(signal)

        result = {'imfs': imfs, 'inst_freq': [], 'inst_amp': []}

        for imf in imfs:
            analytic = hilbert(imf)
            inst_amp = np.abs(analytic)
            inst_phase = np.unwrap(np.angle(analytic))
            inst_freq = np.diff(inst_phase) / (2 * np.pi)

            result['inst_amp'].append(inst_amp)
            result['inst_freq'].append(inst_freq)

        return result

    def extract_trend(self, signal: np.ndarray) -> np.ndarray:
        """
        Extract trend as residual after IMF extraction.
        """
        imfs = self.decompose(signal)
        # Last IMF is typically the trend (residual)
        return imfs[-1]


# Example (if PyEMD available)
if HAS_EMD:
    emd_analyzer = EMDAnalysis()
    signal = np.sin(2 * np.pi * 0.1 * np.arange(500)) + 0.5 * np.sin(2 * np.pi * 0.5 * np.arange(500))
    imfs = emd_analyzer.decompose(signal)
    print(f"Number of IMFs: {len(imfs)}")
```

---

## 3. Kalman Filtering

### 3.1 Theory

State-space model:
$$x_t = F x_{t-1} + w_t, \quad w_t \sim N(0, Q)$$
$$y_t = H x_t + v_t, \quad v_t \sim N(0, R)$$

**Predict**:
$$\hat{x}_{t|t-1} = F \hat{x}_{t-1|t-1}$$
$$P_{t|t-1} = F P_{t-1|t-1} F^T + Q$$

**Update**:
$$K_t = P_{t|t-1} H^T (H P_{t|t-1} H^T + R)^{-1}$$
$$\hat{x}_{t|t} = \hat{x}_{t|t-1} + K_t (y_t - H \hat{x}_{t|t-1})$$
$$P_{t|t} = (I - K_t H) P_{t|t-1}$$

### 3.2 Python Implementation

```python
from filterpy.kalman import KalmanFilter


class FinancialKalmanFilter:
    """
    Kalman filter applications for financial data.
    """

    def __init__(self, dim_state: int = 2, dim_obs: int = 1):
        """
        Initialize Kalman filter.

        Args:
            dim_state: State dimension
            dim_obs: Observation dimension
        """
        self.kf = KalmanFilter(dim_x=dim_state, dim_z=dim_obs)
        self.dim_state = dim_state
        self.dim_obs = dim_obs

    def setup_price_tracking(
        self,
        process_variance: float = 1e-5,
        measurement_variance: float = 1e-3
    ):
        """
        Configure for price/level tracking with velocity.

        State: [price, velocity]
        """
        self.kf.F = np.array([[1, 1], [0, 1]])  # State transition
        self.kf.H = np.array([[1, 0]])           # Observation
        self.kf.Q = np.eye(2) * process_variance # Process noise
        self.kf.R = np.array([[measurement_variance]])  # Measurement noise
        self.kf.P = np.eye(2) * 1.0              # Initial covariance
        self.kf.x = np.array([[0], [0]])         # Initial state

    def setup_spread_tracking(
        self,
        mean_reversion_speed: float = 0.1,
        process_variance: float = 1e-4,
        measurement_variance: float = 1e-3
    ):
        """
        Configure for mean-reverting spread tracking.

        State: [spread]
        """
        self.kf = KalmanFilter(dim_x=1, dim_z=1)
        self.kf.F = np.array([[1 - mean_reversion_speed]])
        self.kf.H = np.array([[1]])
        self.kf.Q = np.array([[process_variance]])
        self.kf.R = np.array([[measurement_variance]])
        self.kf.P = np.array([[1.0]])
        self.kf.x = np.array([[0]])

    def filter(self, observations: np.ndarray) -> pd.DataFrame:
        """
        Run Kalman filter on observations.

        Returns:
            DataFrame with filtered states and predictions
        """
        n = len(observations)
        states = np.zeros((n, self.kf.dim_x))
        predictions = np.zeros(n)
        variances = np.zeros(n)

        for i, obs in enumerate(observations):
            # Predict
            self.kf.predict()
            predictions[i] = self.kf.x[0, 0]

            # Update
            self.kf.update(obs)
            states[i] = self.kf.x.flatten()
            variances[i] = self.kf.P[0, 0]

        result = pd.DataFrame({
            'observation': observations,
            'filtered_state': states[:, 0],
            'prediction': predictions,
            'variance': variances
        })

        if self.kf.dim_x > 1:
            result['velocity'] = states[:, 1]

        return result

    def online_regression(
        self,
        x: np.ndarray,
        y: np.ndarray,
        process_noise: float = 1e-5,
        measurement_noise: float = 1e-3
    ) -> pd.DataFrame:
        """
        Online linear regression via Kalman filter.

        Tracks time-varying regression coefficients.
        """
        # State: [intercept, slope]
        kf = KalmanFilter(dim_x=2, dim_z=1)
        kf.F = np.eye(2)  # Random walk
        kf.Q = np.eye(2) * process_noise
        kf.R = np.array([[measurement_noise]])
        kf.P = np.eye(2) * 1.0
        kf.x = np.array([[0], [0]])

        n = len(y)
        intercepts = np.zeros(n)
        slopes = np.zeros(n)
        residuals = np.zeros(n)

        for i in range(n):
            kf.H = np.array([[1, x[i]]])

            kf.predict()
            pred = (kf.H @ kf.x)[0, 0]

            kf.update(y[i])

            intercepts[i] = kf.x[0, 0]
            slopes[i] = kf.x[1, 0]
            residuals[i] = y[i] - pred

        return pd.DataFrame({
            'x': x, 'y': y,
            'intercept': intercepts,
            'slope': slopes,
            'residual': residuals
        })


# Example
if __name__ == "__main__":
    np.random.seed(42)

    # Price tracking
    true_price = 100 + np.cumsum(np.random.randn(200) * 0.5)
    noisy_price = true_price + np.random.randn(200) * 2

    kf = FinancialKalmanFilter()
    kf.setup_price_tracking()
    result = kf.filter(noisy_price)

    print("Kalman Filter Price Tracking:")
    print(f"  Noise std: {np.std(noisy_price - true_price):.2f}")
    print(f"  Filter error std: {np.std(result['filtered_state'] - true_price):.2f}")
```

---

## 4. Singular Spectrum Analysis (SSA)

### 4.1 Theory

SSA decomposes time series via trajectory matrix SVD:

1. Form trajectory matrix from lagged copies
2. SVD: $X = U \Sigma V^T$
3. Group components (trend, seasonality, noise)
4. Reconstruct via diagonal averaging

### 4.2 Python Implementation

```python
class SingularSpectrumAnalysis:
    """
    SSA for trend and cycle extraction.
    """

    def __init__(self, window_length: int = None):
        """
        Initialize SSA.

        Args:
            window_length: Embedding dimension (default: N//2)
        """
        self.L = window_length

    def decompose(self, signal: np.ndarray, n_components: int = None) -> Dict:
        """
        SSA decomposition.

        Returns:
            Dictionary with components and singular values
        """
        N = len(signal)
        L = self.L or N // 2
        K = N - L + 1

        # Build trajectory matrix
        X = np.array([signal[i:i+L] for i in range(K)]).T

        # SVD
        U, s, Vt = np.linalg.svd(X, full_matrices=False)

        if n_components is None:
            n_components = len(s)

        # Reconstruct each component
        components = []
        for i in range(n_components):
            Xi = s[i] * np.outer(U[:, i], Vt[i, :])
            component = self._diagonal_averaging(Xi, N)
            components.append(component)

        return {
            'components': np.array(components),
            'singular_values': s,
            'explained_variance': s ** 2 / np.sum(s ** 2)
        }

    def _diagonal_averaging(self, X: np.ndarray, N: int) -> np.ndarray:
        """Convert trajectory matrix back to time series."""
        L, K = X.shape
        result = np.zeros(N)
        counts = np.zeros(N)

        for i in range(L):
            for j in range(K):
                result[i + j] += X[i, j]
                counts[i + j] += 1

        return result / counts

    def extract_trend(self, signal: np.ndarray, n_trend: int = 1) -> np.ndarray:
        """Extract trend using first n_trend components."""
        decomp = self.decompose(signal, n_components=n_trend + 5)
        return np.sum(decomp['components'][:n_trend], axis=0)


# Example
if __name__ == "__main__":
    np.random.seed(42)
    t = np.arange(200)
    trend = 0.05 * t
    seasonal = 2 * np.sin(2 * np.pi * t / 20)
    noise = np.random.randn(200) * 0.5
    signal = trend + seasonal + noise

    ssa = SingularSpectrumAnalysis(window_length=40)
    decomp = ssa.decompose(signal, n_components=10)

    print(f"SSA Decomposition:")
    print(f"  Top 5 explained variance: {decomp['explained_variance'][:5]}")

    extracted_trend = ssa.extract_trend(signal, n_trend=2)
    print(f"  Trend extraction MSE: {np.mean((extracted_trend - trend - seasonal)**2):.4f}")
```

---

## 5. Adaptive Filtering

### 5.1 Least Mean Squares (LMS)

Online adaptive filter:
$$w_{n+1} = w_n + \mu e_n x_n$$

where $e_n = d_n - w_n^T x_n$ is the error.

### 5.2 Python Implementation

```python
class AdaptiveFilter:
    """
    Adaptive filtering for real-time signal processing.
    """

    def __init__(self, filter_length: int = 10, step_size: float = 0.01):
        """
        Initialize LMS adaptive filter.

        Args:
            filter_length: Number of filter taps
            step_size: Learning rate (mu)
        """
        self.M = filter_length
        self.mu = step_size
        self.weights = np.zeros(filter_length)

    def reset(self):
        """Reset filter weights."""
        self.weights = np.zeros(self.M)

    def filter(self, signal: np.ndarray, reference: np.ndarray = None) -> Dict:
        """
        Apply adaptive filter.

        Args:
            signal: Input signal
            reference: Desired signal (for system ID) or None for prediction

        Returns:
            Dictionary with output and error signals
        """
        N = len(signal)
        output = np.zeros(N)
        error = np.zeros(N)

        for n in range(self.M, N):
            x = signal[n-self.M:n][::-1]  # Reversed for convolution
            y = np.dot(self.weights, x)
            output[n] = y

            if reference is not None:
                e = reference[n] - y
            else:
                e = signal[n] - y  # One-step prediction

            error[n] = e

            # LMS update
            self.weights += self.mu * e * x

        return {
            'output': output,
            'error': error,
            'weights': self.weights.copy()
        }

    def online_predict(self, new_sample: float, history: np.ndarray) -> float:
        """
        Make one-step ahead prediction and update.
        """
        if len(history) < self.M:
            return history[-1] if len(history) > 0 else 0

        x = history[-self.M:][::-1]
        prediction = np.dot(self.weights, x)

        # Update with actual
        error = new_sample - prediction
        self.weights += self.mu * error * x

        return prediction


# Example
if __name__ == "__main__":
    np.random.seed(42)

    # Signal with changing dynamics
    t = np.arange(500)
    signal = np.sin(0.1 * t) + 0.3 * np.sin(0.05 * t) + 0.2 * np.random.randn(500)

    af = AdaptiveFilter(filter_length=20, step_size=0.01)
    result = af.filter(signal)

    mse = np.mean(result['error'][100:] ** 2)
    print(f"Adaptive Filter MSE (after warmup): {mse:.6f}")
```

---

## 6. Integration Example

```python
class SignalProcessingPipeline:
    """
    Combined signal processing pipeline for trading signals.
    """

    def __init__(self):
        self.wavelet = WaveletAnalysis()
        self.kalman = FinancialKalmanFilter()
        self.ssa = SingularSpectrumAnalysis()

    def process_prices(self, prices: pd.Series) -> pd.DataFrame:
        """
        Full signal processing pipeline on prices.
        """
        result = pd.DataFrame(index=prices.index)
        result['price'] = prices

        # Wavelet denoising
        result['denoised'] = self.wavelet.denoise(prices.values)

        # Kalman filtered
        self.kalman.setup_price_tracking()
        kf_result = self.kalman.filter(prices.values)
        result['kalman_filtered'] = kf_result['filtered_state']
        result['kalman_velocity'] = kf_result.get('velocity', 0)

        # SSA trend
        result['ssa_trend'] = self.ssa.extract_trend(prices.values, n_trend=2)

        # Regime detection
        result['regime_score'] = self.wavelet.detect_regime_changes(prices.values)

        return result
```

---

## 7. Academic References

1. **Mallat, S. (2008)**. *A Wavelet Tour of Signal Processing*. Academic Press.
2. **Huang, N. E., et al. (1998)**. "The Empirical Mode Decomposition." *Proc. Royal Society A*.
3. **Durbin, J., & Koopman, S. J. (2012)**. *Time Series Analysis by State Space Methods*. Oxford.
4. **Golyandina, N., et al. (2001)**. *Analysis of Time Series Structure: SSA and Related Techniques*. Chapman & Hall.
5. **Haykin, S. (2002)**. *Adaptive Filter Theory*. Prentice Hall.

---

## Document Metadata

```yaml
version: "1.0.0"
created: "2025-12-12"
status: "published"
tags: ["signal-processing", "wavelet", "kalman", "emd", "ssa", "adaptive-filter"]
code_lines: 500
```

---

**END OF DOCUMENT**
