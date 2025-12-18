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
