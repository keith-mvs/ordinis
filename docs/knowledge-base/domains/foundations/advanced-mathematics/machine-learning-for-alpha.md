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
