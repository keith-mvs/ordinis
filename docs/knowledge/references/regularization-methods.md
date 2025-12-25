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
