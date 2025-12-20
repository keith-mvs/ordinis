import numpy as np


def calculate_historical_var(returns, confidence_level=0.95):
    """
    Calculate Value at Risk (VaR) using historical simulation.

    Args:
        returns: Array of historical returns
        confidence_level: Confidence level (default 0.95)
    """
    if not len(returns):
        return 0.0
    return np.percentile(returns, 100 * (1 - confidence_level))


def calculate_parametric_var(returns, confidence_level=0.95):
    """
    Calculate VaR using parametric (normal) distribution assumption.
    """
    if not len(returns):
        return 0.0
    mu = np.mean(returns)
    sigma = np.std(returns)
    from scipy.stats import norm

    return norm.ppf(1 - confidence_level, mu, sigma)
