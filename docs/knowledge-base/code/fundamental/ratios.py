def calculate_pe_ratio(price, eps):
    """
    Calculate Price-to-Earnings ratio.
    """
    if eps == 0:
        return float("inf")
    return price / eps


def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """
    Calculate Sharpe Ratio.
    """
    import numpy as np

    excess_returns = np.array(returns) - risk_free_rate
    if len(excess_returns) == 0 or np.std(excess_returns) == 0:
        return 0.0
    return np.mean(excess_returns) / np.std(excess_returns)
