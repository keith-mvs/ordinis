"""
Populate the knowledge base with sample financial algorithms.
"""

from pathlib import Path

KB_ROOT = Path("docs/knowledge-base/code")

FILES = {
    "options/black_scholes.py": """
import math
from scipy.stats import norm

def black_scholes_call(S, K, T, r, sigma):
    '''
    Calculate Black-Scholes price for a call option.

    Args:
        S: Current stock price
        K: Strike price
        T: Time to maturity (years)
        r: Risk-free interest rate
        sigma: Volatility
    '''
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)

def black_scholes_put(S, K, T, r, sigma):
    '''
    Calculate Black-Scholes price for a put option.
    '''
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
""",
    "risk/var_models.py": """
import numpy as np

def calculate_historical_var(returns, confidence_level=0.95):
    '''
    Calculate Value at Risk (VaR) using historical simulation.

    Args:
        returns: Array of historical returns
        confidence_level: Confidence level (default 0.95)
    '''
    if not len(returns):
        return 0.0
    return np.percentile(returns, 100 * (1 - confidence_level))

def calculate_parametric_var(returns, confidence_level=0.95):
    '''
    Calculate VaR using parametric (normal) distribution assumption.
    '''
    if not len(returns):
        return 0.0
    mu = np.mean(returns)
    sigma = np.std(returns)
    from scipy.stats import norm
    return norm.ppf(1 - confidence_level, mu, sigma)
""",
    "fundamental/ratios.py": """
def calculate_pe_ratio(price, eps):
    '''
    Calculate Price-to-Earnings ratio.
    '''
    if eps == 0:
        return float('inf')
    return price / eps

def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    '''
    Calculate Sharpe Ratio.
    '''
    import numpy as np
    excess_returns = np.array(returns) - risk_free_rate
    if len(excess_returns) == 0 or np.std(excess_returns) == 0:
        return 0.0
    return np.mean(excess_returns) / np.std(excess_returns)
""",
}


def main():
    for rel_path, content in FILES.items():
        path = KB_ROOT / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(content.strip())
        print(f"Created {path}")


if __name__ == "__main__":
    main()
