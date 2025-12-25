
import pytest
import numpy as np
import pandas as pd
from ordinis.engines.portfolioopt.optimizer import PortfolioOptimizer

def test_portfolio_optimizer_cuopt_path(monkeypatch):
    """Test that PortfolioOptimizer attempts cuOpt and falls back gracefully."""
    # Create sample data
    assets = ["AAPL", "MSFT", "GOOGL"]
    n_assets = len(assets)
    returns_data = np.random.randn(100, n_assets) * 0.01 + 0.0005
    returns_df = pd.DataFrame(returns_data, columns=assets)
    
    optimizer = PortfolioOptimizer(method="mean_variance")
    
    # Run optimization
    # This will try cuOpt first. If it works, solver will be "cuOpt (GPU)"
    # If it fails (e.g. no GPU in CI or import error), it will be "SciPy (CPU fallback)"
    result = optimizer.optimize(returns_df, constraints={})
    
    assert "weights" in result
    assert len(result["weights"]) == n_assets
    assert abs(sum(result["weights"]) - 1.0) < 1e-5
    assert "solver" in result
    print(f"\nUsed solver: {result['solver']}")

@pytest.mark.gpu
def test_portfolio_optimizer_gpu_verification():
    """Verify GPU solver specifically."""
    import importlib.util
    if importlib.util.find_spec("cuopt") is None:
        pytest.skip("cuopt not installed")
        
    assets = ["AAPL", "MSFT", "GOOGL"]
    returns_data = np.random.randn(100, len(assets)) * 0.01 + 0.0005
    returns_df = pd.DataFrame(returns_data, columns=assets)
    
    optimizer = PortfolioOptimizer(method="mean_variance")
    result = optimizer._try_cuopt(returns_df.values, constraints={})
    
    if result is None:
        pytest.fail("cuOpt optimization failed or was not available despite being installed")
        
    assert result["solver"] == "cuOpt (GPU)"
    assert abs(sum(result["weights"]) - 1.0) < 1e-5
