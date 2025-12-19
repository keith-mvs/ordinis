from typing import Any

import numpy as np


class PortfolioOptimizer:
    """Thin wrapper that tries cuOpt first, then falls back to SciPy."""

    def __init__(self, method: str = "mean_variance"):
        if method not in {"mean_variance", "mean_cvar"}:
            raise ValueError(f"Unsupported method {method}")
        self.method = method

    def _try_cuopt(self, returns: np.ndarray, constraints: dict) -> dict[str, Any]:
        """Attempt GPU optimisation via cuOpt."""
        try:
            from cuopt import QP  # cuOpt QP solver (installed with `pip install cuopt`)
            import cupy as cp

            # Convert to CuPy arrays (GPU memory)
            R = cp.asarray(returns)  # T×N
            mu = cp.mean(R, axis=0)  # expected returns (N,)
            Sigma = cp.cov(R, rowvar=False)  # covariance matrix (N×N)

            # Build QP: minimize ½ wᵀ Σ w  –  μᵀ w   (mean‑variance)
            Q = Sigma
            c = -mu

            # Simple box constraints: 0 ≤ w ≤ 1 (can be overridden by `constraints`)
            lb = cp.zeros(Q.shape[0])
            ub = cp.ones(Q.shape[0])

            # Apply user‑provided bounds if any
            if "bounds" in constraints:
                b = constraints["bounds"]
                lb = cp.asarray(b.get("lower", lb))
                ub = cp.asarray(b.get("upper", ub))

            # Equality: sum(w) = 1
            A = cp.ones((1, Q.shape[0]))
            b_eq = cp.asarray([1.0])

            qp = QP(Q, c, A=A, b=b_eq, lb=lb, ub=ub)
            w_opt = qp.solve()
            w_opt = cp.asnumpy(w_opt)  # back to host

            # Compute risk metrics on CPU for readability
            var = np.var(returns @ w_opt)
            ret = np.mean(returns @ w_opt)

            return {
                "weights": w_opt.tolist(),
                "expected_return": float(ret),
                "variance": float(var),
                "method": self.method,
                "solver": "cuOpt (GPU)",
            }
        except Exception as exc:  # cuOpt not available or GPU error
            # print(f"[PortfolioOptimizer] cuOpt fallback triggered: {exc}")
            return None

    def _fallback_scipy(self, returns: np.ndarray, constraints: dict) -> dict[str, Any]:
        """CPU fallback using SciPy's `minimize`."""
        from scipy.optimize import minimize

        mu = np.mean(returns, axis=0)
        Sigma = np.cov(returns, rowvar=False)

        N = returns.shape[1]

        # Objective: ½ wᵀ Σ w – μᵀ w   (mean‑variance)
        def obj(w):
            return 0.5 * w @ Sigma @ w - mu @ w

        # Equality constraint: sum(w) = 1
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        # Bounds (default 0‑1)
        bounds = [(0.0, 1.0)] * N
        if "bounds" in constraints:
            b = constraints["bounds"]
            bounds = [(b.get("lower", 0.0)[i], b.get("upper", 1.0)[i]) for i in range(N)]

        # Optional sector caps, turnover, etc. can be added as additional constraints
        # (omitted for brevity – plug in `constraints["linear"]` if needed).

        init = np.full(N, 1.0 / N)

        res = minimize(obj, init, method="SLSQP", bounds=bounds, constraints=cons)

        if not res.success:
            raise RuntimeError(f"SciPy optimisation failed: {res.message}")

        w_opt = res.x
        var = np.var(returns @ w_opt)
        ret = np.mean(returns @ w_opt)

        return {
            "weights": w_opt.tolist(),
            "expected_return": float(ret),
            "variance": float(var),
            "method": self.method,
            "solver": "SciPy (CPU fallback)",
        }

    def optimize(self, returns_df, constraints: dict) -> dict[str, Any]:
        """Public entry point – returns a JSON‑serialisable dict."""
        returns = returns_df.values.astype(np.float64)

        # Try GPU first
        result = self._try_cuopt(returns, constraints)
        if result is not None:
            return result

        # Fallback to CPU
        return self._fallback_scipy(returns, constraints)
