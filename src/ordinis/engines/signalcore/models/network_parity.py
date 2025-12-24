"""
Network Risk Parity Strategy.

Uses correlation network centrality to weight positions.
Central assets get reduced weight (higher systemic risk).
Peripheral assets get higher weight (more diversification benefit).

Theory:
- Asset correlations form a network graph
- Central nodes = systemically important = more exposed to market moves
- Peripheral nodes = more idiosyncratic = better diversification
- Weight inversely by centrality for risk parity
- Monitor network stability for regime changes
"""

from dataclasses import dataclass
from datetime import datetime
import logging

import numpy as np
import pandas as pd

from ordinis.engines.signalcore.core.model import Model, ModelConfig
from ordinis.engines.signalcore.core.signal import Direction, Signal, SignalType

logger = logging.getLogger(__name__)


@dataclass
class NetworkConfig:
    """Network Risk Parity configuration."""

    # Network parameters
    corr_lookback: int = 60  # Lookback for correlation calculation
    corr_threshold: float = 0.3  # Minimum correlation for edge
    recalc_frequency: int = 5  # Recalculate network every N bars

    # Centrality parameters
    centrality_method: str = "eigenvector"  # eigenvector, degree, betweenness
    weight_decay: float = 0.5  # Inverse centrality weight power
    min_weight: float = 0.02  # Minimum position weight
    max_weight: float = 0.30  # Maximum position weight

    # Signal parameters (uses underlying alpha source)
    momentum_lookback: int = 20  # Momentum calculation period
    vol_target: float = 0.15  # Annualized volatility target

    # Risk parameters
    max_concentration: float = 0.4  # Max weight in any cluster
    network_stability_threshold: float = 0.3  # Threshold for network change


@dataclass
class NetworkStats:
    """Network analysis statistics."""

    n_nodes: int
    n_edges: int
    density: float
    avg_clustering: float
    avg_centrality: float
    central_assets: list[str]
    peripheral_assets: list[str]


class CorrelationNetwork:
    """
    Correlation network analyzer.

    Builds network from correlation matrix and calculates centrality.
    """

    def __init__(
        self,
        threshold: float = 0.3,
        centrality_method: str = "eigenvector",
    ):
        """
        Initialize network analyzer.

        Args:
            threshold: Minimum correlation for edge
            centrality_method: Method for centrality calculation
        """
        self.threshold = threshold
        self.centrality_method = centrality_method

        self._correlation: pd.DataFrame | None = None
        self._adjacency: np.ndarray | None = None
        self._centrality: dict[str, float] | None = None
        self._assets: list[str] = []

    def fit(self, returns: pd.DataFrame) -> "CorrelationNetwork":
        """
        Build network from returns data.

        Args:
            returns: DataFrame with columns as assets, returns as values

        Returns:
            Self for chaining
        """
        self._assets = returns.columns.tolist()
        n = len(self._assets)

        # Calculate correlation matrix
        self._correlation = returns.corr()

        # Build adjacency matrix (threshold correlations)
        corr_values = self._correlation.values.copy()
        np.fill_diagonal(corr_values, 0)  # No self-loops

        self._adjacency = (np.abs(corr_values) >= self.threshold).astype(float)

        # Weight edges by correlation strength
        self._weighted_adj = np.abs(corr_values) * self._adjacency

        # Calculate centrality
        self._centrality = self._calculate_centrality()

        return self

    def _calculate_centrality(self) -> dict[str, float]:
        """Calculate node centrality scores."""
        n = len(self._assets)

        if self.centrality_method == "degree":
            # Degree centrality (normalized)
            degrees = self._adjacency.sum(axis=1)
            centrality = degrees / (n - 1)

        elif self.centrality_method == "eigenvector":
            # Eigenvector centrality
            centrality = self._eigenvector_centrality()

        elif self.centrality_method == "betweenness":
            # Betweenness centrality (approximation)
            centrality = self._betweenness_centrality()

        else:
            # Default to degree
            degrees = self._adjacency.sum(axis=1)
            centrality = degrees / (n - 1)

        return {asset: cent for asset, cent in zip(self._assets, centrality, strict=False)}

    def _eigenvector_centrality(self, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
        """Calculate eigenvector centrality using power iteration."""
        n = len(self._assets)

        if n == 0:
            return np.array([])

        # Use weighted adjacency
        A = self._weighted_adj

        # Power iteration
        x = np.ones(n) / n

        for _ in range(max_iter):
            x_new = A @ x
            norm = np.linalg.norm(x_new)
            if norm == 0:
                return np.ones(n) / n
            x_new = x_new / norm

            if np.linalg.norm(x_new - x) < tol:
                break
            x = x_new

        # Normalize to [0, 1]
        if x.max() > 0:
            x = x / x.max()

        return x

    def _betweenness_centrality(self) -> np.ndarray:
        """Approximate betweenness centrality."""
        n = len(self._assets)

        if n <= 2:
            return np.ones(n) / n

        # Simplified approximation using correlation structure
        # True betweenness would require shortest path calculation
        centrality = np.zeros(n)

        for i in range(n):
            for j in range(i + 1, n):
                if self._adjacency[i, j] > 0:
                    continue  # Direct connection

                # Check for paths through other nodes
                for k in range(n):
                    if k != i and k != j:
                        if self._adjacency[i, k] > 0 and self._adjacency[k, j] > 0:
                            centrality[k] += 1

        # Normalize
        max_cent = centrality.max()
        if max_cent > 0:
            centrality = centrality / max_cent

        return centrality

    def get_centrality(self) -> dict[str, float]:
        """Get centrality scores for all assets."""
        return self._centrality.copy() if self._centrality else {}

    def get_weights(
        self,
        decay: float = 0.5,
        min_weight: float = 0.02,
        max_weight: float = 0.30,
    ) -> dict[str, float]:
        """
        Calculate risk parity weights (inverse centrality).

        Args:
            decay: Power for inverse weighting
            min_weight: Minimum weight per asset
            max_weight: Maximum weight per asset

        Returns:
            Dictionary of asset to weight
        """
        if not self._centrality:
            return {}

        # Inverse centrality weighting
        # Higher centrality = lower weight
        inv_centrality = {}
        for asset, cent in self._centrality.items():
            # Add small constant to avoid division by zero
            inv_centrality[asset] = 1.0 / (cent + 0.1) ** decay

        # Normalize weights
        total = sum(inv_centrality.values())
        weights = {k: v / total for k, v in inv_centrality.items()}

        # Apply constraints
        for asset in weights:
            weights[asset] = max(min_weight, min(max_weight, weights[asset]))

        # Renormalize after constraints
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}

        return weights

    def get_network_stats(self) -> NetworkStats:
        """Get network summary statistics."""
        n = len(self._assets)

        if n == 0:
            return NetworkStats(0, 0, 0, 0, 0, [], [])

        n_edges = int(self._adjacency.sum() / 2)  # Undirected
        max_edges = n * (n - 1) / 2
        density = n_edges / max_edges if max_edges > 0 else 0

        # Clustering coefficient
        clustering = []
        for i in range(n):
            neighbors = np.where(self._adjacency[i] > 0)[0]
            k = len(neighbors)
            if k < 2:
                clustering.append(0)
                continue

            # Count edges between neighbors
            neighbor_edges = 0
            for j in range(len(neighbors)):
                for l in range(j + 1, len(neighbors)):
                    if self._adjacency[neighbors[j], neighbors[l]] > 0:
                        neighbor_edges += 1

            max_neighbor_edges = k * (k - 1) / 2
            clustering.append(neighbor_edges / max_neighbor_edges if max_neighbor_edges > 0 else 0)

        avg_clustering = np.mean(clustering)
        avg_centrality = np.mean(list(self._centrality.values()))

        # Identify central and peripheral assets
        sorted_assets = sorted(self._centrality.items(), key=lambda x: x[1], reverse=True)
        n_top = max(1, n // 4)
        central_assets = [a for a, _ in sorted_assets[:n_top]]
        peripheral_assets = [a for a, _ in sorted_assets[-n_top:]]

        return NetworkStats(
            n_nodes=n,
            n_edges=n_edges,
            density=density,
            avg_clustering=avg_clustering,
            avg_centrality=avg_centrality,
            central_assets=central_assets,
            peripheral_assets=peripheral_assets,
        )

    def network_distance(self, other: "CorrelationNetwork") -> float:
        """Calculate distance between two networks."""
        if self._adjacency is None or other._adjacency is None:
            return 1.0

        # Frobenius norm of adjacency difference
        if self._adjacency.shape != other._adjacency.shape:
            return 1.0

        diff = np.abs(self._adjacency - other._adjacency)
        n = len(self._assets)
        max_diff = n * (n - 1)  # Maximum possible difference

        return diff.sum() / max_diff if max_diff > 0 else 0.0


class NetworkRiskParityModel(Model):
    """
    Network Risk Parity Strategy.

    Uses correlation network to weight a portfolio, giving lower
    weight to central (systemically important) assets.

    Signal Logic:
    1. Calculate correlation matrix from recent returns
    2. Build network graph (edges where |corr| > threshold)
    3. Calculate centrality for each node
    4. Weight inversely by centrality
    5. Generate signals for underweight (BUY) or overweight (SELL)

    Example:
        config = ModelConfig(
            model_id="network_parity",
            model_type="portfolio",
            parameters={"corr_lookback": 60}
        )
        model = NetworkRiskParityModel(config)
        weights = await model.generate_portfolio_weights(returns_df)
    """

    def __init__(self, config: ModelConfig):
        """Initialize Network Risk Parity model."""
        super().__init__(config)
        params = config.parameters or {}

        self.net_config = NetworkConfig(
            corr_lookback=params.get("corr_lookback", 60),
            corr_threshold=params.get("corr_threshold", 0.3),
            recalc_frequency=params.get("recalc_frequency", 5),
            centrality_method=params.get("centrality_method", "eigenvector"),
            weight_decay=params.get("weight_decay", 0.5),
            min_weight=params.get("min_weight", 0.02),
            max_weight=params.get("max_weight", 0.30),
            momentum_lookback=params.get("momentum_lookback", 20),
            vol_target=params.get("vol_target", 0.15),
            network_stability_threshold=params.get("network_stability_threshold", 0.3),
        )

        # Network cache
        self._network: CorrelationNetwork | None = None
        self._prev_network: CorrelationNetwork | None = None
        self._last_recalc: int = 0
        self._weights: dict[str, float] = {}

    def analyze_network(
        self,
        returns: pd.DataFrame,
    ) -> tuple[dict[str, float], NetworkStats]:
        """
        Analyze network and calculate weights.

        Args:
            returns: DataFrame with columns as assets

        Returns:
            Tuple of (weights dict, network stats)
        """
        # Store previous network for stability check
        self._prev_network = self._network

        # Build new network
        self._network = CorrelationNetwork(
            threshold=self.net_config.corr_threshold,
            centrality_method=self.net_config.centrality_method,
        )
        self._network.fit(returns)

        # Get weights
        weights = self._network.get_weights(
            decay=self.net_config.weight_decay,
            min_weight=self.net_config.min_weight,
            max_weight=self.net_config.max_weight,
        )

        # Get stats
        stats = self._network.get_network_stats()

        self._weights = weights

        return weights, stats

    def check_network_stability(self) -> tuple[bool, float]:
        """
        Check if network has changed significantly.

        Returns:
            Tuple of (is_stable, distance)
        """
        if self._prev_network is None or self._network is None:
            return True, 0.0

        distance = self._network.network_distance(self._prev_network)
        is_stable = distance < self.net_config.network_stability_threshold

        if not is_stable:
            logger.warning(f"Network instability detected: distance={distance:.3f}")

        return is_stable, distance

    async def generate(
        self,
        symbol: str,
        data: pd.DataFrame,
        timestamp: datetime,
    ) -> Signal | None:
        """
        Generate signal for single symbol.

        For portfolio-wide allocation, use generate_portfolio_weights.
        This method returns BUY/SELL based on whether current weight
        should be increased or decreased.
        """
        if symbol not in self._weights:
            return None

        target_weight = self._weights[symbol]
        centrality = self._network.get_centrality().get(symbol, 0.5) if self._network else 0.5

        # Calculate momentum for direction
        returns = data["close"].pct_change(self.net_config.momentum_lookback)
        momentum = returns.iloc[-1] if len(returns) > 0 else 0.0

        # Signal based on weight and momentum
        if target_weight > self.net_config.min_weight * 1.5 and momentum > 0:
            signal_type = SignalType.ENTRY
            direction = Direction.LONG
            score = min(1.0, target_weight / max(self.net_config.max_weight, 1e-10))
        elif target_weight < self.net_config.min_weight * 1.5 or momentum < 0:
            signal_type = SignalType.EXIT
            direction = Direction.NEUTRAL
            score = -min(1.0, target_weight / max(self.net_config.max_weight, 1e-10))
        else:
            signal_type = SignalType.HOLD
            direction = Direction.NEUTRAL
            score = 0.0

        return Signal(
            symbol=symbol,
            timestamp=timestamp,
            signal_type=signal_type,
            direction=direction,
            probability=max(0.0, min(1.0, 1.0 - centrality)),
            score=float(score),
            model_id=self.config.model_id,
            model_version=self.config.version,
            confidence=1.0 - centrality,  # legacy alias
            metadata={
                "strategy": "network_risk_parity",
                "target_weight": target_weight,
                "centrality": centrality,
                "momentum": momentum,
                "direction": 1 if direction == Direction.LONG else (-1 if direction == Direction.SHORT else 0),
            },
        )

    async def generate_portfolio_weights(
        self,
        returns: pd.DataFrame,
        timestamp: datetime,
    ) -> dict[str, Signal]:
        """
        Generate portfolio-wide weight signals.

        Args:
            returns: DataFrame with asset returns
            timestamp: Current timestamp

        Returns:
            Dictionary of symbol to Signal with target weights
        """
        current_bar = len(returns)

        # Recalculate if needed
        if current_bar - self._last_recalc >= self.net_config.recalc_frequency:
            recent_returns = returns.iloc[-self.net_config.corr_lookback :]
            weights, stats = self.analyze_network(recent_returns)
            self._last_recalc = current_bar

            logger.info(
                f"Network updated: {stats.n_nodes} nodes, {stats.n_edges} edges, "
                f"density={stats.density:.2f}"
            )

        weights = self._weights
        centralities = self._network.get_centrality() if self._network else {}

        signals = {}
        for symbol, weight in weights.items():
            centrality = centralities.get(symbol, 0.5)

            # Momentum for signal direction
            if symbol in returns.columns:
                mom = returns[symbol].iloc[-self.net_config.momentum_lookback :].sum()
            else:
                mom = 0.0

            if weight > 0.05 and mom > 0:
                signal_type = SignalType.ENTRY
                direction = Direction.LONG
                score = min(1.0, weight / max(self.net_config.max_weight, 1e-10))
            elif weight < 0.05 or mom < 0:
                signal_type = SignalType.EXIT
                direction = Direction.NEUTRAL
                score = -min(1.0, weight / max(self.net_config.max_weight, 1e-10))
            else:
                signal_type = SignalType.HOLD
                direction = Direction.NEUTRAL
                score = 0.0

            signals[symbol] = Signal(
                symbol=symbol,
                timestamp=timestamp,
                signal_type=signal_type,
                direction=direction,
                probability=max(0.0, min(1.0, 1.0 - centrality)),
                score=float(score),
                model_id=self.config.model_id,
                model_version=self.config.version,
                confidence=1.0 - centrality,  # legacy alias
                metadata={
                    "strategy": "network_risk_parity",
                    "target_weight": weight,
                    "centrality": centrality,
                    "momentum": mom,
                },
            )

        return signals


def analyze_correlation_network(
    returns: pd.DataFrame,
    threshold: float = 0.3,
) -> dict:
    """
    Analyze correlation network of asset returns.

    Args:
        returns: DataFrame with asset returns
        threshold: Minimum correlation for network edge

    Returns:
        Dictionary with network analysis results
    """
    network = CorrelationNetwork(threshold=threshold)
    network.fit(returns)

    stats = network.get_network_stats()
    weights = network.get_weights()
    centrality = network.get_centrality()

    # Correlation matrix
    corr = returns.corr()

    return {
        "stats": stats,
        "weights": weights,
        "centrality": centrality,
        "correlation_matrix": corr,
        "central_assets": stats.central_assets,
        "peripheral_assets": stats.peripheral_assets,
    }


def backtest(
    prices_df: pd.DataFrame,
    config: NetworkConfig | None = None,
    initial_capital: float = 100000,
) -> dict:
    """Backtest Network Risk Parity portfolio strategy."""
    if config is None:
        config = NetworkConfig()

    # Calculate returns
    returns = prices_df.pct_change().dropna()

    if len(returns) < config.corr_lookback + 20:
        return {"error": "Insufficient data"}

    model_config = ModelConfig(
        model_id="network_parity_backtest",
        model_type="portfolio",
        parameters={
            "corr_lookback": config.corr_lookback,
            "corr_threshold": config.corr_threshold,
        },
    )

    model = NetworkRiskParityModel(model_config)

    portfolio_values = [initial_capital]
    weight_history = []

    start_idx = config.corr_lookback

    for i in range(start_idx, len(returns)):
        window = returns.iloc[: i + 1]

        # Get target weights
        import asyncio

        signals = asyncio.get_event_loop().run_until_complete(
            model.generate_portfolio_weights(window, returns.index[i])
        )

        # Calculate portfolio return
        weights = {sym: sig.metadata["target_weight"] for sym, sig in signals.items()}
        weight_history.append(weights)

        daily_return = sum(
            weights.get(col, 0) * returns.iloc[i][col] for col in returns.columns if col in weights
        )

        new_value = portfolio_values[-1] * (1 + daily_return)
        portfolio_values.append(new_value)

    # Calculate metrics
    portfolio_returns = pd.Series(portfolio_values).pct_change().dropna()

    total_return = (portfolio_values[-1] / initial_capital - 1) * 100
    annual_return = ((1 + total_return / 100) ** (252 / len(portfolio_returns)) - 1) * 100
    volatility = portfolio_returns.std() * np.sqrt(252) * 100
    sharpe = annual_return / volatility if volatility > 0 else 0

    max_dd = 0
    peak = portfolio_values[0]
    for v in portfolio_values:
        peak = max(v, peak)
        dd = (peak - v) / peak
        max_dd = max(max_dd, dd)

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_dd * 100,
        "final_value": portfolio_values[-1],
        "portfolio_values": portfolio_values,
        "weight_history": weight_history,
    }


# Visualization helper (optional, uses networkx if available)
def visualize_network(
    returns: pd.DataFrame,
    threshold: float = 0.3,
    figsize: tuple[int, int] = (12, 10),
):
    """
    Visualize correlation network (requires networkx and matplotlib).
    """
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError:
        logger.error("networkx and matplotlib required for visualization")
        return None

    network = CorrelationNetwork(threshold=threshold)
    network.fit(returns)

    centrality = network.get_centrality()
    weights = network.get_weights()

    # Build NetworkX graph
    G = nx.Graph()

    for asset in returns.columns:
        G.add_node(asset, centrality=centrality.get(asset, 0))

    corr = returns.corr()
    for i, asset1 in enumerate(returns.columns):
        for j, asset2 in enumerate(returns.columns):
            if i < j and abs(corr.iloc[i, j]) >= threshold:
                G.add_edge(asset1, asset2, weight=abs(corr.iloc[i, j]))

    fig, ax = plt.subplots(figsize=figsize)

    # Node sizes by inverse centrality (larger = more peripheral)
    node_sizes = [3000 * (1 - centrality.get(n, 0) + 0.2) for n in G.nodes()]

    # Node colors by weight
    node_colors = [weights.get(n, 0) for n in G.nodes()]

    pos = nx.spring_layout(G, k=2, iterations=50)

    nx.draw(
        G,
        pos,
        ax=ax,
        node_size=node_sizes,
        node_color=node_colors,
        cmap=plt.cm.YlOrRd,
        with_labels=True,
        font_size=10,
        font_weight="bold",
        edge_color="gray",
        alpha=0.7,
    )

    ax.set_title("Correlation Network (size=peripheral, color=weight)")
    plt.tight_layout()

    return fig
