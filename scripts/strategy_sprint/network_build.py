"""
Network Correlation Build.

Builds correlation network for portfolio risk parity allocation.
Uses graph-based clustering and centrality measures for diversification.
"""

import asyncio
from datetime import datetime
import logging
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

logger = logging.getLogger(__name__)

# Universe for network analysis
UNIVERSE = [
    # Tech
    "AAPL",
    "MSFT",
    "GOOGL",
    "META",
    "AMZN",
    "NVDA",
    "AMD",
    "TSLA",
    # Finance
    "JPM",
    "BAC",
    "GS",
    "V",
    "MA",
    # Consumer
    "WMT",
    "COST",
    "HD",
    "NKE",
    "KO",
    "PEP",
    # Healthcare
    "JNJ",
    "UNH",
    "PFE",
    "ABBV",
    # Energy
    "XOM",
    "CVX",
    # Industrial
    "CAT",
    "BA",
    "GE",
    # ETFs
    "SPY",
    "QQQ",
    "IWM",
    "TLT",
    "GLD",
]


def load_historical_data(symbol: str, days: int = 500) -> pd.DataFrame:
    """Load historical price data."""
    cache_path = Path(f"data/historical/{symbol}_daily.parquet")

    if cache_path.exists():
        return pd.read_parquet(cache_path)

    csv_path = Path(f"data/historical/{symbol}_daily.csv")
    if csv_path.exists():
        return pd.read_csv(csv_path, parse_dates=["date"], index_col="date")

    return generate_synthetic_prices(symbol, days)


def generate_synthetic_prices(symbol: str, days: int = 500) -> pd.DataFrame:
    """Generate synthetic price data with sector correlations."""
    np.random.seed(hash(symbol) % 2**32)

    # Sector mapping for correlation structure
    sectors = {
        "AAPL": "tech",
        "MSFT": "tech",
        "GOOGL": "tech",
        "META": "tech",
        "AMZN": "tech",
        "NVDA": "tech",
        "AMD": "tech",
        "TSLA": "tech",
        "JPM": "fin",
        "BAC": "fin",
        "GS": "fin",
        "V": "fin",
        "MA": "fin",
        "WMT": "cons",
        "COST": "cons",
        "HD": "cons",
        "NKE": "cons",
        "KO": "cons",
        "PEP": "cons",
        "JNJ": "health",
        "UNH": "health",
        "PFE": "health",
        "ABBV": "health",
        "XOM": "energy",
        "CVX": "energy",
        "CAT": "ind",
        "BA": "ind",
        "GE": "ind",
        "SPY": "market",
        "QQQ": "tech",
        "IWM": "market",
        "TLT": "bond",
        "GLD": "commodity",
    }

    sector = sectors.get(symbol, "other")

    # Sector-specific parameters
    sector_params = {
        "tech": {"drift": 0.0004, "vol": 0.02, "market_beta": 1.2},
        "fin": {"drift": 0.0002, "vol": 0.018, "market_beta": 1.1},
        "cons": {"drift": 0.0002, "vol": 0.012, "market_beta": 0.8},
        "health": {"drift": 0.0002, "vol": 0.014, "market_beta": 0.7},
        "energy": {"drift": 0.0001, "vol": 0.022, "market_beta": 1.0},
        "ind": {"drift": 0.0002, "vol": 0.016, "market_beta": 1.0},
        "market": {"drift": 0.0003, "vol": 0.012, "market_beta": 1.0},
        "bond": {"drift": 0.0001, "vol": 0.008, "market_beta": -0.2},
        "commodity": {"drift": 0.0001, "vol": 0.015, "market_beta": 0.1},
        "other": {"drift": 0.0002, "vol": 0.015, "market_beta": 1.0},
    }

    params = sector_params[sector]

    # Market factor
    np.random.seed(42)  # Same market for all
    market_returns = np.random.normal(0.0003, 0.01, days)

    # Idiosyncratic returns
    np.random.seed(hash(symbol) % 2**32)
    idio_returns = np.random.normal(0, params["vol"] * 0.7, days)

    # Combined returns
    returns = params["drift"] + params["market_beta"] * market_returns + idio_returns

    base_price = np.random.uniform(50, 500)
    prices = base_price * np.exp(np.cumsum(returns))

    dates = pd.date_range(end=datetime.now(), periods=days, freq="D")

    df = pd.DataFrame(index=dates)
    df["close"] = prices
    df["high"] = prices * (1 + np.abs(np.random.normal(0, 0.01, days)))
    df["low"] = prices * (1 - np.abs(np.random.normal(0, 0.01, days)))
    df["open"] = df["close"].shift(1).fillna(df["close"].iloc[0])
    df["volume"] = np.random.randint(1_000_000, 50_000_000, days)

    return df


def build_returns_matrix(universe_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build matrix of returns for correlation analysis."""
    returns_dict = {}

    for symbol, df in universe_data.items():
        returns_dict[symbol] = df["close"].pct_change()

    returns_df = pd.DataFrame(returns_dict).dropna()
    return returns_df


def build_correlation_matrix(returns_df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """Build correlation matrix."""
    if method == "pearson":
        return returns_df.corr()
    if method == "spearman":
        return returns_df.corr(method="spearman")
    if method == "shrinkage":
        # Ledoit-Wolf shrinkage
        corr = returns_df.corr()
        n = len(returns_df.columns)

        # Shrink toward identity
        shrinkage = 0.2
        shrunk = (1 - shrinkage) * corr + shrinkage * np.eye(n)
        return pd.DataFrame(shrunk, index=corr.index, columns=corr.columns)

    return returns_df.corr()


def correlation_to_distance(corr_matrix: pd.DataFrame) -> pd.DataFrame:
    """Convert correlation to distance matrix."""
    # d = sqrt(2 * (1 - rho))
    distance = np.sqrt(2 * (1 - corr_matrix))
    return pd.DataFrame(distance, index=corr_matrix.index, columns=corr_matrix.columns)


def build_mst(distance_matrix: pd.DataFrame) -> list[tuple[str, str, float]]:
    """Build Minimum Spanning Tree using Prim's algorithm."""
    symbols = list(distance_matrix.columns)
    n = len(symbols)

    # Start with first node
    in_tree = {symbols[0]}
    edges = []

    while len(in_tree) < n:
        min_dist = np.inf
        min_edge = None

        for s1 in in_tree:
            for s2 in symbols:
                if s2 not in in_tree:
                    dist = distance_matrix.loc[s1, s2]
                    if dist < min_dist:
                        min_dist = dist
                        min_edge = (s1, s2, dist)

        if min_edge:
            edges.append(min_edge)
            in_tree.add(min_edge[1])

    return edges


def calculate_centrality(mst_edges: list[tuple], symbols: list[str]) -> dict[str, float]:
    """Calculate degree centrality from MST."""
    degree = {s: 0 for s in symbols}

    for s1, s2, _ in mst_edges:
        degree[s1] += 1
        degree[s2] += 1

    # Normalize
    max_degree = max(degree.values()) if degree else 1
    return {s: d / max_degree for s, d in degree.items()}


def hierarchical_clustering(distance_matrix: pd.DataFrame, n_clusters: int = 5) -> dict[str, int]:
    """Simple agglomerative clustering."""
    symbols = list(distance_matrix.columns)
    n = len(symbols)

    # Start with each symbol in its own cluster
    clusters = {s: i for i, s in enumerate(symbols)}
    n_current = n

    # Track cluster sizes
    cluster_sizes = {i: 1 for i in range(n)}

    while n_current > n_clusters:
        # Find closest clusters
        min_dist = np.inf
        merge_pair = None

        for i, s1 in enumerate(symbols):
            for s2 in symbols[i + 1 :]:
                if clusters[s1] != clusters[s2]:
                    dist = distance_matrix.loc[s1, s2]
                    if dist < min_dist:
                        min_dist = dist
                        merge_pair = (clusters[s1], clusters[s2])

        if merge_pair:
            old_cluster, new_cluster = merge_pair[1], merge_pair[0]
            for s in symbols:
                if clusters[s] == old_cluster:
                    clusters[s] = new_cluster
            n_current -= 1

    # Relabel clusters 0 to n_clusters-1
    unique_clusters = list(set(clusters.values()))
    relabel = {old: new for new, old in enumerate(unique_clusters)}
    return {s: relabel[c] for s, c in clusters.items()}


def risk_parity_weights(returns_df: pd.DataFrame) -> dict[str, float]:
    """Calculate risk parity weights (equal risk contribution)."""
    # Simplified: inverse volatility weighting
    vols = returns_df.std() * np.sqrt(252)
    inv_vols = 1 / (vols + 1e-10)
    weights = inv_vols / inv_vols.sum()
    return weights.to_dict()


def cluster_aware_weights(
    returns_df: pd.DataFrame,
    clusters: dict[str, int],
) -> dict[str, float]:
    """Calculate weights with cluster awareness."""
    symbols = list(returns_df.columns)
    n_clusters = len(set(clusters.values()))

    # Equal weight per cluster
    cluster_weight = 1 / n_clusters

    # Count symbols per cluster
    cluster_counts = {}
    for s, c in clusters.items():
        cluster_counts[c] = cluster_counts.get(c, 0) + 1

    # Assign weights
    weights = {}
    for s in symbols:
        c = clusters[s]
        weights[s] = cluster_weight / cluster_counts[c]

    return weights


def backtest_portfolio(
    returns_df: pd.DataFrame,
    weights: dict[str, float],
) -> dict:
    """Backtest portfolio with given weights."""
    weighted_returns = pd.Series(0.0, index=returns_df.index)

    for symbol, weight in weights.items():
        if symbol in returns_df.columns:
            weighted_returns += weight * returns_df[symbol]

    total_return = (1 + weighted_returns).prod() - 1
    volatility = weighted_returns.std() * np.sqrt(252)
    sharpe = (weighted_returns.mean() * 252) / volatility if volatility > 0 else 0

    # Max drawdown
    cumulative = (1 + weighted_returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()

    return {
        "total_return": total_return * 100,
        "volatility": volatility * 100,
        "sharpe": sharpe,
        "max_drawdown": max_dd * 100,
    }


async def run() -> dict:
    """Run network correlation analysis."""
    logger.info("=" * 50)
    logger.info("NETWORK CORRELATION BUILD")
    logger.info("=" * 50)

    # Load data
    logger.info(f"Loading data for {len(UNIVERSE)} symbols...")
    universe_data = {}
    for symbol in UNIVERSE:
        df = load_historical_data(symbol)
        if len(df) > 252:
            universe_data[symbol] = df

    logger.info(f"Loaded {len(universe_data)} symbols")

    # Build returns matrix
    returns_df = build_returns_matrix(universe_data)
    logger.info(f"Returns matrix: {returns_df.shape}")

    # Build correlation matrix
    corr_matrix = build_correlation_matrix(returns_df, method="shrinkage")

    logger.info("\n--- Correlation Statistics ---")
    avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix, k=1)].mean()
    logger.info(f"  Average pairwise correlation: {avg_corr:.3f}")

    # Find highest correlations
    flat_corr = []
    symbols = list(corr_matrix.columns)
    for i, s1 in enumerate(symbols):
        for s2 in symbols[i + 1 :]:
            flat_corr.append((s1, s2, corr_matrix.loc[s1, s2]))

    flat_corr.sort(key=lambda x: -x[2])

    logger.info("\n  Top 10 Correlated Pairs:")
    for s1, s2, corr in flat_corr[:10]:
        logger.info(f"    {s1:5s}-{s2:5s}: {corr:.3f}")

    # Distance matrix and MST
    dist_matrix = correlation_to_distance(corr_matrix)
    mst_edges = build_mst(dist_matrix)

    logger.info("\n--- Minimum Spanning Tree ---")
    logger.info(f"  Edges: {len(mst_edges)}")
    logger.info("  Key connections:")
    for s1, s2, dist in mst_edges[:10]:
        logger.info(f"    {s1:5s} -- {s2:5s} (d={dist:.3f})")

    # Centrality
    centrality = calculate_centrality(mst_edges, symbols)
    sorted_centrality = sorted(centrality.items(), key=lambda x: -x[1])

    logger.info("\n--- Network Centrality ---")
    logger.info("  Most central (diversification hubs):")
    for symbol, cent in sorted_centrality[:10]:
        logger.info(f"    {symbol:5s}: {cent:.2f}")

    logger.info("\n  Most peripheral (diversifiers):")
    for symbol, cent in sorted_centrality[-5:]:
        logger.info(f"    {symbol:5s}: {cent:.2f}")

    # Clustering
    clusters = hierarchical_clustering(dist_matrix, n_clusters=5)

    logger.info("\n--- Cluster Assignment ---")
    for cluster_id in range(5):
        members = [s for s, c in clusters.items() if c == cluster_id]
        logger.info(f"  Cluster {cluster_id}: {', '.join(members)}")

    # Weight schemes
    logger.info("\n--- Portfolio Weight Schemes ---")

    # Equal weight
    eq_weights = {s: 1 / len(symbols) for s in symbols}
    eq_bt = backtest_portfolio(returns_df, eq_weights)
    logger.info(
        f"  Equal Weight: Return={eq_bt['total_return']:.1f}%, "
        f"Sharpe={eq_bt['sharpe']:.2f}, DD={eq_bt['max_drawdown']:.1f}%"
    )

    # Risk parity
    rp_weights = risk_parity_weights(returns_df)
    rp_bt = backtest_portfolio(returns_df, rp_weights)
    logger.info(
        f"  Risk Parity: Return={rp_bt['total_return']:.1f}%, "
        f"Sharpe={rp_bt['sharpe']:.2f}, DD={rp_bt['max_drawdown']:.1f}%"
    )

    # Cluster-aware
    ca_weights = cluster_aware_weights(returns_df, clusters)
    ca_bt = backtest_portfolio(returns_df, ca_weights)
    logger.info(
        f"  Cluster-Aware: Return={ca_bt['total_return']:.1f}%, "
        f"Sharpe={ca_bt['sharpe']:.2f}, DD={ca_bt['max_drawdown']:.1f}%"
    )

    # Best diversifiers (low centrality, different clusters)
    logger.info("\n--- Recommended Diversifiers ---")
    peripheral = sorted_centrality[-10:]
    for symbol, cent in peripheral:
        cluster = clusters[symbol]
        logger.info(f"  {symbol:5s}: Cluster {cluster}, Centrality {cent:.2f}")

    # Save results
    output_dir = Path("artifacts/reports/strategy_sprint")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Correlation matrix
    corr_matrix.to_csv(output_dir / "network_correlation_matrix.csv")

    # Clusters
    cluster_df = pd.DataFrame(
        [{"symbol": s, "cluster": c, "centrality": centrality[s]} for s, c in clusters.items()]
    )
    cluster_df.to_csv(output_dir / "network_clusters.csv", index=False)

    # MST edges
    mst_df = pd.DataFrame(mst_edges, columns=["node1", "node2", "distance"])
    mst_df.to_csv(output_dir / "network_mst_edges.csv", index=False)

    # Weights comparison
    weights_df = pd.DataFrame(
        {
            "symbol": symbols,
            "equal_weight": [eq_weights[s] for s in symbols],
            "risk_parity": [rp_weights[s] for s in symbols],
            "cluster_aware": [ca_weights[s] for s in symbols],
        }
    )
    weights_df.to_csv(output_dir / "network_weight_schemes.csv", index=False)

    return {
        "success": True,
        "summary": f"Built correlation network for {len(symbols)} assets",
        "avg_correlation": avg_corr,
        "n_clusters": 5,
        "best_scheme": "cluster_aware" if ca_bt["sharpe"] > rp_bt["sharpe"] else "risk_parity",
        "best_sharpe": max(eq_bt["sharpe"], rp_bt["sharpe"], ca_bt["sharpe"]),
        "diversifiers": [s for s, _ in peripheral[:5]],
    }


if __name__ == "__main__":
    asyncio.run(run())
