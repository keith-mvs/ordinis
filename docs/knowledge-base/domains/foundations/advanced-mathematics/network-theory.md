# Network Theory for Financial Markets

Network theory provides powerful tools for analyzing interconnections between financial assets, identifying systemic risk, constructing robust portfolios, and understanding market contagion dynamics.

---

## Overview

Financial markets exhibit complex interdependencies that can be modeled as networks. Network analysis enables:

1. **Correlation Networks**: Visualize and analyze asset relationships
2. **Portfolio Construction**: Use network topology for diversification
3. **Systemic Risk**: Identify critical nodes and contagion pathways
4. **Market Regimes**: Detect structural changes through network evolution
5. **Sector Analysis**: Community detection for asset clustering

This document covers foundational network concepts and their applications to systematic trading.

---

## 1. Correlation Networks

### 1.1 Distance Metrics

To construct networks from correlation matrices, we transform correlations into distances:

**Metric Distance** (Mantegna, 1999):
$$d_{ij} = \sqrt{2(1 - \rho_{ij})}$$

where $\rho_{ij}$ is the Pearson correlation between assets $i$ and $j$.

**Properties**:
- $d_{ij} = 0$ when $\rho_{ij} = 1$ (perfect correlation)
- $d_{ij} = \sqrt{2}$ when $\rho_{ij} = 0$ (uncorrelated)
- $d_{ij} = 2$ when $\rho_{ij} = -1$ (perfect anti-correlation)
- Satisfies triangle inequality (metric space)

**Angular Distance**:
$$d_{ij}^{(\theta)} = \arccos(\rho_{ij})$$

**Absolute Correlation Distance**:
$$d_{ij}^{(|r|)} = \sqrt{1 - |\rho_{ij}|}$$

### 1.2 Network Construction Methods

**Threshold Networks**:
- Connect nodes if $\rho_{ij} > \theta$
- Simple but arbitrary threshold selection
- May produce disconnected components

**k-Nearest Neighbors (kNN)**:
- Each node connects to k closest neighbors
- Guarantees minimum connectivity
- Asymmetric (requires symmetrization)

**Minimum Spanning Tree (MST)**:
- Connects all nodes with minimum total distance
- Exactly $n-1$ edges for $n$ nodes
- Hierarchical structure, no cycles

**Planar Maximally Filtered Graph (PMFG)**:
- Extends MST while maintaining planarity
- Exactly $3(n-2)$ edges
- Preserves more correlation structure

### 1.3 Python Implementation

```python
import numpy as np
import pandas as pd
import networkx as nx
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass


@dataclass
class NetworkMetrics:
    """Container for network analysis results."""
    n_nodes: int
    n_edges: int
    density: float
    avg_clustering: float
    avg_path_length: float
    diameter: int
    modularity: float
    n_communities: int


class CorrelationNetwork:
    """
    Correlation-based network analysis for financial assets.

    Constructs and analyzes networks from return correlations
    using various filtering methods (MST, threshold, PMFG).
    """

    def __init__(self, returns: pd.DataFrame):
        """
        Initialize with asset returns.

        Args:
            returns: DataFrame of asset returns (rows=dates, cols=assets)
        """
        self.returns = returns
        self.assets = list(returns.columns)
        self.n_assets = len(self.assets)

        # Compute correlation matrix
        self.correlation = returns.corr()

        # Convert to distance matrix
        self.distance = self._correlation_to_distance(self.correlation)

        # Networks (computed on demand)
        self._mst = None
        self._threshold_network = None
        self._pmfg = None

    def _correlation_to_distance(
        self,
        corr: pd.DataFrame,
        method: str = 'metric'
    ) -> pd.DataFrame:
        """
        Convert correlation matrix to distance matrix.

        Args:
            corr: Correlation matrix
            method: 'metric', 'angular', or 'absolute'

        Returns:
            Distance matrix
        """
        if method == 'metric':
            # Mantegna distance: d = sqrt(2(1 - rho))
            dist = np.sqrt(2 * (1 - corr))
        elif method == 'angular':
            # Angular distance: d = arccos(rho)
            dist = np.arccos(np.clip(corr, -1, 1))
        elif method == 'absolute':
            # Absolute correlation distance
            dist = np.sqrt(1 - np.abs(corr))
        else:
            raise ValueError(f"Unknown method: {method}")

        # Set diagonal to zero
        np.fill_diagonal(dist.values, 0)

        return dist

    def compute_mst(self) -> nx.Graph:
        """
        Compute Minimum Spanning Tree from correlation network.

        Returns:
            NetworkX graph representing MST
        """
        if self._mst is not None:
            return self._mst

        # Create complete graph with distance weights
        G_complete = nx.Graph()

        for i, asset_i in enumerate(self.assets):
            for j, asset_j in enumerate(self.assets):
                if i < j:
                    weight = self.distance.loc[asset_i, asset_j]
                    G_complete.add_edge(asset_i, asset_j, weight=weight)

        # Compute MST using Kruskal's algorithm
        self._mst = nx.minimum_spanning_tree(G_complete, weight='weight')

        # Add correlation as edge attribute
        for u, v in self._mst.edges():
            self._mst[u][v]['correlation'] = self.correlation.loc[u, v]

        return self._mst

    def compute_threshold_network(
        self,
        threshold: float = 0.5,
        use_correlation: bool = True
    ) -> nx.Graph:
        """
        Construct network by thresholding correlation/distance.

        Args:
            threshold: Threshold value
            use_correlation: If True, connect when corr > threshold
                           If False, connect when dist < threshold

        Returns:
            Thresholded network
        """
        G = nx.Graph()
        G.add_nodes_from(self.assets)

        for i, asset_i in enumerate(self.assets):
            for j, asset_j in enumerate(self.assets):
                if i < j:
                    corr = self.correlation.loc[asset_i, asset_j]
                    dist = self.distance.loc[asset_i, asset_j]

                    if use_correlation:
                        if corr > threshold:
                            G.add_edge(asset_i, asset_j,
                                      weight=dist, correlation=corr)
                    else:
                        if dist < threshold:
                            G.add_edge(asset_i, asset_j,
                                      weight=dist, correlation=corr)

        self._threshold_network = G
        return G

    def compute_pmfg(self) -> nx.Graph:
        """
        Compute Planar Maximally Filtered Graph.

        PMFG extends MST while maintaining graph planarity,
        allowing 3(n-2) edges instead of n-1.

        Returns:
            PMFG network
        """
        # Get all edges sorted by distance (ascending)
        edges = []
        for i, asset_i in enumerate(self.assets):
            for j, asset_j in enumerate(self.assets):
                if i < j:
                    dist = self.distance.loc[asset_i, asset_j]
                    corr = self.correlation.loc[asset_i, asset_j]
                    edges.append((asset_i, asset_j, dist, corr))

        edges.sort(key=lambda x: x[2])  # Sort by distance

        # Build PMFG greedily
        G = nx.Graph()
        G.add_nodes_from(self.assets)

        max_edges = 3 * (self.n_assets - 2)

        for asset_i, asset_j, dist, corr in edges:
            if G.number_of_edges() >= max_edges:
                break

            # Add edge temporarily
            G.add_edge(asset_i, asset_j, weight=dist, correlation=corr)

            # Check planarity
            is_planar, _ = nx.check_planarity(G)

            if not is_planar:
                # Remove edge if it violates planarity
                G.remove_edge(asset_i, asset_j)

        self._pmfg = G
        return G

    def get_network_metrics(self, G: nx.Graph) -> NetworkMetrics:
        """
        Compute comprehensive network metrics.

        Args:
            G: NetworkX graph

        Returns:
            NetworkMetrics dataclass
        """
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()

        # Density
        max_edges = n_nodes * (n_nodes - 1) / 2
        density = n_edges / max_edges if max_edges > 0 else 0

        # Clustering coefficient
        avg_clustering = nx.average_clustering(G)

        # Path length and diameter (only for connected graphs)
        if nx.is_connected(G):
            avg_path_length = nx.average_shortest_path_length(G)
            diameter = nx.diameter(G)
        else:
            # Use largest connected component
            largest_cc = max(nx.connected_components(G), key=len)
            G_cc = G.subgraph(largest_cc)
            avg_path_length = nx.average_shortest_path_length(G_cc)
            diameter = nx.diameter(G_cc)

        # Community detection for modularity
        communities = list(nx.community.greedy_modularity_communities(G))
        modularity = nx.community.modularity(G, communities)

        return NetworkMetrics(
            n_nodes=n_nodes,
            n_edges=n_edges,
            density=density,
            avg_clustering=avg_clustering,
            avg_path_length=avg_path_length,
            diameter=diameter,
            modularity=modularity,
            n_communities=len(communities)
        )


# Example usage
if __name__ == "__main__":
    # Generate synthetic returns
    np.random.seed(42)
    n_assets = 20
    n_days = 252

    # Create correlated returns (block structure)
    base_returns = np.random.randn(n_days, 4)  # 4 factors
    loadings = np.random.randn(n_assets, 4) * 0.3
    idiosyncratic = np.random.randn(n_days, n_assets) * 0.5

    returns_data = base_returns @ loadings.T + idiosyncratic

    assets = [f"Asset_{i}" for i in range(n_assets)]
    returns = pd.DataFrame(returns_data, columns=assets)

    # Build network
    network = CorrelationNetwork(returns)

    # Compute MST
    mst = network.compute_mst()
    print(f"MST: {mst.number_of_nodes()} nodes, {mst.number_of_edges()} edges")

    # Compute threshold network
    threshold_net = network.compute_threshold_network(threshold=0.3)
    print(f"Threshold (0.3): {threshold_net.number_of_edges()} edges")

    # Get metrics
    metrics = network.get_network_metrics(mst)
    print(f"\nMST Metrics:")
    print(f"  Density: {metrics.density:.4f}")
    print(f"  Avg Clustering: {metrics.avg_clustering:.4f}")
    print(f"  Avg Path Length: {metrics.avg_path_length:.2f}")
    print(f"  Diameter: {metrics.diameter}")
```

---

## 2. Centrality Measures

### 2.1 Degree Centrality

**Definition**: Number of connections normalized by maximum possible.

$$C_D(v) = \frac{\deg(v)}{n-1}$$

**Trading Interpretation**: Assets with high degree centrality are highly correlated with many others - potential market drivers or systemic risk contributors.

### 2.2 Betweenness Centrality

**Definition**: Fraction of shortest paths passing through a node.

$$C_B(v) = \sum_{s \neq v \neq t} \frac{\sigma_{st}(v)}{\sigma_{st}}$$

where $\sigma_{st}$ is the number of shortest paths from $s$ to $t$, and $\sigma_{st}(v)$ is the number passing through $v$.

**Trading Interpretation**: Assets with high betweenness act as "bridges" between market sectors - their movements can transmit shocks across the network.

### 2.3 Eigenvector Centrality

**Definition**: Proportional to sum of neighbors' centralities.

$$C_E(v) = \frac{1}{\lambda} \sum_{u \in N(v)} C_E(u)$$

Equivalently, the principal eigenvector of the adjacency matrix.

**Trading Interpretation**: Assets connected to other central assets. Captures influence propagation.

### 2.4 PageRank Centrality

**Definition**: Random walk probability with damping factor $\alpha$.

$$PR(v) = \frac{1-\alpha}{n} + \alpha \sum_{u \in N(v)} \frac{PR(u)}{\deg(u)}$$

**Trading Interpretation**: Robust centrality measure that handles disconnected components.

### 2.5 Python Implementation

```python
class CentralityAnalysis:
    """
    Centrality analysis for financial networks.

    Computes various centrality measures and their
    implications for portfolio construction and risk.
    """

    def __init__(self, G: nx.Graph):
        """
        Initialize with network.

        Args:
            G: NetworkX graph (typically MST or filtered network)
        """
        self.G = G
        self._centralities = {}

    def compute_all_centralities(self) -> pd.DataFrame:
        """
        Compute all centrality measures.

        Returns:
            DataFrame with centrality scores for each node
        """
        centralities = {
            'degree': nx.degree_centrality(self.G),
            'betweenness': nx.betweenness_centrality(self.G),
            'closeness': nx.closeness_centrality(self.G),
            'eigenvector': nx.eigenvector_centrality(self.G, max_iter=1000),
            'pagerank': nx.pagerank(self.G),
        }

        # Weighted versions if graph has weights
        if nx.is_weighted(self.G):
            # For distance-weighted graphs, invert weights for some measures
            centralities['weighted_degree'] = dict(self.G.degree(weight='weight'))
            centralities['weighted_betweenness'] = nx.betweenness_centrality(
                self.G, weight='weight'
            )

        self._centralities = centralities

        # Convert to DataFrame
        df = pd.DataFrame(centralities)
        df.index.name = 'asset'

        return df

    def get_central_assets(
        self,
        measure: str = 'eigenvector',
        top_n: int = 5
    ) -> List[str]:
        """
        Get most central assets by specified measure.

        Args:
            measure: Centrality measure name
            top_n: Number of top assets to return

        Returns:
            List of asset names
        """
        if measure not in self._centralities:
            self.compute_all_centralities()

        centrality = self._centralities[measure]
        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)

        return [node for node, _ in sorted_nodes[:top_n]]

    def get_peripheral_assets(
        self,
        measure: str = 'degree',
        bottom_n: int = 5
    ) -> List[str]:
        """
        Get least central (peripheral) assets.

        Peripheral assets may offer diversification benefits.

        Args:
            measure: Centrality measure name
            bottom_n: Number of bottom assets to return

        Returns:
            List of asset names
        """
        if measure not in self._centralities:
            self.compute_all_centralities()

        centrality = self._centralities[measure]
        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1])

        return [node for node, _ in sorted_nodes[:bottom_n]]

    def centrality_based_weights(
        self,
        measure: str = 'eigenvector',
        invert: bool = True
    ) -> Dict[str, float]:
        """
        Compute portfolio weights based on centrality.

        Args:
            measure: Centrality measure to use
            invert: If True, underweight central assets (for diversification)
                   If False, overweight central assets (for momentum)

        Returns:
            Dictionary of asset weights
        """
        if measure not in self._centralities:
            self.compute_all_centralities()

        centrality = self._centralities[measure]

        if invert:
            # Inverse centrality weighting (diversification)
            inv_centrality = {k: 1.0 / (v + 1e-10) for k, v in centrality.items()}
            total = sum(inv_centrality.values())
            weights = {k: v / total for k, v in inv_centrality.items()}
        else:
            # Direct centrality weighting
            total = sum(centrality.values())
            weights = {k: v / total for k, v in centrality.items()}

        return weights

    def compute_systemic_importance(self) -> pd.DataFrame:
        """
        Compute systemic importance score combining multiple centralities.

        Uses eigenvector centrality and betweenness to identify
        assets that are both well-connected and act as bridges.

        Returns:
            DataFrame with systemic importance scores
        """
        if not self._centralities:
            self.compute_all_centralities()

        eigen = pd.Series(self._centralities['eigenvector'])
        between = pd.Series(self._centralities['betweenness'])
        degree = pd.Series(self._centralities['degree'])

        # Normalize each measure
        eigen_norm = (eigen - eigen.min()) / (eigen.max() - eigen.min() + 1e-10)
        between_norm = (between - between.min()) / (between.max() - between.min() + 1e-10)
        degree_norm = (degree - degree.min()) / (degree.max() - degree.min() + 1e-10)

        # Composite systemic importance score
        systemic_score = 0.4 * eigen_norm + 0.4 * between_norm + 0.2 * degree_norm

        return pd.DataFrame({
            'eigenvector': eigen,
            'betweenness': between,
            'degree': degree,
            'systemic_importance': systemic_score
        }).sort_values('systemic_importance', ascending=False)


# Example usage
if __name__ == "__main__":
    # Using network from previous example
    network = CorrelationNetwork(returns)
    mst = network.compute_mst()

    # Centrality analysis
    centrality = CentralityAnalysis(mst)
    df = centrality.compute_all_centralities()

    print("Centrality Measures:")
    print(df.round(4).head(10))

    print("\nMost Central Assets (Eigenvector):")
    print(centrality.get_central_assets('eigenvector', top_n=5))

    print("\nSystemic Importance:")
    systemic = centrality.compute_systemic_importance()
    print(systemic.head(5))
```

---

## 3. Community Detection

### 3.1 Louvain Algorithm

The Louvain method optimizes modularity through hierarchical agglomeration:

**Modularity**:
$$Q = \frac{1}{2m} \sum_{ij} \left[ A_{ij} - \frac{k_i k_j}{2m} \right] \delta(c_i, c_j)$$

where:
- $A_{ij}$ is the adjacency matrix
- $k_i$ is the degree of node $i$
- $m$ is total number of edges
- $c_i$ is the community of node $i$
- $\delta$ is Kronecker delta

**Trading Application**: Identify market sectors or asset clusters that move together.

### 3.2 Hierarchical Clustering

**Single Linkage** (MST-based):
$$d(C_1, C_2) = \min_{i \in C_1, j \in C_2} d_{ij}$$

**Complete Linkage**:
$$d(C_1, C_2) = \max_{i \in C_1, j \in C_2} d_{ij}$$

**Ward's Method**:
Minimizes within-cluster variance increase.

### 3.3 Python Implementation

```python
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from collections import defaultdict
import matplotlib.pyplot as plt


class CommunityDetection:
    """
    Community detection and hierarchical clustering for asset networks.
    """

    def __init__(self, correlation_network: CorrelationNetwork):
        """
        Initialize with correlation network.

        Args:
            correlation_network: CorrelationNetwork instance
        """
        self.network = correlation_network
        self.assets = correlation_network.assets
        self.distance = correlation_network.distance

    def louvain_communities(self, G: nx.Graph = None) -> Dict[str, int]:
        """
        Detect communities using Louvain algorithm.

        Args:
            G: Network to analyze (uses MST if None)

        Returns:
            Dictionary mapping assets to community IDs
        """
        if G is None:
            G = self.network.compute_mst()

        # Louvain community detection
        communities = nx.community.louvain_communities(G, seed=42)

        # Convert to asset -> community mapping
        asset_communities = {}
        for comm_id, community in enumerate(communities):
            for asset in community:
                asset_communities[asset] = comm_id

        return asset_communities

    def hierarchical_clustering(
        self,
        method: str = 'ward',
        n_clusters: int = None,
        distance_threshold: float = None
    ) -> Dict[str, int]:
        """
        Perform hierarchical clustering on assets.

        Args:
            method: Linkage method ('single', 'complete', 'average', 'ward')
            n_clusters: Number of clusters (if specified)
            distance_threshold: Distance cutoff (if n_clusters not specified)

        Returns:
            Dictionary mapping assets to cluster IDs
        """
        # Convert distance matrix to condensed form
        dist_condensed = squareform(self.distance.values)

        # Hierarchical clustering
        Z = linkage(dist_condensed, method=method)

        # Determine clusters
        if n_clusters is not None:
            clusters = fcluster(Z, n_clusters, criterion='maxclust')
        elif distance_threshold is not None:
            clusters = fcluster(Z, distance_threshold, criterion='distance')
        else:
            # Default: use 5 clusters
            clusters = fcluster(Z, 5, criterion='maxclust')

        # Map to assets
        asset_clusters = {
            asset: int(cluster)
            for asset, cluster in zip(self.assets, clusters)
        }

        return asset_clusters

    def get_cluster_composition(
        self,
        clusters: Dict[str, int]
    ) -> Dict[int, List[str]]:
        """
        Get assets in each cluster.

        Args:
            clusters: Asset to cluster mapping

        Returns:
            Dictionary mapping cluster ID to list of assets
        """
        composition = defaultdict(list)
        for asset, cluster_id in clusters.items():
            composition[cluster_id].append(asset)

        return dict(composition)

    def compute_cluster_statistics(
        self,
        clusters: Dict[str, int]
    ) -> pd.DataFrame:
        """
        Compute statistics for each cluster.

        Args:
            clusters: Asset to cluster mapping

        Returns:
            DataFrame with cluster statistics
        """
        composition = self.get_cluster_composition(clusters)

        stats = []
        for cluster_id, assets in composition.items():
            # Intra-cluster correlation
            if len(assets) > 1:
                cluster_corr = self.network.correlation.loc[assets, assets]
                # Get upper triangle (excluding diagonal)
                mask = np.triu(np.ones_like(cluster_corr, dtype=bool), k=1)
                avg_corr = cluster_corr.values[mask].mean()
            else:
                avg_corr = 1.0

            # Cluster volatility (average of asset volatilities)
            cluster_returns = self.network.returns[assets]
            avg_vol = cluster_returns.std().mean() * np.sqrt(252)

            stats.append({
                'cluster_id': cluster_id,
                'n_assets': len(assets),
                'avg_intra_correlation': avg_corr,
                'avg_volatility': avg_vol,
                'assets': ', '.join(assets[:5]) + ('...' if len(assets) > 5 else '')
            })

        return pd.DataFrame(stats).set_index('cluster_id')

    def plot_dendrogram(
        self,
        method: str = 'ward',
        figsize: Tuple[int, int] = (12, 8)
    ) -> None:
        """
        Plot hierarchical clustering dendrogram.

        Args:
            method: Linkage method
            figsize: Figure size
        """
        dist_condensed = squareform(self.distance.values)
        Z = linkage(dist_condensed, method=method)

        plt.figure(figsize=figsize)
        dendrogram(Z, labels=self.assets, leaf_rotation=90)
        plt.title(f'Hierarchical Clustering Dendrogram ({method} linkage)')
        plt.xlabel('Asset')
        plt.ylabel('Distance')
        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    # Using network from previous examples
    community = CommunityDetection(network)

    # Louvain communities
    louvain_clusters = community.louvain_communities()
    print("Louvain Communities:")
    for asset, comm in sorted(louvain_clusters.items(), key=lambda x: x[1]):
        print(f"  {asset}: Community {comm}")

    # Hierarchical clustering
    hier_clusters = community.hierarchical_clustering(method='ward', n_clusters=4)

    # Cluster statistics
    stats = community.compute_cluster_statistics(hier_clusters)
    print("\nCluster Statistics:")
    print(stats)
```

---

## 4. Systemic Risk Analysis

### 4.1 Network-Based Risk Measures

**Absorption Ratio** (Kritzman et al., 2011):
$$AR = \frac{\sum_{i=1}^{k} \lambda_i}{\sum_{j=1}^{n} \lambda_j}$$

where $\lambda_i$ are eigenvalues of correlation matrix, sorted descending.

High AR indicates concentrated risk (assets moving together).

**Network Fragility**:
- Remove central nodes and measure network degradation
- Critical nodes cause rapid fragmentation

**Contagion Simulation**:
- Model shock propagation through network
- Identify cascade risk

### 4.2 Python Implementation

```python
class SystemicRiskAnalysis:
    """
    Network-based systemic risk analysis.

    Measures market fragility, contagion risk, and
    identifies systemically important assets.
    """

    def __init__(self, correlation_network: CorrelationNetwork):
        """
        Initialize with correlation network.

        Args:
            correlation_network: CorrelationNetwork instance
        """
        self.network = correlation_network
        self.correlation = correlation_network.correlation
        self.returns = correlation_network.returns

    def compute_absorption_ratio(self, n_components: int = None) -> float:
        """
        Compute absorption ratio.

        AR measures the fraction of total variance explained
        by a subset of principal components.

        Args:
            n_components: Number of PCs to include (default: 20% of assets)

        Returns:
            Absorption ratio (0 to 1)
        """
        # Eigenvalue decomposition
        eigenvalues = np.linalg.eigvalsh(self.correlation.values)
        eigenvalues = np.sort(eigenvalues)[::-1]  # Descending order

        if n_components is None:
            n_components = max(1, int(0.2 * len(eigenvalues)))

        total_variance = np.sum(eigenvalues)
        explained_variance = np.sum(eigenvalues[:n_components])

        return explained_variance / total_variance

    def rolling_absorption_ratio(
        self,
        window: int = 63,
        n_components: int = None
    ) -> pd.Series:
        """
        Compute rolling absorption ratio for regime detection.

        Args:
            window: Rolling window size (days)
            n_components: Number of PCs

        Returns:
            Series of absorption ratios
        """
        ar_values = []
        dates = self.returns.index[window:]

        for i in range(window, len(self.returns)):
            window_returns = self.returns.iloc[i-window:i]
            window_corr = window_returns.corr()

            eigenvalues = np.linalg.eigvalsh(window_corr.values)
            eigenvalues = np.sort(eigenvalues)[::-1]

            if n_components is None:
                n_comp = max(1, int(0.2 * len(eigenvalues)))
            else:
                n_comp = n_components

            ar = np.sum(eigenvalues[:n_comp]) / np.sum(eigenvalues)
            ar_values.append(ar)

        return pd.Series(ar_values, index=dates, name='absorption_ratio')

    def compute_network_fragility(
        self,
        G: nx.Graph = None,
        removal_order: str = 'betweenness'
    ) -> pd.DataFrame:
        """
        Measure network fragility by sequential node removal.

        Args:
            G: Network to analyze (uses threshold network if None)
            removal_order: Order to remove nodes ('betweenness', 'degree', 'random')

        Returns:
            DataFrame showing network degradation
        """
        if G is None:
            G = self.network.compute_threshold_network(threshold=0.3)

        G_copy = G.copy()

        # Compute removal order
        if removal_order == 'betweenness':
            centrality = nx.betweenness_centrality(G_copy)
        elif removal_order == 'degree':
            centrality = dict(G_copy.degree())
        else:
            centrality = {n: np.random.random() for n in G_copy.nodes()}

        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)

        # Track network properties during removal
        results = []

        for i, (node, _) in enumerate(sorted_nodes):
            # Current network metrics
            n_nodes = G_copy.number_of_nodes()
            n_edges = G_copy.number_of_edges()

            if n_nodes > 0:
                # Size of largest connected component
                if n_nodes > 1:
                    largest_cc = max(nx.connected_components(G_copy), key=len)
                    lcc_size = len(largest_cc) / n_nodes
                else:
                    lcc_size = 1.0
            else:
                lcc_size = 0

            results.append({
                'nodes_removed': i,
                'remaining_nodes': n_nodes,
                'remaining_edges': n_edges,
                'lcc_fraction': lcc_size,
                'removed_node': node
            })

            # Remove node
            if node in G_copy:
                G_copy.remove_node(node)

        return pd.DataFrame(results)

    def simulate_contagion(
        self,
        G: nx.Graph,
        initial_shock_node: str,
        shock_magnitude: float = 0.10,
        transmission_threshold: float = 0.5,
        max_iterations: int = 10
    ) -> Dict:
        """
        Simulate shock contagion through network.

        Args:
            G: Network with correlation edge weights
            initial_shock_node: Asset receiving initial shock
            shock_magnitude: Size of initial shock (as fraction)
            transmission_threshold: Correlation threshold for transmission
            max_iterations: Maximum propagation steps

        Returns:
            Contagion simulation results
        """
        # Initialize shock state
        shocked = {initial_shock_node}
        shock_levels = {initial_shock_node: shock_magnitude}

        propagation_history = [{
            'iteration': 0,
            'newly_shocked': [initial_shock_node],
            'total_shocked': 1
        }]

        for iteration in range(1, max_iterations + 1):
            newly_shocked = set()

            for shocked_node in shocked:
                current_shock = shock_levels.get(shocked_node, 0)

                # Propagate to neighbors
                for neighbor in G.neighbors(shocked_node):
                    if neighbor in shocked:
                        continue

                    # Get correlation
                    edge_data = G.get_edge_data(shocked_node, neighbor)
                    correlation = edge_data.get('correlation', 0)

                    # Transmit if correlation above threshold
                    if abs(correlation) >= transmission_threshold:
                        transmitted_shock = current_shock * abs(correlation)

                        if transmitted_shock > 0.01:  # Minimum shock threshold
                            newly_shocked.add(neighbor)
                            shock_levels[neighbor] = max(
                                shock_levels.get(neighbor, 0),
                                transmitted_shock
                            )

            if not newly_shocked:
                break

            shocked.update(newly_shocked)

            propagation_history.append({
                'iteration': iteration,
                'newly_shocked': list(newly_shocked),
                'total_shocked': len(shocked)
            })

        return {
            'total_affected': len(shocked),
            'affected_fraction': len(shocked) / G.number_of_nodes(),
            'shock_levels': shock_levels,
            'propagation_history': propagation_history
        }

    def identify_critical_nodes(
        self,
        G: nx.Graph = None,
        top_n: int = 5
    ) -> pd.DataFrame:
        """
        Identify critical nodes for systemic risk.

        Combines multiple measures: centrality, contagion potential,
        and network fragility contribution.

        Args:
            G: Network to analyze
            top_n: Number of critical nodes to return

        Returns:
            DataFrame with critical node analysis
        """
        if G is None:
            G = self.network.compute_threshold_network(threshold=0.3)

        # Compute centralities
        betweenness = nx.betweenness_centrality(G)
        eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
        degree = dict(G.degree())

        # Simulate contagion from each node
        contagion_scores = {}
        for node in G.nodes():
            result = self.simulate_contagion(G, node, shock_magnitude=0.10)
            contagion_scores[node] = result['affected_fraction']

        # Combine into DataFrame
        df = pd.DataFrame({
            'betweenness': betweenness,
            'eigenvector': eigenvector,
            'degree': degree,
            'contagion_reach': contagion_scores
        })

        # Normalize and compute composite score
        for col in df.columns:
            df[f'{col}_norm'] = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-10)

        df['critical_score'] = (
            0.3 * df['betweenness_norm'] +
            0.3 * df['eigenvector_norm'] +
            0.2 * df['degree_norm'] +
            0.2 * df['contagion_reach_norm']
        )

        return df.nlargest(top_n, 'critical_score')[
            ['betweenness', 'eigenvector', 'degree', 'contagion_reach', 'critical_score']
        ]


# Example usage
if __name__ == "__main__":
    risk_analysis = SystemicRiskAnalysis(network)

    # Absorption ratio
    ar = risk_analysis.compute_absorption_ratio()
    print(f"Absorption Ratio: {ar:.4f}")

    # Network fragility
    fragility = risk_analysis.compute_network_fragility(removal_order='betweenness')
    print("\nNetwork Fragility (first 5 removals):")
    print(fragility.head())

    # Critical nodes
    threshold_net = network.compute_threshold_network(threshold=0.3)
    critical = risk_analysis.identify_critical_nodes(threshold_net)
    print("\nCritical Nodes:")
    print(critical)
```

---

## 5. Portfolio Construction Applications

### 5.1 Hierarchical Risk Parity (HRP)

HRP uses hierarchical clustering to construct portfolios:

1. Build distance matrix from correlations
2. Perform hierarchical clustering
3. Quasi-diagonalize covariance matrix
4. Recursively bisect and allocate weights

See [HRP Documentation](../../02_signals/quantitative/portfolio_construction/hrp.md) for full implementation.

### 5.2 Network-Based Diversification

```python
class NetworkPortfolioConstruction:
    """
    Portfolio construction using network topology.
    """

    def __init__(self, correlation_network: CorrelationNetwork):
        """
        Initialize with correlation network.

        Args:
            correlation_network: CorrelationNetwork instance
        """
        self.network = correlation_network
        self.returns = correlation_network.returns

    def peripheral_diversification(
        self,
        G: nx.Graph = None,
        n_assets: int = 10
    ) -> Dict[str, float]:
        """
        Select peripheral (low centrality) assets for diversification.

        Args:
            G: Network to analyze
            n_assets: Number of assets to select

        Returns:
            Equal-weighted portfolio of peripheral assets
        """
        if G is None:
            G = self.network.compute_mst()

        # Get peripheral assets (low degree in MST)
        degree = dict(G.degree())
        sorted_assets = sorted(degree.items(), key=lambda x: x[1])

        selected = [asset for asset, _ in sorted_assets[:n_assets]]
        weight = 1.0 / len(selected)

        return {asset: weight for asset in selected}

    def community_balanced_portfolio(
        self,
        n_assets_per_community: int = 2
    ) -> Dict[str, float]:
        """
        Select assets balanced across network communities.

        Args:
            n_assets_per_community: Assets to select from each community

        Returns:
            Portfolio weights
        """
        mst = self.network.compute_mst()
        communities = nx.community.louvain_communities(mst, seed=42)

        selected = []
        for community in communities:
            # Select most central assets within community
            subgraph = mst.subgraph(community)
            centrality = nx.degree_centrality(subgraph)
            sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
            selected.extend([n for n, _ in sorted_nodes[:n_assets_per_community]])

        weight = 1.0 / len(selected)
        return {asset: weight for asset in selected}

    def mst_based_weights(self) -> Dict[str, float]:
        """
        Weight assets inversely to MST degree (hub penalty).

        Central hubs in MST get lower weights for diversification.

        Returns:
            Portfolio weights
        """
        mst = self.network.compute_mst()
        degree = dict(mst.degree())

        # Inverse degree weighting
        inv_degree = {k: 1.0 / (v + 1) for k, v in degree.items()}
        total = sum(inv_degree.values())

        return {k: v / total for k, v in inv_degree.items()}


# Example usage
if __name__ == "__main__":
    portfolio = NetworkPortfolioConstruction(network)

    # Peripheral diversification
    peripheral_weights = portfolio.peripheral_diversification(n_assets=5)
    print("Peripheral Portfolio:")
    for asset, weight in peripheral_weights.items():
        print(f"  {asset}: {weight:.2%}")

    # Community-balanced
    community_weights = portfolio.community_balanced_portfolio()
    print("\nCommunity-Balanced Portfolio:")
    for asset, weight in list(community_weights.items())[:10]:
        print(f"  {asset}: {weight:.2%}")
```

---

## 6. Dynamic Network Analysis

### 6.1 Rolling Networks

Track network evolution through time to detect regime changes:

```python
class DynamicNetworkAnalysis:
    """
    Analyze network evolution over time.
    """

    def __init__(self, returns: pd.DataFrame, window: int = 63):
        """
        Initialize with returns and rolling window.

        Args:
            returns: Asset returns DataFrame
            window: Rolling window size (days)
        """
        self.returns = returns
        self.window = window

    def compute_rolling_network_metrics(self) -> pd.DataFrame:
        """
        Compute network metrics over rolling windows.

        Returns:
            DataFrame with time series of network metrics
        """
        results = []

        for i in range(self.window, len(self.returns)):
            window_returns = self.returns.iloc[i-self.window:i]

            # Build network
            network = CorrelationNetwork(window_returns)
            mst = network.compute_mst()

            # Compute metrics
            metrics = network.get_network_metrics(mst)

            # Absorption ratio
            corr = window_returns.corr()
            eigenvalues = np.linalg.eigvalsh(corr.values)
            eigenvalues = np.sort(eigenvalues)[::-1]
            n_comp = max(1, int(0.2 * len(eigenvalues)))
            ar = np.sum(eigenvalues[:n_comp]) / np.sum(eigenvalues)

            results.append({
                'date': self.returns.index[i],
                'avg_clustering': metrics.avg_clustering,
                'avg_path_length': metrics.avg_path_length,
                'modularity': metrics.modularity,
                'n_communities': metrics.n_communities,
                'absorption_ratio': ar
            })

        return pd.DataFrame(results).set_index('date')

    def detect_network_regime_change(
        self,
        metrics_df: pd.DataFrame,
        z_threshold: float = 2.0
    ) -> pd.DataFrame:
        """
        Detect regime changes from network metric anomalies.

        Args:
            metrics_df: Rolling network metrics
            z_threshold: Z-score threshold for regime detection

        Returns:
            DataFrame with regime signals
        """
        signals = pd.DataFrame(index=metrics_df.index)

        for col in metrics_df.columns:
            rolling_mean = metrics_df[col].rolling(20).mean()
            rolling_std = metrics_df[col].rolling(20).std()

            z_score = (metrics_df[col] - rolling_mean) / (rolling_std + 1e-10)
            signals[f'{col}_zscore'] = z_score
            signals[f'{col}_regime_change'] = np.abs(z_score) > z_threshold

        return signals
```

---

## 7. Integration with Trading Systems

### 7.1 Ordinis Integration Points

```python
# Example: Network-based risk monitoring in RiskGuard

from ordinis.risk import RiskEngine

class NetworkRiskMonitor:
    """
    Real-time network-based risk monitoring.
    """

    def __init__(self, lookback_window: int = 63):
        self.lookback = lookback_window
        self.returns_buffer = []

    def update(self, returns: pd.Series) -> Dict:
        """
        Update risk metrics with new returns.

        Args:
            returns: Today's asset returns

        Returns:
            Risk signals
        """
        self.returns_buffer.append(returns)

        if len(self.returns_buffer) < self.lookback:
            return {'status': 'warming_up'}

        # Keep only lookback window
        self.returns_buffer = self.returns_buffer[-self.lookback:]

        # Build DataFrame
        returns_df = pd.DataFrame(self.returns_buffer)

        # Compute network metrics
        network = CorrelationNetwork(returns_df)
        risk_analysis = SystemicRiskAnalysis(network)

        ar = risk_analysis.compute_absorption_ratio()

        # Generate signals
        signals = {
            'absorption_ratio': ar,
            'risk_level': 'elevated' if ar > 0.7 else 'normal',
            'timestamp': pd.Timestamp.now()
        }

        if ar > 0.8:
            signals['action'] = 'reduce_exposure'

        return signals
```

---

## 8. Academic References

### Foundational Papers

1. **Mantegna, R. N. (1999)**. "Hierarchical Structure in Financial Markets." *European Physical Journal B*, 11, 193-197.
   - Introduces correlation-based MST for financial markets

2. **Tumminello, M., et al. (2005)**. "A Tool for Filtering Information in Complex Systems." *PNAS*, 102(30), 10421-10426.
   - Planar Maximally Filtered Graph (PMFG)

3. **Onnela, J.-P., et al. (2003)**. "Dynamics of Market Correlations: Taxonomy and Portfolio Analysis." *Physical Review E*, 68, 056110.
   - Dynamic correlation networks

4. **Kritzman, M., et al. (2011)**. "Principal Components as a Measure of Systemic Risk." *Journal of Portfolio Management*, 37(4), 112-126.
   - Absorption ratio for systemic risk

5. **Billio, M., et al. (2012)**. "Econometric Measures of Connectedness and Systemic Risk in the Finance and Insurance Sectors." *Journal of Financial Economics*, 104(3), 535-559.
   - Granger causality networks, systemic risk measures

### Network Science

6. **Newman, M. E. J. (2010)**. *Networks: An Introduction*. Oxford University Press.
   - Comprehensive network science textbook

7. **Barabási, A.-L. (2016)**. *Network Science*. Cambridge University Press.
   - Modern network science reference

### Portfolio Applications

8. **López de Prado, M. (2016)**. "Building Diversified Portfolios that Outperform Out of Sample." *Journal of Portfolio Management*, 42(4), 59-69.
   - Hierarchical Risk Parity (HRP)

9. **Peralta, G., & Zareei, A. (2016)**. "A Network Approach to Portfolio Selection." *Journal of Empirical Finance*, 38, 157-180.
   - Network-based portfolio optimization

---

## 9. Cross-References

**Related Knowledge Base Sections**:

- [Extreme Value Theory](extreme_value_theory.md) - Tail dependence, copulas
- [Causal Inference](causal_inference.md) - Granger causality networks
- [Information Theory](information_theory.md) - Mutual information networks
- [HRP Portfolio Construction](../../02_signals/quantitative/portfolio_construction/hrp.md) - Hierarchical clustering
- [Advanced Risk Methods](../../03_risk/advanced_risk_methods.md) - Systemic risk measures

**Integration Points**:

1. **RiskGuard**: Systemic risk monitoring, absorption ratio alerts
2. **SignalCore**: Network-based regime detection signals
3. **Cortex**: Community structure for sector rotation
4. **ProofBench**: Network stability in backtests

---

## Document Metadata

```yaml
version: "1.0.0"
created: "2025-12-12"
last_updated: "2025-12-12"
status: "published"
category: "foundations/advanced_mathematics"
tags: ["network-theory", "correlation-networks", "mst", "systemic-risk", "community-detection", "centrality", "portfolio-construction"]
code_lines: 750
academic_references: 9
implementation_completeness: "production-ready"
```

---

**END OF DOCUMENT**
