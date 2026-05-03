"""Temporal cluster discovery from edge behavior profiles.

Discovers temporal types by clustering edges based on their
velocity-volatility profiles and observed temporal behavior.
Supports HDBSCAN and DPGMM (Dirichlet Process Gaussian Mixture Model).
"""

from dataclasses import dataclass

import hdbscan
import numpy as np
import pandas as pd
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)


@dataclass(frozen=True)
class ClusterResult:
    """Result of temporal cluster discovery."""

    labels: np.ndarray  # cluster assignment per edge
    n_clusters: int
    method: str
    # Cluster centroids in feature space
    centroids: np.ndarray | None = None
    # Quality metrics
    silhouette: float | None = None
    # If ground truth available
    ari: float | None = None
    nmi: float | None = None


def build_temporal_features(df: pd.DataFrame) -> np.ndarray:
    """Build the temporal behavior feature vector for each edge.

    Features (matching paper Section 3.8):
    - velocity: observation frequency of this concept
    - volatility: mean value change between observations
    - log_mean_lifetime: log of average lifetime for edges with this predicate
    - supersession_rate: fraction of edges superseded for this predicate
    - reinforcement_density: reinforcements per unit observed time
    """
    features = df[["velocity", "volatility"]].copy()

    # Log mean lifetime per predicate (log-transform to compress huge range)
    pred_lifetime = df.groupby("predicate")["lifetime_observed"].transform("mean")
    features["log_pred_mean_lifetime"] = np.log1p(pred_lifetime)

    # Fraction of edges superseded per predicate (proxy for change fraction)
    pred_supersession_rate = df.groupby("predicate")["is_censored"].transform(
        lambda x: 1 - x.mean()
    )
    features["pred_supersession_rate"] = pred_supersession_rate

    # Reinforcement density (reinforcements per unit observed time)
    features["reinforcement_density"] = df["n_reinforcements"] / df["lifetime_observed"].clip(lower=1.0)

    return features.values


def build_predicate_features(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Aggregate temporal features to the predicate level for clustering.

    Clustering should operate on predicates (or concepts), not individual edges,
    since edges from the same predicate share temporal behavior by design.

    Returns:
        predicate_df: DataFrame with one row per predicate and its ground truth cluster
        features: feature matrix for clustering
    """
    agg = df.groupby("predicate").agg(
        velocity=("velocity", "mean"),
        volatility=("volatility", "mean"),
        mean_lifetime=("lifetime_observed", "mean"),
        median_lifetime=("lifetime_observed", "median"),
        supersession_rate=("is_censored", lambda x: 1 - x.mean()),
        mean_reinforcements=("n_reinforcements", "mean"),
        n_edges=("edge_id", "count"),
        cluster=("cluster", "first"),  # ground truth (all edges of a predicate share the cluster)
    ).reset_index()

    features = np.column_stack([
        agg["velocity"].values,
        agg["volatility"].values,
        np.log1p(agg["mean_lifetime"].values),
        agg["supersession_rate"].values,
        agg["mean_reinforcements"].values / agg["mean_lifetime"].values.clip(min=1.0),
    ])

    return agg, features


def cluster_hdbscan(
    features: np.ndarray,
    min_cluster_size: int = 50,
    min_samples: int = 10,
) -> ClusterResult:
    """Discover temporal clusters using HDBSCAN.

    HDBSCAN is density-based and automatically determines the number of clusters.
    It can identify noise points (labeled -1) that don't belong to any cluster.
    """
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(features_scaled)

    n_clusters = len(set(labels) - {-1})

    # Compute centroids (excluding noise)
    centroids = None
    if n_clusters > 0:
        centroids = np.array([
            features_scaled[labels == k].mean(axis=0)
            for k in range(n_clusters)
        ])

    # Silhouette score on a subsample (only if >1 cluster and not all noise)
    sil = None
    valid_mask = labels >= 0
    if n_clusters > 1 and valid_mask.sum() > n_clusters:
        valid_features = features_scaled[valid_mask]
        valid_labels = labels[valid_mask]
        sil_n = min(len(valid_labels), 10000)
        if sil_n < len(valid_labels):
            sil_idx = np.random.default_rng(42).choice(len(valid_labels), size=sil_n, replace=False)
            sil = silhouette_score(valid_features[sil_idx], valid_labels[sil_idx])
        else:
            sil = silhouette_score(valid_features, valid_labels)

    return ClusterResult(
        labels=labels,
        n_clusters=n_clusters,
        method="hdbscan",
        centroids=centroids,
        silhouette=sil,
    )


def cluster_dpgmm(
    features: np.ndarray,
    max_components: int = 10,
    max_fit_samples: int = 10000,
    random_state: int = 42,
) -> ClusterResult:
    """Discover temporal clusters using Dirichlet Process Gaussian Mixture Model.

    DPGMM is Bayesian nonparametric -- it automatically infers the number
    of clusters up to max_components. Components with negligible weight
    are effectively pruned.

    For large datasets, fits on a subsample and predicts on all data.
    """
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    rng = np.random.default_rng(random_state)

    # Subsample for fitting if dataset is large
    n = features_scaled.shape[0]
    if n > max_fit_samples:
        fit_idx = rng.choice(n, size=max_fit_samples, replace=False)
        fit_data = features_scaled[fit_idx]
    else:
        fit_data = features_scaled

    dpgmm = BayesianGaussianMixture(
        n_components=max_components,
        covariance_type="full",
        weight_concentration_prior_type="dirichlet_process",
        weight_concentration_prior=1.0,
        max_iter=500,
        random_state=random_state,
        n_init=3,
        reg_covar=1e-4,  # regularization to prevent singular covariance
    )
    dpgmm.fit(fit_data)

    # Predict on full dataset
    labels = dpgmm.predict(features_scaled)

    # Count effective clusters (those with meaningful weight)
    weights = dpgmm.weights_
    effective_mask = weights > 0.01  # threshold for "active" component
    n_clusters = int(effective_mask.sum())

    # Centroids from the mixture means
    centroids = dpgmm.means_[effective_mask]

    # Silhouette score on a subsample (expensive for large n)
    sil = None
    if n_clusters > 1:
        sil_n = min(n, 10000)
        sil_idx = rng.choice(n, size=sil_n, replace=False)
        sil = silhouette_score(features_scaled[sil_idx], labels[sil_idx])

    return ClusterResult(
        labels=labels,
        n_clusters=n_clusters,
        method="dpgmm",
        centroids=centroids,
        silhouette=sil,
    )


def evaluate_clusters(
    result: ClusterResult,
    ground_truth_labels: np.ndarray,
) -> ClusterResult:
    """Evaluate clustering against ground truth labels.

    Computes ARI and NMI. For HDBSCAN, noise points (label=-1) are
    excluded from the evaluation.
    """
    pred = result.labels
    true = ground_truth_labels

    # For HDBSCAN, exclude noise points from evaluation
    if result.method == "hdbscan":
        valid = pred >= 0
        pred = pred[valid]
        true = true[valid]

    ari = adjusted_rand_score(true, pred)
    nmi = normalized_mutual_info_score(true, pred)

    return ClusterResult(
        labels=result.labels,
        n_clusters=result.n_clusters,
        method=result.method,
        centroids=result.centroids,
        silhouette=result.silhouette,
        ari=ari,
        nmi=nmi,
    )


def encode_ground_truth(df: pd.DataFrame) -> np.ndarray:
    """Encode ground truth cluster names as integer labels."""
    cluster_names = sorted(df["cluster"].unique())
    name_to_id = {name: i for i, name in enumerate(cluster_names)}
    return df["cluster"].map(name_to_id).values


def discover_and_evaluate(
    df: pd.DataFrame,
    methods: list[str] | None = None,
    level: str = "predicate",
) -> dict[str, ClusterResult]:
    """Run cluster discovery and evaluate against planted ground truth.

    Args:
        df: Synthetic KG DataFrame
        methods: Clustering methods to use
        level: "predicate" (aggregate first, recommended) or "edge" (cluster raw edges)

    Returns a dict mapping method name to ClusterResult with evaluation metrics.
    """
    if methods is None:
        methods = ["hdbscan", "dpgmm"]

    if level == "predicate":
        pred_df, features = build_predicate_features(df)
        cluster_names = sorted(pred_df["cluster"].unique())
        name_to_id = {name: i for i, name in enumerate(cluster_names)}
        ground_truth = pred_df["cluster"].map(name_to_id).values
    else:
        features = build_temporal_features(df)
        ground_truth = encode_ground_truth(df)

    results = {}

    for method in methods:
        if method == "hdbscan":
            # For predicate-level (small n), use smaller min_cluster_size
            min_cs = 3 if level == "predicate" else 50
            result = cluster_hdbscan(features, min_cluster_size=min_cs, min_samples=2 if level == "predicate" else 10)
        elif method == "dpgmm":
            result = cluster_dpgmm(features)
        else:
            raise ValueError(f"Unknown clustering method: {method}")

        result = evaluate_clusters(result, ground_truth)
        results[method] = result

    return results


if __name__ == "__main__":
    from src.synthetic.generator import generate_synthetic_kg

    print("Generating synthetic KG...")
    df = generate_synthetic_kg()

    print(f"\nRunning predicate-level cluster discovery ({df['predicate'].nunique()} predicates from {len(df)} edges)...")
    results = discover_and_evaluate(df, level="predicate")

    for method, result in results.items():
        print(f"\n{method.upper()}:")
        print(f"  Clusters found: {result.n_clusters}")
        if result.silhouette is not None:
            print(f"  Silhouette: {result.silhouette:.3f}")
        if result.ari is not None:
            print(f"  ARI: {result.ari:.3f}")
        if result.nmi is not None:
            print(f"  NMI: {result.nmi:.3f}")
        if method == "hdbscan":
            noise = (result.labels == -1).sum()
            print(f"  Noise points: {noise} ({100*noise/len(result.labels):.1f}%)")
