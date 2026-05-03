"""Run the full Wikipedia experiment pipeline.

Produces results for the paper's Section 5.2:
1. Cluster discovery on real data (do velocity-volatility clusters emerge?)
2. Survival fitting per discovered cluster
3. Distribution comparison on real data
4. Retrieval comparison (6 conditions)
5. Interpretability analysis (what do the clusters correspond to?)
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.clustering.temporal_clusters import (
    build_predicate_features,
    cluster_hdbscan,
    cluster_dpgmm,
    ClusterResult,
)
from src.models.bayesian.survival_model import (
    fit_weibull,
    compare_distributions,
    compute_survival_weight,
    SurvivalFit,
)
from src.retrieval.temporal_retrieval import (
    score_edges,
    ndcg_at_k,
    mrr,
    precision_at_k,
    TemporalQuery,
    RetrievalMetrics,
)

logger = logging.getLogger(__name__)


def load_wikipedia_kg() -> pd.DataFrame:
    """Load the Wikipedia temporal KG from disk."""
    path = Path("data/wikipedia/temporal_kg.parquet")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run src.wikipedia.pipeline first to build it."
        )
    return pd.read_parquet(path)


def assign_clusters_to_edges(
    df: pd.DataFrame,
    pred_df: pd.DataFrame,
    cluster_result: ClusterResult,
) -> pd.DataFrame:
    """Map predicate-level cluster labels back to edge-level."""
    pred_to_cluster = dict(zip(pred_df["predicate"], cluster_result.labels))
    df = df.copy()
    df["discovered_cluster"] = df["predicate"].map(pred_to_cluster).fillna(-1).astype(int)
    return df


def fit_per_cluster_survival(
    df: pd.DataFrame,
    cluster_col: str = "discovered_cluster",
) -> dict[int, SurvivalFit]:
    """Fit Weibull survival to each discovered cluster."""
    fits = {}
    for cluster_id in sorted(df[cluster_col].unique()):
        if cluster_id < 0:
            continue
        cluster_df = df[df[cluster_col] == cluster_id]
        durations = cluster_df["lifetime_observed"].values
        events = (~cluster_df["is_censored"]).astype(int).values
        if len(durations) >= 5 and events.sum() >= 2:
            fits[cluster_id] = fit_weibull(durations, events)
    return fits


def generate_wikipedia_queries(
    df: pd.DataFrame,
    n_queries: int = 100,
    seed: int = 42,
) -> list[TemporalQuery]:
    """Generate temporal queries for Wikipedia KG.

    Query: "what is the current value of [predicate] for [article]?"
    at a random time point. Ground truth: edges current at query time.
    """
    rng = np.random.default_rng(seed)
    max_time = df["timestamp"].max()
    min_time = df["timestamp"].min()
    mid_time = (max_time + min_time) / 2

    # Get unique (subject, predicate) concepts
    concepts = df.groupby(["subject", "predicate"]).size().reset_index(name="count")
    # Only concepts with some supersession history
    concepts_with_history = concepts[concepts["count"] >= 3]

    if len(concepts_with_history) == 0:
        concepts_with_history = concepts[concepts["count"] >= 2]

    queries = []
    for i in range(min(n_queries, len(concepts_with_history))):
        row = concepts_with_history.iloc[i % len(concepts_with_history)]
        subject = row["subject"]
        predicate = row["predicate"]

        # Query time in second half of the timeline
        q_time = rng.uniform(mid_time, max_time)

        # Find relevant edges
        concept_edges = df[(df["subject"] == subject) & (df["predicate"] == predicate)]
        relevant = {}
        for _, edge in concept_edges.iterrows():
            if edge["timestamp"] > q_time:
                continue
            sup_at = edge["superseded_at"]
            if pd.notna(sup_at) and sup_at <= q_time:
                relevant[int(edge["edge_id"])] = 0.0
            else:
                relevant[int(edge["edge_id"])] = 1.0

        if sum(v > 0 for v in relevant.values()) > 0:
            queries.append(TemporalQuery(
                query_id=f"wq_{i}_{subject}::{predicate}",
                query_time=q_time,
                relevant_edges=relevant,
            ))

    return queries


def run_wikipedia_retrieval_comparison(
    df: pd.DataFrame,
    queries: list[TemporalQuery],
    cluster_fits: dict[int, SurvivalFit],
    cluster_col: str = "discovered_cluster",
) -> dict[str, RetrievalMetrics]:
    """Compare retrieval strategies on Wikipedia data."""
    edge_ids = df["edge_id"].values
    edge_timestamps = df["timestamp"].values
    n_edges = len(df)

    rng = np.random.default_rng(42)
    base_sims = rng.uniform(0.1, 0.4, n_edges)

    # Build concept lookup for semantic sim
    edge_concepts = df["subject"].values + "::" + df["predicate"].values

    def make_sim_fn():
        def sim_fn(query_id):
            # Extract concept from query_id: "wq_{i}_{subject}::{predicate}"
            parts = query_id.split("_", 2)
            query_concept = parts[2] if len(parts) >= 3 else ""
            sims = base_sims.copy()
            mask = edge_concepts == query_concept
            sims[mask] = 1.0
            # Same article but different predicate gets partial match
            query_article = query_concept.split("::")[0] if "::" in query_concept else ""
            article_mask = np.array([c.startswith(query_article + "::") for c in edge_concepts])
            sims[article_mask & ~mask] = np.maximum(sims[article_mask & ~mask], 0.6)
            return sims
        return sim_fn

    sim_fn = make_sim_fn()
    results = {}

    def run_queries(taus, shapes, alpha, beta, name):
        all_ndcg5, all_ndcg10, all_mrr, all_p5, all_p10 = [], [], [], [], []
        for query in queries:
            ages = query.query_time - edge_timestamps
            sims = sim_fn(query.query_id)
            scores = score_edges(ages, sims, taus, shapes, alpha, beta)
            relevances = np.zeros(n_edges)
            for i, eid in enumerate(edge_ids):
                relevances[i] = query.relevant_edges.get(int(eid), 0.0)
            all_ndcg5.append(ndcg_at_k(scores, relevances, k=5))
            all_ndcg10.append(ndcg_at_k(scores, relevances, k=10))
            all_mrr.append(mrr(scores, relevances))
            all_p5.append(precision_at_k(scores, relevances, k=5))
            all_p10.append(precision_at_k(scores, relevances, k=10))
        results[name] = RetrievalMetrics(
            ndcg_5=float(np.mean(all_ndcg5)),
            ndcg_10=float(np.mean(all_ndcg10)),
            mrr_val=float(np.mean(all_mrr)),
            precision_5=float(np.mean(all_p5)),
            precision_10=float(np.mean(all_p10)),
            n_queries=len(queries),
        )

    # Baseline 1: No temporal weighting
    run_queries(
        np.ones(n_edges) * 1e10, np.ones(n_edges),
        alpha=1.0, beta=0.0, name="no_temporal",
    )

    # Baseline 2: Uniform exponential
    median_life = np.median(df["lifetime_observed"].values)
    run_queries(
        np.ones(n_edges) * median_life, np.ones(n_edges),
        alpha=1.0, beta=1.0, name="uniform_exponential",
    )

    # Baseline 3: Uniform half-life
    run_queries(
        np.ones(n_edges) * median_life * 0.5, np.ones(n_edges),
        alpha=1.0, beta=1.0, name="uniform_halflife",
    )

    # Type-level heterogeneous: use discovered cluster fits
    taus_type = np.ones(n_edges) * median_life  # default
    shapes_type = np.ones(n_edges)
    for cluster_id, fit in cluster_fits.items():
        mask = df[cluster_col].values == cluster_id
        if np.isfinite(fit.tau) and fit.tau > 0:
            taus_type[mask] = fit.tau
            shapes_type[mask] = fit.shape

    run_queries(taus_type, shapes_type, alpha=1.0, beta=1.0, name="type_level")

    return results


def run_full_wikipedia_experiment(verbose: bool = True) -> dict:
    """Run the complete Wikipedia experiment pipeline."""
    results = {}

    # ===== 1. Load data =====
    if verbose:
        print("=" * 60)
        print("STEP 1: Loading Wikipedia temporal KG")
        print("=" * 60)

    df = load_wikipedia_kg()
    if verbose:
        print(f"  Loaded {len(df)} edges, {df['subject'].nunique()} articles, "
              f"{df['predicate'].nunique()} predicates")
        print(f"  Supersession rate: {(~df['is_censored']).mean():.1%}")

    results["data"] = {
        "total_edges": len(df),
        "articles": int(df["subject"].nunique()),
        "predicates": int(df["predicate"].nunique()),
        "supersession_rate": float((~df["is_censored"]).mean()),
    }

    # ===== 2. Cluster discovery =====
    if verbose:
        print("\n" + "=" * 60)
        print("STEP 2: Cluster discovery on Wikipedia predicates")
        print("=" * 60)

    pred_df, features = build_predicate_features(df)

    # HDBSCAN
    hdbscan_result = cluster_hdbscan(features, min_cluster_size=3, min_samples=2)
    if verbose:
        noise = (hdbscan_result.labels == -1).sum()
        print(f"  HDBSCAN: {hdbscan_result.n_clusters} clusters, "
              f"silhouette={hdbscan_result.silhouette:.3f}" if hdbscan_result.silhouette else "",
              f"noise={noise}")

    # DPGMM
    dpgmm_result = cluster_dpgmm(features)
    if verbose:
        print(f"  DPGMM: {dpgmm_result.n_clusters} clusters"
              + (f", silhouette={dpgmm_result.silhouette:.3f}" if dpgmm_result.silhouette else ""))

    # Use DPGMM clusters (assigns every point, no noise)
    cluster_result = dpgmm_result
    df = assign_clusters_to_edges(df, pred_df, cluster_result)

    results["clustering"] = {
        "hdbscan": {
            "n_clusters": hdbscan_result.n_clusters,
            "silhouette": hdbscan_result.silhouette,
            "noise_pct": float((hdbscan_result.labels == -1).mean()),
        },
        "dpgmm": {
            "n_clusters": dpgmm_result.n_clusters,
            "silhouette": dpgmm_result.silhouette,
        },
    }

    # ===== 3. Cluster interpretation =====
    if verbose:
        print("\n" + "=" * 60)
        print("STEP 3: Cluster interpretation")
        print("=" * 60)

    cluster_profiles = {}
    for cid in sorted(df["discovered_cluster"].unique()):
        if cid < 0:
            continue
        cdf = df[df["discovered_cluster"] == cid]
        profile = {
            "n_edges": len(cdf),
            "n_predicates": int(cdf["predicate"].nunique()),
            "mean_volatility": float(cdf["volatility"].mean()),
            "mean_velocity": float(cdf["velocity"].mean()),
            "median_lifetime_days": float(cdf["lifetime_observed"].median()),
            "supersession_rate": float((~cdf["is_censored"]).mean()),
            "sample_predicates": cdf["predicate"].value_counts().head(5).index.tolist(),
        }
        cluster_profiles[int(cid)] = profile

        if verbose:
            print(f"\n  Cluster {cid}: {profile['n_edges']} edges, {profile['n_predicates']} predicates")
            print(f"    volatility={profile['mean_volatility']:.3f}, "
                  f"velocity={profile['mean_velocity']:.3f}")
            print(f"    median_lifetime={profile['median_lifetime_days']:.0f}d, "
                  f"supersession_rate={profile['supersession_rate']:.2f}")
            print(f"    sample predicates: {profile['sample_predicates'][:3]}")

    results["cluster_profiles"] = cluster_profiles

    # ===== 4. Survival fitting =====
    if verbose:
        print("\n" + "=" * 60)
        print("STEP 4: Survival fitting per cluster")
        print("=" * 60)

    cluster_fits = fit_per_cluster_survival(df)

    results["survival"] = {}
    for cid, fit in sorted(cluster_fits.items()):
        results["survival"][int(cid)] = {
            "tau": fit.tau,
            "shape": fit.shape,
            "n_observations": fit.n_observations,
            "aic": fit.aic,
        }
        if verbose:
            lindy = "Lindy" if fit.shape < 1 else ("aging" if fit.shape > 1 else "memoryless")
            print(f"  Cluster {cid}: tau={fit.tau:.1f}d, shape={fit.shape:.3f} ({lindy}), "
                  f"n={fit.n_observations}")

    # Distribution comparison per cluster
    if verbose:
        print("\n  Distribution comparison (AIC):")
    results["distribution_comparison"] = {}
    for cid in sorted(df["discovered_cluster"].unique()):
        if cid < 0:
            continue
        cdf = df[df["discovered_cluster"] == cid]
        dists = compare_distributions(
            cdf["lifetime_observed"].values,
            (~cdf["is_censored"]).astype(int).values,
        )
        if dists:
            best = min(dists, key=dists.get)
            results["distribution_comparison"][int(cid)] = {
                "aic": {k: float(v) for k, v in dists.items()},
                "best": best,
            }
            if verbose:
                print(f"    Cluster {cid}: best={best}, "
                      f"weibull={dists.get('weibull', 'N/A'):.0f}, "
                      f"lognormal={dists.get('lognormal', 'N/A'):.0f}")

    # ===== 5. Retrieval comparison =====
    if verbose:
        print("\n" + "=" * 60)
        print("STEP 5: Retrieval comparison")
        print("=" * 60)

    queries = generate_wikipedia_queries(df, n_queries=200)
    retrieval_results = run_wikipedia_retrieval_comparison(df, queries, cluster_fits)

    results["retrieval"] = {"n_queries": len(queries)}

    if verbose:
        print(f"\n  {'Method':<25} {'NDCG@5':>8} {'NDCG@10':>8} {'MRR':>8} {'P@5':>8} {'P@10':>8}")
        print("  " + "-" * 75)

    for method, metrics in retrieval_results.items():
        results["retrieval"][method] = {
            "ndcg_5": metrics.ndcg_5,
            "ndcg_10": metrics.ndcg_10,
            "mrr": metrics.mrr_val,
            "precision_5": metrics.precision_5,
            "precision_10": metrics.precision_10,
        }
        if verbose:
            print(f"  {method:<25} {metrics.ndcg_5:>8.3f} {metrics.ndcg_10:>8.3f} "
                  f"{metrics.mrr_val:>8.3f} {metrics.precision_5:>8.3f} {metrics.precision_10:>8.3f}")

    return results


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    results = run_full_wikipedia_experiment(verbose=True)

    import os
    os.makedirs("results", exist_ok=True)
    with open("results/wikipedia_experiment.json", "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    print("\n\nResults saved to results/wikipedia_experiment.json")
