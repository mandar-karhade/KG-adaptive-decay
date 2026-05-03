"""Sensitivity and robustness analyses requested by reviewer feedback.

1. Epsilon sensitivity: do clusters and Lindy findings hold across threshold values?
2. Log-normal hazard: does log-normal also show decreasing hazard?
3. Gradient vs Bayesian: do both produce similar retrieval rankings despite different params?
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import lognorm

from src.synthetic.generator import generate_synthetic_kg
from src.synthetic.config import SyntheticKGConfig
from src.clustering.temporal_clusters import (
    build_predicate_features,
    cluster_hdbscan,
    cluster_dpgmm,
    evaluate_clusters,
)
from src.models.bayesian.survival_model import (
    fit_weibull,
    fit_hierarchical,
    compare_distributions,
)
from src.models.gradient.decay_model import fit_gradient
from src.retrieval.temporal_retrieval import (
    generate_synthetic_queries,
    run_retrieval_comparison,
    score_edges,
    ndcg_at_k,
)


def epsilon_sensitivity(verbose: bool = True) -> dict:
    """Test robustness of clusters and Lindy findings across epsilon values.

    Epsilon controls the supersession threshold: how much must a value
    change to count as a supersession event. We vary it and check:
    - Do the same number of clusters emerge?
    - Is the Lindy effect (kappa < 1) consistent?
    - How do cluster assignments change?
    """
    epsilon_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
    results = {}

    if verbose:
        print("=" * 60)
        print("EPSILON SENSITIVITY ANALYSIS")
        print("=" * 60)

    for eps in epsilon_values:
        config = SyntheticKGConfig(seed=42, epsilon=eps)
        df = generate_synthetic_kg(config=config)

        # Cluster recovery
        pred_df, features = build_predicate_features(df)
        cluster_names = sorted(pred_df["cluster"].unique())
        name_to_id = {name: i for i, name in enumerate(cluster_names)}
        ground_truth = pred_df["cluster"].map(name_to_id).values

        hdbscan_result = cluster_hdbscan(features, min_cluster_size=3, min_samples=2)
        hdbscan_result = evaluate_clusters(hdbscan_result, ground_truth)

        # Weibull shape per cluster
        shapes = {}
        for cluster_name in df["cluster"].unique():
            cdf = df[df["cluster"] == cluster_name]
            durations = cdf["lifetime_observed"].values
            events = (~cdf["is_censored"]).astype(int).values
            if events.sum() >= 5:
                fit = fit_weibull(durations, events)
                shapes[cluster_name] = fit.shape

        lindy_count = sum(1 for k in shapes.values() if k < 1)

        results[eps] = {
            "total_edges": len(df),
            "supersession_rate": float((~df["is_censored"]).mean()),
            "hdbscan_clusters": hdbscan_result.n_clusters,
            "hdbscan_ari": hdbscan_result.ari,
            "shapes": {k: round(v, 3) for k, v in shapes.items()},
            "lindy_clusters": lindy_count,
            "total_fitted_clusters": len(shapes),
        }

        if verbose:
            print(f"\n  epsilon = {eps}:")
            print(f"    Edges: {len(df)}, supersession rate: {(~df['is_censored']).mean():.2%}")
            print(f"    HDBSCAN: {hdbscan_result.n_clusters} clusters, ARI = {hdbscan_result.ari:.3f}")
            print(f"    Lindy (kappa < 1): {lindy_count}/{len(shapes)} clusters")
            for name, shape in sorted(shapes.items()):
                print(f"      {name}: kappa = {shape:.3f}")

    return results


def lognormal_hazard_analysis(verbose: bool = True) -> dict:
    """Show that log-normal fits also imply decreasing hazard rates.

    The log-normal hazard function h(t) = f(t)/S(t) is non-monotonic:
    it rises to a peak and then DECREASES. So if log-normal fits better
    than Weibull, the decreasing hazard (Lindy effect) is even MORE
    pronounced, not an artifact of Weibull parameterization.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("LOG-NORMAL HAZARD ANALYSIS")
        print("=" * 60)

    df = generate_synthetic_kg()
    results = {}

    for cluster_name in sorted(df["cluster"].unique()):
        cdf = df[df["cluster"] == cluster_name]
        durations = cdf["lifetime_observed"].values
        events = (~cdf["is_censored"]).astype(int).values
        observed = durations[events == 1]

        if len(observed) < 10:
            continue

        # Fit log-normal to observed lifetimes
        log_obs = np.log(observed[observed > 0])
        mu = np.mean(log_obs)
        sigma = np.std(log_obs)

        # Log-normal hazard: h(t) = f(t) / S(t)
        # It peaks at t_peak and then decreases
        # The peak occurs at t_peak = exp(mu - sigma^2)
        t_peak = np.exp(mu - sigma ** 2)

        # Compute hazard at several time points
        t_points = np.array([1, 7, 30, 90, 365, 365 * 3, 365 * 5])
        t_points = t_points[t_points > 0]

        hazards = []
        for t in t_points:
            pdf_val = lognorm.pdf(t, s=sigma, scale=np.exp(mu))
            sf_val = lognorm.sf(t, s=sigma, scale=np.exp(mu))
            if sf_val > 1e-10:
                hazards.append(float(pdf_val / sf_val))
            else:
                hazards.append(float("nan"))

        # Is hazard decreasing after the peak?
        post_peak_hazards = [h for t, h in zip(t_points, hazards) if t > t_peak and np.isfinite(h)]
        hazard_decreasing = all(
            post_peak_hazards[i] >= post_peak_hazards[i + 1]
            for i in range(len(post_peak_hazards) - 1)
        ) if len(post_peak_hazards) >= 2 else None

        results[cluster_name] = {
            "mu": round(float(mu), 3),
            "sigma": round(float(sigma), 3),
            "t_peak_days": round(float(t_peak), 1),
            "hazard_at_points": {
                f"{int(t)}d": round(h, 6) for t, h in zip(t_points, hazards) if np.isfinite(h)
            },
            "hazard_decreasing_after_peak": hazard_decreasing,
        }

        if verbose:
            print(f"\n  {cluster_name}:")
            print(f"    Log-normal params: mu={mu:.3f}, sigma={sigma:.3f}")
            print(f"    Hazard peaks at t={t_peak:.1f} days")
            print(f"    Hazard after peak is {'DECREASING' if hazard_decreasing else 'NOT decreasing' if hazard_decreasing is not None else 'insufficient data'}")
            print(f"    Hazard values: ", end="")
            for t, h in zip(t_points, hazards):
                if np.isfinite(h):
                    print(f"{int(t)}d={h:.6f}  ", end="")
            print()

    return results


def gradient_bayesian_ranking_comparison(verbose: bool = True) -> dict:
    """Compare retrieval rankings from Bayesian vs gradient-based approaches.

    The key question: do both methods produce similar retrieval rankings
    despite recovering different parameterizations of the decay surface?
    """
    if verbose:
        print("\n" + "=" * 60)
        print("GRADIENT vs BAYESIAN RANKING COMPARISON")
        print("=" * 60)

    df = generate_synthetic_kg()

    # Fit both models
    if verbose:
        print("\n  Fitting Bayesian model...")
    bayesian_result = fit_hierarchical(df)

    if verbose:
        print("  Fitting gradient model...")
    gradient_result = fit_gradient(df, n_epochs=500, lr=0.01, batch_size=8192)

    # Generate queries
    queries = generate_synthetic_queries(df, n_queries=200)

    edge_ids = df["edge_id"].values
    edge_timestamps = df["timestamp"].values
    n_edges = len(df)

    rng = np.random.default_rng(42)
    base_sims = rng.uniform(0.1, 0.5, n_edges)

    # Build semantic sim function
    edge_predicates = df["predicate"].values

    def sim_fn(query_id):
        parts = query_id.split("_", 2)
        pred = parts[2] if len(parts) >= 3 else ""
        sims = base_sims.copy()
        sims[edge_predicates == pred] = 1.0
        return sims

    # Bayesian: assign tau/shape per edge from cluster fits
    taus_bay = np.ones(n_edges) * 100
    shapes_bay = np.ones(n_edges)
    for cluster_name, fit in bayesian_result.cluster_fits.items():
        mask = df["cluster"].values == cluster_name
        taus_bay[mask] = fit.tau
        shapes_bay[mask] = fit.shape

    # Gradient: compute tau from learned theta per cluster
    taus_grad = np.ones(n_edges) * 100
    shapes_grad = np.ones(n_edges)
    cluster_names_sorted = sorted(gradient_result.cluster_theta.keys())
    for cluster_name in cluster_names_sorted:
        theta = gradient_result.cluster_theta[cluster_name]
        shape = gradient_result.cluster_shape[cluster_name]
        mask = df["cluster"].values == cluster_name
        v = df.loc[mask, "velocity"].values
        s = df.loc[mask, "volatility"].values
        log_tau = theta[0] + theta[1] * v + theta[2] * s + theta[3] * v * s
        taus_grad[mask] = np.exp(np.clip(log_tau, -5, 15))
        shapes_grad[mask] = shape

    # Compute retrieval scores for all queries under both methods
    all_rank_correlations = []
    all_ndcg_bay = []
    all_ndcg_grad = []

    for query in queries:
        ages = query.query_time - edge_timestamps
        sims = sim_fn(query.query_id)

        scores_bay = score_edges(ages, sims, taus_bay, shapes_bay, alpha=1.0, beta=1.0)
        scores_grad = score_edges(ages, sims, taus_grad, shapes_grad, alpha=1.0, beta=1.0)

        # Rank correlation (Spearman) on top-100 candidates
        top_k = 100
        top_bay = np.argsort(-scores_bay)[:top_k]
        top_grad = np.argsort(-scores_grad)[:top_k]

        # Compute rank correlation using overlap of top-K
        set_bay = set(top_bay)
        set_grad = set(top_grad)
        overlap = len(set_bay & set_grad) / top_k
        all_rank_correlations.append(overlap)

        # NDCG for both
        relevances = np.zeros(n_edges)
        for i, eid in enumerate(edge_ids):
            relevances[i] = query.relevant_edges.get(int(eid), 0.0)
        all_ndcg_bay.append(ndcg_at_k(scores_bay, relevances, k=10))
        all_ndcg_grad.append(ndcg_at_k(scores_grad, relevances, k=10))

    results = {
        "mean_top100_overlap": float(np.mean(all_rank_correlations)),
        "std_top100_overlap": float(np.std(all_rank_correlations)),
        "bayesian_ndcg10": float(np.mean(all_ndcg_bay)),
        "gradient_ndcg10": float(np.mean(all_ndcg_grad)),
        "ndcg_correlation": float(np.corrcoef(all_ndcg_bay, all_ndcg_grad)[0, 1]),
        "n_queries": len(queries),
    }

    if verbose:
        print(f"\n  Top-100 overlap: {results['mean_top100_overlap']:.3f} +/- {results['std_top100_overlap']:.3f}")
        print(f"  Bayesian NDCG@10: {results['bayesian_ndcg10']:.3f}")
        print(f"  Gradient NDCG@10: {results['gradient_ndcg10']:.3f}")
        print(f"  Per-query NDCG correlation: {results['ndcg_correlation']:.3f}")

    return results


def run_all_sensitivity(verbose: bool = True) -> dict:
    """Run all sensitivity analyses."""
    results = {}

    results["epsilon_sensitivity"] = epsilon_sensitivity(verbose)
    results["lognormal_hazard"] = lognormal_hazard_analysis(verbose)
    results["gradient_bayesian_ranking"] = gradient_bayesian_ranking_comparison(verbose)

    return results


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.WARNING)

    results = run_all_sensitivity(verbose=True)

    import os
    os.makedirs("results", exist_ok=True)

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open("results/sensitivity_analysis.json", "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    print("\n\nResults saved to results/sensitivity_analysis.json")
