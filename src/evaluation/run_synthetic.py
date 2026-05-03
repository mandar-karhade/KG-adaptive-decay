"""Run the full synthetic experiment pipeline.

Produces all results needed for the paper's Section 5.1:
1. Parameter recovery (planted vs recovered tau/shape)
2. Cluster recovery (ARI, NMI)
3. Distribution comparison (Weibull vs Exponential vs LogNormal)
4. Retrieval comparison (6 conditions)
5. Inference method comparison (Bayesian vs gradient-based)
"""

import json
import time

import numpy as np
import pandas as pd

from src.synthetic.generator import generate_synthetic_kg, summarize_synthetic_kg
from src.synthetic.config import DEFAULT_CLUSTERS
from src.clustering.temporal_clusters import discover_and_evaluate
from src.models.bayesian.survival_model import fit_hierarchical
from src.models.gradient.decay_model import fit_gradient
from src.retrieval.temporal_retrieval import (
    generate_synthetic_queries,
    run_retrieval_comparison,
)


def run_full_synthetic_experiment(verbose: bool = True) -> dict:
    """Run the complete synthetic experiment pipeline."""
    results = {}

    # ===== 1. Generate synthetic KG =====
    if verbose:
        print("=" * 60)
        print("STEP 1: Generating synthetic temporal KG")
        print("=" * 60)

    t0 = time.time()
    df = generate_synthetic_kg()
    summary = summarize_synthetic_kg(df)
    gen_time = time.time() - t0

    results["generation"] = {
        "summary": summary,
        "time_seconds": gen_time,
    }

    if verbose:
        print(f"  Generated {summary['total_edges']} edges in {gen_time:.1f}s")
        print(f"  Clusters: {summary['clusters']}, Entities: {summary['entities']}")

    # ===== 2. Cluster discovery =====
    if verbose:
        print("\n" + "=" * 60)
        print("STEP 2: Cluster discovery (predicate-level)")
        print("=" * 60)

    t0 = time.time()
    cluster_results = discover_and_evaluate(df, level="predicate")
    cluster_time = time.time() - t0

    results["clustering"] = {}
    for method, result in cluster_results.items():
        results["clustering"][method] = {
            "n_clusters": result.n_clusters,
            "ari": result.ari,
            "nmi": result.nmi,
            "silhouette": result.silhouette,
        }
        if verbose:
            print(f"  {method}: clusters={result.n_clusters}, "
                  f"ARI={result.ari:.3f}, NMI={result.nmi:.3f}, "
                  f"silhouette={result.silhouette:.3f}")

    results["clustering"]["time_seconds"] = cluster_time

    # ===== 3. Bayesian survival model =====
    if verbose:
        print("\n" + "=" * 60)
        print("STEP 3: Hierarchical Bayesian survival model")
        print("=" * 60)

    t0 = time.time()
    bayesian_result = fit_hierarchical(df)
    bayesian_time = time.time() - t0

    # Parameter recovery: compare planted vs recovered
    planted_params = {}
    for cluster in DEFAULT_CLUSTERS:
        planted_params[cluster.name] = {
            "theta": cluster.theta.tolist(),
            "shape": cluster.shape,
        }

    recovered_params = {}
    for cluster_name, fit in bayesian_result.cluster_fits.items():
        recovered_params[cluster_name] = {
            "tau": fit.tau,
            "shape": fit.shape,
            "n_observations": fit.n_observations,
            "n_superseded": fit.n_superseded,
            "aic": fit.aic,
        }
        if cluster_name in bayesian_result.cluster_surface_fits:
            sf = bayesian_result.cluster_surface_fits[cluster_name]
            recovered_params[cluster_name]["theta"] = sf.theta.tolist()
            recovered_params[cluster_name]["theta_r_squared"] = sf.r_squared

    # Distribution comparison
    dist_comparison = {}
    for cluster_name, dists in bayesian_result.distribution_comparison.items():
        best = min(dists, key=dists.get) if dists else "N/A"
        dist_comparison[cluster_name] = {
            "aic_scores": {k: float(v) for k, v in dists.items()},
            "best_distribution": best,
        }

    results["bayesian"] = {
        "planted_params": planted_params,
        "recovered_params": recovered_params,
        "distribution_comparison": dist_comparison,
        "n_cohort_fits": len(bayesian_result.cohort_fits),
        "n_individual_fits": len(bayesian_result.individual_fits),
        "time_seconds": bayesian_time,
    }

    if verbose:
        print(f"  Fitted in {bayesian_time:.1f}s")
        print(f"  Cluster fits: {len(recovered_params)}")
        print(f"  Cohort fits: {len(bayesian_result.cohort_fits)}")
        print(f"  Individual fits: {len(bayesian_result.individual_fits)}")
        print("\n  Tau recovery:")
        for name in sorted(recovered_params.keys()):
            rp = recovered_params[name]
            print(f"    {name}: tau={rp['tau']:.1f}, shape={rp['shape']:.3f}")
        print("\n  Best distribution per cluster:")
        for name, dc in sorted(dist_comparison.items()):
            print(f"    {name}: {dc['best_distribution']}")

    # ===== 4. Gradient-based model =====
    if verbose:
        print("\n" + "=" * 60)
        print("STEP 4: Gradient-based model")
        print("=" * 60)

    t0 = time.time()
    gradient_result = fit_gradient(df, n_epochs=500, lr=0.01, batch_size=8192, verbose=verbose)
    gradient_time = time.time() - t0

    results["gradient"] = {
        "cluster_theta": {k: v.tolist() for k, v in gradient_result.cluster_theta.items()},
        "cluster_shape": gradient_result.cluster_shape,
        "final_loss": gradient_result.final_loss,
        "time_seconds": gradient_time,
    }

    if verbose:
        print(f"\n  Fitted in {gradient_time:.1f}s")
        for name in sorted(gradient_result.cluster_theta.keys()):
            theta = gradient_result.cluster_theta[name]
            shape = gradient_result.cluster_shape[name]
            print(f"    {name}: theta={np.round(theta, 2)}, shape={shape:.3f}")

    # ===== 5. Retrieval comparison =====
    if verbose:
        print("\n" + "=" * 60)
        print("STEP 5: Retrieval comparison")
        print("=" * 60)

    t0 = time.time()
    queries = generate_synthetic_queries(df, n_queries=200)
    retrieval_results = run_retrieval_comparison(df, queries, bayesian_result)
    retrieval_time = time.time() - t0

    results["retrieval"] = {
        "n_queries": len(queries),
        "time_seconds": retrieval_time,
    }

    if verbose:
        print(f"\n  {'Method':<25} {'NDCG@5':>8} {'NDCG@10':>8} {'MRR':>8} {'P@5':>8} {'P@10':>8}")
        print("  " + "-" * 75)

    for method_name, metrics in retrieval_results.items():
        results["retrieval"][method_name] = {
            "ndcg_5": metrics.ndcg_5,
            "ndcg_10": metrics.ndcg_10,
            "mrr": metrics.mrr_val,
            "precision_5": metrics.precision_5,
            "precision_10": metrics.precision_10,
        }
        if verbose:
            print(f"  {method_name:<25} {metrics.ndcg_5:>8.3f} {metrics.ndcg_10:>8.3f} "
                  f"{metrics.mrr_val:>8.3f} {metrics.precision_5:>8.3f} {metrics.precision_10:>8.3f}")

    # ===== 6. Inference comparison =====
    if verbose:
        print("\n" + "=" * 60)
        print("STEP 6: Inference method comparison")
        print("=" * 60)

    shape_comparison = {}
    for cluster_name in sorted(bayesian_result.cluster_fits.keys()):
        bay_shape = bayesian_result.cluster_fits[cluster_name].shape
        grad_shape = gradient_result.cluster_shape.get(cluster_name, float("nan"))
        shape_comparison[cluster_name] = {
            "bayesian_shape": bay_shape,
            "gradient_shape": grad_shape,
            "difference": abs(bay_shape - grad_shape),
        }
        if verbose:
            print(f"  {cluster_name}: bayesian={bay_shape:.3f}, gradient={grad_shape:.3f}, "
                  f"diff={abs(bay_shape - grad_shape):.3f}")

    results["inference_comparison"] = {
        "shape_comparison": shape_comparison,
        "bayesian_time": bayesian_time,
        "gradient_time": gradient_time,
    }

    # Both methods agree on Lindy effect?
    perm_bay = bayesian_result.cluster_fits["permanent_facts"].shape
    perm_grad = gradient_result.cluster_shape["permanent_facts"]
    results["inference_comparison"]["lindy_agreement"] = (perm_bay < 1.0) and (perm_grad < 1.0)

    if verbose:
        print(f"\n  Lindy effect agreement (permanent_facts shape < 1): "
              f"{'YES' if results['inference_comparison']['lindy_agreement'] else 'NO'}")

    return results


if __name__ == "__main__":
    results = run_full_synthetic_experiment(verbose=True)

    # Save results
    import os
    os.makedirs("results", exist_ok=True)

    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            result = convert_numpy(obj)
            if result is not obj:
                return result
            return super().default(obj)

    with open("results/synthetic_experiment.json", "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    print("\n\nResults saved to results/synthetic_experiment.json")
