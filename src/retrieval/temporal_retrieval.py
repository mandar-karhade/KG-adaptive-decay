"""Temporal retrieval scoring and evaluation.

Implements the retrieval scoring function:
    score(e, query, t_q) = sim(e, query)^alpha * S(t_q - t_e | tau, shape)^beta

And metrics: NDCG, MRR, Precision@K on temporal queries.
"""

import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class TemporalQuery:
    """A temporal query with ground truth relevance."""

    query_id: str
    query_time: float  # timestamp of the query
    # Ground truth: edge_id -> relevance score (higher is more relevant)
    relevant_edges: dict[int, float]


def survival_weight(age: float, tau: float, shape: float) -> float:
    """Weibull survival function S(t) = exp(-(t/tau)^shape)."""
    if tau <= 0 or age < 0 or not np.isfinite(tau):
        return 0.0
    if age == 0:
        return 1.0
    return np.exp(-((age / tau) ** shape))


def score_edges(
    edge_ages: np.ndarray,
    semantic_sims: np.ndarray,
    taus: np.ndarray,
    shapes: np.ndarray,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> np.ndarray:
    """Compute retrieval scores for a batch of candidate edges.

    score = sim^alpha * S(age | tau, shape)^beta
    """
    tau_safe = np.maximum(taus, 1e-6)
    z = np.maximum(edge_ages, 0) / tau_safe
    temporal_weights = np.exp(-(z ** shapes))

    scores = (semantic_sims ** alpha) * (temporal_weights ** beta)
    return scores


def ndcg_at_k(
    scores: np.ndarray,
    relevances: np.ndarray,
    k: int = 10,
) -> float:
    """Compute NDCG@K."""
    if len(scores) == 0 or k == 0:
        return 0.0

    order = np.argsort(-scores)
    sorted_rels = relevances[order][:k]

    positions = np.arange(1, len(sorted_rels) + 1)
    dcg = np.sum(sorted_rels / np.log2(positions + 1))

    ideal_order = np.argsort(-relevances)
    ideal_rels = relevances[ideal_order][:k]
    ideal_positions = np.arange(1, len(ideal_rels) + 1)
    idcg = np.sum(ideal_rels / np.log2(ideal_positions + 1))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def mrr(
    scores: np.ndarray,
    relevances: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Compute Mean Reciprocal Rank."""
    order = np.argsort(-scores)
    sorted_rels = relevances[order]
    for i, rel in enumerate(sorted_rels):
        if rel >= threshold:
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(
    scores: np.ndarray,
    relevances: np.ndarray,
    k: int = 10,
    threshold: float = 0.5,
) -> float:
    """Compute Precision@K."""
    if len(scores) == 0 or k == 0:
        return 0.0
    order = np.argsort(-scores)
    top_k_rels = relevances[order][:k]
    return np.mean(top_k_rels >= threshold)


@dataclass(frozen=True)
class RetrievalMetrics:
    """Aggregated retrieval metrics across queries."""

    ndcg_5: float
    ndcg_10: float
    mrr_val: float
    precision_5: float
    precision_10: float
    n_queries: int


def run_retrieval_comparison(
    df,
    queries: list[TemporalQuery],
    bayesian_result,
    gradient_result=None,
) -> dict[str, RetrievalMetrics]:
    """Run retrieval comparison across multiple temporal weighting strategies.

    Baselines:
    1. No temporal weighting (semantic only)
    2. Uniform exponential decay (single tau for all edges)
    3. Type-level heterogeneous (Level 1 only)
    4. Type + cohort (Levels 1+2)
    5. Full hierarchical (Levels 1+2+3)
    """
    edge_ids = df["edge_id"].values
    edge_timestamps = df["timestamp"].values
    n_edges = len(df)

    # Simulated semantic similarity: edges matching the query predicate get sim=1.0,
    # others get a random lower similarity
    rng = np.random.default_rng(42)
    base_sims = rng.uniform(0.1, 0.5, n_edges)

    def make_sim_fn(query_predicate_map):
        """Create semantic sim function that returns 1.0 for matching predicate."""
        def sim_fn(query_id):
            pred = query_predicate_map.get(query_id)
            sims = base_sims.copy()
            if pred is not None:
                mask = df["predicate"].values == pred
                sims[mask] = 1.0
            return sims
        return sim_fn

    # Build query -> predicate mapping
    query_pred_map = {}
    for q in queries:
        # Extract predicate from query_id format "q_{i}_{predicate}"
        parts = q.query_id.split("_", 2)
        if len(parts) >= 3:
            query_pred_map[q.query_id] = parts[2]

    sim_fn = make_sim_fn(query_pred_map)

    results = {}

    # --- Baseline 1: No temporal weighting (beta=0) ---
    taus_uniform = np.ones(n_edges) * 1e10  # effectively no decay
    shapes_uniform = np.ones(n_edges)
    results["no_temporal"] = _run_queries(
        queries, edge_timestamps, edge_ids, sim_fn,
        taus_uniform, shapes_uniform, alpha=1.0, beta=0.0,
    )

    # --- Baseline 2: Uniform exponential decay ---
    median_lifetime = np.median(df["lifetime_observed"].values)
    taus_uniform_exp = np.ones(n_edges) * median_lifetime
    results["uniform_exponential"] = _run_queries(
        queries, edge_timestamps, edge_ids, sim_fn,
        taus_uniform_exp, shapes_uniform, alpha=1.0, beta=1.0,
    )

    # --- Baseline 3: Uniform half-life ---
    taus_halflife = np.ones(n_edges) * (median_lifetime * 0.5)
    results["uniform_halflife"] = _run_queries(
        queries, edge_timestamps, edge_ids, sim_fn,
        taus_halflife, shapes_uniform, alpha=1.0, beta=1.0,
    )

    # --- Level 1: Type-level heterogeneous ---
    taus_type = np.zeros(n_edges)
    shapes_type = np.zeros(n_edges)
    for cluster_name, fit in bayesian_result.cluster_fits.items():
        mask = df["cluster"].values == cluster_name
        taus_type[mask] = fit.tau
        shapes_type[mask] = fit.shape
    results["type_level"] = _run_queries(
        queries, edge_timestamps, edge_ids, sim_fn,
        taus_type, shapes_type, alpha=1.0, beta=1.0,
    )

    # --- Levels 1+2: Type + cohort ---
    taus_cohort = np.zeros(n_edges)
    shapes_cohort = np.zeros(n_edges)
    for (cluster_name, cohort_name), fit in bayesian_result.cohort_fits.items():
        mask = (df["cluster"].values == cluster_name) & (df["cohort"].values == cohort_name)
        taus_cohort[mask] = fit.tau
        shapes_cohort[mask] = fit.shape
    # Fill any missing with type-level
    missing = taus_cohort == 0
    taus_cohort[missing] = taus_type[missing]
    shapes_cohort[missing] = shapes_type[missing]
    results["type_cohort"] = _run_queries(
        queries, edge_timestamps, edge_ids, sim_fn,
        taus_cohort, shapes_cohort, alpha=1.0, beta=1.0,
    )

    # --- Levels 1+2+3: Full hierarchical ---
    taus_full = taus_cohort.copy()
    shapes_full = shapes_cohort.copy()
    for (cluster, cohort, entity), fit in bayesian_result.individual_fits.items():
        mask = (
            (df["cluster"].values == cluster)
            & (df["cohort"].values == cohort)
            & (df["entity"].values == entity)
        )
        if np.isfinite(fit.tau) and fit.tau > 0:
            taus_full[mask] = fit.tau
            shapes_full[mask] = fit.shape
    results["full_hierarchical"] = _run_queries(
        queries, edge_timestamps, edge_ids, sim_fn,
        taus_full, shapes_full, alpha=1.0, beta=1.0,
    )

    return results


def _run_queries(
    queries, edge_timestamps, edge_ids, sim_fn,
    taus, shapes, alpha, beta,
) -> RetrievalMetrics:
    """Run retrieval on all queries with given parameters."""
    all_ndcg5, all_ndcg10, all_mrr, all_p5, all_p10 = [], [], [], [], []

    for query in queries:
        ages = query.query_time - edge_timestamps
        sims = sim_fn(query.query_id)
        scores = score_edges(ages, sims, taus, shapes, alpha, beta)

        relevances = np.zeros(len(edge_ids))
        for i, eid in enumerate(edge_ids):
            relevances[i] = query.relevant_edges.get(int(eid), 0.0)

        all_ndcg5.append(ndcg_at_k(scores, relevances, k=5))
        all_ndcg10.append(ndcg_at_k(scores, relevances, k=10))
        all_mrr.append(mrr(scores, relevances))
        all_p5.append(precision_at_k(scores, relevances, k=5))
        all_p10.append(precision_at_k(scores, relevances, k=10))

    return RetrievalMetrics(
        ndcg_5=float(np.mean(all_ndcg5)),
        ndcg_10=float(np.mean(all_ndcg10)),
        mrr_val=float(np.mean(all_mrr)),
        precision_5=float(np.mean(all_p5)),
        precision_10=float(np.mean(all_p10)),
        n_queries=len(queries),
    )


def generate_synthetic_queries(
    df,
    n_queries: int = 100,
    seed: int = 42,
) -> list[TemporalQuery]:
    """Generate temporal queries from a synthetic KG.

    For each query, we pick a random time point and a random predicate.
    Relevant edges are those CURRENT at query time (not yet superseded).
    """
    rng = np.random.default_rng(seed)
    total_days = df["timestamp"].max()
    predicates = df["predicate"].unique()

    queries = []
    for i in range(n_queries):
        q_time = rng.uniform(total_days * 0.5, total_days)
        pred = rng.choice(predicates)

        pred_edges = df[df["predicate"] == pred]

        relevant = {}
        for _, edge in pred_edges.iterrows():
            if edge["timestamp"] > q_time:
                continue
            sup_at = edge["superseded_at"]
            if sup_at is not None and not np.isnan(sup_at) and sup_at <= q_time:
                relevant[int(edge["edge_id"])] = 0.0
            else:
                relevant[int(edge["edge_id"])] = 1.0

        if sum(v > 0 for v in relevant.values()) > 0:
            queries.append(TemporalQuery(
                query_id=f"q_{i}_{pred}",
                query_time=q_time,
                relevant_edges=relevant,
            ))

    return queries
