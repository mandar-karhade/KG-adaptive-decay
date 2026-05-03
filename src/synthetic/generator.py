"""Synthetic temporal KG generator with planted hierarchical parameters.

Generates a temporal knowledge graph where:
- Edges belong to planted temporal clusters (types)
- Each cluster has known decay surface parameters
- Cohorts within clusters shift the surface
- Individuals within cohorts add further variation
- Edges are created, reinforced, or superseded over a simulated timeline

This provides ground truth for parameter recovery experiments.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.synthetic.config import (
    ClusterConfig,
    CohortConfig,
    SyntheticKGConfig,
    DEFAULT_CLUSTERS,
    DEFAULT_COHORTS,
    get_default_config,
)


@dataclass
class Edge:
    """A single edge in the temporal KG."""

    edge_id: int
    subject: str
    predicate: str
    object_value: float  # simplified: numeric value for synthetic data
    timestamp: float  # days from simulation start
    cluster: str
    cohort: str
    entity: str
    # Ground truth parameters
    true_tau: float
    true_shape: float
    true_theta: np.ndarray
    # Observed outcomes
    superseded_at: float | None = None  # timestamp when superseded
    superseded_by: int | None = None  # edge_id of superseding edge
    reinforcement_times: list = None  # timestamps of same-value re-observations

    def __post_init__(self):
        if self.reinforcement_times is None:
            self.reinforcement_times = []

    @property
    def lifetime(self) -> float | None:
        """Observed lifetime if superseded, None if right-censored."""
        if self.superseded_at is not None:
            return self.superseded_at - self.timestamp
        return None

    @property
    def is_superseded(self) -> bool:
        return self.superseded_at is not None

    @property
    def is_reinforced(self) -> bool:
        return len(self.reinforcement_times) > 0


def compute_tau(velocity: float, volatility: float, theta: np.ndarray) -> float:
    """Compute characteristic lifetime from velocity, volatility, and surface params.

    tau(v, sigma) = exp(theta_0 + theta_1*v + theta_2*sigma + theta_3*v*sigma)
    """
    log_tau = theta[0] + theta[1] * velocity + theta[2] * volatility + theta[3] * velocity * volatility
    return np.exp(np.clip(log_tau, -5, 15))  # clip for numerical stability


def sample_weibull_lifetime(tau: float, shape: float, rng: np.random.Generator) -> float:
    """Sample a lifetime from Weibull(tau, shape).

    Parameterization: f(t) = (k/tau)(t/tau)^(k-1) exp(-(t/tau)^k)
    where k = shape, tau = scale.
    """
    # numpy's weibull takes shape param 'a' and samples from Weibull with scale=1
    # To get Weibull(tau, shape): sample = tau * numpy_weibull(shape)
    return tau * rng.weibull(shape)


def generate_synthetic_kg(
    clusters: list[ClusterConfig] | None = None,
    cohorts: dict[str, list[CohortConfig]] | None = None,
    config: SyntheticKGConfig | None = None,
) -> pd.DataFrame:
    """Generate a synthetic temporal KG with planted parameters.

    Returns a DataFrame with one row per edge, including ground truth
    parameters and observed outcomes (superseded, reinforced, or active).
    """
    if config is None:
        config = get_default_config()
    if clusters is None:
        clusters = DEFAULT_CLUSTERS
    if cohorts is None:
        cohorts = DEFAULT_COHORTS

    rng = np.random.default_rng(config.seed)
    total_days = config.n_years * 365.25

    edges: list[dict] = []
    edge_id = 0

    for cluster in clusters:
        cluster_cohorts = cohorts.get(cluster.name, [])

        for cohort in cluster_cohorts:
            # Cohort-level theta = cluster theta + cohort offset
            cohort_theta = cluster.theta + cohort.theta_offset

            for entity_idx in range(cohort.n_entities):
                entity_name = f"{cluster.name}_{cohort.name}_e{entity_idx}"

                # Individual-level theta = cohort theta + random noise
                individual_theta = cohort_theta + rng.normal(
                    0, config.individual_sigma, size=4
                )

                # Generate edges for each predicate in this cluster
                for pred_idx in range(cluster.n_predicates):
                    predicate = f"{cluster.name}_pred{pred_idx}"

                    # Sample velocity and volatility for this concept
                    velocity = max(0.001, rng.normal(cluster.velocity_mean, cluster.velocity_std))
                    volatility = max(0.001, rng.normal(cluster.volatility_mean, cluster.volatility_std))

                    # Compute the true tau for this (entity, predicate)
                    true_tau = compute_tau(velocity, volatility, individual_theta)

                    # Simulate the observation process over the timeline
                    # Mean inter-observation interval from velocity (observations per day)
                    mean_interval = 1.0 / max(velocity, 0.001)

                    current_time = rng.exponential(mean_interval)  # first observation
                    current_value = rng.normal(0, 1)  # initial value

                    # Sample the true lifetime of this first value
                    true_lifetime = sample_weibull_lifetime(true_tau, cluster.shape, rng)

                    first_edge_id = edge_id
                    current_edge = {
                        "edge_id": edge_id,
                        "subject": entity_name,
                        "predicate": predicate,
                        "object_value": current_value,
                        "timestamp": current_time,
                        "cluster": cluster.name,
                        "cohort": cohort.name,
                        "entity": entity_name,
                        "true_tau": true_tau,
                        "true_shape": cluster.shape,
                        "true_theta_0": individual_theta[0],
                        "true_theta_1": individual_theta[1],
                        "true_theta_2": individual_theta[2],
                        "true_theta_3": individual_theta[3],
                        "velocity": velocity,
                        "volatility": volatility,
                        "superseded_at": None,
                        "superseded_by": None,
                        "reinforcement_times": [],
                        "lifetime_observed": None,
                        "is_censored": True,
                    }
                    edge_id += 1

                    # Step through time, generating observations
                    obs_time = current_time
                    while obs_time < total_days:
                        # Next observation time
                        interval = rng.exponential(mean_interval)
                        obs_time += interval

                        if obs_time >= total_days:
                            break

                        # Has the true value changed by this observation?
                        # The value changes when elapsed time since last change > true_lifetime
                        elapsed_since_creation = obs_time - current_edge["timestamp"]

                        if elapsed_since_creation >= true_lifetime:
                            # Value has been superseded -- generate new value
                            # The new value differs by at least epsilon (meaningful change)
                            value_delta = rng.normal(0, 1) * max(volatility, config.epsilon + 0.1)
                            if abs(value_delta) < config.epsilon:
                                value_delta = np.sign(value_delta) * (config.epsilon + 0.05)
                            new_value = current_value + value_delta

                            # Mark current edge as superseded
                            current_edge["superseded_at"] = obs_time
                            current_edge["superseded_by"] = edge_id
                            current_edge["lifetime_observed"] = elapsed_since_creation
                            current_edge["is_censored"] = False
                            edges.append(current_edge)

                            # Create new edge
                            current_value = new_value
                            true_lifetime = sample_weibull_lifetime(true_tau, cluster.shape, rng)

                            current_edge = {
                                "edge_id": edge_id,
                                "subject": entity_name,
                                "predicate": predicate,
                                "object_value": current_value,
                                "timestamp": obs_time,
                                "cluster": cluster.name,
                                "cohort": cohort.name,
                                "entity": entity_name,
                                "true_tau": true_tau,
                                "true_shape": cluster.shape,
                                "true_theta_0": individual_theta[0],
                                "true_theta_1": individual_theta[1],
                                "true_theta_2": individual_theta[2],
                                "true_theta_3": individual_theta[3],
                                "velocity": velocity,
                                "volatility": volatility,
                                "superseded_at": None,
                                "superseded_by": None,
                                "reinforcement_times": [],
                                "lifetime_observed": None,
                                "is_censored": True,
                            }
                            edge_id += 1
                        else:
                            # Value hasn't changed -- this is a reinforcement
                            current_edge["reinforcement_times"].append(obs_time)

                    # Final edge is right-censored (still active at end of simulation)
                    current_edge["lifetime_observed"] = total_days - current_edge["timestamp"]
                    edges.append(current_edge)

    df = pd.DataFrame(edges)

    # Convert reinforcement_times lists to counts for the main DataFrame
    df["n_reinforcements"] = df["reinforcement_times"].apply(len)

    return df


def summarize_synthetic_kg(df: pd.DataFrame) -> dict:
    """Summarize key statistics of the generated synthetic KG."""
    return {
        "total_edges": len(df),
        "superseded_edges": int(df["is_censored"].eq(False).sum()),
        "censored_edges": int(df["is_censored"].sum()),
        "clusters": df["cluster"].nunique(),
        "cohorts": df.groupby("cluster")["cohort"].nunique().to_dict(),
        "entities": df["entity"].nunique(),
        "predicates": df["predicate"].nunique(),
        "mean_lifetime_by_cluster": df.groupby("cluster")["lifetime_observed"].mean().to_dict(),
        "median_lifetime_by_cluster": df.groupby("cluster")["lifetime_observed"].median().to_dict(),
        "supersession_rate_by_cluster": (
            df.groupby("cluster")["is_censored"]
            .apply(lambda x: 1 - x.mean())
            .to_dict()
        ),
        "mean_reinforcements_by_cluster": (
            df.groupby("cluster")["n_reinforcements"].mean().to_dict()
        ),
    }


if __name__ == "__main__":
    print("Generating synthetic temporal KG...")
    df = generate_synthetic_kg()
    summary = summarize_synthetic_kg(df)

    print(f"\nGenerated {summary['total_edges']} edges:")
    print(f"  Superseded: {summary['superseded_edges']}")
    print(f"  Censored (still active): {summary['censored_edges']}")
    print(f"  Clusters: {summary['clusters']}")
    print(f"  Entities: {summary['entities']}")
    print(f"  Predicates: {summary['predicates']}")

    print("\nMean lifetime by cluster (days):")
    for cluster, lifetime in summary["mean_lifetime_by_cluster"].items():
        rate = summary["supersession_rate_by_cluster"][cluster]
        reinforcements = summary["mean_reinforcements_by_cluster"][cluster]
        print(f"  {cluster}: {lifetime:.1f} days (supersession rate: {rate:.2f}, mean reinforcements: {reinforcements:.1f})")
