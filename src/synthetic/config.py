"""Configuration for synthetic temporal KG generation.

Defines the planted temporal clusters with known velocity-volatility
regimes and hierarchical parameters for parameter recovery experiments.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class ClusterConfig:
    """Configuration for a single temporal cluster (type-level)."""

    name: str
    # Decay surface parameters: tau = exp(theta_0 + theta_1*v + theta_2*sigma + theta_3*v*sigma)
    theta: np.ndarray  # [theta_0, theta_1, theta_2, theta_3]
    # Weibull shape parameter
    shape: float
    # Velocity distribution for edges in this cluster
    velocity_mean: float
    velocity_std: float
    # Volatility distribution for edges in this cluster
    volatility_mean: float
    volatility_std: float
    # How many predicates belong to this cluster
    n_predicates: int = 5

    def __eq__(self, other):
        if not isinstance(other, ClusterConfig):
            return NotImplemented
        return (
            self.name == other.name
            and np.array_equal(self.theta, other.theta)
            and self.shape == other.shape
        )

    def __hash__(self):
        return hash(self.name)


@dataclass(frozen=True)
class CohortConfig:
    """Configuration for a cohort within a cluster."""

    name: str
    # Offset from cluster-level theta (cohort shifts the surface)
    theta_offset: np.ndarray  # [delta_0, delta_1, delta_2, delta_3]
    n_entities: int = 20

    def __eq__(self, other):
        if not isinstance(other, CohortConfig):
            return NotImplemented
        return self.name == other.name and np.array_equal(
            self.theta_offset, other.theta_offset
        )

    def __hash__(self):
        return hash(self.name)


@dataclass(frozen=True)
class SyntheticKGConfig:
    """Full configuration for synthetic temporal KG generation."""

    # Simulation parameters
    n_years: float = 5.0
    time_step_days: float = 1.0  # granularity of simulation

    # Individual-level noise
    individual_sigma: float = 0.2  # std of individual theta offset from cohort

    # Cohort-level noise
    cohort_sigma: float = 0.3  # std of cohort theta offset from cluster

    # Supersession threshold (embedding distance)
    epsilon: float = 0.3

    # Random seed
    seed: Optional[int] = 42


# --- Default planted configurations ---

# Cluster 1: Permanent facts (genetic mutations, demographics)
# Very high baseline lifetime, velocity and volatility have minimal effect
PERMANENT_FACTS = ClusterConfig(
    name="permanent_facts",
    theta=np.array([8.0, -0.1, -0.1, 0.0]),  # tau ~ exp(8) ~ 2981 days ~ 8+ years
    shape=0.5,  # k < 1: Lindy effect -- older facts are MORE stable
    velocity_mean=0.01,  # rarely mentioned
    velocity_std=0.005,
    volatility_mean=0.02,  # value almost never changes
    volatility_std=0.01,
    n_predicates=5,
)

# Cluster 2: Current state (treatment regimens, employment, address)
# Moderate lifetime, volatility strongly reduces it
CURRENT_STATE = ClusterConfig(
    name="current_state",
    theta=np.array([5.5, -0.3, -1.5, 0.1]),  # tau ~ exp(5.5) ~ 245 days baseline
    shape=1.2,  # k > 1: slight aging -- longer-held states are slightly more likely to change
    velocity_mean=0.1,  # mentioned periodically
    velocity_std=0.05,
    volatility_mean=0.4,  # values change moderately
    volatility_std=0.15,
    n_predicates=5,
)

# Cluster 3: Volatile measurements (vitals, daily labs)
# Short lifetime, high velocity, high volatility
VOLATILE_MEASUREMENTS = ClusterConfig(
    name="volatile_measurements",
    theta=np.array([3.0, -0.5, -2.0, 0.2]),  # tau ~ exp(3) ~ 20 days baseline
    shape=1.0,  # k = 1: memoryless (exponential)
    velocity_mean=0.8,  # measured frequently
    velocity_std=0.3,
    volatility_mean=0.7,  # values change a lot
    volatility_std=0.2,
    n_predicates=5,
)

# Cluster 4: Periodic assessments (quarterly labs, annual reviews)
# Medium lifetime, moderate velocity, moderate volatility
PERIODIC_ASSESSMENTS = ClusterConfig(
    name="periodic_assessments",
    theta=np.array([4.5, -0.2, -1.0, 0.05]),  # tau ~ exp(4.5) ~ 90 days baseline
    shape=0.8,  # k < 1: slight Lindy effect
    velocity_mean=0.03,  # infrequent
    velocity_std=0.01,
    volatility_mean=0.3,  # moderate change between observations
    volatility_std=0.1,
    n_predicates=5,
)

# Default cohorts for each cluster
DEFAULT_COHORTS = {
    "permanent_facts": [
        CohortConfig("genomic", np.array([0.5, 0.0, 0.0, 0.0]), n_entities=30),
        CohortConfig("demographic", np.array([0.0, 0.0, 0.0, 0.0]), n_entities=30),
        CohortConfig("established_knowledge", np.array([1.0, 0.0, 0.0, 0.0]), n_entities=20),
    ],
    "current_state": [
        CohortConfig("aggressive_disease", np.array([-1.0, -0.1, -0.5, 0.0]), n_entities=20),
        CohortConfig("stable_chronic", np.array([1.0, 0.1, 0.2, 0.0]), n_entities=25),
        CohortConfig("routine_care", np.array([0.0, 0.0, 0.0, 0.0]), n_entities=25),
    ],
    "volatile_measurements": [
        CohortConfig("icu", np.array([-1.0, -0.3, -0.5, 0.1]), n_entities=15),
        CohortConfig("inpatient_ward", np.array([0.0, 0.0, 0.0, 0.0]), n_entities=20),
        CohortConfig("outpatient", np.array([1.5, 0.2, 0.3, 0.0]), n_entities=25),
    ],
    "periodic_assessments": [
        CohortConfig("quarterly_monitoring", np.array([0.0, 0.0, 0.0, 0.0]), n_entities=20),
        CohortConfig("annual_review", np.array([1.0, 0.1, 0.1, 0.0]), n_entities=20),
        CohortConfig("specialist_consult", np.array([0.5, -0.1, -0.2, 0.0]), n_entities=15),
    ],
}

DEFAULT_CLUSTERS = [PERMANENT_FACTS, CURRENT_STATE, VOLATILE_MEASUREMENTS, PERIODIC_ASSESSMENTS]


def get_default_config() -> SyntheticKGConfig:
    """Return the default synthetic KG configuration."""
    return SyntheticKGConfig()
