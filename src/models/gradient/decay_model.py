"""Gradient-based hierarchical decay model using PyTorch.

Learns the decay surface parameters by directly optimizing a survival
likelihood loss. Compares against the Bayesian approach as an ablation.

The model learns:
- Per-cluster theta parameters (decay surface)
- Per-cluster Weibull shape parameters
- Per-cohort offsets from cluster theta (with L2 regularization)
- Per-individual offsets from cohort theta (with L2 regularization)
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class HierarchicalDecayModel(nn.Module):
    """Learnable hierarchical decay surface.

    Parameters:
        n_clusters: number of temporal type clusters
        n_cohorts_per_cluster: list of cohort counts per cluster
        n_individuals_per_cohort: nested list of individual counts
    """

    def __init__(
        self,
        cluster_names: list[str],
        cohort_names: dict[str, list[str]],
        entity_names: dict[tuple[str, str], list[str]],
    ):
        super().__init__()

        self.cluster_names = cluster_names
        self.cohort_names = cohort_names
        self.entity_names = entity_names

        n_clusters = len(cluster_names)

        # Level 1: Cluster-level theta [theta_0, theta_1, theta_2, theta_3] per cluster
        self.cluster_theta = nn.Parameter(torch.zeros(n_clusters, 4))
        # Level 1: Cluster-level Weibull shape (log-parameterized for positivity)
        self.cluster_log_shape = nn.Parameter(torch.zeros(n_clusters))

        # Level 2: Cohort offsets from cluster theta
        cohort_list = []
        self._cohort_to_idx = {}
        self._cohort_to_cluster_idx = {}
        idx = 0
        for ci, cluster in enumerate(cluster_names):
            for cohort in cohort_names.get(cluster, []):
                self._cohort_to_idx[(cluster, cohort)] = idx
                self._cohort_to_cluster_idx[(cluster, cohort)] = ci
                idx += 1
                cohort_list.append((cluster, cohort))
        n_cohorts = len(cohort_list)
        self.cohort_offset = nn.Parameter(torch.zeros(n_cohorts, 4))

        # Level 3: Individual offsets from cohort theta
        entity_list = []
        self._entity_to_idx = {}
        self._entity_to_cohort_idx = {}
        idx = 0
        for (cluster, cohort), entities in entity_names.items():
            cohort_idx = self._cohort_to_idx.get((cluster, cohort))
            if cohort_idx is None:
                continue
            for entity in entities:
                self._entity_to_idx[(cluster, cohort, entity)] = idx
                self._entity_to_cohort_idx[(cluster, cohort, entity)] = cohort_idx
                idx += 1
                entity_list.append((cluster, cohort, entity))
        n_entities = len(entity_list)
        self.entity_offset = nn.Parameter(torch.zeros(n_entities, 4))

        # Build index mapping tensors for vectorized lookups
        self._cluster_idx_for_cohort = torch.tensor(
            [self._cohort_to_cluster_idx[k] for k in cohort_list], dtype=torch.long
        )
        self._cohort_idx_for_entity = torch.tensor(
            [self._entity_to_cohort_idx[k] for k in entity_list], dtype=torch.long
        )

        # Initialize with reasonable values
        nn.init.normal_(self.cluster_theta, mean=5.0, std=1.0)
        nn.init.zeros_(self.cluster_log_shape)  # shape = exp(0) = 1.0

    def get_theta(
        self,
        cluster_indices: torch.Tensor,
        cohort_indices: torch.Tensor | None = None,
        entity_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Get effective theta for a batch of edges.

        Theta = cluster_theta + cohort_offset + entity_offset
        """
        theta = self.cluster_theta[cluster_indices]  # (batch, 4)

        if cohort_indices is not None:
            valid_cohort = cohort_indices >= 0
            if valid_cohort.any():
                cohort_contrib = torch.zeros_like(theta)
                cohort_contrib[valid_cohort] = self.cohort_offset[cohort_indices[valid_cohort]]
                theta = theta + cohort_contrib

        if entity_indices is not None:
            valid_entity = entity_indices >= 0
            if valid_entity.any():
                entity_contrib = torch.zeros_like(theta)
                entity_contrib[valid_entity] = self.entity_offset[entity_indices[valid_entity]]
                theta = theta + entity_contrib

        return theta

    def compute_tau(
        self,
        theta: torch.Tensor,
        velocity: torch.Tensor,
        volatility: torch.Tensor,
    ) -> torch.Tensor:
        """Compute tau = exp(theta_0 + theta_1*v + theta_2*sigma + theta_3*v*sigma)."""
        log_tau = (
            theta[:, 0]
            + theta[:, 1] * velocity
            + theta[:, 2] * volatility
            + theta[:, 3] * velocity * volatility
        )
        return torch.exp(torch.clamp(log_tau, min=-5, max=15))

    def weibull_log_likelihood(
        self,
        durations: torch.Tensor,
        event_observed: torch.Tensor,
        tau: torch.Tensor,
        shape: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Weibull log-likelihood with right-censoring.

        For observed events: log f(t) = log(k/tau) + (k-1)*log(t/tau) - (t/tau)^k
        For censored: log S(t) = -(t/tau)^k
        """
        # Avoid numerical issues
        t_safe = torch.clamp(durations, min=1e-6)
        tau_safe = torch.clamp(tau, min=1e-6)
        shape_safe = torch.clamp(shape, min=1e-3)

        z = t_safe / tau_safe  # normalized time

        # log S(t) = -(t/tau)^k  (survival for all)
        log_survival = -(z ** shape_safe)

        # log f(t) = log(k/tau) + (k-1)*log(t/tau) - (t/tau)^k  (density for events)
        log_density = (
            torch.log(shape_safe) - torch.log(tau_safe)
            + (shape_safe - 1) * torch.log(z)
            + log_survival
        )

        # Combine: observed events use density, censored use survival
        log_lik = event_observed * log_density + (1 - event_observed) * log_survival

        return log_lik

    def hierarchical_regularization(
        self,
        lambda_cohort: float = 1.0,
        lambda_entity: float = 1.0,
    ) -> torch.Tensor:
        """L2 regularization encouraging lower levels toward higher levels.

        Cohort offsets -> 0 (toward cluster)
        Entity offsets -> 0 (toward cohort)
        """
        reg = lambda_cohort * (self.cohort_offset ** 2).sum()
        reg += lambda_entity * (self.entity_offset ** 2).sum()
        return reg

    def forward(
        self,
        cluster_indices: torch.Tensor,
        cohort_indices: torch.Tensor,
        entity_indices: torch.Tensor,
        velocity: torch.Tensor,
        volatility: torch.Tensor,
        durations: torch.Tensor,
        event_observed: torch.Tensor,
    ) -> torch.Tensor:
        """Compute negative log-likelihood loss for a batch of edges."""
        theta = self.get_theta(cluster_indices, cohort_indices, entity_indices)
        tau = self.compute_tau(theta, velocity, volatility)

        # Shape per cluster
        shape = torch.exp(self.cluster_log_shape[cluster_indices])

        log_lik = self.weibull_log_likelihood(durations, event_observed, tau, shape)

        return -log_lik.mean()  # negative log-likelihood


def prepare_training_data(df: pd.DataFrame) -> dict:
    """Convert DataFrame to PyTorch tensors with index mappings."""
    # Build name -> index mappings
    cluster_names = sorted(df["cluster"].unique())
    cluster_to_idx = {name: i for i, name in enumerate(cluster_names)}

    cohort_names_dict = {}
    for cluster in cluster_names:
        cohort_names_dict[cluster] = sorted(
            df[df["cluster"] == cluster]["cohort"].unique()
        )

    cohort_to_idx = {}
    idx = 0
    for cluster in cluster_names:
        for cohort in cohort_names_dict[cluster]:
            cohort_to_idx[(cluster, cohort)] = idx
            idx += 1

    entity_names_dict = {}
    entity_to_idx = {}
    idx = 0
    for (cluster, cohort), group in df.groupby(["cluster", "cohort"]):
        entities = sorted(group["entity"].unique())
        entity_names_dict[(cluster, cohort)] = entities
        for entity in entities:
            entity_to_idx[(cluster, cohort, entity)] = idx
            idx += 1

    # Build index tensors
    cluster_indices = torch.tensor(
        [cluster_to_idx[c] for c in df["cluster"]], dtype=torch.long
    )
    cohort_indices = torch.tensor(
        [cohort_to_idx.get((c, h), -1) for c, h in zip(df["cluster"], df["cohort"])],
        dtype=torch.long,
    )
    entity_indices = torch.tensor(
        [entity_to_idx.get((c, h, e), -1) for c, h, e in zip(df["cluster"], df["cohort"], df["entity"])],
        dtype=torch.long,
    )

    velocity = torch.tensor(df["velocity"].values, dtype=torch.float32)
    volatility = torch.tensor(df["volatility"].values, dtype=torch.float32)
    durations = torch.tensor(df["lifetime_observed"].values, dtype=torch.float32)
    event_observed = torch.tensor(
        (~df["is_censored"]).astype(float).values, dtype=torch.float32
    )

    return {
        "cluster_names": cluster_names,
        "cohort_names": cohort_names_dict,
        "entity_names": entity_names_dict,
        "cluster_indices": cluster_indices,
        "cohort_indices": cohort_indices,
        "entity_indices": entity_indices,
        "velocity": velocity,
        "volatility": volatility,
        "durations": durations,
        "event_observed": event_observed,
    }


@dataclass
class GradientFitResult:
    """Result of gradient-based fitting."""

    cluster_theta: dict[str, np.ndarray]  # cluster_name -> theta[4]
    cluster_shape: dict[str, float]  # cluster_name -> Weibull shape
    final_loss: float
    n_epochs: int
    loss_history: list[float]


def fit_gradient(
    df: pd.DataFrame,
    n_epochs: int = 500,
    lr: float = 0.01,
    lambda_cohort: float = 1.0,
    lambda_entity: float = 1.0,
    batch_size: int | None = None,
    verbose: bool = False,
) -> GradientFitResult:
    """Fit the hierarchical decay model via gradient descent.

    Optimizes the Weibull survival likelihood directly.
    """
    data = prepare_training_data(df)

    model = HierarchicalDecayModel(
        cluster_names=data["cluster_names"],
        cohort_names=data["cohort_names"],
        entity_names=data["entity_names"],
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n = len(data["durations"])
    loss_history = []

    for epoch in range(n_epochs):
        if batch_size and batch_size < n:
            # Mini-batch
            perm = torch.randperm(n)[:batch_size]
            batch_cluster = data["cluster_indices"][perm]
            batch_cohort = data["cohort_indices"][perm]
            batch_entity = data["entity_indices"][perm]
            batch_vel = data["velocity"][perm]
            batch_vol = data["volatility"][perm]
            batch_dur = data["durations"][perm]
            batch_evt = data["event_observed"][perm]
        else:
            batch_cluster = data["cluster_indices"]
            batch_cohort = data["cohort_indices"]
            batch_entity = data["entity_indices"]
            batch_vel = data["velocity"]
            batch_vol = data["volatility"]
            batch_dur = data["durations"]
            batch_evt = data["event_observed"]

        optimizer.zero_grad()

        nll = model(
            batch_cluster, batch_cohort, batch_entity,
            batch_vel, batch_vol, batch_dur, batch_evt,
        )
        reg = model.hierarchical_regularization(lambda_cohort, lambda_entity)
        loss = nll + reg / n

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        loss_val = loss.item()
        loss_history.append(loss_val)

        if verbose and (epoch % 100 == 0 or epoch == n_epochs - 1):
            print(f"  Epoch {epoch}: loss={loss_val:.4f}, nll={nll.item():.4f}, reg={reg.item():.4f}")

    # Extract results
    with torch.no_grad():
        cluster_theta = {}
        cluster_shape = {}
        for i, name in enumerate(data["cluster_names"]):
            cluster_theta[name] = model.cluster_theta[i].numpy().copy()
            cluster_shape[name] = float(torch.exp(model.cluster_log_shape[i]).item())

    return GradientFitResult(
        cluster_theta=cluster_theta,
        cluster_shape=cluster_shape,
        final_loss=loss_history[-1],
        n_epochs=n_epochs,
        loss_history=loss_history,
    )


if __name__ == "__main__":
    from src.synthetic.generator import generate_synthetic_kg

    print("Generating synthetic KG...")
    df = generate_synthetic_kg()

    print("Fitting gradient-based model...")
    result = fit_gradient(df, n_epochs=500, lr=0.01, batch_size=8192, verbose=True)

    print("\n=== Gradient-based cluster parameters ===")
    for cluster in sorted(result.cluster_theta.keys()):
        theta = result.cluster_theta[cluster]
        shape = result.cluster_shape[cluster]
        tau_baseline = np.exp(theta[0])
        print(f"\n{cluster}:")
        print(f"  theta = {theta}")
        print(f"  tau(v=0,s=0) = {tau_baseline:.1f} days")
        print(f"  shape = {shape:.3f}")
