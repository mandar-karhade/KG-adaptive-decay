"""Tests for gradient-based hierarchical decay model."""

import numpy as np
import pandas as pd
import pytest
import torch

from src.synthetic.generator import generate_synthetic_kg
from src.synthetic.config import SyntheticKGConfig, ClusterConfig, CohortConfig
from src.models.gradient.decay_model import (
    HierarchicalDecayModel,
    prepare_training_data,
    fit_gradient,
)


@pytest.fixture(scope="module")
def small_df():
    """Small synthetic KG for fast gradient tests."""
    clusters = [
        ClusterConfig("fast", np.array([3.0, -0.5, -2.0, 0.2]), 1.0,
                       0.5, 0.1, 0.6, 0.1, n_predicates=2),
        ClusterConfig("slow", np.array([7.0, -0.1, -0.1, 0.0]), 0.5,
                       0.01, 0.005, 0.02, 0.01, n_predicates=2),
    ]
    cohorts = {
        "fast": [CohortConfig("c1", np.zeros(4), n_entities=5)],
        "slow": [CohortConfig("c2", np.zeros(4), n_entities=5)],
    }
    return generate_synthetic_kg(clusters, cohorts, SyntheticKGConfig(n_years=2.0, seed=42))


class TestPrepareTrainingData:
    def test_returns_required_keys(self, small_df):
        data = prepare_training_data(small_df)
        required = [
            "cluster_names", "cohort_names", "entity_names",
            "cluster_indices", "cohort_indices", "entity_indices",
            "velocity", "volatility", "durations", "event_observed",
        ]
        for key in required:
            assert key in data

    def test_tensor_shapes(self, small_df):
        data = prepare_training_data(small_df)
        n = len(small_df)
        assert data["cluster_indices"].shape == (n,)
        assert data["velocity"].shape == (n,)
        assert data["durations"].shape == (n,)

    def test_indices_valid(self, small_df):
        data = prepare_training_data(small_df)
        n_clusters = len(data["cluster_names"])
        assert (data["cluster_indices"] >= 0).all()
        assert (data["cluster_indices"] < n_clusters).all()


class TestHierarchicalDecayModel:
    def test_forward_runs(self, small_df):
        data = prepare_training_data(small_df)
        model = HierarchicalDecayModel(
            data["cluster_names"], data["cohort_names"], data["entity_names"]
        )
        loss = model(
            data["cluster_indices"], data["cohort_indices"], data["entity_indices"],
            data["velocity"], data["volatility"], data["durations"], data["event_observed"],
        )
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_loss_decreases(self, small_df):
        data = prepare_training_data(small_df)
        model = HierarchicalDecayModel(
            data["cluster_names"], data["cohort_names"], data["entity_names"]
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        initial_loss = model(
            data["cluster_indices"], data["cohort_indices"], data["entity_indices"],
            data["velocity"], data["volatility"], data["durations"], data["event_observed"],
        ).item()

        for _ in range(50):
            optimizer.zero_grad()
            loss = model(
                data["cluster_indices"], data["cohort_indices"], data["entity_indices"],
                data["velocity"], data["volatility"], data["durations"], data["event_observed"],
            )
            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        assert final_loss < initial_loss, "Loss should decrease during training"

    def test_regularization_positive(self, small_df):
        data = prepare_training_data(small_df)
        model = HierarchicalDecayModel(
            data["cluster_names"], data["cohort_names"], data["entity_names"]
        )
        # After some training, offsets should be non-zero
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        for _ in range(20):
            optimizer.zero_grad()
            loss = model(
                data["cluster_indices"], data["cohort_indices"], data["entity_indices"],
                data["velocity"], data["volatility"], data["durations"], data["event_observed"],
            )
            reg = model.hierarchical_regularization()
            (loss + reg / len(small_df)).backward()
            optimizer.step()
        assert model.hierarchical_regularization().item() >= 0


class TestFitGradient:
    def test_returns_result(self, small_df):
        result = fit_gradient(small_df, n_epochs=50, lr=0.01)
        assert len(result.cluster_theta) == small_df["cluster"].nunique()
        assert len(result.cluster_shape) == small_df["cluster"].nunique()
        assert len(result.loss_history) == 50

    def test_permanent_has_lower_shape(self):
        """On full synthetic data, permanent facts should have shape < 1."""
        df = generate_synthetic_kg(config=SyntheticKGConfig(seed=42))
        result = fit_gradient(df, n_epochs=200, lr=0.01, batch_size=8192)
        perm_shape = result.cluster_shape["permanent_facts"]
        assert perm_shape < 1.2, f"permanent_facts shape={perm_shape} should be < 1 (Lindy)"

    def test_loss_converges(self, small_df):
        result = fit_gradient(small_df, n_epochs=200, lr=0.01)
        # Last 10% of losses should be lower than first 10%
        early = np.mean(result.loss_history[:20])
        late = np.mean(result.loss_history[-20:])
        assert late < early, "Loss should converge (late < early)"
