"""Tests for synthetic temporal KG generator."""

import numpy as np
import pandas as pd
import pytest

from src.synthetic.config import (
    ClusterConfig,
    CohortConfig,
    SyntheticKGConfig,
    DEFAULT_CLUSTERS,
    DEFAULT_COHORTS,
    PERMANENT_FACTS,
    VOLATILE_MEASUREMENTS,
)
from src.synthetic.generator import (
    compute_tau,
    sample_weibull_lifetime,
    generate_synthetic_kg,
    summarize_synthetic_kg,
)


class TestComputeTau:
    def test_baseline_only(self):
        theta = np.array([5.0, 0.0, 0.0, 0.0])
        tau = compute_tau(0.0, 0.0, theta)
        assert np.isclose(tau, np.exp(5.0))

    def test_velocity_effect(self):
        theta = np.array([5.0, -1.0, 0.0, 0.0])
        tau_low_v = compute_tau(0.1, 0.0, theta)
        tau_high_v = compute_tau(1.0, 0.0, theta)
        # Higher velocity with negative theta_1 should give shorter lifetime
        assert tau_high_v < tau_low_v

    def test_volatility_effect(self):
        theta = np.array([5.0, 0.0, -2.0, 0.0])
        tau_low_vol = compute_tau(0.0, 0.1, theta)
        tau_high_vol = compute_tau(0.0, 0.8, theta)
        # Higher volatility with negative theta_2 should give shorter lifetime
        assert tau_high_vol < tau_low_vol

    def test_interaction_term(self):
        theta = np.array([5.0, 0.0, 0.0, 1.0])
        tau_no_interaction = compute_tau(0.0, 0.0, theta)
        tau_with_interaction = compute_tau(1.0, 1.0, theta)
        # Positive interaction: high v AND high sigma together increase tau
        assert tau_with_interaction > tau_no_interaction

    def test_numerical_stability(self):
        # Very large values should be clipped, not overflow
        theta = np.array([100.0, 0.0, 0.0, 0.0])
        tau = compute_tau(0.0, 0.0, theta)
        assert np.isfinite(tau)

        # Very negative values should also be handled
        theta = np.array([-100.0, 0.0, 0.0, 0.0])
        tau = compute_tau(0.0, 0.0, theta)
        assert np.isfinite(tau)
        assert tau > 0


class TestSampleWeibullLifetime:
    def test_positive_lifetime(self):
        rng = np.random.default_rng(42)
        for _ in range(100):
            lifetime = sample_weibull_lifetime(100.0, 1.0, rng)
            assert lifetime > 0

    def test_mean_scales_with_tau(self):
        rng = np.random.default_rng(42)
        lifetimes_small = [sample_weibull_lifetime(10.0, 1.0, rng) for _ in range(1000)]
        rng = np.random.default_rng(42)
        lifetimes_large = [sample_weibull_lifetime(100.0, 1.0, rng) for _ in range(1000)]
        # Mean lifetime should scale proportionally with tau
        ratio = np.mean(lifetimes_large) / np.mean(lifetimes_small)
        assert 8.0 < ratio < 12.0  # should be ~10

    def test_shape_affects_variance(self):
        rng = np.random.default_rng(42)
        lifetimes_k1 = [sample_weibull_lifetime(100.0, 1.0, rng) for _ in range(2000)]
        rng = np.random.default_rng(42)
        lifetimes_k3 = [sample_weibull_lifetime(100.0, 3.0, rng) for _ in range(2000)]
        # Higher shape -> lower coefficient of variation
        cv_k1 = np.std(lifetimes_k1) / np.mean(lifetimes_k1)
        cv_k3 = np.std(lifetimes_k3) / np.mean(lifetimes_k3)
        assert cv_k3 < cv_k1


class TestGenerateSyntheticKG:
    @pytest.fixture
    def small_config(self):
        return SyntheticKGConfig(n_years=2.0, seed=42)

    @pytest.fixture
    def small_clusters(self):
        return [
            ClusterConfig(
                name="fast",
                theta=np.array([3.0, -0.5, -2.0, 0.2]),
                shape=1.0,
                velocity_mean=0.5,
                velocity_std=0.1,
                volatility_mean=0.6,
                volatility_std=0.1,
                n_predicates=2,
            ),
            ClusterConfig(
                name="slow",
                theta=np.array([7.0, -0.1, -0.1, 0.0]),
                shape=0.5,
                velocity_mean=0.01,
                velocity_std=0.005,
                volatility_mean=0.02,
                volatility_std=0.01,
                n_predicates=2,
            ),
        ]

    @pytest.fixture
    def small_cohorts(self):
        return {
            "fast": [
                CohortConfig("cohort_a", np.array([0.0, 0.0, 0.0, 0.0]), n_entities=5),
            ],
            "slow": [
                CohortConfig("cohort_b", np.array([0.0, 0.0, 0.0, 0.0]), n_entities=5),
            ],
        }

    def test_generates_dataframe(self, small_config, small_clusters, small_cohorts):
        df = generate_synthetic_kg(small_clusters, small_cohorts, small_config)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_required_columns(self, small_config, small_clusters, small_cohorts):
        df = generate_synthetic_kg(small_clusters, small_cohorts, small_config)
        required = [
            "edge_id", "subject", "predicate", "object_value", "timestamp",
            "cluster", "cohort", "entity", "true_tau", "true_shape",
            "velocity", "volatility", "is_censored", "lifetime_observed",
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_has_both_superseded_and_censored(self, small_config, small_clusters, small_cohorts):
        df = generate_synthetic_kg(small_clusters, small_cohorts, small_config)
        assert df["is_censored"].any(), "Expected some censored edges"
        assert (~df["is_censored"]).any(), "Expected some superseded edges"

    def test_fast_cluster_shorter_lifetime(self, small_config, small_clusters, small_cohorts):
        df = generate_synthetic_kg(small_clusters, small_cohorts, small_config)
        fast_lifetime = df[df["cluster"] == "fast"]["lifetime_observed"].median()
        slow_lifetime = df[df["cluster"] == "slow"]["lifetime_observed"].median()
        assert fast_lifetime < slow_lifetime, (
            f"Fast cluster ({fast_lifetime:.0f}d) should have shorter lifetime "
            f"than slow ({slow_lifetime:.0f}d)"
        )

    def test_timestamps_within_simulation(self, small_config, small_clusters, small_cohorts):
        df = generate_synthetic_kg(small_clusters, small_cohorts, small_config)
        total_days = small_config.n_years * 365.25
        assert df["timestamp"].min() >= 0
        assert df["timestamp"].max() <= total_days

    def test_supersession_chains(self, small_config, small_clusters, small_cohorts):
        df = generate_synthetic_kg(small_clusters, small_cohorts, small_config)
        superseded = df[df["superseded_by"].notna()]
        for _, edge in superseded.iterrows():
            successor = df[df["edge_id"] == edge["superseded_by"]]
            assert len(successor) == 1
            assert successor.iloc[0]["timestamp"] > edge["timestamp"]
            assert successor.iloc[0]["subject"] == edge["subject"]
            assert successor.iloc[0]["predicate"] == edge["predicate"]

    def test_deterministic_with_seed(self, small_config, small_clusters, small_cohorts):
        df1 = generate_synthetic_kg(small_clusters, small_cohorts, small_config)
        df2 = generate_synthetic_kg(small_clusters, small_cohorts, small_config)
        pd.testing.assert_frame_equal(df1, df2)

    def test_default_config_generates(self):
        df = generate_synthetic_kg()
        summary = summarize_synthetic_kg(df)
        assert summary["total_edges"] > 1000
        assert summary["clusters"] == 4


class TestSummarizeSyntheticKG:
    def test_summary_keys(self):
        df = generate_synthetic_kg()
        summary = summarize_synthetic_kg(df)
        expected_keys = [
            "total_edges", "superseded_edges", "censored_edges",
            "clusters", "cohorts", "entities", "predicates",
            "mean_lifetime_by_cluster", "supersession_rate_by_cluster",
        ]
        for key in expected_keys:
            assert key in summary, f"Missing key: {key}"
