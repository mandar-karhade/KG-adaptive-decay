"""Tests for hierarchical Bayesian survival model."""

import numpy as np
import pandas as pd
import pytest

from src.synthetic.generator import generate_synthetic_kg
from src.synthetic.config import SyntheticKGConfig
from src.models.bayesian.survival_model import (
    fit_weibull,
    fit_decay_surface,
    fit_hierarchical,
    compute_survival_weight,
    compute_tau_from_surface,
    compare_distributions,
)


@pytest.fixture(scope="module")
def synthetic_df():
    return generate_synthetic_kg(config=SyntheticKGConfig(seed=42))


class TestFitWeibull:
    def test_basic_fit(self):
        rng = np.random.default_rng(42)
        # Generate Weibull data with known params: tau=100, shape=1.5
        true_tau, true_shape = 100.0, 1.5
        durations = true_tau * rng.weibull(true_shape, size=500)
        events = np.ones(500)  # all observed (no censoring)

        fit = fit_weibull(durations, events)
        assert 70 < fit.tau < 140, f"tau={fit.tau} not near 100"
        assert 1.0 < fit.shape < 2.0, f"shape={fit.shape} not near 1.5"
        assert fit.n_observations == 500
        assert fit.n_superseded == 500

    def test_with_censoring(self):
        rng = np.random.default_rng(42)
        durations = 100.0 * rng.weibull(1.0, size=200)
        # Censor 50% of observations
        events = np.zeros(200)
        events[:100] = 1
        fit = fit_weibull(durations, events)
        assert fit.n_superseded == 100
        assert fit.n_censored == 100
        assert np.isfinite(fit.tau)

    def test_too_few_observations(self):
        fit = fit_weibull(np.array([10.0, 20.0]), np.array([1, 0]))
        assert np.isnan(fit.tau)

    def test_aic_computed(self):
        rng = np.random.default_rng(42)
        durations = 50.0 * rng.weibull(1.0, size=100)
        events = np.ones(100)
        fit = fit_weibull(durations, events)
        assert fit.aic is not None
        assert np.isfinite(fit.aic)


class TestFitDecaySurface:
    def test_recovers_known_surface(self):
        rng = np.random.default_rng(42)
        n = 50
        v = rng.uniform(0, 1, n)
        s = rng.uniform(0, 1, n)
        true_theta = np.array([5.0, -1.0, -2.0, 0.5])
        log_taus = true_theta[0] + true_theta[1] * v + true_theta[2] * s + true_theta[3] * v * s
        log_taus += rng.normal(0, 0.1, n)  # small noise

        fit = fit_decay_surface(v, s, log_taus)
        assert fit.r_squared > 0.9, f"R²={fit.r_squared} too low"
        for i in range(4):
            assert abs(fit.theta[i] - true_theta[i]) < 0.5, (
                f"theta[{i}]={fit.theta[i]:.2f} not near {true_theta[i]}"
            )

    def test_with_regularization(self):
        rng = np.random.default_rng(42)
        n = 10
        v = rng.uniform(0, 1, n)
        s = rng.uniform(0, 1, n)
        log_taus = 5.0 - v - 2 * s + rng.normal(0, 0.5, n)

        fit_noreg = fit_decay_surface(v, s, log_taus, regularization=0.0)
        fit_reg = fit_decay_surface(v, s, log_taus, regularization=1.0)

        # Regularization should shrink coefficients
        assert np.linalg.norm(fit_reg.theta) <= np.linalg.norm(fit_noreg.theta) + 0.1

    def test_too_few_points(self):
        fit = fit_decay_surface(
            np.array([0.1, 0.2]),
            np.array([0.3, 0.4]),
            np.array([5.0, 4.5]),
        )
        assert np.isnan(fit.theta[0])


class TestHierarchicalFit:
    def test_fits_all_clusters(self, synthetic_df):
        result = fit_hierarchical(synthetic_df)
        expected_clusters = set(synthetic_df["cluster"].unique())
        assert set(result.cluster_fits.keys()) == expected_clusters

    def test_cluster_tau_ordering(self, synthetic_df):
        result = fit_hierarchical(synthetic_df)
        # permanent_facts should have much longer tau than volatile_measurements
        perm_tau = result.cluster_fits["permanent_facts"].tau
        vol_tau = result.cluster_fits["volatile_measurements"].tau
        assert perm_tau > vol_tau * 10, (
            f"permanent ({perm_tau:.0f}) should be >> volatile ({vol_tau:.0f})"
        )

    def test_cohort_fits_present(self, synthetic_df):
        result = fit_hierarchical(synthetic_df)
        assert len(result.cohort_fits) > 0
        # Each cohort fit should have a (cluster, cohort) key
        for key in result.cohort_fits:
            assert len(key) == 2

    def test_cohort_variation_within_cluster(self, synthetic_df):
        result = fit_hierarchical(synthetic_df)
        # Within volatile_measurements, ICU should have shorter tau than outpatient
        icu = result.cohort_fits.get(("volatile_measurements", "icu"))
        outpatient = result.cohort_fits.get(("volatile_measurements", "outpatient"))
        if icu and outpatient:
            assert icu.tau < outpatient.tau, (
                f"ICU tau ({icu.tau:.1f}) should be < outpatient ({outpatient.tau:.1f})"
            )

    def test_individual_fits_present(self, synthetic_df):
        result = fit_hierarchical(synthetic_df)
        assert len(result.individual_fits) > 0

    def test_distribution_comparison(self, synthetic_df):
        result = fit_hierarchical(synthetic_df)
        assert result.distribution_comparison is not None
        for cluster in synthetic_df["cluster"].unique():
            assert cluster in result.distribution_comparison
            dists = result.distribution_comparison[cluster]
            assert len(dists) > 0

    def test_shape_parameter_lindy(self, synthetic_df):
        """Permanent facts should have shape < 1 (Lindy effect)."""
        result = fit_hierarchical(synthetic_df)
        perm_shape = result.cluster_fits["permanent_facts"].shape
        assert perm_shape < 1.0, f"permanent_facts shape={perm_shape} should be < 1 (Lindy)"


class TestComputeSurvivalWeight:
    def test_zero_age(self):
        w = compute_survival_weight(0.0, 100.0, 1.0)
        assert w == 1.0

    def test_decays_with_age(self):
        w1 = compute_survival_weight(10.0, 100.0, 1.0)
        w2 = compute_survival_weight(50.0, 100.0, 1.0)
        assert w1 > w2

    def test_longer_tau_slower_decay(self):
        w_short = compute_survival_weight(50.0, 10.0, 1.0)
        w_long = compute_survival_weight(50.0, 1000.0, 1.0)
        assert w_long > w_short

    def test_invalid_tau(self):
        assert compute_survival_weight(10.0, 0.0, 1.0) == 0.0
        assert compute_survival_weight(10.0, -5.0, 1.0) == 0.0
        assert compute_survival_weight(10.0, np.nan, 1.0) == 0.0


class TestComputeTauFromSurface:
    def test_matches_compute_tau(self):
        from src.synthetic.generator import compute_tau
        theta = np.array([5.0, -1.0, -2.0, 0.5])
        v, s = 0.3, 0.6
        tau1 = compute_tau(v, s, theta)
        tau2 = compute_tau_from_surface(v, s, theta)
        assert np.isclose(tau1, tau2, rtol=1e-5)
