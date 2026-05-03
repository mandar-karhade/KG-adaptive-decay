"""Tests for temporal cluster discovery."""

import numpy as np
import pandas as pd
import pytest

from src.synthetic.generator import generate_synthetic_kg
from src.synthetic.config import SyntheticKGConfig
from src.clustering.temporal_clusters import (
    build_temporal_features,
    build_predicate_features,
    cluster_hdbscan,
    cluster_dpgmm,
    evaluate_clusters,
    encode_ground_truth,
    discover_and_evaluate,
)


@pytest.fixture(scope="module")
def synthetic_df():
    """Generate synthetic KG once for all clustering tests."""
    return generate_synthetic_kg(config=SyntheticKGConfig(seed=42))


class TestBuildTemporalFeatures:
    def test_shape(self, synthetic_df):
        features = build_temporal_features(synthetic_df)
        assert features.shape[0] == len(synthetic_df)
        assert features.shape[1] == 5  # velocity, volatility, pred_mean_lifetime, pred_supersession_rate, reinforcement_density

    def test_no_nans(self, synthetic_df):
        features = build_temporal_features(synthetic_df)
        assert not np.any(np.isnan(features))

    def test_features_differ_by_cluster(self, synthetic_df):
        features = build_temporal_features(synthetic_df)
        # Permanent facts and volatile measurements should have very different feature means
        perm_mask = synthetic_df["cluster"] == "permanent_facts"
        vol_mask = synthetic_df["cluster"] == "volatile_measurements"
        perm_mean = features[perm_mask.values].mean(axis=0)
        vol_mean = features[vol_mask.values].mean(axis=0)
        # At least velocity and volatility should differ substantially
        assert abs(perm_mean[0] - vol_mean[0]) > 0.1  # velocity
        assert abs(perm_mean[1] - vol_mean[1]) > 0.1  # volatility


class TestClusterHDBSCAN:
    def test_finds_clusters_predicate_level(self, synthetic_df):
        _, features = build_predicate_features(synthetic_df)
        result = cluster_hdbscan(features, min_cluster_size=3, min_samples=2)
        assert result.n_clusters >= 2
        assert result.method == "hdbscan"

    def test_labels_shape(self, synthetic_df):
        _, features = build_predicate_features(synthetic_df)
        result = cluster_hdbscan(features, min_cluster_size=3, min_samples=2)
        assert len(result.labels) == features.shape[0]

    def test_has_silhouette(self, synthetic_df):
        _, features = build_predicate_features(synthetic_df)
        result = cluster_hdbscan(features, min_cluster_size=3, min_samples=2)
        if result.n_clusters > 1:
            assert result.silhouette is not None
            assert -1 <= result.silhouette <= 1


class TestClusterDPGMM:
    def test_finds_clusters(self, synthetic_df):
        _, features = build_predicate_features(synthetic_df)
        result = cluster_dpgmm(features)
        assert result.n_clusters >= 2
        assert result.method == "dpgmm"

    def test_labels_shape(self, synthetic_df):
        _, features = build_predicate_features(synthetic_df)
        result = cluster_dpgmm(features)
        assert len(result.labels) == features.shape[0]

    def test_no_noise_labels(self, synthetic_df):
        _, features = build_predicate_features(synthetic_df)
        result = cluster_dpgmm(features)
        assert (result.labels >= 0).all()


class TestEvaluation:
    def test_ari_nmi_computed(self, synthetic_df):
        pred_df, features = build_predicate_features(synthetic_df)
        result = cluster_dpgmm(features)
        cluster_names = sorted(pred_df["cluster"].unique())
        name_to_id = {name: i for i, name in enumerate(cluster_names)}
        ground_truth = pred_df["cluster"].map(name_to_id).values
        evaluated = evaluate_clusters(result, ground_truth)
        assert evaluated.ari is not None
        assert evaluated.nmi is not None
        assert -0.5 <= evaluated.ari <= 1.0
        assert 0.0 <= evaluated.nmi <= 1.0

    def test_perfect_labels_give_high_ari(self, synthetic_df):
        """If we use the ground truth as predictions, ARI should be 1.0."""
        ground_truth = encode_ground_truth(synthetic_df)
        from src.clustering.temporal_clusters import ClusterResult
        perfect = ClusterResult(
            labels=ground_truth,
            n_clusters=len(np.unique(ground_truth)),
            method="dpgmm",
        )
        evaluated = evaluate_clusters(perfect, ground_truth)
        assert evaluated.ari > 0.99

    def test_planted_clusters_recoverable(self, synthetic_df):
        """The planted clusters should be recoverable with ARI > 0.8 at predicate level."""
        results = discover_and_evaluate(synthetic_df, level="predicate")
        best_ari = max(r.ari for r in results.values() if r.ari is not None)
        assert best_ari > 0.8, f"Best ARI ({best_ari:.3f}) too low -- planted clusters not recoverable"


class TestEncodeGroundTruth:
    def test_encodes_to_ints(self, synthetic_df):
        labels = encode_ground_truth(synthetic_df)
        assert labels.dtype in [np.int64, np.int32, int]
        assert len(labels) == len(synthetic_df)

    def test_correct_number_of_classes(self, synthetic_df):
        labels = encode_ground_truth(synthetic_df)
        assert len(np.unique(labels)) == synthetic_df["cluster"].nunique()
