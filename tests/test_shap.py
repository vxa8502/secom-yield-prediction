"""
Unit tests for SHAP explainability utilities.
"""

import numpy as np
import pytest

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.visualization.shap_plots import (
    create_shap_explainer,
    compute_shap_values,
    get_top_shap_features,
    clear_explainer_cache,
    get_cached_explainer,
)


class TestCreateShapExplainer:
    """Tests for create_shap_explainer function."""

    def test_creates_linear_explainer_for_logreg(self):
        """Test that LinearExplainer is created for LogisticRegression."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = (X[:, 0] > 0).astype(int)

        model = LogisticRegression(random_state=42, max_iter=200)
        model.fit(X, y)

        explainer = create_shap_explainer(model, X, model_type='auto')

        # Check explainer was created (LinearExplainer for LogReg)
        assert explainer is not None

    def test_creates_tree_explainer_for_rf(self):
        """Test that TreeExplainer is created for RandomForest."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = (X[:, 0] > 0).astype(int)

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        explainer = create_shap_explainer(model, X, model_type='auto')

        assert explainer is not None

    def test_explicit_model_type_override(self):
        """Test that explicit model_type overrides auto-detection."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = (X[:, 0] > 0).astype(int)

        model = LogisticRegression(random_state=42, max_iter=200)
        model.fit(X, y)

        # Force kernel explainer even for linear model
        explainer = create_shap_explainer(model, X, model_type='kernel', max_background_samples=20)

        assert explainer is not None

    def test_subsamples_large_background(self):
        """Test that large background datasets are subsampled."""
        np.random.seed(42)
        X = np.random.randn(500, 5)
        y = (X[:, 0] > 0).astype(int)

        model = LogisticRegression(random_state=42, max_iter=200)
        model.fit(X, y)

        # Should subsample to max_background_samples
        explainer = create_shap_explainer(model, X, model_type='linear', max_background_samples=50)

        assert explainer is not None

    def test_invalid_model_type_raises(self):
        """Test that invalid model_type raises ValueError."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = (X[:, 0] > 0).astype(int)

        model = LogisticRegression(random_state=42, max_iter=200)
        model.fit(X, y)

        with pytest.raises(ValueError, match="Invalid model_type"):
            create_shap_explainer(model, X, model_type='invalid')


class TestComputeShapValues:
    """Tests for compute_shap_values function."""

    def test_returns_shap_values(self):
        """Test that SHAP values are computed."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = (X[:, 0] > 0).astype(int)

        model = LogisticRegression(random_state=42, max_iter=200)
        model.fit(X, y)

        explainer = create_shap_explainer(model, X[:50], model_type='linear')
        shap_values, indices = compute_shap_values(explainer, X[:10])

        assert shap_values is not None
        assert shap_values.shape == (10, 5)

    def test_subsamples_large_explain_set(self):
        """Test that large explain sets are subsampled."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = (X[:, 0] > 0).astype(int)

        model = LogisticRegression(random_state=42, max_iter=200)
        model.fit(X, y)

        explainer = create_shap_explainer(model, X[:50], model_type='linear')
        shap_values, indices = compute_shap_values(explainer, X, max_samples=20)

        assert shap_values.shape[0] == 20
        assert indices is not None
        assert len(indices) == 20

    def test_returns_none_indices_when_no_subsampling(self):
        """Test that indices is None when no subsampling needed."""
        np.random.seed(42)
        X = np.random.randn(50, 5)
        y = (X[:, 0] > 0).astype(int)

        model = LogisticRegression(random_state=42, max_iter=200)
        model.fit(X, y)

        explainer = create_shap_explainer(model, X[:30], model_type='linear')
        shap_values, indices = compute_shap_values(explainer, X[:10])

        assert indices is None


class TestGetTopShapFeatures:
    """Tests for get_top_shap_features function."""

    def test_returns_top_n_features(self):
        """Test that correct number of top features returned."""
        np.random.seed(42)
        shap_values = np.random.randn(100, 10)
        feature_names = [f'feature_{i}' for i in range(10)]

        result = get_top_shap_features(shap_values, feature_names, top_n=5)

        assert len(result) == 5
        assert 'feature' in result.columns
        assert 'mean_abs_shap' in result.columns
        assert 'rank' in result.columns

    def test_features_sorted_by_importance(self):
        """Test that features are sorted by mean absolute SHAP."""
        # Create SHAP values where feature 0 is most important
        shap_values = np.zeros((100, 5))
        shap_values[:, 0] = 10.0  # Most important
        shap_values[:, 1] = 5.0
        shap_values[:, 2] = 1.0
        shap_values[:, 3] = 0.5
        shap_values[:, 4] = 0.1

        feature_names = ['feat_a', 'feat_b', 'feat_c', 'feat_d', 'feat_e']
        result = get_top_shap_features(shap_values, feature_names, top_n=3)

        assert result.iloc[0]['feature'] == 'feat_a'
        assert result.iloc[1]['feature'] == 'feat_b'
        assert result.iloc[2]['feature'] == 'feat_c'

    def test_generates_default_feature_names(self):
        """Test that default feature names generated when not provided."""
        shap_values = np.random.randn(50, 5)

        result = get_top_shap_features(shap_values, feature_names=None, top_n=3)

        assert result.iloc[0]['feature'].startswith('feature_')

    def test_rank_column_is_sequential(self):
        """Test that rank column is 1-indexed and sequential."""
        shap_values = np.random.randn(50, 10)
        result = get_top_shap_features(shap_values, top_n=5)

        assert list(result['rank']) == [1, 2, 3, 4, 5]


class TestExplainerCache:
    """Tests for SHAP explainer caching functionality."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_explainer_cache()

    def test_clear_cache(self):
        """Test that cache is cleared."""
        # This mainly tests that clear doesn't raise
        clear_explainer_cache()

    def test_cached_explainer_reused(self):
        """Test that cached explainer is returned on second call."""
        np.random.seed(42)
        X = np.random.randn(50, 5)
        y = (X[:, 0] > 0).astype(int)

        model = LogisticRegression(random_state=42, max_iter=200)
        model.fit(X, y)

        # First call creates explainer
        explainer1 = get_cached_explainer(model, X)

        # Second call should return same object
        explainer2 = get_cached_explainer(model, X)

        assert explainer1 is explainer2

    def test_different_models_get_different_explainers(self):
        """Test that different models get different explainers."""
        np.random.seed(42)
        X = np.random.randn(50, 5)
        y = (X[:, 0] > 0).astype(int)

        model1 = LogisticRegression(random_state=42, max_iter=200)
        model1.fit(X, y)

        model2 = LogisticRegression(random_state=43, max_iter=200)
        model2.fit(X, y)

        explainer1 = get_cached_explainer(model1, X)
        explainer2 = get_cached_explainer(model2, X)

        # Different models should have different explainers
        assert explainer1 is not explainer2


class TestShapValuesShape:
    """Tests for SHAP value array shapes and properties."""

    def test_shap_values_match_input_shape(self):
        """Test that SHAP values match input feature dimensions."""
        np.random.seed(42)
        n_samples, n_features = 100, 8
        X = np.random.randn(n_samples, n_features)
        y = (X[:, 0] > 0).astype(int)

        model = LogisticRegression(random_state=42, max_iter=200)
        model.fit(X, y)

        explainer = create_shap_explainer(model, X[:50], model_type='linear')
        shap_values, _ = compute_shap_values(explainer, X[:20])

        assert shap_values.shape == (20, n_features)

    def test_shap_values_finite(self):
        """Test that SHAP values contain no NaN or Inf."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = (X[:, 0] > 0).astype(int)

        model = LogisticRegression(random_state=42, max_iter=200)
        model.fit(X, y)

        explainer = create_shap_explainer(model, X[:50], model_type='linear')
        shap_values, _ = compute_shap_values(explainer, X[:10])

        assert np.all(np.isfinite(shap_values))
