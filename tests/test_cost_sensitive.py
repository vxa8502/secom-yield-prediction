"""
Unit tests for cost-sensitive threshold optimization.
"""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.evaluation.metrics import calculate_gmean, calculate_weighted_gmean
from src.evaluation.threshold import find_optimal_threshold, find_optimal_threshold_cv
from src.config import DEFAULT_COST_RATIO, COST_PROFILES


class TestCalculateWeightedGmean:
    """Tests for calculate_weighted_gmean function."""

    @pytest.fixture
    def balanced_predictions(self):
        """Balanced binary predictions."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 0, 1, 1])  # 1 FP, 1 FN
        return y_true, y_pred

    def test_cost_ratio_1_equals_standard_gmean(self, balanced_predictions):
        """Test that cost_ratio=1.0 gives same result as standard G-Mean."""
        y_true, y_pred = balanced_predictions

        standard_gmean, sens, spec = calculate_gmean(y_true, y_pred)
        weighted_gmean, w_sens, w_spec = calculate_weighted_gmean(y_true, y_pred, cost_ratio=1.0)

        assert np.isclose(standard_gmean, weighted_gmean, rtol=1e-10)
        assert sens == w_sens
        assert spec == w_spec

    def test_higher_cost_ratio_favors_sensitivity(self, balanced_predictions):
        """Test that higher cost_ratio penalizes FN more (favors sensitivity)."""
        y_true, y_pred = balanced_predictions

        # Get weighted G-Mean at different cost ratios
        _, sens_1, spec_1 = calculate_weighted_gmean(y_true, y_pred, cost_ratio=1.0)
        _, sens_5, spec_5 = calculate_weighted_gmean(y_true, y_pred, cost_ratio=5.0)

        # Sensitivity and specificity values shouldn't change
        assert sens_1 == sens_5
        assert spec_1 == spec_5

    def test_perfect_classification(self):
        """Test weighted G-Mean with perfect classification."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 1, 1])

        gmean, sens, spec = calculate_weighted_gmean(y_true, y_pred, cost_ratio=5.0)

        assert gmean == 1.0
        assert sens == 1.0
        assert spec == 1.0

    def test_all_false_negatives(self):
        """Test weighted G-Mean when all positives are missed."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 0, 0, 0])  # All FN

        gmean, sens, spec = calculate_weighted_gmean(y_true, y_pred, cost_ratio=5.0)

        assert gmean == 0.0  # Zero sensitivity = zero weighted G-Mean
        assert sens == 0.0
        assert spec == 1.0

    def test_invalid_cost_ratio_raises(self):
        """Test that invalid cost_ratio raises ValueError."""
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])

        with pytest.raises(ValueError, match="must be positive"):
            calculate_weighted_gmean(y_true, y_pred, cost_ratio=0)

        with pytest.raises(ValueError, match="must be positive"):
            calculate_weighted_gmean(y_true, y_pred, cost_ratio=-1.0)

    def test_weighted_gmean_formula(self):
        """Test weighted G-Mean follows the formula: (sens^beta * spec)^(1/(1+beta))."""
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 0, 1, 1, 1])  # 1 FP, 1 FN

        # sens = 3/4 = 0.75, spec = 3/4 = 0.75
        beta = 5.0
        expected_sens = 0.75
        expected_spec = 0.75
        expected_weighted = (expected_sens ** beta * expected_spec) ** (1 / (1 + beta))

        gmean, sens, spec = calculate_weighted_gmean(y_true, y_pred, cost_ratio=beta)

        assert np.isclose(sens, expected_sens)
        assert np.isclose(spec, expected_spec)
        assert np.isclose(gmean, expected_weighted)


class TestCostProfiles:
    """Tests for COST_PROFILES configuration."""

    def test_cost_profiles_defined(self):
        """Test that standard cost profiles are defined."""
        assert 'balanced' in COST_PROFILES
        assert 'manufacturing_typical' in COST_PROFILES
        assert 'manufacturing_critical' in COST_PROFILES

    def test_balanced_is_default(self):
        """Test that balanced profile equals default cost ratio."""
        assert COST_PROFILES['balanced'] == DEFAULT_COST_RATIO
        assert DEFAULT_COST_RATIO == 1.0

    def test_manufacturing_ratios_greater_than_balanced(self):
        """Test that manufacturing profiles have higher cost ratios."""
        assert COST_PROFILES['manufacturing_typical'] > COST_PROFILES['balanced']
        assert COST_PROFILES['manufacturing_critical'] > COST_PROFILES['manufacturing_typical']


class TestFindOptimalThresholdWithCostRatio:
    """Tests for find_optimal_threshold with cost_ratio parameter."""

    @pytest.fixture
    def separable_data(self):
        """Create data with some overlap."""
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.3, 0.4, 0.45, 0.55, 0.6, 0.7, 0.8, 0.9])
        return y_true, y_proba

    def test_cost_ratio_in_result(self, separable_data):
        """Test that cost_ratio is included in result dict."""
        y_true, y_proba = separable_data

        result = find_optimal_threshold(y_true, y_proba, cost_ratio=5.0)

        assert 'cost_ratio' in result
        assert result['cost_ratio'] == 5.0

    def test_higher_cost_ratio_lowers_threshold(self, separable_data):
        """Test that higher cost_ratio tends to lower the threshold."""
        y_true, y_proba = separable_data

        result_balanced = find_optimal_threshold(y_true, y_proba, cost_ratio=1.0)
        result_high_cost = find_optimal_threshold(y_true, y_proba, cost_ratio=10.0)

        # Higher cost_ratio should favor catching more positives (lower threshold)
        # This may not always hold, but with typical data it should
        assert result_high_cost['optimal_threshold'] <= result_balanced['optimal_threshold']

    def test_backward_compatible_default(self, separable_data):
        """Test that default behavior (no cost_ratio) equals cost_ratio=1.0."""
        y_true, y_proba = separable_data

        result_default = find_optimal_threshold(y_true, y_proba)
        result_explicit = find_optimal_threshold(y_true, y_proba, cost_ratio=1.0)

        assert result_default['optimal_threshold'] == result_explicit['optimal_threshold']
        assert result_default['optimal_value'] == result_explicit['optimal_value']


class TestFindOptimalThresholdCVWithCostRatio:
    """Tests for find_optimal_threshold_cv with cost_ratio parameter."""

    @pytest.fixture
    def simple_pipeline(self):
        """Create a simple logistic regression pipeline."""
        return Pipeline([
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ])

    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        return X, y

    def test_cost_ratio_in_cv_result(self, simple_pipeline, sample_data):
        """Test that cost_ratio is included in CV result dict."""
        X, y = sample_data

        result = find_optimal_threshold_cv(simple_pipeline, X, y, cost_ratio=5.0)

        assert 'cost_ratio' in result
        assert result['cost_ratio'] == 5.0

    def test_cv_backward_compatible(self, simple_pipeline, sample_data):
        """Test CV function with default parameters matches explicit cost_ratio=1.0."""
        X, y = sample_data

        result_default = find_optimal_threshold_cv(simple_pipeline, X, y)
        result_explicit = find_optimal_threshold_cv(simple_pipeline, X, y, cost_ratio=1.0)

        # Thresholds should match
        assert result_default['optimal_threshold'] == result_explicit['optimal_threshold']
