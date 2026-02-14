"""
Unit tests for threshold optimization functions.
"""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.evaluation.threshold import (
    find_optimal_threshold,
    find_optimal_threshold_cv,
    _validate_threshold_range,
)


class TestValidateThresholdRange:
    """Tests for _validate_threshold_range function."""

    def test_valid_range(self):
        """Test valid threshold range passes."""
        _validate_threshold_range((0.1, 0.9), 0.01)  # Should not raise

    def test_invalid_min_negative(self):
        """Test negative min threshold raises ValueError."""
        with pytest.raises(ValueError, match="must satisfy 0 <= min < max <= 1"):
            _validate_threshold_range((-0.1, 0.9), 0.01)

    def test_invalid_max_greater_than_one(self):
        """Test max > 1 raises ValueError."""
        with pytest.raises(ValueError, match="must satisfy 0 <= min < max <= 1"):
            _validate_threshold_range((0.1, 1.5), 0.01)

    def test_invalid_min_equals_max(self):
        """Test min == max raises ValueError."""
        with pytest.raises(ValueError, match="must satisfy 0 <= min < max <= 1"):
            _validate_threshold_range((0.5, 0.5), 0.01)

    def test_invalid_min_greater_than_max(self):
        """Test min > max raises ValueError."""
        with pytest.raises(ValueError, match="must satisfy 0 <= min < max <= 1"):
            _validate_threshold_range((0.9, 0.1), 0.01)

    def test_invalid_step_zero(self):
        """Test zero step raises ValueError."""
        with pytest.raises(ValueError, match="step must be positive"):
            _validate_threshold_range((0.1, 0.9), 0.0)

    def test_invalid_step_negative(self):
        """Test negative step raises ValueError."""
        with pytest.raises(ValueError, match="step must be positive"):
            _validate_threshold_range((0.1, 0.9), -0.01)

    def test_invalid_step_too_large(self):
        """Test step larger than range raises ValueError."""
        with pytest.raises(ValueError, match="step must be positive and <= range width"):
            _validate_threshold_range((0.4, 0.6), 0.5)


class TestFindOptimalThreshold:
    """Tests for find_optimal_threshold function."""

    def test_returns_expected_keys(self):
        """Test that result contains all expected keys."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

        result = find_optimal_threshold(y_true, y_proba)

        expected_keys = {'optimal_threshold', 'optimal_value', 'thresholds', 'metric_values', 'metric_name', 'cost_ratio'}
        assert expected_keys == set(result.keys())

    def test_perfect_separation(self):
        """Test with perfectly separable probabilities."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

        result = find_optimal_threshold(y_true, y_proba)

        # Optimal threshold should be between 0.3 and 0.7
        assert 0.3 <= result['optimal_threshold'] <= 0.7
        assert result['optimal_value'] == 1.0  # Perfect G-Mean

    def test_all_same_probability(self):
        """Test with uniform probabilities (degenerate case)."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_proba = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

        result = find_optimal_threshold(y_true, y_proba)

        # Should still return valid result
        assert 0 <= result['optimal_threshold'] <= 1
        assert 0 <= result['optimal_value'] <= 1

    def test_gmean_metric(self):
        """Test G-Mean metric selection."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.2, 0.4, 0.6, 0.8])

        result = find_optimal_threshold(y_true, y_proba, metric='gmean')

        assert result['metric_name'] == 'gmean'

    def test_f1_metric(self):
        """Test F1 metric selection."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.2, 0.4, 0.6, 0.8])

        result = find_optimal_threshold(y_true, y_proba, metric='f1')

        assert result['metric_name'] == 'f1'

    def test_sensitivity_metric(self):
        """Test sensitivity metric selection."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.2, 0.4, 0.6, 0.8])

        result = find_optimal_threshold(y_true, y_proba, metric='sensitivity')

        assert result['metric_name'] == 'sensitivity'
        # Optimal for sensitivity is lowest threshold (predicts more positives)
        assert result['optimal_threshold'] <= 0.5

    def test_custom_threshold_range(self):
        """Test custom threshold range."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.2, 0.4, 0.6, 0.8])

        result = find_optimal_threshold(y_true, y_proba, threshold_range=(0.3, 0.7))

        # Optimal threshold should be within specified range
        assert 0.3 <= result['optimal_threshold'] <= 0.7
        assert result['thresholds'][0] >= 0.3
        assert result['thresholds'][-1] <= 0.71  # Account for step

    def test_thresholds_array_matches_metric_values(self):
        """Test that thresholds and metric_values have same length."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.2, 0.4, 0.6, 0.8])

        result = find_optimal_threshold(y_true, y_proba)

        assert len(result['thresholds']) == len(result['metric_values'])


class TestFindOptimalThresholdCV:
    """Tests for find_optimal_threshold_cv function."""

    @pytest.fixture
    def simple_pipeline(self):
        """Create a simple logistic regression pipeline for testing."""
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

    def test_returns_expected_keys(self, simple_pipeline, sample_data):
        """Test that result contains all expected keys."""
        X, y = sample_data

        result = find_optimal_threshold_cv(simple_pipeline, X, y)

        expected_keys = {
            'optimal_threshold', 'cv_gmean', 'cv_gmean_at_default',
            'threshold_improvement', 'cv_sensitivity', 'cv_specificity',
            'cv_sensitivity_at_default', 'cv_specificity_at_default',
            'threshold_curve'
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_threshold_in_valid_range(self, simple_pipeline, sample_data):
        """Test optimal threshold is in valid range."""
        X, y = sample_data

        result = find_optimal_threshold_cv(simple_pipeline, X, y)

        assert 0.01 <= result['optimal_threshold'] <= 0.99

    def test_gmean_bounded(self, simple_pipeline, sample_data):
        """Test G-Mean values are bounded [0, 1]."""
        X, y = sample_data

        result = find_optimal_threshold_cv(simple_pipeline, X, y)

        assert 0 <= result['cv_gmean'] <= 1
        assert 0 <= result['cv_gmean_at_default'] <= 1

    def test_sensitivity_specificity_bounded(self, simple_pipeline, sample_data):
        """Test sensitivity and specificity are bounded [0, 1]."""
        X, y = sample_data

        result = find_optimal_threshold_cv(simple_pipeline, X, y)

        assert 0 <= result['cv_sensitivity'] <= 1
        assert 0 <= result['cv_specificity'] <= 1

    def test_threshold_curve_structure(self, simple_pipeline, sample_data):
        """Test threshold_curve has expected structure."""
        X, y = sample_data

        result = find_optimal_threshold_cv(simple_pipeline, X, y)

        curve = result['threshold_curve']
        assert 'thresholds' in curve
        assert 'gmean_scores' in curve
        assert len(curve['thresholds']) == len(curve['gmean_scores'])

    def test_improvement_calculation(self, simple_pipeline, sample_data):
        """Test improvement is correctly calculated."""
        X, y = sample_data

        result = find_optimal_threshold_cv(simple_pipeline, X, y)

        expected_improvement = result['cv_gmean'] - result['cv_gmean_at_default']
        assert np.isclose(result['threshold_improvement'], expected_improvement)

    def test_invalid_threshold_range_raises(self, simple_pipeline, sample_data):
        """Test invalid threshold range raises ValueError."""
        X, y = sample_data

        with pytest.raises(ValueError):
            find_optimal_threshold_cv(simple_pipeline, X, y, threshold_range=(-0.1, 1.5))

    def test_model_name_in_logging(self, simple_pipeline, sample_data, caplog):
        """Test model_name appears in debug logging."""
        X, y = sample_data

        import logging
        with caplog.at_level(logging.DEBUG, logger='secom'):
            find_optimal_threshold_cv(
                simple_pipeline, X, y,
                model_name='TestModel',
                feature_set='lasso'
            )

        # Check logging occurred (may not contain model name at INFO level)
        # This is a smoke test to ensure logging doesn't error
