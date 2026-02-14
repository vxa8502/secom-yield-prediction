"""
Unit tests for evaluation metrics.
"""

import numpy as np
from src.evaluation.metrics import calculate_gmean, gmean_scorer_func, evaluate_model


class TestCalculateGMean:
    """Tests for calculate_gmean function."""

    def test_perfect_predictions(self):
        """Test G-Mean = 1.0 for perfect predictions."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 1, 1])

        gmean, sensitivity, specificity = calculate_gmean(y_true, y_pred)

        assert gmean == 1.0
        assert sensitivity == 1.0
        assert specificity == 1.0

    def test_all_wrong_predictions(self):
        """Test G-Mean = 0.0 when all predictions are wrong."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 0, 0, 0])

        gmean, sensitivity, specificity = calculate_gmean(y_true, y_pred)

        assert gmean == 0.0
        assert sensitivity == 0.0
        assert specificity == 0.0

    def test_all_predict_positive(self):
        """Test when model predicts all positive (sensitivity=1, specificity=0)."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 1, 1, 1])

        gmean, sensitivity, specificity = calculate_gmean(y_true, y_pred)

        assert gmean == 0.0  # sqrt(1.0 * 0.0) = 0
        assert sensitivity == 1.0
        assert specificity == 0.0

    def test_all_predict_negative(self):
        """Test when model predicts all negative (sensitivity=0, specificity=1)."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 0, 0, 0])

        gmean, sensitivity, specificity = calculate_gmean(y_true, y_pred)

        assert gmean == 0.0  # sqrt(0.0 * 1.0) = 0
        assert sensitivity == 0.0
        assert specificity == 1.0

    def test_balanced_errors(self):
        """Test G-Mean with balanced errors."""
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 0, 1, 1])

        gmean, sensitivity, specificity = calculate_gmean(y_true, y_pred)

        # 2 TN, 2 FP, 2 FN, 2 TP
        # Sensitivity = 2/4 = 0.5
        # Specificity = 2/4 = 0.5
        # G-Mean = sqrt(0.5 * 0.5) = 0.5
        assert sensitivity == 0.5
        assert specificity == 0.5
        assert gmean == 0.5

    def test_imbalanced_classes(self):
        """Test G-Mean with imbalanced class distribution."""
        # 10 negatives, 2 positives (5:1 ratio)
        y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
        # Correctly predict all negatives, miss 1 positive
        y_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

        gmean, sensitivity, specificity = calculate_gmean(y_true, y_pred)

        # Sensitivity = 1/2 = 0.5
        # Specificity = 10/10 = 1.0
        # G-Mean = sqrt(0.5 * 1.0) = 0.707...
        assert sensitivity == 0.5
        assert specificity == 1.0
        assert np.isclose(gmean, np.sqrt(0.5))

    def test_empty_positive_class(self):
        """Test edge case when no true positives exist."""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 0, 1, 1])

        gmean, sensitivity, specificity = calculate_gmean(y_true, y_pred)

        # No positives -> sensitivity = 0 by definition
        assert sensitivity == 0.0
        assert specificity == 0.5
        assert gmean == 0.0


class TestGMeanScorer:
    """Tests for gmean_scorer_func."""

    def test_scorer_returns_float(self):
        """Test that scorer returns a float value."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1])

        result = gmean_scorer_func(y_true, y_pred)

        assert isinstance(result, float)

    def test_scorer_matches_calculate_gmean(self):
        """Test that scorer matches calculate_gmean output."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 0, 1, 1])

        scorer_result = gmean_scorer_func(y_true, y_pred)
        gmean, _, _ = calculate_gmean(y_true, y_pred)

        assert scorer_result == gmean


class TestEvaluateModel:
    """Tests for evaluate_model function."""

    def test_returns_all_metrics(self):
        """Test that all expected metrics are returned."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 0, 1, 1])

        results = evaluate_model(y_true, y_pred, model_name="TestModel")

        expected_keys = {'model', 'gmean', 'sensitivity', 'specificity',
                        'precision', 'f1_score', 'accuracy'}
        assert expected_keys.issubset(set(results.keys()))
        assert results['model'] == "TestModel"

    def test_with_probabilities(self):
        """Test that AUC metrics are included when probabilities provided."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.6, 0.4, 0.8, 0.9])

        results = evaluate_model(y_true, y_pred, y_proba, model_name="TestModel")

        assert 'auc_roc' in results
        assert 'auc_pr' in results
        assert 0 <= results['auc_roc'] <= 1
        assert 0 <= results['auc_pr'] <= 1

    def test_metrics_are_bounded(self):
        """Test that all metrics are in valid ranges."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 0, 1, 1])

        results = evaluate_model(y_true, y_pred)

        for key in ['gmean', 'sensitivity', 'specificity', 'precision', 'f1_score', 'accuracy']:
            assert 0 <= results[key] <= 1, f"{key} out of bounds: {results[key]}"
