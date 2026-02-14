"""
Unit tests for visualization utilities.
"""

import numpy as np
import pytest
import tempfile
from pathlib import Path

from src.visualization.plots import (
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_confusion_matrix,
    plot_calibration_curve,
    compare_feature_distributions,
)


class TestPlotROCCurve:
    """Tests for plot_roc_curve function."""

    def test_returns_auc_metric(self):
        """Test that AUC-ROC is returned correctly."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])

        result = plot_roc_curve(y_true, y_proba, model_name="Test")

        assert 'auc_roc' in result
        assert 0 <= result['auc_roc'] <= 1

    def test_perfect_predictions(self):
        """Test AUC-ROC = 1.0 for perfect separation."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_proba = np.array([0.0, 0.1, 0.2, 0.8, 0.9, 1.0])

        result = plot_roc_curve(y_true, y_proba)

        assert result['auc_roc'] == 1.0

    def test_random_predictions(self):
        """Test AUC-ROC ~ 0.5 for random predictions."""
        np.random.seed(42)
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_proba = np.random.rand(10)

        result = plot_roc_curve(y_true, y_proba)

        # Random should be close to 0.5 (within reasonable variance)
        assert 0.2 <= result['auc_roc'] <= 0.8

    def test_saves_figure(self):
        """Test that figure is saved when save_path provided."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.2, 0.3, 0.7, 0.8])

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "roc_curve.png"
            plot_roc_curve(y_true, y_proba, save_path=save_path)

            assert save_path.exists()
            assert save_path.stat().st_size > 0

    def test_returns_curve_data(self):
        """Test that FPR and TPR arrays are returned."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.2, 0.3, 0.7, 0.8])

        result = plot_roc_curve(y_true, y_proba)

        assert 'fpr' in result
        assert 'tpr' in result
        assert len(result['fpr']) > 0
        assert len(result['tpr']) > 0


class TestPlotPrecisionRecallCurve:
    """Tests for plot_precision_recall_curve function."""

    def test_returns_auc_pr_metric(self):
        """Test that AUC-PR is returned correctly."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])

        result = plot_precision_recall_curve(y_true, y_proba)

        assert 'auc_pr' in result
        assert 0 <= result['auc_pr'] <= 1

    def test_perfect_predictions(self):
        """Test AUC-PR = 1.0 for perfect separation."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_proba = np.array([0.0, 0.1, 0.2, 0.8, 0.9, 1.0])

        result = plot_precision_recall_curve(y_true, y_proba)

        assert result['auc_pr'] == 1.0

    def test_saves_figure(self):
        """Test that figure is saved when save_path provided."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.2, 0.3, 0.7, 0.8])

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "pr_curve.png"
            plot_precision_recall_curve(y_true, y_proba, save_path=save_path)

            assert save_path.exists()

    def test_returns_curve_data(self):
        """Test that precision and recall arrays are returned."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.2, 0.3, 0.7, 0.8])

        result = plot_precision_recall_curve(y_true, y_proba)

        assert 'precision' in result
        assert 'recall' in result


class TestPlotConfusionMatrix:
    """Tests for plot_confusion_matrix function."""

    def test_returns_matrix_values(self):
        """Test that confusion matrix values are returned."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 0, 1, 1])

        result = plot_confusion_matrix(y_true, y_pred)

        assert 'tn' in result
        assert 'fp' in result
        assert 'fn' in result
        assert 'tp' in result
        assert 'total' in result

    def test_matrix_values_sum_to_total(self):
        """Test that TP+TN+FP+FN = total."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 0, 1, 1])

        result = plot_confusion_matrix(y_true, y_pred)

        assert result['tn'] + result['fp'] + result['fn'] + result['tp'] == result['total']

    def test_perfect_predictions(self):
        """Test confusion matrix for perfect predictions."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 1, 1])

        result = plot_confusion_matrix(y_true, y_pred)

        assert result['tn'] == 3
        assert result['tp'] == 3
        assert result['fp'] == 0
        assert result['fn'] == 0

    def test_saves_figure(self):
        """Test that figure is saved when save_path provided."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1])

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "confusion_matrix.png"
            plot_confusion_matrix(y_true, y_pred, save_path=save_path)

            assert save_path.exists()


class TestPlotCalibrationCurve:
    """Tests for plot_calibration_curve function."""

    def test_returns_calibration_error(self):
        """Test that calibration error is returned."""
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.6, 0.7, 0.8, 0.9])

        result = plot_calibration_curve(y_true, y_proba)

        assert 'calibration_error' in result
        assert 0 <= result['calibration_error'] <= 1

    def test_perfect_calibration(self):
        """Test calibration error ~ 0 for well-calibrated predictions."""
        # Create well-calibrated probabilities
        y_true = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
        y_proba = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.7, 0.8, 0.9])

        result = plot_calibration_curve(y_true, y_proba, n_bins=3)

        # Well-calibrated should have low error
        assert result['calibration_error'] < 0.5

    def test_returns_curve_data(self):
        """Test that probability curves are returned."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

        result = plot_calibration_curve(y_true, y_proba, n_bins=2)

        assert 'prob_true' in result
        assert 'prob_pred' in result


class TestCompareFeatureDistributions:
    """Tests for compare_feature_distributions function."""

    def test_returns_ks_results(self):
        """Test that KS test results are returned."""
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        X_test = np.random.randn(50, 5)

        results = compare_feature_distributions(X_train, X_test)

        assert len(results) == 5
        assert all('ks_statistic' in r for r in results)
        assert all('p_value' in r for r in results)
        assert all('drift' in r for r in results)

    def test_detects_drift(self):
        """Test that significant distribution shift is detected."""
        np.random.seed(42)
        X_train = np.random.randn(100, 3)
        # Add significant shift to one feature
        X_test = np.random.randn(50, 3)
        X_test[:, 0] += 5.0  # Large shift in feature 0

        results = compare_feature_distributions(X_train, X_test)

        # Feature 0 should show drift
        feature_0_result = [r for r in results if r['feature'] == 'feature_0'][0]
        assert feature_0_result['drift']  # np.True_ == True but is not True

    def test_no_drift_same_distribution(self):
        """Test that similar distributions show no drift."""
        np.random.seed(42)
        X_train = np.random.randn(200, 3)
        X_test = np.random.randn(200, 3)

        results = compare_feature_distributions(X_train, X_test)

        # Most features should not show drift for same distribution
        drift_count = sum(1 for r in results if r['drift'])
        # Allow some false positives due to random sampling
        assert drift_count <= 1

    def test_saves_figure(self):
        """Test that figure is saved when save_path provided."""
        np.random.seed(42)
        X_train = np.random.randn(50, 6)
        X_test = np.random.randn(30, 6)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "distributions.png"
            compare_feature_distributions(X_train, X_test, save_path=save_path)

            assert save_path.exists()
