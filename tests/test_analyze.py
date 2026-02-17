"""
Unit tests for interpretability analysis pipeline.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import sys
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pipelines.analyze import (
    analyze_misclassifications,
    analyze_residuals,
    analyze_missingness_performance,
    analyze_fn_clusters,
    load_feature_names,
    generate_analysis_report,
    _report_header,
    _report_misclassification,
    _report_residuals,
    _report_missingness,
    _report_clusters,
    _report_recommendations,
)


class TestAnalyzeMisclassifications:
    """Tests for analyze_misclassifications function."""

    @pytest.fixture
    def trained_pipeline(self):
        """Create a trained pipeline for testing."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = (X[:, 0] > 0).astype(int)

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(random_state=42, max_iter=200))
        ])
        pipeline.fit(X, y)
        return pipeline

    @pytest.fixture
    def test_data_with_fn(self):
        """Create test data guaranteed to have false negatives."""
        np.random.seed(42)
        X_test = np.random.randn(50, 5)
        # Create labels where some positives are hard to predict
        y_test = np.zeros(50, dtype=int)
        y_test[40:50] = 1  # Last 10 are positives

        # Probabilities that will cause FNs (low proba for some positives)
        y_proba = np.random.rand(50) * 0.5  # All below 0.5
        y_proba[45:50] = 0.8  # Only 5 positives get high proba

        y_pred = (y_proba >= 0.5).astype(int)

        return X_test, y_test, y_pred, y_proba

    @pytest.fixture
    def test_data_no_fn(self):
        """Create test data with no false negatives."""
        np.random.seed(42)
        X_test = np.random.randn(50, 5)
        y_test = np.zeros(50, dtype=int)
        y_test[40:50] = 1

        # All positives correctly predicted
        y_proba = np.zeros(50)
        y_proba[40:50] = 0.9

        y_pred = (y_proba >= 0.5).astype(int)

        return X_test, y_test, y_pred, y_proba

    def test_returns_dataframe(self, trained_pipeline, test_data_with_fn):
        """Test that function returns a DataFrame."""
        X_test, y_test, y_pred, y_proba = test_data_with_fn
        feature_names = [f'feature_{i}' for i in range(5)]

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('pipelines.analyze.FIGURES_DIR', Path(tmpdir)):
                with patch('pipelines.analyze.REPORTS_DIR', Path(tmpdir)):
                    result = analyze_misclassifications(
                        trained_pipeline, X_test, y_test, y_pred, y_proba,
                        feature_names, threshold=0.5, n_examples=3
                    )

        assert isinstance(result, pd.DataFrame)

    def test_fn_analysis_columns(self, trained_pipeline, test_data_with_fn):
        """Test that result has expected columns."""
        X_test, y_test, y_pred, y_proba = test_data_with_fn
        feature_names = [f'feature_{i}' for i in range(5)]

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('pipelines.analyze.FIGURES_DIR', Path(tmpdir)):
                with patch('pipelines.analyze.REPORTS_DIR', Path(tmpdir)):
                    result = analyze_misclassifications(
                        trained_pipeline, X_test, y_test, y_pred, y_proba,
                        feature_names, threshold=0.5, n_examples=3
                    )

        if len(result) > 0:
            expected_cols = {'rank', 'test_index', 'true_label', 'predicted_label',
                            'predicted_proba', 'error_type', 'distance_to_threshold'}
            assert expected_cols.issubset(set(result.columns))

    def test_handles_zero_false_negatives(self, trained_pipeline, test_data_no_fn):
        """Test graceful handling when no false negatives exist."""
        X_test, y_test, y_pred, y_proba = test_data_no_fn
        feature_names = [f'feature_{i}' for i in range(5)]

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('pipelines.analyze.FIGURES_DIR', Path(tmpdir)):
                with patch('pipelines.analyze.REPORTS_DIR', Path(tmpdir)):
                    result = analyze_misclassifications(
                        trained_pipeline, X_test, y_test, y_pred, y_proba,
                        feature_names, threshold=0.5, n_examples=5
                    )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_respects_n_examples_limit(self, trained_pipeline, test_data_with_fn):
        """Test that n_examples limits output."""
        X_test, y_test, y_pred, y_proba = test_data_with_fn
        feature_names = [f'feature_{i}' for i in range(5)]

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('pipelines.analyze.FIGURES_DIR', Path(tmpdir)):
                with patch('pipelines.analyze.REPORTS_DIR', Path(tmpdir)):
                    result = analyze_misclassifications(
                        trained_pipeline, X_test, y_test, y_pred, y_proba,
                        feature_names, threshold=0.5, n_examples=2
                    )

        assert len(result) <= 2

    def test_sorted_by_predicted_probability(self, trained_pipeline, test_data_with_fn):
        """Test that FNs are sorted by predicted probability (worst first)."""
        X_test, y_test, y_pred, y_proba = test_data_with_fn
        feature_names = [f'feature_{i}' for i in range(5)]

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('pipelines.analyze.FIGURES_DIR', Path(tmpdir)):
                with patch('pipelines.analyze.REPORTS_DIR', Path(tmpdir)):
                    result = analyze_misclassifications(
                        trained_pipeline, X_test, y_test, y_pred, y_proba,
                        feature_names, threshold=0.5, n_examples=10
                    )

        if len(result) > 1:
            # Probabilities should be increasing (worst = lowest proba first)
            probas = result['predicted_proba'].values
            assert all(probas[i] <= probas[i+1] for i in range(len(probas)-1))


class TestAnalyzeResiduals:
    """Tests for analyze_residuals function."""

    @pytest.fixture
    def balanced_predictions(self):
        """Create balanced predictions for testing."""
        y_test = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.6, 0.7, 0.8, 0.9])
        y_pred = (y_proba >= 0.5).astype(int)
        return y_test, y_proba, y_pred

    @pytest.fixture
    def overconfident_predictions(self):
        """Create predictions where model is overconfident on passes."""
        y_test = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        # Very low probabilities for passes (too confident)
        y_proba = np.array([0.01, 0.02, 0.01, 0.02, 0.01, 0.6, 0.7, 0.8, 0.9, 0.95])
        y_pred = (y_proba >= 0.5).astype(int)
        return y_test, y_proba, y_pred

    def test_returns_dict(self, balanced_predictions):
        """Test that function returns a dictionary."""
        y_test, y_proba, y_pred = balanced_predictions

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('pipelines.analyze.FIGURES_DIR', Path(tmpdir)):
                with patch('pipelines.analyze.REPORTS_DIR', Path(tmpdir)):
                    result = analyze_residuals(y_test, y_proba, y_pred)

        assert isinstance(result, dict)

    def test_has_expected_keys(self, balanced_predictions):
        """Test that result contains expected sections."""
        y_test, y_proba, y_pred = balanced_predictions

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('pipelines.analyze.FIGURES_DIR', Path(tmpdir)):
                with patch('pipelines.analyze.REPORTS_DIR', Path(tmpdir)):
                    result = analyze_residuals(y_test, y_proba, y_pred)

        assert 'overall' in result
        assert 'passes' in result
        assert 'failures' in result
        assert 'interpretation' in result

    def test_overall_statistics(self, balanced_predictions):
        """Test that overall statistics are computed correctly."""
        y_test, y_proba, y_pred = balanced_predictions

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('pipelines.analyze.FIGURES_DIR', Path(tmpdir)):
                with patch('pipelines.analyze.REPORTS_DIR', Path(tmpdir)):
                    result = analyze_residuals(y_test, y_proba, y_pred)

        overall = result['overall']
        assert 'mean' in overall
        assert 'std' in overall
        assert 'median' in overall
        assert 'skewness' in overall
        assert 'kurtosis' in overall

    def test_residuals_computed_correctly(self, balanced_predictions):
        """Test that residuals are y_true - y_proba."""
        y_test, y_proba, y_pred = balanced_predictions
        expected_mean = np.mean(y_test - y_proba)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('pipelines.analyze.FIGURES_DIR', Path(tmpdir)):
                with patch('pipelines.analyze.REPORTS_DIR', Path(tmpdir)):
                    result = analyze_residuals(y_test, y_proba, y_pred)

        assert np.isclose(result['overall']['mean'], expected_mean)

    def test_interpretation_values(self, balanced_predictions):
        """Test that interpretation contains valid bias labels."""
        y_test, y_proba, y_pred = balanced_predictions

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('pipelines.analyze.FIGURES_DIR', Path(tmpdir)):
                with patch('pipelines.analyze.REPORTS_DIR', Path(tmpdir)):
                    result = analyze_residuals(y_test, y_proba, y_pred)

        valid_biases = {'overconfident', 'underconfident', 'calibrated'}
        assert result['interpretation']['pass_bias'] in valid_biases
        assert result['interpretation']['fail_bias'] in valid_biases

    def test_saves_json_report(self, balanced_predictions):
        """Test that JSON report is saved."""
        y_test, y_proba, y_pred = balanced_predictions

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('pipelines.analyze.FIGURES_DIR', Path(tmpdir)):
                with patch('pipelines.analyze.REPORTS_DIR', Path(tmpdir)):
                    analyze_residuals(y_test, y_proba, y_pred)
                    json_path = Path(tmpdir) / 'residual_analysis.json'
                    assert json_path.exists()

    def test_all_same_class(self):
        """Test handling when all samples are same class."""
        y_test = np.array([0, 0, 0, 0, 0])
        y_proba = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        y_pred = np.zeros(5, dtype=int)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('pipelines.analyze.FIGURES_DIR', Path(tmpdir)):
                with patch('pipelines.analyze.REPORTS_DIR', Path(tmpdir)):
                    result = analyze_residuals(y_test, y_proba, y_pred)

        # Should still return valid result
        assert result['failures']['n_samples'] == 0


class TestAnalyzeFnClusters:
    """Tests for analyze_fn_clusters function."""

    @pytest.fixture
    def data_with_multiple_fn(self):
        """Create test data with multiple false negatives for clustering."""
        np.random.seed(42)
        X_test = np.random.randn(100, 5)
        y_test = np.zeros(100, dtype=int)
        y_test[80:100] = 1  # 20 positives

        # Create predictions that miss some positives
        y_proba = np.random.rand(100) * 0.4
        y_proba[90:100] = 0.8  # Only last 10 predicted correctly

        y_pred = (y_proba >= 0.5).astype(int)
        feature_names = [f'feature_{i}' for i in range(5)]

        return X_test, y_test, y_pred, y_proba, feature_names

    @pytest.fixture
    def data_with_few_fn(self):
        """Create test data with fewer FNs than clusters."""
        np.random.seed(42)
        X_test = np.random.randn(50, 5)
        y_test = np.zeros(50, dtype=int)
        y_test[48:50] = 1  # Only 2 positives

        y_proba = np.zeros(50)
        y_proba[48] = 0.3  # 1 FN
        y_proba[49] = 0.8  # Correctly predicted

        y_pred = (y_proba >= 0.5).astype(int)
        feature_names = [f'feature_{i}' for i in range(5)]

        return X_test, y_test, y_pred, y_proba, feature_names

    def test_returns_dataframe(self, data_with_multiple_fn):
        """Test that function returns a DataFrame."""
        X_test, y_test, y_pred, y_proba, feature_names = data_with_multiple_fn

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('pipelines.analyze.FIGURES_DIR', Path(tmpdir)):
                with patch('pipelines.analyze.REPORTS_DIR', Path(tmpdir)):
                    result = analyze_fn_clusters(
                        X_test, y_test, y_pred, y_proba, feature_names, n_clusters=3
                    )

        assert isinstance(result, pd.DataFrame)

    def test_cluster_analysis_columns(self, data_with_multiple_fn):
        """Test that result has expected columns."""
        X_test, y_test, y_pred, y_proba, feature_names = data_with_multiple_fn

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('pipelines.analyze.FIGURES_DIR', Path(tmpdir)):
                with patch('pipelines.analyze.REPORTS_DIR', Path(tmpdir)):
                    result = analyze_fn_clusters(
                        X_test, y_test, y_pred, y_proba, feature_names, n_clusters=3
                    )

        if len(result) > 0:
            expected_cols = {'cluster', 'n_samples', 'mean_proba', 'distinctive_features'}
            assert expected_cols.issubset(set(result.columns))

    def test_handles_insufficient_fn(self, data_with_few_fn):
        """Test graceful handling when FNs < n_clusters."""
        X_test, y_test, y_pred, y_proba, feature_names = data_with_few_fn

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('pipelines.analyze.FIGURES_DIR', Path(tmpdir)):
                with patch('pipelines.analyze.REPORTS_DIR', Path(tmpdir)):
                    result = analyze_fn_clusters(
                        X_test, y_test, y_pred, y_proba, feature_names, n_clusters=5
                    )

        # Should return empty DataFrame when insufficient samples
        assert isinstance(result, pd.DataFrame)

    def test_cluster_count_respects_data(self, data_with_multiple_fn):
        """Test that actual clusters <= requested clusters."""
        X_test, y_test, y_pred, y_proba, feature_names = data_with_multiple_fn

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('pipelines.analyze.FIGURES_DIR', Path(tmpdir)):
                with patch('pipelines.analyze.REPORTS_DIR', Path(tmpdir)):
                    result = analyze_fn_clusters(
                        X_test, y_test, y_pred, y_proba, feature_names, n_clusters=3
                    )

        if len(result) > 0:
            assert len(result) <= 3

    def test_saves_csv_report(self, data_with_multiple_fn):
        """Test that CSV report is saved."""
        X_test, y_test, y_pred, y_proba, feature_names = data_with_multiple_fn

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('pipelines.analyze.FIGURES_DIR', Path(tmpdir)):
                with patch('pipelines.analyze.REPORTS_DIR', Path(tmpdir)):
                    analyze_fn_clusters(
                        X_test, y_test, y_pred, y_proba, feature_names, n_clusters=3
                    )
                    csv_path = Path(tmpdir) / 'fn_cluster_analysis.csv'
                    assert csv_path.exists()


class TestAnalyzeMissingnessPerformance:
    """Tests for analyze_missingness_performance function."""

    def test_handles_missing_raw_data(self):
        """Test graceful handling when raw data file not found."""
        np.random.seed(42)
        X_test = np.random.randn(50, 5)
        y_test = np.zeros(50, dtype=int)
        y_pred = np.zeros(50, dtype=int)
        y_proba = np.random.rand(50)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('pipelines.analyze.SECOM_RAW_DIR', Path(tmpdir) / 'nonexistent'):
                with patch('pipelines.analyze.FIGURES_DIR', Path(tmpdir)):
                    with patch('pipelines.analyze.REPORTS_DIR', Path(tmpdir)):
                        result = analyze_missingness_performance(
                            X_test, y_test, y_pred, y_proba
                        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0  # Empty when no raw data


class TestLoadFeatureNames:
    """Tests for load_feature_names function."""

    def test_pca_feature_names(self):
        """Test PCA feature name generation."""
        result = load_feature_names('pca', n_features=5)

        assert len(result) == 5
        assert result[0] == 'PC1'
        assert result[4] == 'PC5'

    def test_generic_feature_names(self):
        """Test generic feature name generation for unknown feature sets."""
        result = load_feature_names('unknown', n_features=3)

        assert len(result) == 3
        assert result[0] == 'feature_0'
        assert result[2] == 'feature_2'


class TestReportGeneration:
    """Tests for report section generation functions."""

    def test_report_header_format(self):
        """Test report header contains expected information."""
        metadata = {
            'model_name': 'SVM',
            'feature_set': 'lasso',
            'optimal_threshold': 0.07,
            'test_gmean': 0.7134
        }

        result = _report_header(metadata)

        assert 'SVM' in result
        assert 'lasso' in result
        assert '0.07' in result
        assert '71' in result  # G-Mean as percentage

    def test_report_misclassification_with_data(self):
        """Test misclassification section with FN data."""
        fn_df = pd.DataFrame({
            'rank': [1, 2],
            'test_index': [10, 20],
            'predicted_proba': [0.1, 0.2],
            'distance_to_threshold': [0.4, 0.3]
        })

        result = _report_misclassification(fn_df, n_total_failures=5)

        assert 'False Negatives' in result
        assert '2/5' in result

    def test_report_misclassification_empty(self):
        """Test misclassification section with no FNs."""
        fn_df = pd.DataFrame()

        result = _report_misclassification(fn_df)

        assert 'None in test set' in result

    def test_report_residuals_format(self):
        """Test residual section formatting."""
        residual_analysis = {
            'overall': {'mean': 0.1, 'std': 0.3, 'skewness': 0.5, 'kurtosis': 1.2},
            'passes': {'mean': -0.1, 'std': 0.2, 'n_samples': 100},
            'failures': {'mean': 0.8, 'std': 0.1, 'n_samples': 10},
            'interpretation': {'pass_bias': 'calibrated', 'fail_bias': 'underconfident'}
        }

        result = _report_residuals(residual_analysis)

        assert 'Calibration' in result
        assert 'underconfident' in result

    def test_report_missingness_with_data(self):
        """Test missingness section returns empty (omitted in concise report)."""
        missingness_df = pd.DataFrame({
            'quartile': ['Q1', 'Q2'],
            'gmean': [0.7, 0.65]
        })

        result = _report_missingness(missingness_df)

        assert result == ""

    def test_report_missingness_empty(self):
        """Test missingness section returns empty."""
        missingness_df = pd.DataFrame()

        result = _report_missingness(missingness_df)

        assert result == ""

    def test_report_clusters_with_data(self):
        """Test cluster section with data."""
        cluster_df = pd.DataFrame({
            'cluster': [0, 1],
            'n_samples': [3, 2],
            'distinctive_features': ['feat_1', 'feat_2']
        })

        result = _report_clusters(cluster_df)

        assert 'FN Clusters' in result

    def test_report_clusters_empty(self):
        """Test cluster section without sufficient data."""
        cluster_df = pd.DataFrame()

        result = _report_clusters(cluster_df)

        assert result == ""

    def test_report_recommendations(self):
        """Test recommendations section is footer only."""
        result = _report_recommendations()

        assert 'Generated by' in result


class TestGenerateAnalysisReport:
    """Tests for generate_analysis_report function."""

    def test_creates_markdown_file(self):
        """Test that markdown report is created."""
        fn_df = pd.DataFrame({'rank': [1], 'test_index': [10], 'predicted_proba': [0.1], 'distance_to_threshold': [0.4]})
        residual_analysis = {
            'overall': {'mean': 0.1, 'std': 0.3, 'skewness': 0.5, 'kurtosis': 1.2},
            'passes': {'mean': -0.1, 'std': 0.2, 'n_samples': 100},
            'failures': {'mean': 0.8, 'std': 0.1, 'n_samples': 10},
            'interpretation': {'pass_bias': 'calibrated', 'fail_bias': 'underconfident'}
        }
        missingness_df = pd.DataFrame()
        cluster_df = pd.DataFrame()
        metadata = {
            'model_name': 'SVM',
            'feature_set': 'lasso',
            'optimal_threshold': 0.07,
            'test_gmean': 0.7134
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('pipelines.analyze.REPORTS_DIR', Path(tmpdir)):
                generate_analysis_report(
                    fn_df, residual_analysis, missingness_df, cluster_df, metadata
                )
                report_path = Path(tmpdir) / 'interpretability_report.md'
                assert report_path.exists()

                content = report_path.read_text()
                assert '# Interpretability Report' in content

    def test_report_contains_all_sections(self):
        """Test that report contains expected sections."""
        fn_df = pd.DataFrame()
        residual_analysis = {
            'overall': {'mean': 0.0, 'std': 0.1, 'skewness': 0.0, 'kurtosis': 0.0},
            'passes': {'mean': 0.0, 'std': 0.1, 'n_samples': 50},
            'failures': {'mean': 0.0, 'std': 0.1, 'n_samples': 5},
            'interpretation': {'pass_bias': 'calibrated', 'fail_bias': 'calibrated'}
        }
        missingness_df = pd.DataFrame()
        cluster_df = pd.DataFrame()
        metadata = {
            'model_name': 'TestModel',
            'feature_set': 'test',
            'optimal_threshold': 0.5,
            'test_gmean': 0.8
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('pipelines.analyze.REPORTS_DIR', Path(tmpdir)):
                generate_analysis_report(
                    fn_df, residual_analysis, missingness_df, cluster_df, metadata
                )
                report_path = Path(tmpdir) / 'interpretability_report.md'
                content = report_path.read_text()

                assert '## False Negatives' in content
                assert '## Calibration' in content
                assert 'Generated by' in content


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_test_set(self):
        """Test handling of empty test set."""
        X_test = np.array([]).reshape(0, 5)
        y_test = np.array([], dtype=int)
        y_pred = np.array([], dtype=int)
        y_proba = np.array([])
        feature_names = [f'feature_{i}' for i in range(5)]

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('pipelines.analyze.FIGURES_DIR', Path(tmpdir)):
                with patch('pipelines.analyze.REPORTS_DIR', Path(tmpdir)):
                    result = analyze_fn_clusters(
                        X_test, y_test, y_pred, y_proba, feature_names
                    )

        assert len(result) == 0

    def test_single_sample(self):
        """Test handling of single sample."""
        y_test = np.array([1])
        y_proba = np.array([0.3])
        y_pred = np.array([0])

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('pipelines.analyze.FIGURES_DIR', Path(tmpdir)):
                with patch('pipelines.analyze.REPORTS_DIR', Path(tmpdir)):
                    result = analyze_residuals(y_test, y_proba, y_pred)

        assert result is not None
        assert result['overall']['mean'] == 0.7  # 1 - 0.3

    def test_extreme_probabilities(self):
        """Test handling of extreme probability values."""
        y_test = np.array([0, 0, 1, 1])
        y_proba = np.array([0.0, 0.0, 1.0, 1.0])
        y_pred = np.array([0, 0, 1, 1])

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('pipelines.analyze.FIGURES_DIR', Path(tmpdir)):
                with patch('pipelines.analyze.REPORTS_DIR', Path(tmpdir)):
                    result = analyze_residuals(y_test, y_proba, y_pred)

        # Perfect calibration at extremes
        assert result['overall']['mean'] == 0.0
        assert result['passes']['mean'] == 0.0  # 0 - 0 = 0
        assert result['failures']['mean'] == 0.0  # 1 - 1 = 0
