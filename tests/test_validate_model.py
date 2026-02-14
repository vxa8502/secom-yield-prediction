"""
Unit tests for model validation CI gate.
"""

import json
import pytest
from pathlib import Path

from pipelines.validate_model import (
    validate_model_performance,
    load_metadata,
    format_validation_result,
)


class TestLoadMetadata:
    """Tests for load_metadata function."""

    def test_load_valid_metadata(self, tmp_path):
        """Test loading valid metadata file."""
        metadata = {
            'cv_gmean': 0.75,
            'model_name': 'LogReg',
            'feature_set': 'lasso',
        }
        metadata_path = tmp_path / 'metadata.json'
        metadata_path.write_text(json.dumps(metadata))

        result = load_metadata(metadata_path)

        assert result == metadata

    def test_load_nonexistent_file(self, tmp_path):
        """Test loading nonexistent file returns None."""
        result = load_metadata(tmp_path / 'nonexistent.json')
        assert result is None


class TestValidateModelPerformance:
    """Tests for validate_model_performance function."""

    @pytest.fixture
    def good_metadata(self, tmp_path):
        """Create metadata file with good performance."""
        metadata = {
            'cv_gmean': 0.80,
            'model_name': 'XGBoost',
            'feature_set': 'lasso',
            'sampling_strategy': 'smote',
        }
        path = tmp_path / 'good_metadata.json'
        path.write_text(json.dumps(metadata))
        return path

    @pytest.fixture
    def poor_metadata(self, tmp_path):
        """Create metadata file with poor performance."""
        metadata = {
            'cv_gmean': 0.55,
            'model_name': 'LogReg',
            'feature_set': 'lasso',
            'sampling_strategy': 'native',
        }
        path = tmp_path / 'poor_metadata.json'
        path.write_text(json.dumps(metadata))
        return path

    @pytest.fixture
    def warn_metadata(self, tmp_path):
        """Create metadata file with warning-level performance."""
        metadata = {
            'cv_gmean': 0.68,
            'model_name': 'SVM',
            'feature_set': 'lasso',
            'sampling_strategy': 'native',
        }
        path = tmp_path / 'warn_metadata.json'
        path.write_text(json.dumps(metadata))
        return path

    def test_pass_with_good_performance(self, good_metadata):
        """Test validation passes with good G-Mean."""
        passed, errors, warnings = validate_model_performance(
            good_metadata,
            min_gmean=0.65,
            warn_gmean=0.70
        )

        assert passed is True
        assert len(errors) == 0
        assert len(warnings) == 0

    def test_fail_with_poor_performance(self, poor_metadata):
        """Test validation fails with poor G-Mean."""
        passed, errors, warnings = validate_model_performance(
            poor_metadata,
            min_gmean=0.65,
            warn_gmean=0.70
        )

        assert passed is False
        assert len(errors) == 1
        assert 'below minimum threshold' in errors[0]

    def test_warn_with_borderline_performance(self, warn_metadata):
        """Test validation warns with borderline G-Mean."""
        passed, errors, warnings = validate_model_performance(
            warn_metadata,
            min_gmean=0.65,
            warn_gmean=0.70
        )

        assert passed is True
        assert len(errors) == 0
        assert len(warnings) == 1
        assert 'below warning threshold' in warnings[0]

    def test_fail_with_missing_file(self, tmp_path):
        """Test validation fails when metadata file doesn't exist."""
        passed, errors, warnings = validate_model_performance(
            tmp_path / 'nonexistent.json',
            min_gmean=0.65,
            warn_gmean=0.70
        )

        assert passed is False
        assert len(errors) == 1
        assert 'not found' in errors[0]

    def test_fail_with_missing_gmean_field(self, tmp_path):
        """Test validation fails when cv_gmean field is missing."""
        metadata = {'model_name': 'LogReg'}
        path = tmp_path / 'incomplete.json'
        path.write_text(json.dumps(metadata))

        passed, errors, warnings = validate_model_performance(
            path,
            min_gmean=0.65,
            warn_gmean=0.70
        )

        assert passed is False
        assert len(errors) == 1
        assert 'missing' in errors[0].lower()


class TestRegressionDetection:
    """Tests for model regression detection."""

    @pytest.fixture
    def current_metadata(self, tmp_path):
        """Create current model metadata."""
        metadata = {'cv_gmean': 0.75, 'model_name': 'Current'}
        path = tmp_path / 'current.json'
        path.write_text(json.dumps(metadata))
        return path

    @pytest.fixture
    def better_previous(self, tmp_path):
        """Create previous model with better performance."""
        metadata = {'cv_gmean': 0.82, 'model_name': 'Previous'}
        path = tmp_path / 'previous_better.json'
        path.write_text(json.dumps(metadata))
        return path

    @pytest.fixture
    def worse_previous(self, tmp_path):
        """Create previous model with worse performance."""
        metadata = {'cv_gmean': 0.70, 'model_name': 'Previous'}
        path = tmp_path / 'previous_worse.json'
        path.write_text(json.dumps(metadata))
        return path

    def test_detect_significant_regression(self, current_metadata, better_previous):
        """Test detection of significant performance regression."""
        passed, errors, warnings = validate_model_performance(
            current_metadata,
            min_gmean=0.65,
            warn_gmean=0.70,
            previous_metadata_path=better_previous,
            max_regression=0.02
        )

        assert passed is False
        assert any('regression' in e.lower() for e in errors)

    def test_improvement_no_warning(self, current_metadata, worse_previous):
        """Test no warning when model improves."""
        passed, errors, warnings = validate_model_performance(
            current_metadata,
            min_gmean=0.65,
            warn_gmean=0.70,
            previous_metadata_path=worse_previous,
            max_regression=0.02
        )

        assert passed is True
        assert len(errors) == 0
        assert not any('regression' in w.lower() for w in warnings)


class TestFormatValidationResult:
    """Tests for format_validation_result function."""

    def test_format_passed_result(self):
        """Test formatting passed result."""
        metadata = {
            'model_name': 'XGBoost',
            'feature_set': 'lasso',
            'sampling_strategy': 'smote',
            'cv_gmean': 0.80,
            'test_gmean': 0.78,
            'optimal_threshold': 0.35,
        }

        result = format_validation_result(
            passed=True,
            errors=[],
            warnings=[],
            metadata=metadata
        )

        assert 'PASSED' in result
        assert 'XGBoost' in result
        assert '0.8000' in result

    def test_format_failed_result_with_errors(self):
        """Test formatting failed result with errors."""
        result = format_validation_result(
            passed=False,
            errors=['CV G-Mean below threshold'],
            warnings=[],
            metadata={'model_name': 'LogReg', 'cv_gmean': 0.55}
        )

        assert 'FAILED' in result
        assert 'ERRORS' in result
        assert 'below threshold' in result

    def test_format_with_warnings(self):
        """Test formatting result with warnings."""
        result = format_validation_result(
            passed=True,
            errors=[],
            warnings=['Minor performance drop'],
            metadata={'model_name': 'SVM', 'cv_gmean': 0.68}
        )

        assert 'PASSED' in result
        assert 'WARNINGS' in result
        assert 'performance drop' in result
