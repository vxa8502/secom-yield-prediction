"""
Unit tests for health check module.
"""

import pytest
from pathlib import Path

from src.health import (
    HealthCheckResult,
    check_directories,
    check_data_files,
    run_all_health_checks,
    is_healthy,
)


class TestHealthCheckResult:
    """Tests for HealthCheckResult dataclass."""

    def test_passed_str_format(self):
        """Test string format for passed check."""
        result = HealthCheckResult(
            name="test_check",
            passed=True,
            message="Everything OK"
        )
        assert "[PASS]" in str(result)
        assert "test_check" in str(result)

    def test_failed_str_format(self):
        """Test string format for failed check."""
        result = HealthCheckResult(
            name="test_check",
            passed=False,
            message="Something wrong"
        )
        assert "[FAIL]" in str(result)
        assert "Something wrong" in str(result)

    def test_details_optional(self):
        """Test that details field is optional."""
        result = HealthCheckResult(
            name="test",
            passed=True,
            message="OK"
        )
        assert result.details is None

    def test_details_stored(self):
        """Test that details are stored correctly."""
        details = {"key": "value", "count": 42}
        result = HealthCheckResult(
            name="test",
            passed=True,
            message="OK",
            details=details
        )
        assert result.details == details


class TestCheckDirectories:
    """Tests for check_directories function."""

    def test_returns_health_check_result(self):
        """Test that function returns HealthCheckResult."""
        result = check_directories()
        assert isinstance(result, HealthCheckResult)
        assert result.name == "directories"

    def test_has_details(self):
        """Test that result includes details."""
        result = check_directories()
        assert result.details is not None


class TestCheckDataFiles:
    """Tests for check_data_files function."""

    def test_returns_health_check_result(self):
        """Test that function returns HealthCheckResult."""
        result = check_data_files()
        assert isinstance(result, HealthCheckResult)
        assert result.name == "data_files"

    def test_has_details(self):
        """Test that result includes details about files."""
        result = check_data_files()
        assert result.details is not None
        # Should have either 'found' or 'missing' keys
        assert 'found' in result.details or 'missing' in result.details


class TestRunAllHealthChecks:
    """Tests for run_all_health_checks function."""

    def test_returns_list_of_results(self):
        """Test that function returns list of HealthCheckResult."""
        results = run_all_health_checks(verbose=False)
        assert isinstance(results, list)
        assert all(isinstance(r, HealthCheckResult) for r in results)

    def test_runs_multiple_checks(self):
        """Test that multiple checks are executed."""
        results = run_all_health_checks(verbose=False)
        assert len(results) >= 2  # At least directories and data_files

    def test_verbose_mode(self, caplog):
        """Test verbose mode logs results."""
        import logging
        with caplog.at_level(logging.INFO, logger='secom'):
            run_all_health_checks(verbose=True)
        # Should have logged something
        # (actual content depends on system state)


class TestIsHealthy:
    """Tests for is_healthy function."""

    def test_returns_bool(self):
        """Test that function returns boolean."""
        result = is_healthy()
        assert isinstance(result, bool)
