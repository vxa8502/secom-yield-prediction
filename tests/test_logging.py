"""
Unit tests for structured logging functionality.

Tests JSON logging format, execution timing, and metric logging.
"""

from __future__ import annotations

import json
import logging
import sys
from io import StringIO
from pathlib import Path


# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import (
    LogRecord,
    JSONFormatter,
    setup_json_logging,
    log_execution_time,
    log_metric,
    log_pipeline_step,
)


class TestLogRecord:
    """Tests for LogRecord dataclass."""

    def test_log_record_basic_fields(self):
        """Test basic LogRecord creation."""
        record = LogRecord(
            timestamp="2026-02-07T12:00:00Z",
            level="INFO",
            message="Test message",
        )

        assert record.timestamp == "2026-02-07T12:00:00Z"
        assert record.level == "INFO"
        assert record.message == "Test message"
        assert record.logger == "secom"

    def test_log_record_with_metrics(self):
        """Test LogRecord with metrics."""
        record = LogRecord(
            timestamp="2026-02-07T12:00:00Z",
            level="INFO",
            message="Test",
            metrics={"accuracy": 0.95, "gmean": 0.71},
        )

        assert record.metrics["accuracy"] == 0.95
        assert record.metrics["gmean"] == 0.71

    def test_log_record_to_json(self):
        """Test JSON serialization."""
        record = LogRecord(
            timestamp="2026-02-07T12:00:00Z",
            level="INFO",
            message="Test message",
            duration_ms=123.45,
        )

        json_str = record.to_json()
        parsed = json.loads(json_str)

        assert parsed["timestamp"] == "2026-02-07T12:00:00Z"
        assert parsed["level"] == "INFO"
        assert parsed["message"] == "Test message"
        assert parsed["duration_ms"] == 123.45

    def test_log_record_to_json_excludes_none(self):
        """Test that None values are excluded from JSON."""
        record = LogRecord(
            timestamp="2026-02-07T12:00:00Z",
            level="INFO",
            message="Test",
            module=None,
            function=None,
        )

        json_str = record.to_json()
        parsed = json.loads(json_str)

        assert "module" not in parsed
        assert "function" not in parsed

    def test_log_record_to_json_excludes_empty_dicts(self):
        """Test that empty metrics/context dicts are excluded."""
        record = LogRecord(
            timestamp="2026-02-07T12:00:00Z",
            level="INFO",
            message="Test",
        )

        json_str = record.to_json()
        parsed = json.loads(json_str)

        assert "metrics" not in parsed
        assert "context" not in parsed


class TestJSONFormatter:
    """Tests for JSONFormatter."""

    def test_formatter_produces_valid_json(self):
        """Test that formatter produces valid JSON."""
        formatter = JSONFormatter()

        log_record = logging.LogRecord(
            name="secom",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(log_record)
        parsed = json.loads(formatted)

        assert parsed["level"] == "INFO"
        assert parsed["message"] == "Test message"
        assert parsed["line"] == 42

    def test_formatter_includes_custom_fields(self):
        """Test that formatter includes custom fields."""
        formatter = JSONFormatter()

        log_record = logging.LogRecord(
            name="secom",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )
        log_record.duration_ms = 50.5
        log_record.metrics = {"gmean": 0.71}

        formatted = formatter.format(log_record)
        parsed = json.loads(formatted)

        assert parsed["duration_ms"] == 50.5
        assert parsed["metrics"]["gmean"] == 0.71

    def test_formatter_timestamp_format(self):
        """Test that timestamp is ISO format with Z suffix."""
        formatter = JSONFormatter()

        log_record = logging.LogRecord(
            name="secom",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(log_record)
        parsed = json.loads(formatted)

        assert parsed["timestamp"].endswith("Z")
        assert "T" in parsed["timestamp"]


class TestSetupJsonLogging:
    """Tests for setup_json_logging function."""

    def test_setup_creates_logger(self):
        """Test that setup creates a logger."""
        logger = setup_json_logging(console=False)

        assert logger.name == "secom"
        assert isinstance(logger, logging.Logger)

    def test_setup_with_file(self, tmp_path: Path):
        """Test logging to file."""
        log_file = tmp_path / "test.jsonl"
        logger = setup_json_logging(log_file=log_file, console=False)

        logger.info("Test message")

        assert log_file.exists()
        content = log_file.read_text()
        parsed = json.loads(content.strip())
        assert parsed["message"] == "Test message"

    def test_setup_console_output(self, capsys):
        """Test console output in JSON format."""
        logger = setup_json_logging(console=True)
        logger.handlers.clear()

        # Create a StringIO handler to capture output
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)

        logger.info("Console test")

        output = stream.getvalue()
        parsed = json.loads(output.strip())
        assert parsed["message"] == "Console test"

    def test_setup_suppresses_library_logs(self):
        """Test that library loggers are suppressed."""
        setup_json_logging(console=False)

        assert logging.getLogger("mlflow").level == logging.WARNING
        assert logging.getLogger("optuna").level == logging.WARNING


class TestLogExecutionTime:
    """Tests for log_execution_time context manager."""

    def test_logs_duration(self, tmp_path: Path):
        """Test that execution time is logged."""
        log_file = tmp_path / "test.jsonl"
        logger = setup_json_logging(log_file=log_file, console=False)

        with log_execution_time(logger, "test_operation"):
            pass  # Simulate work

        content = log_file.read_text()
        parsed = json.loads(content.strip())

        assert "duration_ms" in parsed
        assert parsed["duration_ms"] >= 0
        assert "test_operation" in parsed["message"]

    def test_collects_metrics(self, tmp_path: Path):
        """Test that metrics can be collected during execution."""
        log_file = tmp_path / "test.jsonl"
        logger = setup_json_logging(log_file=log_file, console=False)

        with log_execution_time(logger, "training") as metrics:
            metrics["n_samples"] = 100
            metrics["n_features"] = 6

        content = log_file.read_text()
        parsed = json.loads(content.strip())

        assert parsed["metrics"]["n_samples"] == 100
        assert parsed["metrics"]["n_features"] == 6

    def test_includes_extra_context(self, tmp_path: Path):
        """Test that extra context is included."""
        log_file = tmp_path / "test.jsonl"
        logger = setup_json_logging(log_file=log_file, console=False)

        with log_execution_time(logger, "prediction", extra_context={"model": "SVM"}):
            pass

        content = log_file.read_text()
        parsed = json.loads(content.strip())

        assert parsed["context"]["model"] == "SVM"
        assert parsed["context"]["operation"] == "prediction"


class TestLogMetric:
    """Tests for log_metric function."""

    def test_logs_single_metric(self, tmp_path: Path):
        """Test logging a single metric."""
        log_file = tmp_path / "test.jsonl"
        logger = setup_json_logging(log_file=log_file, console=False)

        log_metric(logger, "gmean", 0.713)

        content = log_file.read_text()
        parsed = json.loads(content.strip())

        assert parsed["metrics"]["gmean"] == 0.713
        assert "gmean" in parsed["message"]

    def test_logs_metric_with_step(self, tmp_path: Path):
        """Test logging metric with step number."""
        log_file = tmp_path / "test.jsonl"
        logger = setup_json_logging(log_file=log_file, console=False)

        log_metric(logger, "loss", 0.5, step=10)

        content = log_file.read_text()
        parsed = json.loads(content.strip())

        assert parsed["metrics"]["loss"] == 0.5
        assert parsed["metrics"]["step"] == 10

    def test_logs_metric_with_context(self, tmp_path: Path):
        """Test logging metric with context."""
        log_file = tmp_path / "test.jsonl"
        logger = setup_json_logging(log_file=log_file, console=False)

        log_metric(logger, "auc", 0.75, context={"fold": 3, "model": "XGBoost"})

        content = log_file.read_text()
        parsed = json.loads(content.strip())

        assert parsed["context"]["fold"] == 3
        assert parsed["context"]["model"] == "XGBoost"


class TestLogPipelineStep:
    """Tests for log_pipeline_step function."""

    def test_logs_step_started(self, tmp_path: Path):
        """Test logging pipeline step start."""
        log_file = tmp_path / "test.jsonl"
        logger = setup_json_logging(log_file=log_file, console=False)

        log_pipeline_step(logger, "preprocessing", "started")

        content = log_file.read_text()
        parsed = json.loads(content.strip())

        assert parsed["context"]["step"] == "preprocessing"
        assert parsed["context"]["status"] == "started"

    def test_logs_step_completed_with_duration(self, tmp_path: Path):
        """Test logging pipeline step completion with duration."""
        log_file = tmp_path / "test.jsonl"
        logger = setup_json_logging(log_file=log_file, console=False)

        log_pipeline_step(
            logger,
            "tuning",
            "completed",
            duration_ms=15000.0,
            metrics={"n_experiments": 36, "best_gmean": 0.71}
        )

        content = log_file.read_text()
        parsed = json.loads(content.strip())

        assert parsed["duration_ms"] == 15000.0
        assert parsed["metrics"]["n_experiments"] == 36
        assert parsed["metrics"]["best_gmean"] == 0.71

    def test_logs_step_failed(self, tmp_path: Path):
        """Test logging pipeline step failure."""
        log_file = tmp_path / "test.jsonl"
        logger = setup_json_logging(log_file=log_file, console=False)

        log_pipeline_step(logger, "selection", "failed", level=logging.ERROR)

        content = log_file.read_text()
        parsed = json.loads(content.strip())

        assert parsed["level"] == "ERROR"
        assert parsed["context"]["status"] == "failed"


class TestLoggingIntegration:
    """Integration tests for logging with other components."""

    def test_multiple_log_entries(self, tmp_path: Path):
        """Test multiple log entries in same file."""
        log_file = tmp_path / "test.jsonl"
        logger = setup_json_logging(log_file=log_file, console=False)

        log_pipeline_step(logger, "preprocess", "started")
        log_metric(logger, "missing_rate", 4.5)
        log_pipeline_step(logger, "preprocess", "completed", duration_ms=1000)

        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 3

        for line in lines:
            parsed = json.loads(line)
            assert "timestamp" in parsed
            assert "level" in parsed

    def test_log_levels_respected(self, tmp_path: Path):
        """Test that log levels are respected."""
        log_file = tmp_path / "test.jsonl"
        logger = setup_json_logging(level=logging.WARNING, log_file=log_file, console=False)

        logger.info("Should not appear")
        logger.warning("Should appear")

        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 1
        assert "Should appear" in lines[0]
