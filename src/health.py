"""
Health check utilities for SECOM yield prediction project.

Provides diagnostic functions to validate system state and artifacts
at startup or during runtime monitoring.
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from .config import (
    DATA_DIR,
    MODELS_DIR,
    PRODUCTION_ARTIFACTS,
    FEATURE_SETS,
)

logger = logging.getLogger('secom')

# Maximum allowed size for JSON config files (1 MB should be plenty for threshold/metadata)
MAX_JSON_FILE_SIZE_BYTES = 1 * 1024 * 1024  # 1 MB


def _validate_json_file(
    path: Path,
    required_keys: set[str] | None = None,
    max_size_bytes: int = MAX_JSON_FILE_SIZE_BYTES
) -> tuple[bool, str, dict[str, Any] | None]:
    """
    Validate a JSON file exists, is reasonably sized, and contains required keys.

    Args:
        path: Path to JSON file
        required_keys: Optional set of keys that must be present
        max_size_bytes: Maximum allowed file size

    Returns:
        Tuple of (is_valid, error_message_or_empty, parsed_data_or_none)
    """
    if not path.exists():
        return False, f"{path.name} missing", None

    try:
        file_size = path.stat().st_size
        if file_size > max_size_bytes:
            return False, (
                f"{path.name} unexpectedly large: {file_size / 1024:.1f} KB "
                f"(max: {max_size_bytes / 1024:.0f} KB)"
            ), None

        with open(path, encoding='utf-8') as f:
            data = json.load(f)

        if required_keys:
            missing_keys = required_keys - set(data.keys())
            if missing_keys:
                return False, f"{path.name} missing keys: {missing_keys}", data

        return True, "", data

    except PermissionError:
        return False, f"Permission denied reading {path.name}", None
    except (OSError, IOError) as e:
        return False, f"{path.name} I/O error: {e}", None
    except json.JSONDecodeError as e:
        return False, f"{path.name} JSON invalid: {e}", None


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    passed: bool
    message: str
    details: dict[str, Any] | None = None

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.name}: {self.message}"


def check_data_files() -> HealthCheckResult:
    """Check if preprocessed data files exist."""
    missing = []
    found = []

    for fs_name, fs_config in FEATURE_SETS.items():
        train_path = DATA_DIR / fs_config['train_file']
        test_path = DATA_DIR / fs_config['test_file']

        if train_path.exists():
            found.append(str(train_path.name))
        else:
            missing.append(str(train_path.name))

        if test_path.exists():
            found.append(str(test_path.name))
        else:
            missing.append(str(test_path.name))

    # Check labels
    for label_file in ['y_train.csv', 'y_test.csv']:
        path = DATA_DIR / label_file
        if path.exists():
            found.append(label_file)
        else:
            missing.append(label_file)

    if missing:
        return HealthCheckResult(
            name="data_files",
            passed=False,
            message=f"Missing {len(missing)} data files",
            details={"missing": missing, "found": found}
        )

    return HealthCheckResult(
        name="data_files",
        passed=True,
        message=f"All {len(found)} data files present",
        details={"found": found}
    )


def check_production_model() -> HealthCheckResult:
    """Check if production model and metadata exist and are valid."""
    model_path = PRODUCTION_ARTIFACTS['model']
    threshold_path = PRODUCTION_ARTIFACTS['threshold']
    metadata_path = PRODUCTION_ARTIFACTS['metadata']

    issues = []

    # Check model file
    if not model_path.exists():
        issues.append("production_model.pkl missing")
    else:
        try:
            model = joblib.load(model_path)
            if not hasattr(model, 'predict'):
                issues.append("Model lacks predict() method")
        except PermissionError:
            issues.append("Permission denied reading model file")
        except (EOFError, pickle.UnpicklingError, KeyError, ModuleNotFoundError) as e:
            issues.append(f"Model file incompatible: {type(e).__name__}")
        except (OSError, IOError) as e:
            issues.append(f"Model I/O error: {e}")

    # Check threshold file
    threshold_valid, threshold_error, _ = _validate_json_file(
        threshold_path,
        required_keys={'optimal_threshold', 'model_name', 'feature_set'}
    )
    if not threshold_valid:
        issues.append(threshold_error)

    # Check metadata file
    metadata_valid, metadata_error, _ = _validate_json_file(metadata_path)
    if not metadata_valid:
        issues.append(metadata_error)

    if issues:
        return HealthCheckResult(
            name="production_model",
            passed=False,
            message=f"{len(issues)} issues found",
            details={"issues": issues}
        )

    return HealthCheckResult(
        name="production_model",
        passed=True,
        message="Production model and metadata valid",
        details={"model_path": str(model_path)}
    )


def check_preprocessing_pipeline() -> HealthCheckResult:
    """Check if preprocessing pipeline is loadable."""
    pipeline_path = PRODUCTION_ARTIFACTS['preprocessing_pipeline']

    if not pipeline_path.exists():
        return HealthCheckResult(
            name="preprocessing_pipeline",
            passed=False,
            message="Preprocessing pipeline not found",
            details={"path": str(pipeline_path)}
        )

    try:
        pipeline = joblib.load(pipeline_path)
        steps = [name for name, _ in pipeline.steps]
        return HealthCheckResult(
            name="preprocessing_pipeline",
            passed=True,
            message=f"Pipeline loaded with {len(steps)} steps",
            details={"steps": steps}
        )
    except PermissionError:
        return HealthCheckResult(
            name="preprocessing_pipeline",
            passed=False,
            message="Permission denied reading pipeline file",
            details={"path": str(pipeline_path)}
        )
    except (EOFError, pickle.UnpicklingError, KeyError, ModuleNotFoundError) as e:
        # KeyError can occur with pickle protocol version mismatches
        # ModuleNotFoundError if pickle references unavailable modules
        return HealthCheckResult(
            name="preprocessing_pipeline",
            passed=False,
            message=f"Pipeline file incompatible: {type(e).__name__}",
            details={"error": str(e), "path": str(pipeline_path)}
        )
    except (OSError, IOError) as e:
        return HealthCheckResult(
            name="preprocessing_pipeline",
            passed=False,
            message=f"Pipeline I/O error: {e}",
            details={"error": str(e), "path": str(pipeline_path)}
        )
    except AttributeError as e:
        return HealthCheckResult(
            name="preprocessing_pipeline",
            passed=False,
            message="Pipeline missing expected structure",
            details={"error": str(e), "path": str(pipeline_path)}
        )


def check_directories() -> HealthCheckResult:
    """Check if required directories exist and are writable."""
    directories = {
        'data': DATA_DIR,
        'models': MODELS_DIR,
    }

    issues = []
    for name, path in directories.items():
        if not path.exists():
            issues.append(f"{name} directory missing: {path}")
        elif not path.is_dir():
            issues.append(f"{name} is not a directory: {path}")

    if issues:
        return HealthCheckResult(
            name="directories",
            passed=False,
            message=f"{len(issues)} directory issues",
            details={"issues": issues}
        )

    return HealthCheckResult(
        name="directories",
        passed=True,
        message="All directories accessible",
        details={"directories": list(directories.keys())}
    )


def run_all_health_checks(verbose: bool = True) -> list[HealthCheckResult]:
    """
    Run all health checks and return results.

    Args:
        verbose: If True, log each check result

    Returns:
        List of HealthCheckResult objects
    """
    checks = [
        check_directories,
        check_data_files,
        check_preprocessing_pipeline,
        check_production_model,
    ]

    results = []
    for check_fn in checks:
        result = check_fn()
        results.append(result)
        if verbose:
            log_fn = logger.info if result.passed else logger.warning
            log_fn(str(result))

    passed = sum(1 for r in results if r.passed)
    total = len(results)

    if verbose:
        if passed == total:
            logger.info(f"Health check: {passed}/{total} checks passed")
        else:
            logger.warning(f"Health check: {passed}/{total} checks passed")

    return results


def is_healthy() -> bool:
    """Quick check if system is healthy (all checks pass)."""
    results = run_all_health_checks(verbose=False)
    return all(r.passed for r in results)
