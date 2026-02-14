#!/usr/bin/env python
"""
Model Validation CI Gate

Validates production model meets performance thresholds before deployment.
Used in CI pipeline to prevent regressions.

Usage:
    python -m pipelines.validate_model
    python -m pipelines.validate_model --min-gmean=0.65 --warn-gmean=0.70

Exit codes:
    0: Validation passed
    1: Validation failed (below min thresholds)
    2: Skipped (no artifacts found)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Add project root to path for module imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import PRODUCTION_ARTIFACTS, MODEL_VALIDATION


def load_metadata(metadata_path: Path) -> dict[str, Any] | None:
    """
    Load production model metadata.

    Args:
        metadata_path: Path to metadata JSON file

    Returns:
        Metadata dict or None if file doesn't exist or is invalid JSON
    """
    if not metadata_path.exists():
        return None

    try:
        with open(metadata_path) as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


def validate_model_performance(
    metadata_path: Path,
    min_gmean: float = 0.65,
    warn_gmean: float = 0.70,
    previous_metadata_path: Path | None = None,
    max_regression: float = 0.02
) -> tuple[bool, list[str], list[str]]:
    """
    Validate production model meets performance thresholds.

    Args:
        metadata_path: Path to production_model_metadata.json
        min_gmean: Minimum acceptable CV G-Mean (fail if below)
        warn_gmean: Warning threshold for CV G-Mean
        previous_metadata_path: Optional path to previous model metadata for regression check
        max_regression: Maximum allowed regression from previous model

    Returns:
        Tuple of (passed: bool, errors: list[str], warnings: list[str])
    """
    errors: list[str] = []
    warnings: list[str] = []

    metadata = load_metadata(metadata_path)
    if metadata is None:
        errors.append(f"Metadata file not found: {metadata_path}")
        return False, errors, warnings

    cv_gmean = metadata.get('cv_gmean')
    if cv_gmean is None:
        errors.append("Metadata missing 'cv_gmean' field")
        return False, errors, warnings

    model_name = metadata.get('model_name', 'Unknown')
    feature_set = metadata.get('feature_set', 'unknown')
    sampling = metadata.get('sampling_strategy', 'unknown')

    # Check minimum threshold
    if cv_gmean < min_gmean:
        errors.append(
            f"CV G-Mean {cv_gmean:.4f} below minimum threshold {min_gmean:.4f} "
            f"(model: {model_name}, features: {feature_set}, sampling: {sampling})"
        )

    # Check warning threshold
    if cv_gmean < warn_gmean and cv_gmean >= min_gmean:
        warnings.append(
            f"CV G-Mean {cv_gmean:.4f} below warning threshold {warn_gmean:.4f}"
        )

    # Check for regression against previous model
    if previous_metadata_path is not None:
        prev_metadata = load_metadata(previous_metadata_path)
        if prev_metadata is not None:
            prev_gmean = prev_metadata.get('cv_gmean')
            if prev_gmean is not None:
                regression = prev_gmean - cv_gmean
                if regression > max_regression:
                    errors.append(
                        f"Model regression detected: CV G-Mean dropped {regression:.4f} "
                        f"(from {prev_gmean:.4f} to {cv_gmean:.4f}, max allowed: {max_regression:.4f})"
                    )
                elif regression > 0:
                    warnings.append(
                        f"Minor regression: CV G-Mean dropped {regression:.4f} "
                        f"(from {prev_gmean:.4f} to {cv_gmean:.4f})"
                    )

    passed = len(errors) == 0
    return passed, errors, warnings


def format_validation_result(
    passed: bool,
    errors: list[str],
    warnings: list[str],
    metadata: dict[str, Any] | None
) -> str:
    """Format validation results for display."""
    lines = []

    if metadata:
        lines.append("Model Validation Report")
        lines.append("=" * 50)
        lines.append(f"Model: {metadata.get('model_name', 'Unknown')}")
        lines.append(f"Feature Set: {metadata.get('feature_set', 'unknown')}")
        lines.append(f"Sampling: {metadata.get('sampling_strategy', 'unknown')}")
        lines.append(f"CV G-Mean: {metadata.get('cv_gmean', 0):.4f}")
        lines.append(f"Test G-Mean: {metadata.get('test_gmean', 0):.4f}")
        lines.append(f"Threshold: {metadata.get('optimal_threshold', 0.5):.3f}")
        lines.append("")

    if errors:
        lines.append("ERRORS:")
        for err in errors:
            lines.append(f"  - {err}")
        lines.append("")

    if warnings:
        lines.append("WARNINGS:")
        for warn in warnings:
            lines.append(f"  - {warn}")
        lines.append("")

    status = "PASSED" if passed else "FAILED"
    lines.append(f"Status: {status}")

    return "\n".join(lines)


def main() -> int:
    """
    CLI entry point for model validation.

    Returns:
        Exit code: 0=pass, 1=fail, 2=skip (no artifacts)
    """
    parser = argparse.ArgumentParser(
        description='Validate production model performance for CI gate'
    )
    parser.add_argument(
        '--min-gmean',
        type=float,
        default=MODEL_VALIDATION['min_cv_gmean'],
        help=f"Minimum CV G-Mean threshold (default: {MODEL_VALIDATION['min_cv_gmean']})"
    )
    parser.add_argument(
        '--warn-gmean',
        type=float,
        default=MODEL_VALIDATION['warn_cv_gmean'],
        help=f"Warning CV G-Mean threshold (default: {MODEL_VALIDATION['warn_cv_gmean']})"
    )
    parser.add_argument(
        '--max-regression',
        type=float,
        default=MODEL_VALIDATION['max_regression'],
        help=f"Max allowed regression from previous model (default: {MODEL_VALIDATION['max_regression']})"
    )
    parser.add_argument(
        '--previous-metadata',
        type=Path,
        default=None,
        help="Path to previous model metadata for regression check"
    )
    parser.add_argument(
        '--metadata-path',
        type=Path,
        default=PRODUCTION_ARTIFACTS['metadata'],
        help="Path to production model metadata"
    )
    args = parser.parse_args()

    metadata_path = args.metadata_path

    # Check if artifacts exist
    if not metadata_path.exists():
        print(f"No production model found at {metadata_path}")
        print("Status: SKIPPED (run 'make pipeline' to train a model)")
        return 2

    # Validate
    passed, errors, warnings = validate_model_performance(
        metadata_path=metadata_path,
        min_gmean=args.min_gmean,
        warn_gmean=args.warn_gmean,
        previous_metadata_path=args.previous_metadata,
        max_regression=args.max_regression
    )

    # Load metadata for display
    metadata = load_metadata(metadata_path)

    # Print results
    print(format_validation_result(passed, errors, warnings, metadata))

    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
