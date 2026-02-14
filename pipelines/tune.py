#!/usr/bin/env python
"""
SECOM Hyperparameter Tuning Pipeline

Studies are persisted to SQLite (optuna_studies.db), enabling:
- Crash recovery: Resume interrupted tuning runs automatically
- Progress monitoring: Check status while tuning runs in background
- Reproducibility: Inspect historical optimization results

Usage:
    # Check status of existing studies
    python -m pipelines.tune --status

    # Run tuning (resumes from previous progress by default)
    python -m pipelines.tune --sampling=smote

    # Start fresh, ignoring previous progress
    python -m pipelines.tune --sampling=all --fresh --parallel

    # Run all strategies in parallel
    python -m pipelines.tune --sampling=all --parallel --verbose
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import mlflow

# Suppress specific expected warnings (not blanket suppression)
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
warnings.filterwarnings('ignore', category=UserWarning, message='.*lbfgs.*')
warnings.filterwarnings('ignore', category=UserWarning, message='.*Solver.*')

# Add project root to path for module imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import REPORTS_DIR, setup_logging, ensure_directories, OPTUNA_STORAGE_PATH
from src.data import load_labels, load_all_feature_sets
from src.tuning_utils import (
    run_all_experiments_for_sampling,
    list_existing_studies,
    get_study_progress,
)
from src.mlflow_utils import setup_mlflow

logger = logging.getLogger('secom')


def run_tuning_for_strategy(
    sampling_strategy: str,
    n_trials: int = 100,
    resume: bool = True,
    fresh: bool = False,
) -> dict[str, Any]:
    """
    Run tuning for a single sampling strategy.

    Args:
        sampling_strategy: Resampling method ('native', 'smote', 'adasyn')
        n_trials: Number of Optuna trials per experiment
        resume: If True, continue from existing study progress
        fresh: If True, delete existing studies and start fresh

    Returns:
        Dict with best model configuration and timing info
    """
    setup_logging(logging.INFO)
    setup_mlflow()
    y_train, _ = load_labels()
    feature_sets = load_all_feature_sets()

    start = time.time()
    try:
        results = run_all_experiments_for_sampling(
            sampling_strategy=sampling_strategy,
            feature_sets=feature_sets,
            y_train=y_train,
            n_trials=n_trials,
            resume=resume,
            fresh=fresh,
        )
    except Exception as e:
        raise RuntimeError(
            f"Tuning failed for strategy '{sampling_strategy}': {type(e).__name__}: {e}"
        ) from e
    elapsed = time.time() - start
    logger.info(f"Tuning '{sampling_strategy}' completed: {len(results)} experiments in {elapsed/60:.1f} minutes")

    if not results:
        raise RuntimeError(
            f"No tuning results returned for strategy '{sampling_strategy}'. "
            f"Check if data is loaded correctly and models can be trained."
        )

    results_df = pd.DataFrame(results)

    if results_df.empty:
        raise RuntimeError(
            f"Empty results DataFrame for strategy '{sampling_strategy}'. "
            f"All experiments may have failed."
        )

    if 'cv_gmean' not in results_df.columns:
        raise RuntimeError(
            f"Invalid results structure for strategy '{sampling_strategy}': "
            f"missing 'cv_gmean' column. Got columns: {list(results_df.columns)}"
        )

    output_path = REPORTS_DIR / f'tuning_{sampling_strategy}_results.csv'
    results_df.to_csv(output_path, index=False)

    best = results_df.loc[results_df['cv_gmean'].idxmax()]

    with mlflow.start_run(run_name=f"tuning_{sampling_strategy}_summary"):
        mlflow.log_param("sampling_strategy", sampling_strategy)
        mlflow.log_param("n_experiments", len(results_df))
        mlflow.log_metric("best_cv_gmean", best['cv_gmean'])
        mlflow.log_param("best_model", best['model'])
        mlflow.log_param("best_feature_set", best['feature_set'])
        mlflow.log_artifact(str(output_path))

    return {
        'sampling_strategy': sampling_strategy,
        'best_model': best['model'],
        'best_feature_set': best['feature_set'],
        'best_cv_gmean': best['cv_gmean'],
        'elapsed_minutes': elapsed / 60
    }


def run_worker(args: tuple[str, int, bool, bool]) -> dict[str, Any]:
    """Worker for parallel execution."""
    sampling_strategy, n_trials, resume, fresh = args
    return run_tuning_for_strategy(sampling_strategy, n_trials, resume, fresh)


def show_tuning_status() -> None:
    """Display status of all Optuna studies."""
    studies = list_existing_studies()

    if not studies:
        print(f"No studies found in {OPTUNA_STORAGE_PATH}")
        return

    print(f"\nOptuna Studies ({OPTUNA_STORAGE_PATH}):")
    print("-" * 70)
    print(f"{'Study Name':<40} {'Done':>6} {'Pruned':>7} {'Best':>10}")
    print("-" * 70)

    for study_name in sorted(studies):
        progress = get_study_progress(study_name)
        if progress:
            best_str = f"{progress['best_value']:.4f}" if progress['best_value'] else "N/A"
            print(
                f"{study_name:<40} "
                f"{progress['n_trials_completed']:>6} "
                f"{progress['n_trials_pruned']:>7} "
                f"{best_str:>10}"
            )

    print("-" * 70)
    print(f"Total: {len(studies)} studies\n")


def main() -> None:
    """CLI entrypoint for hyperparameter tuning."""
    parser = argparse.ArgumentParser(
        description='SECOM hyperparameter tuning with persistent Optuna storage',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check status of existing studies
  python -m pipelines.tune --status

  # Run tuning (resumes from previous progress by default)
  python -m pipelines.tune --sampling=smote

  # Start fresh, ignoring previous progress
  python -m pipelines.tune --sampling=all --fresh --parallel

  # Run all strategies in parallel
  python -m pipelines.tune --sampling=all --parallel
        """
    )
    parser.add_argument(
        '--sampling',
        type=str,
        choices=['native', 'smote', 'adasyn', 'all'],
        help='Sampling strategy to tune'
    )
    parser.add_argument('--parallel', action='store_true', help='Run strategies in parallel')
    parser.add_argument('--n-trials', type=int, default=100, help='Optuna trials per experiment')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument(
        '--fresh',
        action='store_true',
        help='Delete existing studies and start fresh (default: resume)'
    )
    parser.add_argument(
        '--status',
        action='store_true',
        help='Show status of existing Optuna studies and exit'
    )
    args = parser.parse_args()

    # Handle --status flag
    if args.status:
        show_tuning_status()
        return

    # Require --sampling if not showing status
    if not args.sampling:
        parser.error("--sampling is required unless using --status")

    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    logger = logging.getLogger('secom')
    ensure_directories()

    strategies = ['native', 'smote', 'adasyn'] if args.sampling == 'all' else [args.sampling]

    if not strategies:
        raise ValueError("No sampling strategies specified")

    total_experiments = len(strategies) * 12
    resume = not args.fresh
    mode = "fresh" if args.fresh else "resume"
    logger.info(f"Running {total_experiments} experiments ({mode} mode)")

    start = time.time()

    # Initialize MLflow SQLite schema in main process before spawning workers.
    # Without this, parallel workers race to create the DB and crash.
    setup_mlflow()

    if args.parallel and len(strategies) > 1:
        with ProcessPoolExecutor(max_workers=len(strategies)) as executor:
            futures = {
                executor.submit(run_worker, (s, args.n_trials, resume, args.fresh)): s
                for s in strategies
            }
            results: list[dict[str, Any]] = []
            failures: list[tuple[str, Exception]] = []
            for future in as_completed(futures):
                strategy = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"{strategy}: {result['best_model']}+{result['best_feature_set']}={result['best_cv_gmean']:.4f}")
                except Exception as e:
                    failures.append((strategy, e))
                    logger.error(f"{strategy} FAILED: {e}")

            if failures:
                failed_strategies = [f[0] for f in failures]
                raise RuntimeError(f"Tuning failed for strategies: {failed_strategies}")
    else:
        results = []
        for strategy in strategies:
            result = run_tuning_for_strategy(strategy, args.n_trials, resume, args.fresh)
            results.append(result)
            logger.info(f"{strategy}: {result['best_model']}+{result['best_feature_set']}={result['best_cv_gmean']:.4f}")

    if not results:
        raise RuntimeError("No tuning results collected")

    elapsed = (time.time() - start) / 60
    best = max(results, key=lambda x: x['best_cv_gmean'])

    logger.info(f"Best: {best['best_model']}+{best['best_feature_set']}+{best['sampling_strategy']}={best['best_cv_gmean']:.4f} ({elapsed:.1f}min)")


if __name__ == '__main__':
    main()
