.PHONY: preprocess tune tune-status select analyze pipeline pipeline-quick dashboard mlflow reset test eda help ci lint

help:
	@echo "SECOM Yield Prediction"
	@echo ""
	@echo "Pipeline:"
	@echo "  make pipeline       Full pipeline (preprocess + tune + select + analyze)"
	@echo "  make pipeline-quick Skip preprocessing (data already processed)"
	@echo "  make preprocess     Data preprocessing only"
	@echo "  make tune           Hyperparameter tuning (36 experiments, resumes by default)"
	@echo "  make tune-status    Check Optuna tuning progress"
	@echo "  make select         Production model selection"
	@echo ""
	@echo "Analysis:"
	@echo "  make eda           Run EDA script and generate markdown report"
	@echo "  make analyze       Interpretability analysis (SHAP, residuals, clusters)"
	@echo ""
	@echo "Applications:"
	@echo "  make dashboard     Streamlit dashboard"
	@echo "  make mlflow        MLflow experiment UI"
	@echo ""
	@echo "Testing:"
	@echo "  make ci            Run lint + tests"
	@echo "  make lint          Run ruff linter"
	@echo "  make test          Run unit tests"
	@echo ""
	@echo "Maintenance:"
	@echo "  make reset         Reset project (clears models, reports, MLflow, cache)"

# Full pipeline (preprocess + tune + select + analyze)
pipeline:
	@if [ ! -f data/raw/secom/secom.data ]; then \
		echo "ERROR: Raw data not found at data/raw/secom/secom.data"; \
		exit 1; \
	fi
	.venv/bin/python -m pipelines.preprocess
	.venv/bin/python -m pipelines.tune --sampling=all --parallel
	.venv/bin/python -m pipelines.select
	.venv/bin/python -m pipelines.analyze

# Skip preprocessing (data already processed)
pipeline-quick:
	.venv/bin/python -m pipelines.tune --sampling=all --parallel
	.venv/bin/python -m pipelines.select
	.venv/bin/python -m pipelines.analyze

# Individual pipeline steps
preprocess:
	.venv/bin/python -m pipelines.preprocess

tune:
	.venv/bin/python -m pipelines.tune --sampling=all --parallel

tune-status:
	.venv/bin/python -m pipelines.tune --status

select:
	.venv/bin/python -m pipelines.select

# Interpretability analysis
analyze:
	@echo "Running interpretability analysis..."
	.venv/bin/python -m pipelines.analyze
	@echo ""
	@echo "Report: reports/interpretability_report.md"
	@echo "Figures: reports/figures/"

# EDA analysis and report generation
eda:
	@echo "Running EDA and generating markdown report..."
	.venv/bin/python notebooks/eda.py
	@echo ""
	@echo "Report: reports/eda_report.md"
	@echo "Figures: reports/figures/"

# Applications
dashboard:
	.venv/bin/streamlit run streamlit_app/Overview.py

mlflow:
	.venv/bin/mlflow ui

# Testing
test:
	.venv/bin/python -m pytest tests/ -v

lint:
	.venv/bin/ruff check src/ pipelines/ streamlit_app/ notebooks/eda.py

ci: lint test

# Maintenance
reset:
	@echo "SECOM Project Reset"
	@echo "==================="
	@echo ""
	@echo "Clearing: processed data, models, reports, MLflow, Optuna, cache"
	@echo "Keeping:  raw data, source code"
	@echo ""
	@echo -n "Processed data... "
	@find data/processed -type f \( -name "*.npy" -o -name "*.csv" -o -name "*.txt" \) -delete 2>/dev/null || true
	@echo "done"
	@echo -n "Models... "
	@find models -type f \( -name "*.pkl" -o -name "*.joblib" -o -name "*.json" \) -delete 2>/dev/null || true
	@echo "done"
	@echo -n "Reports... "
	@find reports -type f \( -name "*.png" -o -name "*.csv" -o -name "*.txt" -o -name "*.json" \) -delete 2>/dev/null || true
	@echo "done"
	@echo -n "MLflow... "
	@rm -rf mlruns mlflow.db notebooks/mlruns notebooks/mlflow.db 2>/dev/null || true
	@echo "done"
	@echo -n "Optuna... "
	@rm -f optuna_studies.db 2>/dev/null || true
	@echo "done"
	@echo -n "Cache... "
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete 2>/dev/null || true
	@rm -rf .pytest_cache 2>/dev/null || true
	@echo "done"
	@echo -n "Outputs... "
	@rm -f modeling_pipeline_output*.txt 2>/dev/null || true
	@echo "done"
	@echo ""
	@echo "Reset complete. Run: make pipeline"
