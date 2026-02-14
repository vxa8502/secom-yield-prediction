# Semiconductor Yield Prediction

> **Predicts semiconductor wafer defects with 72% G-Mean on 14:1 imbalanced manufacturing data, reducing downstream yield loss through early FDC-based fault detection.**

<!-- Model performance visualization -->
<p align="center">
  <img src="reports/figures/confusion_matrix.png" alt="Confusion Matrix" width="500"/>
</p>

<p align="center"><em>Production model confusion matrix: 14 defects caught, 7 missed, 64 false alarms</em></p>

---

## The Problem

Semiconductor fabs process thousands of wafers daily. A single defective wafer that escapes detection can:
- Consume expensive downstream processing resources
- Contaminate batches and reduce overall yield
- Reach customers and damage reputation

**The challenge:** Detecting defects early from 590 noisy sensor readings when only 6.6% of wafers actually fail.

---

## The Solution

This system predicts wafer failures in real-time using machine learning, enabling:
- **Early intervention** before defective wafers consume resources
- **Explainable predictions** so engineers understand *why* a wafer was flagged
- **Tunable thresholds** to balance missed defects vs. false alarms

---

## Results

| Metric | Value | What It Means |
|--------|-------|---------------|
| **G-Mean** | 72% | Balanced detection of both failures and passes |
| **AUC-ROC** | 78% | Strong discrimination between good and bad wafers |
| **Feature Reduction** | 590 to 8 | Only 8 critical sensors needed for prediction |
| **Inference Time** | 0.1ms | Fast enough for real-time production use |

**Test Set Performance (n=314):**
- Sensitivity: 67% (catches 2/3 of actual defects)
- Specificity: 78% (correctly identifies passing wafers)
- Threshold (0.54) tuned to maximize G-Mean

---

## Quick Start

```bash
# Setup
git clone https://github.com/vxa8502/secom-yield-prediction.git
cd secom-yield-prediction
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train model (5-10 minutes)
make pipeline

# Launch dashboard
make dashboard
```

Open http://localhost:8501 to use the interactive prediction interface.

---

## Dashboard Features

### For Process Engineers

**Prediction Page** - Paste sensor readings and get:
- Pass/Fail prediction with confidence score
- Recommended actions based on result
- Feature values summary for troubleshooting

**Explainability Page** - Understand predictions:
- Which sensors contributed most to this decision?
- SHAP waterfall plots for individual wafers
- Feature importance rankings

### For Data Scientists

**Overview Page** - Model performance at a glance:
- Confusion matrix with false negative/positive breakdown
- Prediction distribution across test set
- Known limitations and coverage gaps

**Exploration Page** - Data quality insights:
- Feature distributions by class
- Correlation analysis
- Missing data patterns

---

## How It Works

```
FDC Sensor Data (590 process parameters)
    --> Feature Selection (LASSO reduces to 8 critical parameters)
    --> SVM Classifier with ADASYN resampling
    --> Probability Score
    --> Threshold Decision (0.54)
    --> PASS or FAIL (wafer disposition)
```

**Key Design Decisions:**

1. **G-Mean over Accuracy** - With 14:1 class imbalance, accuracy is misleading. G-Mean balances sensitivity (defect detection) and specificity (false alarm rate).

2. **Train/Test Split First** - FDC data preparation happens *after* splitting to prevent data leakage (a common mistake in academic implementations).

3. **Threshold Optimization** - Default 0.5 threshold is suboptimal for yield classification. We tune it on cross-validation predictions to maximize G-Mean.

4. **LASSO Feature Selection** - Reduces 590 correlated process parameters to 8 critical ones, improving interpretability for process engineers.

---

## Project Structure

```
secom-yield-prediction/
├── streamlit_app/           # Interactive dashboard
├── pipelines/               # Training pipeline (tune.py, select.py)
├── src/                     # Core utilities
├── models/                  # Saved model artifacts
├── data/                    # Processed datasets
├── Makefile                 # Pipeline automation
└── Dockerfile               # Container deployment
```

**Commands:**
```bash
make pipeline      # Full training pipeline
make dashboard     # Launch Streamlit app
make tune          # Hyperparameter tuning only
make select        # Model selection only
make reset         # Clear generated files
```

---

## Technical Details

### Dataset

[UCI SECOM Dataset](https://archive.ics.uci.edu/ml/datasets/SECOM) - Real semiconductor manufacturing data:
- 1,567 wafers (104 failures, 1,463 passes)
- 590 anonymized sensor features
- 4.5% missing values

### Models Evaluated

| Model | Feature Set | Sampling | CV G-Mean |
|-------|-------------|----------|-----------|
| **SVM (RBF)** | LASSO | ADASYN | **71.0%** |
| LogReg | LASSO | native | 69.7% |
| LogReg | LASSO | ADASYN | 69.5% |
| SVM (linear) | LASSO | native | 68.8% |

24 of 36 experiments completed (SMOTE terminated after 60 hours - computationally infeasible for SVM + large feature sets).

### Technologies

- **scikit-learn 1.8** - Models and preprocessing
- **Optuna** - Hyperparameter optimization
- **SHAP** - Model explainability
- **Streamlit** - Interactive dashboard
- **MLflow** - Experiment tracking
- **Docker** - Deployment

---

## Known Limitations

This model does **not** capture:
- Equipment drift over time
- Lot-to-lot variation
- Recipe or process changes
- Environmental factors
- Operator effects

**Recommendation:** Retrain quarterly or when process changes occur.

---

## Deployment

### Docker Build & Run

```bash
# Build image
docker build -t secom-yield-prediction:latest .

# Run dashboard (production)
docker run -d \
  --name secom-dashboard \
  -p 8501:8501 \
  -v $(pwd)/models:/app/models:ro \
  -v $(pwd)/data/processed:/app/data/processed:ro \
  secom-yield-prediction:latest

# Run with custom project root
docker run -d \
  -e SECOM_PROJECT_ROOT=/app \
  -p 8501:8501 \
  secom-yield-prediction:latest
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SECOM_PROJECT_ROOT` | Auto-detected | Project root directory path |
| `MLFLOW_TRACKING_URI` | `sqlite:///mlruns/mlflow.db` | MLflow backend URI |
| `STREAMLIT_SERVER_PORT` | `8501` | Dashboard port |

### Volume Mounts

| Container Path | Purpose | Mode |
|----------------|---------|------|
| `/app/models` | Trained model artifacts | Read-only (`:ro`) |
| `/app/data/processed` | Preprocessed features | Read-only |
| `/app/reports` | Generated reports/figures | Read-write |
| `/app/mlruns` | MLflow tracking data | Read-write |

### Health Checks

```bash
# Streamlit health endpoint
curl http://localhost:8501/_stcore/health

# Application health check
python -c "from src.health import is_healthy; print('OK' if is_healthy() else 'FAIL')"

# Docker health status
docker inspect --format='{{.State.Health.Status}}' secom-dashboard
```

### Pre-Deployment Checklist

1. **Model artifacts exist:**
   ```bash
   ls models/production_model.pkl models/production_threshold.json
   ```

2. **Data files exist:**
   ```bash
   ls data/processed/X_test_lasso.npy data/processed/y_test.csv
   ```

3. **Run health checks:**
   ```bash
   python -c "from src.health import run_all_health_checks; run_all_health_checks(verbose=True)"
   ```

4. **Verify model loads:**
   ```bash
   python -c "import joblib; m = joblib.load('models/production_model.pkl'); print('Model OK')"
   ```

### Monitoring

**Key Metrics to Track:**
- Prediction latency (p50, p95, p99)
- Prediction distribution (% fail predictions)
- Feature drift (compare live data to training distribution)
- Model confidence scores

**Logging Configuration:**

Enable JSON structured logging for log aggregation:

```python
from src.config import setup_json_logging, log_execution_time

logger = setup_json_logging(log_file=Path("logs/secom.jsonl"))

with log_execution_time(logger, "prediction") as metrics:
    result = model.predict(X)
    metrics["n_samples"] = len(X)
```

**Log Format Example:**
```json
{
  "timestamp": "2026-02-07T12:00:00.000Z",
  "level": "INFO",
  "message": "prediction completed",
  "duration_ms": 2.5,
  "metrics": {"n_samples": 100},
  "context": {"operation": "prediction"}
}
```

### Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| `ModuleNotFoundError: src` | Wrong working directory | Set `SECOM_PROJECT_ROOT` or run from project root |
| `FileNotFoundError: production_model.pkl` | Model not trained | Run `make pipeline` first |
| Port 8501 in use | Another Streamlit instance | Kill process or use `-p 8502:8501` |
| SQLite database locked | Parallel MLflow access | Use PostgreSQL backend for production |
| SHAP computation slow | Large test set | Reduce `max_samples` in explainability settings |

### Pipeline Execution in Container

```bash
# Run full pipeline inside container
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/reports:/app/reports \
  --entrypoint python \
  secom-yield-prediction:latest \
  -m pipelines.preprocess

# Run tuning (resource-intensive)
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/reports:/app/reports \
  --entrypoint python \
  secom-yield-prediction:latest \
  -m pipelines.tune --sampling=all
```

---

## References

- Park et al. (2024) - "Effect of Data Preprocessing on SECOM" - *Sensors*
- Salem et al. (2018) - "In-painting KNN-Imputation" - *Big Data and Cognitive Computing*

