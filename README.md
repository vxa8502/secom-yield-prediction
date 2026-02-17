# Semiconductor Yield Prediction

**72.2% G-Mean on 14:1 imbalanced manufacturing data using SVM + LASSO feature selection.**

<p align="center">
  <img src="reports/figures/dashboard_preview.png" alt="Dashboard Preview" width="800"/>
</p>

## Results

| Metric | Value |
|--------|-------|
| **G-Mean** | 72.2% |
| **AUC-ROC** | 78% |
| **Features** | 590 → 8 |
| **Inference** | 0.1ms |

| Model | Feature Set | Sampling | CV G-Mean |
|-------|-------------|----------|-----------|
| **SVM (RBF)** | LASSO | ADASYN | **71.0%** |
| LogReg | LASSO | native | 69.7% |
| SVM (linear) | LASSO | native | 68.8% |

## Quick Start

```bash
git clone https://github.com/vxa8502/secom-yield-prediction.git
cd secom-yield-prediction
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make pipeline && make dashboard
```

Dashboard: http://localhost:8501

## Dashboard

| Page | Purpose |
|------|---------|
| **Overview** | Confusion matrix, prediction distribution |
| **Prediction** | Manual entry or CSV upload |
| **Exploration** | Feature distributions, correlations |
| **Explainability** | SHAP waterfall plots |

## Pipeline

```
590 sensors → LASSO (8 features) → SVM + ADASYN → Threshold (0.54) → PASS/FAIL
```

```bash
make pipeline      # Full training
make dashboard     # Launch app
make reset         # Clear artifacts
```

## Limitations

Does not capture: equipment drift, lot variation, recipe changes, environmental factors, operator effects. Retrain quarterly.

## Deployment

```bash
docker build -t secom-yield-prediction:latest .
docker run -d -p 8501:8501 \
  -v $(pwd)/models:/app/models:ro \
  -v $(pwd)/data/processed:/app/data/processed:ro \
  secom-yield-prediction:latest
```

## Documentation

| Document | Content |
|----------|---------|
| [USAGE.md](USAGE.md) | API code examples |
| [docs/data_card.md](docs/data_card.md) | Dataset details |
| [docs/model_selection.md](docs/model_selection.md) | Model rationale |
