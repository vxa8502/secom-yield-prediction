# SECOM Defect Prediction Dashboard

Interactive web dashboard for semiconductor wafer quality prediction using production-grade machine learning.

## Features

### 1. Overview Page (Home)
- Model performance metrics (CV G-Mean: 71.0%, Test G-Mean: 72.2%)
- Production model architecture summary (SVM + ADASYN)
- LASSO feature selection details (8 features)
- Technical approach overview
- Quick navigation guide

### 2. Prediction Page
- **Manual Entry:** Input 8 sensor readings for instant prediction
- **Example Values:** Pre-loaded passing/failing/borderline cases
- **CSV Upload:** Batch prediction support
- **Threshold Selection:** Default threshold (0.54) or custom
- **Results Display:** Probability, confidence, and actionable recommendations

### 3. Explainability Page
- Feature importance visualization (SHAP values)
- LASSO dimensionality reduction (590 to 8 features)
- Top features: feature_59, feature_129, feature_103
- SHAP waterfall and beeswarm plots
- Actionable insights for process engineers

### 4. Monitoring Page (Coming Soon)
- Model performance tracking
- Feature drift detection (KS test)
- Calibration curve monitoring
- Prediction latency benchmarks
- Alert system for degradation

## Quick Start

### Local Development

```bash
# From project root
make dashboard
```

Or manually:

```bash
.venv/bin/streamlit run streamlit_app/Overview.py
```

Dashboard will be available at `http://localhost:8501`

### Docker Deployment

```bash
# Build image
docker build -t secom-dashboard .

# Run container
docker run -p 8501:8501 secom-dashboard
```

## Project Structure

```
streamlit_app/
├── Overview.py               # Main entry point (Overview page)
├── pages/
│   ├── 1_Prediction.py       # Interactive prediction interface
│   ├── 2_Exploration.py      # Data exploration and visualization
│   └── 3_Explainability.py   # Feature importance & SHAP
├── utils/
│   ├── config.py             # Configuration constants
│   ├── artifact_loader.py    # Model and artifact loading
│   ├── styling.py            # CSS styling and theming
│   └── __init__.py
├── __init__.py
└── README.md                 # This file
```

## Configuration

Edit `streamlit_app/utils/config.py` to customize:

- **PAGE_CONFIG:** Streamlit page settings (title, layout)
- **STYLE_CSS:** Custom CSS styling for headers, buttons, tables
- **FEATURE_DESCRIPTIONS:** Human-readable labels for anonymized sensors
- **get_threshold_options():** Threshold variants (conservative/default/aggressive)

Colors are imported from `src/config.py` (single source of truth).
Model metrics are loaded dynamically from `models/` artifacts via `artifact_loader.py`.

## Model Integration

The dashboard expects these files in the `models/` directory:

- `production_model.pkl` - Trained production classifier
- `production_threshold.json` - Optimal threshold configuration
- `preprocessing_pipeline_lasso.pkl` - LASSO preprocessing pipeline

Generate these by running the full pipeline:

```bash
make pipeline
```

Or run individual stages:

```bash
make preprocess  # Data preprocessing
make tune        # Hyperparameter tuning (36 experiments)
make select      # Production model selection
```

## Adding SHAP Visualizations

To enable SHAP plots in the Explainability page:

1. Install dependencies (already in requirements.txt):
   ```bash
   pip install shap streamlit-shap
   ```

2. Add SHAP computation to `pages/3_Explainability.py`:
   ```python
   import shap
   from streamlit_shap import st_shap
   import numpy as np

   # Load model and test data
   model = load_model("production")
   X_test = np.load("data/processed/X_test_lasso.npy")

   # Compute SHAP values (use KernelExplainer for SVM)
   # Note: KernelExplainer is model-agnostic but slower than TreeExplainer
   # For SVM with RBF kernel, KernelExplainer is required
   background = shap.sample(X_test, 100)  # Use subset for efficiency
   explainer = shap.KernelExplainer(model.predict_proba, background)
   shap_values = explainer.shap_values(X_test[:50])  # Limit samples for speed

   # Display waterfall plot (for single prediction)
   st_shap(shap.plots.waterfall(shap.Explanation(
       values=shap_values[1][0],  # Class 1 (fail) SHAP values
       base_values=explainer.expected_value[1],
       data=X_test[0]
   )), height=400)

   # Display summary plot
   st_shap(shap.summary_plot(shap_values[1], X_test[:50], plot_type="bar"))
   ```

## Customization

### Adding New Pages

1. Create `pages/N_PageName.py` (N = number for ordering)
2. Follow existing page structure (imports, config, layout)
3. Streamlit auto-detects pages in the `pages/` directory

### Styling

Modify `STYLE_CSS` in `utils/config.py` for custom CSS styling.

### Metrics

Update `MODEL_INFO` in `utils/config.py` after model retraining.

## Deployment Options

### Streamlit Community Cloud
1. Push to GitHub
2. Connect repository at share.streamlit.io
3. Deploy with one click (free tier available)

### Heroku
```bash
# Add Procfile
echo "web: streamlit run streamlit_app/Overview.py --server.port=\$PORT" > Procfile

# Deploy
heroku create secom-dashboard
git push heroku main
```

### AWS/GCP/Azure
Use provided Dockerfile for container-based deployment:
- AWS ECS/Fargate
- GCP Cloud Run
- Azure Container Instances

## Performance Optimization

For large datasets or high traffic:

1. **Caching:** Use `@st.cache_resource` for model loading
2. **Session State:** Store computed values in `st.session_state`
3. **Lazy Loading:** Compute SHAP values on-demand, not page load
4. **Database:** Move test data to database instead of CSV
5. **Load Balancing:** Deploy multiple instances behind load balancer

## Troubleshooting

### Models Not Found
Ensure you've run the pipeline to generate model artifacts:
```bash
make pipeline
```

### Import Errors
Check that project root is in Python path:
```python
sys.path.insert(0, str(Path(__file__).parent.parent))
```

### Port Already in Use
Change port in run command:
```bash
streamlit run streamlit_app/Overview.py --server.port=8502
```

## Technology Stack

- **Streamlit 1.28+** - Web framework
- **scikit-learn 1.3+** - ML models
- **SHAP 0.43+** - Model explainability
- **Matplotlib/Plotly** - Visualizations
- **Pandas/NumPy** - Data manipulation

## License

This dashboard is part of the SECOM Yield Prediction project.

## Contact

For questions or issues, refer to the main project README.
