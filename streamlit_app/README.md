# SECOM Dashboard

```bash
make dashboard  # http://localhost:8501
```

## Structure

```
streamlit_app/
├── Overview.py           # Entry point
├── pages/
│   ├── 1_Prediction.py
│   ├── 2_Exploration.py
│   └── 3_Explainability.py
└── utils/
    ├── config.py         # Settings, CSS
    ├── artifact_loader.py
    └── styling.py
```

## Required Artifacts

```bash
make pipeline  # generates models/*.pkl
```

## Config

Edit `utils/config.py` for page settings, styling, feature labels.
