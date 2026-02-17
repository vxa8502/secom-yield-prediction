# Production Usage

## Load Model

```python
import json, joblib, pandas as pd

model = joblib.load('models/production_model.pkl')
with open('models/production_threshold.json') as f:
    config = json.load(f)
THRESHOLD = config['optimal_threshold']
preprocessor = joblib.load(f"models/preprocessing_pipeline_{config['feature_set']}.pkl")
```

## Predict

```python
# Single
X = pd.DataFrame([{'feature_59': -1.3, 'feature_129': 3.1, ...}])
prob = model.predict_proba(preprocessor.transform(X))[0, 1]
result = "FAIL" if prob >= THRESHOLD else "PASS"

# Batch
batch = pd.read_csv('wafers.csv')
probs = model.predict_proba(preprocessor.transform(batch))[:, 1]
```

## REST API

```python
from flask import Flask, request, jsonify

@app.route('/predict', methods=['POST'])
def predict():
    X = pd.DataFrame([request.get_json()['features']])
    prob = model.predict_proba(preprocessor.transform(X))[0, 1]
    return jsonify({'prediction': "FAIL" if prob >= THRESHOLD else "PASS", 'probability': float(prob)})
```

## Artifacts

| File | Size |
|------|------|
| `production_model.pkl` | ~193 KB |
| `production_threshold.json` | - |
| `preprocessing_pipeline_lasso.pkl` | ~18 KB |

**Performance:** 0.1ms latency, ~10K predictions/sec, ~10 MB memory
