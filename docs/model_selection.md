# Model Selection Rationale

## Why SVM Over XGBoost/LSTM

**SVM excelled** with LASSO-reduced features (8 dims) where kernel methods shine. ADASYN resampling near decision boundaries improved SVM margins.

**LSTM rejected:** SECOM is tabular snapshots, not time series. No temporal dependencies; 1,567 samples insufficient for deep learning. (Grinsztajn et al., NeurIPS 2022)

---

## Why G-Mean Over Accuracy

With 14:1 imbalance, "always pass" achieves 93.4% accuracy with zero defect detection.

**G-Mean = sqrt(Sensitivity x Specificity)** requires both classes to be predicted well.

| Metric | Issue |
|--------|-------|
| Accuracy | Majority-dominated |
| F1 | Ignores true negatives |
| AUC | Threshold-independent |

---

## Rejected Alternatives

| Rejected | Reason |
|----------|--------|
| LSTM | Tabular data, insufficient samples |
| PCA-only | Lower G-Mean, lost interpretability |
| Mean imputation | Worse than KNN for correlated features |
| SMOTE | 60+ hour runtime with SVM |
