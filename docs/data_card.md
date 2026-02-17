# Data Card: SECOM Dataset

## Overview

| Attribute | Value |
|-----------|-------|
| **Source** | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/SECOM) |
| **Samples** | 1,567 wafers (104 fail / 1,463 pass) |
| **Features** | 590 anonymized sensors |
| **Imbalance** | 14:1 |
| **Missing** | 4.54% (41,951 cells) |

---

## Preprocessing

| Step | Method |
|------|--------|
| Split | 80/20 stratified, BEFORE preprocessing |
| Missing removal | >50% threshold (~31 features) |
| Imputation | KNN (k=5) |
| Selection | LASSO → 8 features |
| Scaling | StandardScaler |

---

## Selected Features

| Feature | Mean SHAP |
|---------|-----------|
| feature_59 | 0.0597 |
| feature_129 | 0.0480 |
| feature_103 | 0.0448 |
| feature_21 | 0.0378 |
| feature_64 | 0.0357 |
| feature_510 | 0.0215 |
| feature_75 | 0.0196 |
| feature_114 | 0.0036 |

---

## Limitations

- Anonymized features (no physical validation)
- Single fab, 30-day window (no drift/seasonal data)
- Selection bias: only end-of-line tested wafers
- MNAR possible (sensors may fail with process)

---

## References

1. McCann & Johnston (2008). SECOM Dataset. UCI ML Repository.
2. Park et al. (2024). Effect of data preprocessing on SECOM classification. *Sensors*.
