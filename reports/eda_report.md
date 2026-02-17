# EDA Report

## Key Findings

| Finding | Action |
|---------|--------|
| 14:1 class imbalance | G-Mean + ADASYN |
| 4.5% missing (non-random) | KNN imputation |
| 214 features VIF > 10 | LASSO selection |
| Max correlation ~0.15 | Ensemble methods |

---

## Visualizations

![Class Imbalance](figures/class_imbalance_over_time.png)

![Missing Data](figures/missing_data_matrix.png)

![Correlations](figures/correlation_heatmap_top20.png)

![Distributions](figures/top_features_distributions.png)

---

*Generated from notebooks/eda.py*
