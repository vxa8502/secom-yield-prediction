# Exploratory Data Analysis Report
## SECOM Semiconductor Defect Prediction

**Author:** Victoria A.
**Dataset:** UCI SECOM (Semiconductor Manufacturing)

---

## Executive Summary

This report presents the exploratory data analysis of the SECOM semiconductor manufacturing dataset. The analysis reveals critical challenges that inform the modeling strategy:

- **Severe class imbalance** (14:1 Pass:Fail ratio) requiring balanced metrics
- **Complex missing data patterns** with non-random characteristics
- **High multicollinearity** among sensor features requiring dimensionality reduction
- **Weak individual feature-target correlations** suggesting ensemble methods

---

## 1. Dataset Overview

| Metric | Value |
|--------|-------|
| Total Samples | 1,567 wafers |
| Features | 590 sensor measurements |
| Target | Binary (Pass=0, Fail=1) |
| Pass Rate | 93.4% |
| Fail Rate | 6.6% |
| Date Range | ~30 days |

---

## 2. Class Imbalance Analysis

The dataset exhibits severe class imbalance with a 14:1 ratio of passing to failing wafers.

![Class Imbalance Over Time](figures/class_imbalance_over_time.png)

**Key Observations:**
- Failure rate varies between 0% and 100% across production days
- Some days have very few samples, affecting reliability of daily statistics
- Overall imbalance requires G-Mean or AUC-ROC metrics (not accuracy)

**Implications for Modeling:**
- Cannot use accuracy as primary metric (naive baseline = 93.4%)
- Must apply resampling (SMOTE/ADASYN) or class weighting
- Threshold optimization critical for balanced sensitivity/specificity

---

## 3. Missing Data Analysis

Overall missing rate: **4.54%** (41,951 missing values across all cells)

![Missing Data Matrix](figures/missing_data_matrix.png)

![Missing Data Heatmap](figures/missing_data_heatmap.png)

**Key Findings:**
- 541 features have complete data (no missing values)
- 16 features have >60% missing values
- Missingness patterns show correlation with target (MAR/MNAR detected)
- Groups of features missing together suggest sensor clusters

**Preprocessing Strategy:**
- Remove features with >50% missing values
- Apply KNN imputation (preserves correlational structure)
- Consider missingness indicators as engineered features

---

## 4. Feature-Target Correlation

Individual feature correlations with target are weak, with maximum correlation ~0.15.

![Correlation Heatmap](figures/correlation_heatmap_top20.png)

**Key Observations:**
- No single feature provides strong predictive signal
- Top features show mild separation between Pass/Fail distributions
- Ensemble methods or feature combinations needed

---

## 5. Feature Distribution Analysis

![Feature Distributions](figures/top_features_distributions.png)

![Feature Boxplots](figures/top_features_boxplots.png)

**Statistical Tests:**
- Mann-Whitney U tests confirm significant distribution differences
- Top correlated features show measurable separation
- Outliers present in most features (IQR method)

---

## 6. Multicollinearity Analysis

Severe multicollinearity detected: **214 features with VIF > 10**

**Remediation Strategy:**
1. Hierarchical clustering of high-VIF features by correlation
2. Select best feature (highest target correlation) from each cluster
3. Reduces feature set by ~67% while preserving predictive information

**Feature Selection Recommendation:**
- Use LASSO regularization for automatic selection
- Alternative: PCA for dimensionality reduction (sacrifices interpretability)

---

## 7. Key Findings Summary

| Finding | Impact | Recommendation |
|---------|--------|----------------|
| 14:1 class imbalance | Accuracy misleading | Use G-Mean, SMOTE/ADASYN |
| 4.5% missing data (non-random) | Potential information loss | KNN imputation |
| High multicollinearity | Unstable coefficients | LASSO or PCA |
| Weak individual correlations | No single strong predictor | Ensemble methods |

---

## 8. Preprocessing Pipeline

Based on this EDA, the recommended preprocessing pipeline:

```
1. Train/Test Split (80/20, stratified) - BEFORE any preprocessing
2. Remove features with >50% missing values
3. Apply KNN imputation (k=5) on training set
4. Transform test set using training parameters
5. Standardize features (StandardScaler)
6. Apply LASSO for feature selection OR PCA for reduction
7. Apply SMOTE/ADASYN on training set only
```

---

## Next Steps

Run the preprocessing and modeling pipeline:
```bash
make pipeline
```

Or run stages individually:
```bash
make preprocess  # Data preprocessing
make tune        # Hyperparameter tuning
make select      # Production model selection
```

---

*Report generated from notebooks/eda.py*
