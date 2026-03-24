# ML Classification & Regression on High-Dimensional Data

A machine learning project tackling both **classification** and **regression** tasks on a dataset with 400 input features, 1 binary class label, and 1 continuous output variable (2000 samples).

## Overview

### Data
- 2000 samples × 402 columns (Class, Output, Input1–Input400)
- Balanced classes (Class 0: 987, Class 1: 1013)
- Output variable is approximately normally distributed (Shapiro-Wilk p > 0.05)

---

## Classification

**Goal:** Predict `Class` (binary)

| Model | Test Accuracy |
|---|---|
| Logistic Regression (baseline) | 0.52 |
| Random Forest (default) | 0.65 |
| Random Forest (tuned) | **0.80** |

### Approach
1. **Feature selection** – Used baseline RFC feature importances; kept features with importance > 0.004 (19 features selected)
2. **Hyperparameter tuning** – Optimized with [Optuna](https://optuna.org/) over 50 trials

**Best parameters:**
```
n_estimators=50, max_features='sqrt', bootstrap=True,
class_weight='balanced', max_depth=19
```

**Result:** ~55% relative improvement over baseline logistic regression

---

## Regression

**Goal:** Predict continuous `Output`

| Model | Test R² | RMSE |
|---|---|---|
| Linear Regression (baseline) | 0.38 | 2.82 |
| Lasso (CV-tuned, α=0.115) | **0.51** | **2.50** |
| Ridge on Lasso-selected features | 0.48 | 2.56 |
| PCA + Lasso/Ridge | ≤0.31 | — |

### Approach
1. **Scaling** – StandardScaler fit on training set only
2. **Lasso with cross-validation** – alpha tuned via `LassoCV` (50 log-spaced alphas, 10-fold CV)
3. **Feature reduction** – Lasso selected 36 of 400 features

**Top predictive features:** Input83, Input223, Input167, Input193, Input342

**Result:** ~35% relative improvement in R² over baseline

---

## Dependencies
```
pandas, numpy, scikit-learn, optuna, matplotlib, seaborn, statsmodels, scipy
```

## Usage

Open the notebook in Google Colab or locally and run cells sequentially. Place the data file `zad2_wum_data_for_students.csv` (semicolon-delimited) in the working directory.