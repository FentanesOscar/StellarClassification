# StellarClassification
Stellar Classification Using Machine Learning: Stars, Quasars, or Galaxies. 


# Stellar Classification Using Machine Learning

**Stars, Quasars, or Galaxies**  
An attempt to apply machine learning algorithms to classify Stars, Quasars, and Galaxies based on spectral characteristics using data from the Sloan Digital Sky Survey.

---

## Overview

- **Task**  
  Classify an object as a Star, Quasar, or Galaxy based on 18 features from SDSS DR17.
- **Models**  
  - Random Forest  
  - XGBoost (best-performing)
- **Best Accuracy**  
  97% using XGBoost with hyperparameter tuning.

---

## Data

- **Type:** CSV  
- **Features:** 1 categorical (target) + 16 numerical spectral features  
- **Instances:** 100,000 rows  
- **Train/Test Split:** 80/20 (80 000 train, 20 000 test)  
- **CV Folds:** 5  

---

## Preprocessing / Cleanup

1. **Drop Identifiers**  
   Removed administrative columns (object IDs, run IDs, dates, camera IDs) to avoid overfitting survey‐specific metadata.  
2. **Handle Outliers**  
   A lone photometric “outlier” used as a missing‐value sentinel was removed.  
3. **Encode Target**  
   Converted `STAR`, `QSO`, `GALAXY` → 2, 1, 0 using label encoding.

---

## Data Visualization

- **Class Imbalance**  
  Galaxies far outnumber Quasars and Stars.  
- **High Correlation**  
  Several photometric bands are strongly correlated.  
- **Outliers**  
  Quasars show more extreme values (due to intrinsic brightness).

---

## Problem Foundation

- **Type:** Supervised multinomial classification  
- **Inputs:** 17 features per object  
- **Output:** 1 label (Star, Quasar, Galaxy)  
- **Preprocessing:** ID removal, outlier drop, label encoding  
- **Models Explored:**  
  - **Random Forest** (bagging, less overfit prone)  
  - **XGBoost** (state-of-the-art booster)

---

## Hyperparameter Tuning (XGBoost)

```python
best_params = {
    'colsample_bytree': 1.0,
    'learning_rate':    0.1,
    'max_depth':        6,
    'n_estimators':     300,
    'subsample':        0.8
}
