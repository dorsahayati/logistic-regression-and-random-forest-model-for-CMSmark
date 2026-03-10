## CMSmark Basic Models

This repository contains the training and testing workflow for the CMSmark basic machine learning classification models based on gene expression features.

### Requirements

- **Python**: 3.10+  
- **Python packages**:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`

You can install the required packages with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Data and Splits
- **Input data**: `trainset.csv` and `testset.csv` (under `../data/`), with samples labeled by CMS class.
- **Train / external test split**: The script performs an internal stratified train/validation split on the training set and uses an external, untouched test set for the final evaluation.
- **Class balance**: Class distributions for train and test are visualized in `result/class_distribution_train_test.png` (see the included class‑distribution figure).

### Feature Selection Strategy
We use several univariate feature selection analyses to define subsets of informative features, and **decided to continue with the ANOVA F‑test–derived feature sets for the final models**.

- **ANOVA F‑test** (`anova_f_test.csv`):
  - `Excellent`
  - `Good`
  - `Excellent_Good` (combination of Excellent and Good features)
- **Mutual Information** (`mutual_information.csv`):
  - `Good`
  - `Moderate`
  - `Good_Moderate` (Good + Moderate features)
- **Feature–Label Correlation** (`feature_label_correlation.csv`):
  - `Good`
  - `Moderate`
  - `Good_Moderate` (Good + Moderate features)

For each analysis and group, we intersect the selected features with the available training features and train models on:

- **Excellent / Good feature sets individually**, and  
- **Combined sets** (e.g. Excellent+Good or Good+Moderate), capturing both highly and moderately informative features.

### Models and Training
For every selected feature group we train and evaluate:

- **Logistic Regression**
  - `StandardScaler` preprocessing.
  - Optional PCA (see below).
  - Hyperparameter tuning with `GridSearchCV` over `C` and penalty (`l1`, `l2`).
- **Random Forest Classifier**
  - `StandardScaler` preprocessing.
  - Optional PCA.
  - Hyperparameter tuning with `GridSearchCV` over number of trees, maximum depth and minimum samples split.

Both models are trained inside `sklearn` `Pipeline`s, ensuring identical preprocessing across cross‑validation, validation, and external test evaluation.

### PCA Experiments
To study dimensionality reduction, we optionally apply **Principal Component Analysis (PCA)** on the selected features before classification:

- **PCA components tested**: 10, 50, 100, 200, 300 and 400.
- For each `n_components`, new result folders (e.g. `pca10`, `pca50`, …, `pca400`) are created.
- PCA is fitted only on the training data within the pipeline, and the same transformation is applied to validation and test sets.

### Evaluation on the Test Set
For both Logistic Regression and Random Forest, we evaluate on the external test set using the same feature subsets and (optionally) PCA settings. Our **final reported results are based on ANOVA F‑test feature selection**. For each configuration we compute:

- **Accuracy**
- **Macro‑averaged Precision**
- **Macro‑averaged Recall**
- **Macro‑averaged F1‑score**
- **Confusion matrices** (validation and test)
- **One‑vs‑rest ROC curves** per CMS class, including AUC values

Key test‑set visualizations include:

- **Class distribution (train vs test)**: `class_distribution_train_test.png`  
  (see the supplied class‑distribution bar plot).
- **Logistic Regression ROC curves (test set)**: `lr_roc_curve_test.png`  
  (see the attached ROC plot where all CMS classes show high AUC).
- **Random Forest ROC curves (test set)**: `rf_roc_curve_test.png`  
  (see the attached ROC plot with similarly high AUC values).

  ### How to Run

1. **Prepare data**
   - Place `trainset.csv` and `testset.csv` in the `../data/` directory relative to this folder.
2. **Run the training script**
   - From this `basic` directory, execute:
   ```bash
   python training_lr_rf.py
   ```
3. **Inspect results**
   - Outputs (metrics tables, confusion matrices, ROC curves, feature‑importance plots, PCA subfolders, etc.) will be written into the `../result/` subdirectories, organized by feature‑selection method, feature group, and PCA setting.

These results demonstrate that, using ANOVA‑based feature subsets (Excellent, Good and their combinations) and exploring PCA dimensions from 10 up to 400, both Logistic Regression and Random Forest achieve strong performance on the held‑out test set.

### Final Test‑Set Performance (ANOVA Features)
The table below summarizes the final performance on the test set when using ANOVA‑selected features:

| Metric      | Logistic Regression | Random Forest |
|------------|---------------------|---------------|
| Accuracy   | 0.94                | 0.92          |
| Precision  | 0.96                | 0.94          |
| Recall     | 0.93                | 0.90          |
| F1‑score   | 0.94                | 0.92          |


<img width="800" height="700" alt="rf_roc_curve_test" src="https://github.com/user-attachments/assets/df20555d-f466-40fa-bfda-a5edf8584841" />
<img width="800" height="700" alt="lr_roc_curve_test" src="https://github.com/user-attachments/assets/e7f6d25a-d104-4d3a-ab2c-da75977eeb1a" />



