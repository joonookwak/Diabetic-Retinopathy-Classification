## Diabetic Retinopathy Classification: F1-Score Optimization

### 📌 Project Overview

This project is an in-class Kaggle competition for IMEN 415 (Industrial Management Engineering). The objective is to build a binary classification model to detect Diabetic Retinopathy, with the final performance strictly evaluated by the **F1-Score**.

### 👤 Author

* **Joonoo Kwak**

### 🚀 Core Strategy & Algorithm

**Algorithm:** Logistic Regression with L1 Regularization (LASSO) via `glmnet`

Instead of relying on standard hyperparameter tuning, this model maximizes the F1-Score through extensive feature engineering, direct Out-of-Fold (OOF) metric optimization, and a robust multi-seed ensemble approach.

### 🧠 Key Methodologies

#### 1. Advanced Feature Engineering



* **Distribution Correction:** Applied `log1p` and `sqrt` transformations to highly skewed variables (e.g., exudate and microaneurysm counts).
* **Aggregation Metrics:** Created summary features such as `sum_ma`, `max_ma`, and non-zero counts to capture overall lesion severity.
* **Centered Interaction Terms:** Engineered interaction terms (e.g., $ma \times log\_ex$) by centering them around the train-set mean to completely eliminate multicollinearity issues.
* **Clinical & Geometric Features:** Incorporated interactions based on macula-to-optic disc distances, optic disc diameter, and AM/FM classification results.

#### 2. Direct OOF F1 Optimization



* Bypassed traditional AUC-based lambda selection.
* Directly evaluated and selected the optimal `Lambda` that maximizes the **Out-of-Fold (OOF) F1-Score**.
* Conducted a highly granular threshold search (steps of 0.005 from 0.01 to 0.99) to pinpoint the exact classification boundary.

#### 3. Multi-Seed Stratified Ensemble



* Implemented a Stratified 10-Fold Cross-Validation to maintain consistent class distribution across all folds.
* Repeated the training process across **20 different random seeds**.
* Averaged the prediction probabilities across all 20 models to significantly reduce variance and improve generalization on the unseen test set.

### 🛠 Tech Stack

* **Language:** R


* **Libraries:** `glmnet` (binomial family), `ggplot2`
