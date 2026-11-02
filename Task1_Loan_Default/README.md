# Task 1: Loan Default Risk Prediction
3D Club Induction Task 2026 (Sophomore Batch)

Predicts whether a borrower will become seriously delinquent (90+ days past due) within two years. Defaults are rare (~6.7%), so the notebook focuses on imbalance-aware methods and metrics.

## Notebook
`loan_default_risk_prediction.ipynb` follows the required sections in order: Problem Statement, Dataset Description, Data Loading and Cleaning, Exploratory Data Analysis, Feature Engineering, Train-Test Split and Imbalance Handling, Model Selection and Justification, Model Training and Hyperparameter Tuning, Threshold Selection, Model Evaluation, Feature Importance.

## Approach
- Cleaning: removed invalid ages, treated 96/98 codes as missing, median imputation with missing flags, outlier clipping (statistics from the training split only).
- Imbalance: stratified split, class weighting, PR-AUC as the main metric.
- Models: logistic regression, random forest, histogram gradient boosting; the best is tuned with RandomizedSearchCV.
- Threshold chosen from out-of-fold predictions to reach a target recall.
- Feature importance via permutation importance.

## How to run
1. Download `cs-training.csv` from the Kaggle competition "Give Me Some Credit" and place it next to the notebook.
2. `pip install -r requirements.txt`
3. Run `jupyter notebook` and execute all cells.

## Results
Add your final test-set metrics (ROC-AUC, PR-AUC, precision, recall, F1) here.

## Source
Give Me Some Credit, Kaggle competition dataset.
