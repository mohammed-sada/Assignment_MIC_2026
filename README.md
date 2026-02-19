# Regression Models Assignment - Part 1: Before Feature Selection

This repository contains the implementation and results for regression models trained on all features (BEFORE feature subset selection).

## Files

- `mic_regression_solution.py` - Main Python script implementing all regression models
- `Data.csv` - Dataset with 39 features (F1-F39) and TARGET variable
- `results_before_feature_selection.csv` - Results table with metrics for all models

## Models Implemented

1. Linear Regression
2. LASSO Regression
3. RIDGE Regression
4. ElasticNet Regression
5. Polynomial Regression (degree 2)
6. Random Forest Regressor

## Usage

To run models with all features:
```bash
python mic_regression_solution.py
```

## Results

All models were evaluated using:
- 5-fold cross-validation on training data (90% of dataset)
- Test set evaluation (10% of dataset)
- Metrics: Correlation, MAE, RMSE, R²

Results are saved in `results_before_feature_selection.csv`.
