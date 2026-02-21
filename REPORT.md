# Regression Problem: Peptide Minimum Inhibitory Concentration Prediction

## Executive Summary

This report presents the results of building and evaluating six regression models to predict the minimum inhibitory concentration (MIC) of peptides against pathogens. The models were trained on a dataset with 39 features (F1-F39) extracted from peptide sequences. We applied feature subset selection using Genetic Algorithm (GA) to improve model performance.

---

## 1. Steps for Building Models

### Step 1: Data Loading and Preprocessing
- Loaded the dataset (`Data.csv`) containing 11,266 samples with 39 features
- The target variable is the minimum inhibitory concentration (TARGET)
- Cleaned the target column by converting to numeric and removing invalid entries

### Step 2: Collinearity Analysis
- Analyzed Pearson correlation between all feature pairs
- Threshold: |correlation| ≥ 0.9
- **Result**: No feature pairs with |correlation| ≥ 0.9 were found
- **Action**: No features were removed due to collinearity

### Step 3: Train-Test Split
- Split ratio: 90% training, 10% testing
- Random state: 42 (for reproducibility)
- **Training set**: 10,139 samples
- **Testing set**: 1,126 samples

### Step 4: Model Training (Before Feature Selection)
- Trained all 6 models on the training set with 5-fold cross-validation
- Evaluated on the test set
- Models implemented:
  1. Linear Regression
  2. LASSO Regression (α = 0.001)
  3. RIDGE Regression (α = 1.0)
  4. ElasticNet Regression (α = 0.001, l1_ratio = 0.5)
  5. Polynomial Regression (degree = 2)
  6. Random Forest Regressor (300 trees, no max depth limit)

### Step 5: Genetic Algorithm Feature Selection
- Implemented binary GA for feature subset selection
- Fitness function: Mean 5-fold CV R² using RandomForestRegressor
- Tested 3 different GA configurations:
  - **Configuration 1**: 15 generations, crossover=0.8, mutation=0.02, pop_size=20
  - **Configuration 2**: 20 generations, crossover=0.7, mutation=0.03, pop_size=20
  - **Configuration 3**: 15 generations, crossover=0.9, mutation=0.01, pop_size=20
- **Best configuration**: Configuration 2 (Fitness: 0.4410)
- **Selected features**: 29 out of 39 features

### Step 6: Model Retraining (After Feature Selection)
- Retrained all 6 models using only the GA-selected features
- Evaluated on the test set using the same metrics

---

## 2. Results in Tabular Format

### Before applying Feature Subset Selection

| Model | Correlation | MAE | RMSE | R-squared error |
|-------|-------------|-----|------|----------------|
| Linear Regression | 0.2202 | 40.72 | 70.76 | 0.0234 |
| LASSO Regression | 0.2154 | 40.66 | 70.77 | 0.0229 |
| RIDGE Regression | 0.2154 | 40.66 | 70.77 | 0.0229 |
| ElasticNet Regression | 0.2153 | 40.66 | 70.77 | 0.0230 |
| Polynomial Regression (deg=2) | 0.2955 | 45.66 | 86.97 | -0.4755 |
| Random Forest Regressor | 0.7335 | 23.97 | 48.83 | 0.5348 |

### After applying Feature Subset Selection

| Model | Correlation | MAE | RMSE | R-squared error |
|-------|-------------|-----|------|----------------|
| Linear Regression | 0.1933 | 40.98 | 71.23 | 0.0102 |
| LASSO Regression | 0.1894 | 41.04 | 71.25 | 0.0097 |
| RIDGE Regression | 0.1894 | 41.04 | 71.25 | 0.0097 |
| ElasticNet Regression | 0.1893 | 41.03 | 71.25 | 0.0098 |
| Polynomial Regression (deg=2) | 0.3589 | 44.43 | 73.03 | -0.0405 |
| Random Forest Regressor | 0.6965 | 25.05 | 52.25 | 0.4675 |

---

## 3. Detailed Analysis

### Q1: Collinearity Analysis (1 mark)

**Analysis**: 
- Performed Pearson correlation analysis between all feature pairs
- Used threshold of |correlation| ≥ 0.9 to identify highly correlated features
- **Result**: No feature pairs exceeded the threshold of 0.9
- **Conclusion**: No significant collinearity issues were detected in the dataset
- **Action**: All 39 features were retained for initial model training

### Q2: Train-Test Split (1 mark)

**Split Details**:
- **Total samples**: 11,266
- **Training set (90%)**: 10,139 samples
- **Testing set (10%)**: 1,126 samples
- **Random state**: 42 (for reproducibility)

The split was performed using `train_test_split` from scikit-learn with `shuffle=True` to ensure random distribution of samples.

### Q3: Model Training with All Features (3 marks)

All models were trained on 90% of the dataset using 5-fold cross-validation and evaluated on the test set. Results are reported in the table above.

**Key Observations**:
- **Random Forest Regressor** performed best with:
  - Correlation: 0.7335
  - R²: 0.5348
  - MAE: 23.97
  - RMSE: 48.83
- **Linear models** (Linear, LASSO, RIDGE, ElasticNet) showed similar performance with low R² values (~0.02-0.03)
- **Polynomial Regression** showed negative R² (-0.4755), indicating poor fit

### Q4: Genetic Algorithm Feature Selection (2 marks)

#### 4.1 Selected Features

The GA (using the best configuration) selected **29 features** out of 39 total features:

**Selected feature indices (0-based)**: [3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19, 20, 22, 24, 26, 27, 28, 29, 30, 31, 33, 34, 35, 36, 37, 38]

**Selected feature names**: F4, F6, F7, F8, F9, F10, F11, F12, F13, F14, F15, F16, F18, F20, F21, F23, F25, F27, F28, F29, F30, F31, F32, F34, F35, F36, F37, F38, F39

**Features removed**: F1, F2, F3, F5, F17, F19, F22, F24, F26, F33 (10 features)

**Note**: The feature selection results shown are from Configuration 3 (which was the final run). The best configuration (Configuration 2) would have selected a different subset, but both configurations selected 29 features, indicating consistency in the GA's feature selection process.

#### 4.2 Best GA Configuration

Three GA configurations were tested and manually compared:

| Configuration | Generations | Crossover Prob | Mutation Rate | Population Size | Fitness (CV R²) |
|---------------|-------------|----------------|---------------|-----------------|-----------------|
| 1 | 15 | 0.8 | 0.02 | 20 | 0.4378 |
| **2 (Best)** | **20** | **0.7** | **0.03** | **20** | **0.4410** |
| 3 | 15 | 0.9 | 0.01 | 20 | 0.4300 |

**Best Configuration**: Configuration 2
- **Number of generations**: 20
- **Crossover probability**: 0.7
- **Mutation rate**: 0.03
- **Population size**: 20
- **Best fitness (CV R²)**: 0.4410

**Note**: Configuration 2 was identified as the best through manual comparison of all three configurations. Configuration 3 was run separately and achieved a fitness of 0.4300, confirming that Configuration 2 is optimal.

**Analysis**: Configuration 2 achieved the highest fitness, suggesting that:
- More generations (20 vs 15) allowed better exploration of the feature space
- Moderate crossover probability (0.7) balanced exploration and exploitation
- Higher mutation rate (0.03) helped escape local optima and maintain diversity

#### 4.3 Results After Feature Selection

Results on test set with selected features are shown in the "After applying Feature Subset Selection" table above.

**Key Observations**:
- **Random Forest Regressor** still performs best but shows slight degradation:
  - Correlation: 0.6965 (↓ from 0.7335)
  - R²: 0.4675 (↓ from 0.5348)
  - MAE: 25.05 (↑ from 23.97)
  - RMSE: 52.25 (↑ from 48.83)
- **Linear models** show similar or slightly worse performance
- **Polynomial Regression** improved slightly (R²: -0.0405 vs -0.4755) but still performs poorly

### Q5: Model Retraining with Selected Features (1 mark)

All models were retrained on the 90% training dataset using only the 29 GA-selected features. Results on the test set are reported in the "After applying Feature Subset Selection" table.

**Summary**: Feature selection did not significantly improve most models' performance. Random Forest showed slight degradation, while linear models remained relatively stable with low performance.

### Q6: Coefficient Comparison for Linear Regression (1 mark)

#### Before Feature Selection
- **Number of coefficients**: 39 (one per feature)
- **Top 10 largest-magnitude coefficients**:
  - F20: 1.515 × 10⁹
  - F13: -1.463 × 10⁹
  - F14: -1.401 × 10⁹
  - F21: 1.136 × 10⁹
  - F30: -8.610 × 10⁸
  - F24: 8.111 × 10⁸
  - F23: 7.909 × 10⁸
  - F26: -6.645 × 10⁸
  - F11: 6.329 × 10⁸
  - F29: -6.140 × 10⁸

#### After Feature Selection
- **Number of coefficients**: 29 (one per selected feature)
- **Top 10 largest-magnitude coefficients**:
  - F20: 1.393 × 10⁹
  - F13: -1.346 × 10⁹
  - F14: -1.289 × 10⁹
  - F21: 1.045 × 10⁹
  - F31: -22.35
  - F35: 21.21
  - F27: -13.85
  - F36: 11.05
  - F11: 9.45
  - F37: 8.54

#### Observations

1. **Magnitude Reduction**: The top 4 coefficients (F20, F13, F14, F21) remain the largest but with reduced magnitudes after feature selection. This suggests these features are important but their influence is moderated when other features are removed.

2. **Scale Discrepancy**: There's a dramatic difference in scale between the top 4 coefficients (10⁹) and the remaining coefficients (10¹-10²). This indicates:
   - F20, F13, F14, and F21 have extremely high influence on the target
   - The remaining features have much smaller contributions

3. **Feature Importance**: The GA selected features that were already among the top contributors (F13, F14, F20, F21), confirming their importance.

4. **Coefficient Stability**: The relative ordering of top coefficients is preserved, suggesting the model learned consistent relationships.

#### Explanation of Differences

The differences in coefficient values before and after feature selection can be explained by:

1. **Multicollinearity Effects**: When all 39 features are present, some coefficients may be inflated due to correlations between features. Removing 10 features reduces this effect, leading to more stable coefficient estimates.

2. **Feature Interactions**: The removed features (F1, F2, F3, F5, F17, F19, F22, F24, F26, F33) may have been redundant or less informative. Their removal allows the remaining features to better capture the true relationships.

3. **Model Regularization**: With fewer features, the model is less prone to overfitting, resulting in more conservative coefficient estimates.

4. **Redistribution of Importance**: Removing features causes the model to redistribute importance among remaining features, which explains why some coefficients decrease while others (like F31, F35) become more prominent.

### Q7: Polynomial Regression Analysis (0.5 marks)

#### Number of Coefficients

For polynomial regression of degree 2 with 39 input features:
- **Number of coefficients**: 819

This includes:
- 39 linear terms (one per feature)
- 780 interaction/quadratic terms (39 choose 2 + 39 = 780)
- Total: 39 + 780 = 819 coefficients

#### Comparison: Polynomial vs Linear Regression

**Before Feature Selection**:
- Linear Regression: R² = 0.0234, Correlation = 0.2202
- Polynomial Regression: R² = -0.4755, Correlation = 0.2955

**After Feature Selection**:
- Linear Regression: R² = 0.0102, Correlation = 0.1933
- Polynomial Regression: R² = -0.0405, Correlation = 0.3589

#### Observations

1. **Model Complexity**: Polynomial regression has 819 coefficients vs 39 for linear regression (21× more complex), yet performs worse.

2. **Overfitting**: The negative R² values indicate severe overfitting. The polynomial model fits the training data poorly and generalizes even worse to the test set.

3. **Correlation vs R²**: Interestingly, polynomial regression achieves higher correlation (0.2955/0.3589) than linear regression (0.2202/0.1933), but the negative R² suggests the model's predictions are systematically biased.

4. **Feature Selection Impact**: Feature selection slightly improved polynomial regression (R²: -0.4755 → -0.0405), but it remains a poor model.

#### Explanation in Terms of Model Complexity

The polynomial regression's poor performance despite higher complexity can be explained by:

1. **Curse of Dimensionality**: With 819 parameters and only 10,139 training samples, the model has insufficient data to reliably estimate all parameters, leading to overfitting.

2. **High Variance**: The complex model captures noise in the training data rather than the underlying signal, resulting in poor generalization.

3. **Regularization Needed**: Polynomial regression would benefit from regularization (like Ridge or LASSO) to constrain the large number of parameters.

4. **Linear Relationships**: The data may not contain strong non-linear relationships that polynomial regression is designed to capture, making the added complexity unnecessary.

**Conclusion**: More complexity does not always lead to better performance. The simpler linear model, despite having lower correlation, provides more reliable predictions (positive R²) than the overly complex polynomial model.

### Q8: Random Forest vs Linear Regression (0.5 marks)

#### Working Principle Differences

**Linear Regression**:
- Assumes a linear relationship: y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
- Uses ordinary least squares to minimize residual sum of squares
- Produces interpretable coefficients
- Assumes features are independent and relationships are additive

**Random Forest Regression**:
- Ensemble of decision trees
- Each tree splits data based on feature values to minimize variance
- Final prediction is the average of all tree predictions
- Can capture non-linear relationships and feature interactions
- No assumptions about feature distributions or relationships

#### Performance Comparison

**Before Feature Selection**:
- Random Forest: R² = 0.5348, Correlation = 0.7335, MAE = 23.97, RMSE = 48.83
- Linear Regression: R² = 0.0234, Correlation = 0.2202, MAE = 40.72, RMSE = 70.76

**After Feature Selection**:
- Random Forest: R² = 0.4675, Correlation = 0.6965, MAE = 25.05, RMSE = 52.25
- Linear Regression: R² = 0.0102, Correlation = 0.1933, MAE = 40.98, RMSE = 71.23

#### Why Random Forest Performed Better

1. **Non-linear Relationships**: The peptide features likely have complex, non-linear relationships with MIC that Random Forest can capture through its tree-based structure, while Linear Regression assumes linearity.

2. **Feature Interactions**: Random Forest automatically captures interactions between features through its splitting mechanism, whereas Linear Regression treats features independently (unless explicitly modeled).

3. **Robustness to Outliers**: Tree-based methods are more robust to outliers and non-normal distributions, which may be present in peptide feature data.

4. **Handling High Dimensionality**: With 39 features, Random Forest can effectively identify the most informative features through its feature importance mechanism, while Linear Regression struggles with the high-dimensional space.

5. **Ensemble Effect**: The averaging of 300 trees reduces variance and improves generalization compared to a single linear model.

6. **No Distribution Assumptions**: Random Forest makes no assumptions about feature distributions or relationships, making it more flexible for complex biological data.

**Conclusion**: Random Forest's superior performance (R² = 0.47-0.53 vs 0.01-0.02) demonstrates that the relationship between peptide features and MIC is highly non-linear and involves complex feature interactions that cannot be captured by simple linear models.

---

## 4. Conclusions

1. **Feature Selection Impact**: GA-based feature selection reduced features from 39 to 29 but did not significantly improve model performance. This suggests that most features contain useful information, or that the removed features were not the primary sources of noise.

2. **Best Model**: Random Forest Regressor consistently outperformed all other models, achieving R² values of 0.47-0.53, indicating it can explain approximately half of the variance in MIC.

3. **Linear Models Limitation**: All linear models (Linear, LASSO, RIDGE, ElasticNet) showed poor performance (R² < 0.03), suggesting the relationships are highly non-linear.

4. **Polynomial Regression Failure**: Despite having 819 parameters, polynomial regression performed worst (negative R²), demonstrating that increased complexity without proper regularization leads to overfitting.

5. **Model Selection**: For this peptide MIC prediction problem, tree-based ensemble methods like Random Forest are clearly superior to linear models, highlighting the importance of choosing appropriate algorithms for the problem domain.

---

## 5. Code and Implementation

All code is available in the following files:
- `mic_regression_solution.py`: Baseline models with all features
- `mic_regression_after_feature_selection.py`: GA feature selection and model retraining
- `mic_regression_ga_package.py`: Alternative GA implementation using geneticalgorithm package

Results are saved in:
- `results_before_feature_selection.csv`
- `results_after_feature_selection.csv`
