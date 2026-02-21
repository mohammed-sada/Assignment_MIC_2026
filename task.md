**REGRESSION PROBLEM**

**Total Marks: 10**

**Description**

We have a dataset for Peptides (Data.csv). This dataset is composed of a range of features/information (39 features, F1, F2, …, F39) from peptide sequence. And the last column represents the minimum inhibitory concentration (Target) of peptide against pathogens (i.e., bacteria).

You need to build following regression models to predict the target value.

1.  Linear Regression
2.  LASSO Regression
3.  RIDGE Regression
4.  ElasticNet Regression
5.  Polynomial Regression with order 2
6.  Random Forest Regressor

Then you need to apply feature subset selection based on Genetic Algorithm to find the best set of features and re-run the regressor.

Your model will be evaluated based on following:

1.  Correlation (actual vs. predicted target value)
2.  MAE (Mean absolute error)
3.  RMSE (Root Mean squared error)
4.  R-squared error

Please feel free to use any programming language that you know.

You need to upload (a) your code, (b) a report with following:

1.  Steps for building model
2.  Results in Tabular Format

**Before applying Feature Subset Selection**

|     |     |     |     |     |
| --- | --- | --- | --- | --- |
| Model | Correlation | MAE | RMSE | R-squared error |
|     |     |     |     |     |
|     |     |     |     |     |

**After applying Feature Subset Selection**

|     |     |     |     |     |
| --- | --- | --- | --- | --- |
| Model | Correlation | MAE | RMSE | R-squared error |
|     |     |     |     |     |
|     |     |     |     |     |

**Q1 (Marks 10)**

**Here is the mark distribution for each question**

<div class="joplin-table-wrapper"><table><tbody><tr><td><p><strong>No</strong></p></td><td><p><strong>Question</strong></p></td><td><p><strong>Marks</strong></p></td></tr><tr><td><p>1</p></td><td><p>Analyze collinearity among input variables. If there is collinearity in dataset, avoid this issue.</p></td><td><p>1</p></td></tr><tr><td><p>2</p></td><td><p>Split the dataset into 90% for training; 10% for testing. Count the number of records in training and testing set.</p></td><td><p>1</p></td></tr><tr><td><p>3</p></td><td><p>Train all models on 90% dataset with 5-fold cross validation.</p><p></p><p>Report results on test set for all models (with all feature)</p></td><td><p>3</p></td></tr><tr><td><p>4</p></td><td><ol><li>Apply GA to select subset of features and report the selected features</li></ol><p></p><ol><li>Report which configuration of GA gives you the best results. You need to change (i)number of generations, (ii)cross over probability, (iii)mutation rate.</li></ol><p></p><ol><li>Report results on test set for all models (with selected features).</li></ol><p></p></td><td><p>2=1+.5+.5</p></td></tr><tr><td><p>5</p></td><td><p>Train models on 90% dataset with selected features from GA. Report results on test set for all Models (With selected features proposed by GA)</p><p></p></td><td><p>1</p></td></tr><tr><td><p>6</p></td><td><p>Compare the co-efficient values of features for Linear regression (a) before feature selection, (b) after feature selection. What is your observation of co-efficient values?</p><p>How do you explain the difference of co-efficient values before and after feature subset selection?</p><p></p></td><td><p>1</p></td></tr><tr><td><p>7</p></td><td><p>How many co-efficient were there in the polynomial regressor? What do you observe in terms of results from polynomial regressor vs. the linear regressor? Explain your thoughts in terms of model complexity.</p><p></p></td><td><p>0.5</p></td></tr><tr><td><p>8</p></td><td><p>Is the working principle of Random Forest Regression different from Linear Regressor? Explain why Random Forest Regressor provided you better (or worse) results for this dataset.</p></td><td><p>0.5</p></td></tr></tbody></table></div>