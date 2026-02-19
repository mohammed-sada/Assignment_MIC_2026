import numpy as np
import pandas as pd
import pickle
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATA_PATH = Path(__file__).with_name("Data.csv")
RANDOM_STATE = 42


def load_data():
    df = pd.read_csv(DATA_PATH)
    # Assume last column is the target named 'TARGET'
    feature_cols = df.columns[:-1]
    target_col = df.columns[-1]

    # Clean target column: coerce to numeric and drop problematic rows
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col])

    X = df[feature_cols].values
    y = df[target_col].values
    return df, feature_cols.tolist(), target_col, X, y


def analyze_collinearity(df, feature_cols, threshold: float = 0.9):
    """
    Simple collinearity analysis based on Pearson correlation between features.
    Returns a DataFrame of highly correlated feature pairs.
    """
    corr = df[feature_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    high_corr_pairs = (
        upper.stack()
        .reset_index()
        .rename(columns={"level_0": "feature_1", "level_1": "feature_2", 0: "abs_corr"})
    )
    high_corr_pairs = high_corr_pairs[high_corr_pairs["abs_corr"] >= threshold]
    return high_corr_pairs.sort_values("abs_corr", ascending=False)


def train_test_split_data(X, y, test_size: float = 0.1):
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        shuffle=True,
    )


def correlation_metric(y_true, y_pred):
    """Pearson correlation between actual and predicted target."""
    if y_true.ndim > 1:
        y_true = y_true.ravel()
    if y_pred.ndim > 1:
        y_pred = y_pred.ravel()
    return np.corrcoef(y_true, y_pred)[0, 1]


def evaluate_model(name, model, X_train, y_train, X_test, y_test, cv_splits=5):
    """
    Fit model with 5-fold CV on training data, then evaluate on held-out test set.
    Returns dict with metrics on the test set.
    """
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    corr = correlation_metric(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return {
        "Model": name,
        "CV_R2_mean": np.mean(cv_scores),
        "CV_R2_std": np.std(cv_scores),
        "Correlation": corr,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }, model


def build_models():
    """
    Build all required regression models as sklearn estimators.
    Most will use StandardScaler to ensure fair treatment of coefficients.
    """
    models = {}

    # 1. Linear Regression
    models["Linear Regression"] = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("reg", LinearRegression()),
        ]
    )

    # 2. LASSO Regression
    models["LASSO Regression"] = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("reg", Lasso(alpha=0.001, random_state=RANDOM_STATE, max_iter=10000)),
        ]
    )

    # 3. RIDGE Regression
    models["RIDGE Regression"] = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
        ]
    )

    # 4. ElasticNet Regression
    models["ElasticNet Regression"] = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("reg", ElasticNet(
                alpha=0.001,
                l1_ratio=0.5,
                random_state=RANDOM_STATE,
                max_iter=10000,
            )),
        ]
    )

    # 5. Polynomial Regression (order 2)
    models["Polynomial Regression (deg=2)"] = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("reg", LinearRegression()),
        ]
    )

    # 6. Random Forest Regressor
    models["Random Forest Regressor"] = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return models


def run_all_models(X_train, y_train, X_test, y_test):
    models = build_models()
    results = []
    fitted_models = {}

    for name, model in models.items():
        metrics, fitted = evaluate_model(
            name, model, X_train, y_train, X_test, y_test
        )
        results.append(metrics)
        fitted_models[name] = fitted

    results_df = pd.DataFrame(results)
    return results_df, fitted_models


def count_polynomial_features(n_features: int, degree: int = 2, include_bias: bool = False):
    """
    Use sklearn's PolynomialFeatures to compute how many output features we get.
    """
    poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
    dummy = np.zeros((1, n_features))
    transformed = poly.fit_transform(dummy)
    return transformed.shape[1]


def run_before_feature_selection():
    """Run models with all features (BEFORE feature selection)"""
    df, feature_cols, target_col, X, y = load_data()

    # 1. Collinearity analysis
    print("\n=== 1. Collinearity analysis among input variables ===")
    high_corr_pairs = analyze_collinearity(df, feature_cols, threshold=0.9)
    if high_corr_pairs.empty:
        print("No feature pairs with |correlation| >= 0.9 found.")
    else:
        print("Highly correlated feature pairs (|corr| >= 0.9):")
        print(high_corr_pairs.to_string(index=False))

    # 2. Train/test split
    X_train, X_test, y_train, y_test = train_test_split_data(X, y, test_size=0.1)
    print("\n=== 2. Train / Test split (90% / 10%) ===")
    print(f"Total samples: {len(X)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # Save split data for later use
    split_data_path = Path(__file__).with_name("split_data.pkl")
    with open(split_data_path, "wb") as f:
        pickle.dump({
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "feature_cols": feature_cols,
            "target_col": target_col,
        }, f)
    print(f"\nSplit data saved to: {split_data_path.name}")

    # 3. Train all models with all features
    print("\n=== 3. Baseline models with all features (5-fold CV on train, metrics on test) ===")
    baseline_results_df, baseline_models = run_all_models(X_train, y_train, X_test, y_test)
    baseline_display = baseline_results_df[
        ["Model", "Correlation", "MAE", "RMSE", "R2"]
    ].copy()
    print("\nResults BEFORE feature subset selection (test set):")
    print(baseline_display.to_string(index=False))

    # Save baseline models and results
    baseline_models_path = Path(__file__).with_name("baseline_models.pkl")
    with open(baseline_models_path, "wb") as f:
        pickle.dump(baseline_models, f)
    print(f"Baseline models saved to: {baseline_models_path.name}")

    # Save results to CSV
    baseline_csv_path = Path(__file__).with_name("results_before_feature_selection.csv")
    baseline_results_df.to_csv(baseline_csv_path, index=False)
    print(f"Baseline results saved to: {baseline_csv_path.name}")

    # 7. Polynomial regression coefficients / complexity
    print("\n=== 7. Polynomial Regression model complexity ===")
    n_poly_features = count_polynomial_features(n_features=len(feature_cols), degree=2, include_bias=False)
    print(f"Number of coefficients (features) in polynomial regressor of degree 2 (excluding bias): {n_poly_features}")

    print("\n" + "="*60)
    print("COMPLETE: Models with all features")
    print("="*60)
    print("\nTo run GA feature selection and models with selected features, run:")
    print("  python mic_regression_after_feature_selection.py")


def main():
    """Main function - runs models with all features"""
    run_before_feature_selection()


if __name__ == "__main__":
    main()

