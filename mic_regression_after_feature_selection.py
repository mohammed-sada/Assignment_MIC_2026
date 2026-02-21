"""
Part 2: GA Feature Selection and Models with Selected Features

This script runs after the baseline models have been trained.
It performs genetic algorithm-based feature selection and retrains all models
on the selected feature subset.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path

from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Import shared functions from main script
from mic_regression_solution import (
    RANDOM_STATE,
    build_models,
    run_all_models,
    evaluate_model,
)


def genetic_algorithm_feature_selection(
    X_train,
    y_train,
    n_generations=30,
    pop_size=30,
    crossover_prob=0.8,
    mutation_rate=0.02,
    base_model=None,
):
    """
    Simple binary GA for feature subset selection.
    Fitness = mean 5-fold CV R^2 on training data using base_model.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    n_features = X_train.shape[1]

    if base_model is None:
        base_model = RandomForestRegressor(
            n_estimators=100,  # Reduced for faster GA evaluation
            max_depth=10,  # Limit depth for speed
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    def init_population():
        # ensure at least one feature in each chromosome
        pop = rng.integers(0, 2, size=(pop_size, n_features), endpoint=True)
        for ind in pop:
            if ind.sum() == 0:
                ind[rng.integers(0, n_features)] = 1
        return pop

    def fitness(individual):
        if individual.sum() == 0:
            return -np.inf
        selected = individual.astype(bool)
        scores = cross_val_score(
            base_model,
            X_train[:, selected],
            y_train,
            cv=cv,
            scoring="r2",
            n_jobs=-1,
        )
        return float(np.mean(scores))

    def tournament_selection(pop, fits, k=3):
        indices = rng.choice(len(pop), size=k, replace=False)
        best_idx = indices[0]
        best_fit = fits[best_idx]
        for idx in indices[1:]:
            if fits[idx] > best_fit:
                best_idx = idx
                best_fit = fits[idx]
        return pop[best_idx].copy()

    def crossover(parent1, parent2):
        if rng.random() > crossover_prob:
            return parent1.copy(), parent2.copy()
        point = rng.integers(1, n_features)
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        return child1, child2

    def mutate(individual):
        for i in range(n_features):
            if rng.random() < mutation_rate:
                individual[i] = 1 - individual[i]
        if individual.sum() == 0:
            individual[rng.integers(0, n_features)] = 1
        return individual

    population = init_population()
    fitness_values = np.array([fitness(ind) for ind in population])

    best_idx = int(np.argmax(fitness_values))
    best_individual = population[best_idx].copy()
    best_fitness = fitness_values[best_idx]

    history = []
    history.append(best_fitness)

    for gen in range(n_generations):
        new_population = []
        # elitism: keep the best individual
        new_population.append(best_individual.copy())

        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, fitness_values)
            parent2 = tournament_selection(population, fitness_values)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)
            new_population.extend([child1, child2])

        population = np.array(new_population[:pop_size])
        fitness_values = np.array([fitness(ind) for ind in population])

        current_best_idx = int(np.argmax(fitness_values))
        current_best_fit = fitness_values[current_best_idx]
        if current_best_fit > best_fitness:
            best_fitness = current_best_fit
            best_individual = population[current_best_idx].copy()

        history.append(best_fitness)

    return {
        "best_individual": best_individual,
        "best_fitness": best_fitness,
        "history": history,
    }


def search_best_ga_configuration(X_train, y_train):
    """
    Try several GA configurations and return the best one.
    Optimized for faster execution with fewer configurations.
    """
    configs = [
        {
            "n_generations": 15,
            "crossover_prob": 0.8,
            "mutation_rate": 0.02,
            "pop_size": 20,
        },
        {
            "n_generations": 20,
            "crossover_prob": 0.7,
            "mutation_rate": 0.03,
            "pop_size": 20,
        },
        {
            "n_generations": 15,
            "crossover_prob": 0.9,
            "mutation_rate": 0.01,
            "pop_size": 20,
        },
    ]

    best_overall = None

    X_train_np = np.asarray(X_train)
    y_train_np = np.asarray(y_train)

    print(f"Testing {len(configs)} GA configurations...")
    for i, cfg in enumerate(configs, 1):
        print(f"  Configuration {i}/{len(configs)}: gen={cfg['n_generations']}, "
              f"crossover={cfg['crossover_prob']}, mutation={cfg['mutation_rate']}, "
              f"pop_size={cfg['pop_size']}")
        ga_result = genetic_algorithm_feature_selection(
            X_train_np,
            y_train_np,
            n_generations=cfg["n_generations"],
            pop_size=cfg["pop_size"],
            crossover_prob=cfg["crossover_prob"],
            mutation_rate=cfg["mutation_rate"],
        )
        print(f"    -> Fitness (CV R2): {ga_result['best_fitness']:.4f}")
        if best_overall is None or ga_result["best_fitness"] > best_overall["best_fitness"]:
            best_overall = {
                "config": cfg,
                "ga_result": ga_result,
            }

    return best_overall


def extract_linear_coefficients(pipeline, feature_names):
    """
    Extract coefficients from a fitted sklearn Pipeline containing a LinearRegression as final step.
    Returns a pandas Series indexed by expanded feature names (after polynomial step if present).
    """
    steps = dict(pipeline.named_steps)
    if "poly" in steps:
        poly: PolynomialFeatures = steps["poly"]
        expanded_names = poly.get_feature_names_out(feature_names)
    else:
        expanded_names = np.array(feature_names)

    reg: LinearRegression = steps["reg"]
    coefs = reg.coef_.ravel()
    return pd.Series(coefs, index=expanded_names, name="coefficient")


def run_after_feature_selection():
    """Run GA feature selection and models with selected features (AFTER feature selection)"""
    # Load saved split data
    split_data_path = Path(__file__).with_name("split_data.pkl")
    if not split_data_path.exists():
        print("ERROR: split_data.pkl not found. Please run 'before' first:")
        print("  python mic_regression_solution.py")
        return

    with open(split_data_path, "rb") as f:
        split_data = pickle.load(f)
    X_train = split_data["X_train"]
    X_test = split_data["X_test"]
    y_train = split_data["y_train"]
    y_test = split_data["y_test"]
    feature_cols = split_data["feature_cols"]

    # Load baseline models
    baseline_models_path = Path(__file__).with_name("baseline_models.pkl")
    if not baseline_models_path.exists():
        print("ERROR: baseline_models.pkl not found. Please run 'before' first:")
        print("  python mic_regression_solution.py")
        return

    with open(baseline_models_path, "rb") as f:
        baseline_models = pickle.load(f)

    # 4. GA-based feature subset selection
    print("\n=== 4. Genetic Algorithm for feature subset selection (on training data only) ===")
    best_ga = search_best_ga_configuration(X_train, y_train)
    best_cfg = best_ga["config"]
    best_ind = best_ga["ga_result"]["best_individual"]
    best_fit = best_ga["ga_result"]["best_fitness"]

    selected_mask = best_ind.astype(bool)
    selected_features = [f for f, flag in zip(feature_cols, selected_mask) if flag]

    print("\nBest GA configuration:")
    print(best_cfg)
    print(f"Best GA CV R2 (fitness): {best_fit:.4f}")
    print(f"Number of selected features: {selected_mask.sum()} / {len(feature_cols)}")
    print("Selected feature indices (0-based):", np.where(selected_mask)[0].tolist())
    print("Selected feature names:", selected_features)

    X_train_sel = X_train[:, selected_mask]
    X_test_sel = X_test[:, selected_mask]

    print("\n=== 5. Models retrained on GA-selected feature subset ===")
    selected_results_df, selected_models = run_all_models(
        X_train_sel, y_train, X_test_sel, y_test
    )
    selected_display = selected_results_df[
        ["Model", "Correlation", "MAE", "RMSE", "R2"]
    ].copy()
    print("\nResults AFTER feature subset selection (test set):")
    print(selected_display.to_string(index=False))

    # Save results to CSV
    selected_csv_path = Path(__file__).with_name("results_after_feature_selection.csv")
    selected_results_df.to_csv(selected_csv_path, index=False)
    print(f"\nSelected-features results saved to: {selected_csv_path.name}")

    # 6. Linear regression coefficients before vs after feature selection
    print("\n=== 6. Linear Regression coefficients: before vs after feature selection ===")
    lin_before = baseline_models["Linear Regression"]
    coef_before = extract_linear_coefficients(lin_before, feature_cols)
    lin_after = selected_models["Linear Regression"]
    coef_after = extract_linear_coefficients(lin_after, selected_features)

    print("\nNumber of coefficients in Linear Regression BEFORE feature selection:", coef_before.shape[0])
    print("Number of coefficients in Linear Regression AFTER feature selection:", coef_after.shape[0])

    print("\nTop 10 largest-magnitude coefficients BEFORE feature selection:")
    print(coef_before.reindex(coef_before.abs().sort_values(ascending=False).index[:10]).to_string())

    print("\nTop 10 largest-magnitude coefficients AFTER feature selection:")
    print(coef_after.reindex(coef_after.abs().sort_values(ascending=False).index[:10]).to_string())

    print("\n" + "="*60)
    print("PART 2 COMPLETE: Models with GA-selected features")
    print("="*60)


if __name__ == "__main__":
    run_after_feature_selection()
