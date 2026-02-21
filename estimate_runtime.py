"""
Runtime estimation for mic_regression_after_feature_selection.py
"""
import numpy as np

# Dataset info
total_samples = 11266
n_features = 39
train_ratio = 0.9
train_samples = int(total_samples * train_ratio)

# GA configuration
n_configs = 3
configs = [
    {"n_generations": 15, "pop_size": 20},
    {"n_generations": 20, "pop_size": 20},
    {"n_generations": 15, "pop_size": 20},
]

# Fitness evaluation details
cv_folds = 5
rf_trees = 100
rf_max_depth = 10
samples_per_fold = int(train_samples * 0.8)  # 80% for training in each fold

print("=" * 60)
print("RUNTIME ESTIMATION for mic_regression_after_feature_selection.py")
print("=" * 60)
print(f"\nDataset: {total_samples:,} samples, {n_features} features")
print(f"Training set: ~{train_samples:,} samples (90%)")
print(f"Test set: ~{total_samples - train_samples:,} samples (10%)")

print("\n" + "-" * 60)
print("GA FEATURE SELECTION PHASE")
print("-" * 60)

total_fitness_evals = 0
for i, cfg in enumerate(configs, 1):
    n_gen = cfg["n_generations"]
    pop_size = cfg["pop_size"]
    # Initial population + generations * population size
    evals_per_config = pop_size + (n_gen * pop_size)
    total_fitness_evals += evals_per_config
    print(f"\nConfiguration {i}:")
    print(f"  Generations: {n_gen}, Population: {pop_size}")
    print(f"  Fitness evaluations: {evals_per_config:,}")
    print(f"    - Initial population: {pop_size}")
    print(f"    - Per generation: {pop_size} individuals")
    print(f"    - Total: {pop_size} + ({n_gen} × {pop_size}) = {evals_per_config}")

print(f"\nTotal fitness evaluations across all configs: {total_fitness_evals:,}")

print("\n" + "-" * 60)
print("FITNESS EVALUATION DETAILS")
print("-" * 60)
print(f"Each fitness evaluation:")
print(f"  - 5-fold cross-validation")
print(f"  - RandomForest: {rf_trees} trees, max_depth={rf_max_depth}")
print(f"  - Training samples per fold: ~{samples_per_fold:,}")
print(f"  - Uses n_jobs=-1 (parallelized)")

print("\n" + "-" * 60)
print("TIME ESTIMATES")
print("-" * 60)

# Estimate time per fitness evaluation
# RandomForest with 100 trees, max_depth=10 on ~8K samples
# With parallelization (n_jobs=-1), this might take:
# - Fast CPU (8+ cores): 3-8 seconds per CV fold = 15-40 seconds total per eval
# - Medium CPU (4 cores): 5-12 seconds per CV fold = 25-60 seconds total per eval
# - Slow CPU (2 cores): 8-20 seconds per CV fold = 40-100 seconds total per eval

print("\nEstimated time per fitness evaluation (with parallelization):")
print("  - Fast CPU (8+ cores): 5-15 seconds")
print("  - Medium CPU (4 cores): 10-25 seconds")
print("  - Slow CPU (2 cores): 20-50 seconds")

print("\nTotal GA phase time:")
fast_time = total_fitness_evals * 10  # Average 10 seconds
medium_time = total_fitness_evals * 17.5  # Average 17.5 seconds
slow_time = total_fitness_evals * 35  # Average 35 seconds

print(f"  - Fast CPU: {fast_time/60:.1f} - {fast_time*1.5/60:.1f} minutes ({fast_time/3600:.2f} - {fast_time*1.5/3600:.2f} hours)")
print(f"  - Medium CPU: {medium_time/60:.1f} - {medium_time*1.5/60:.1f} minutes ({medium_time/3600:.2f} - {medium_time*1.5/3600:.2f} hours)")
print(f"  - Slow CPU: {slow_time/60:.1f} - {slow_time*1.5/60:.1f} minutes ({slow_time/3600:.2f} - {slow_time*1.5/3600:.2f} hours)")

print("\n" + "-" * 60)
print("POST-GA MODEL TRAINING")
print("-" * 60)
print("After GA, retraining 6 models on selected features:")
print("  - Linear, LASSO, RIDGE, ElasticNet, Polynomial, RandomForest")
print("  - Estimated time: 1-3 minutes")

print("\n" + "=" * 60)
print("TOTAL ESTIMATED RUNTIME")
print("=" * 60)
print(f"  - Fast CPU: {fast_time/3600 + 0.02:.2f} - {fast_time*1.5/3600 + 0.05:.2f} hours")
print(f"  - Medium CPU: {medium_time/3600 + 0.02:.2f} - {medium_time*1.5/3600 + 0.05:.2f} hours")
print(f"  - Slow CPU: {slow_time/3600 + 0.02:.2f} - {slow_time*1.5/3600 + 0.05:.2f} hours")

print("\n" + "=" * 60)
print("RECOMMENDATIONS")
print("=" * 60)
print("1. Run overnight or when you have 2-6 hours available")
print("2. The script uses parallelization (n_jobs=-1), so more CPU cores = faster")
print("3. You can reduce runtime by:")
print("   - Reducing n_generations (e.g., 10 instead of 15-20)")
print("   - Reducing pop_size (e.g., 15 instead of 20)")
print("   - Testing fewer configurations (e.g., 1-2 instead of 3)")
print("4. Consider using the package-based version (mic_regression_ga_package.py)")
print("   which might be optimized better")
