"""
Hyperparameter Tuning with Optuna

This module provides automated hyperparameter optimization for all BrewMatch models
using Optuna's Bayesian optimization framework.

Optimizes:
- Neural network: embedding_dim, hidden_dim, learning_rate, margin, batch_size, dropout
- Classical ML: n_neighbors, method (knn/cosine), normalization
- Baseline: strategy selection

Uses cross-validation for robust evaluation and early pruning for efficiency.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import pandas as pd
import torch

from brewmatch.config import (
    CHECKPOINTS_DIR,
    K_VALUES,
    PROJECT_ROOT,
    TASTE_FEATURES,
)
from brewmatch.data import load_processed_data
from brewmatch.models import (
    NaiveBaselineRecommender,
    ClassicalMLRecommender,
    NeuralRecommender,
)
from brewmatch.evaluation import evaluate_model


TUNING_DIR = PROJECT_ROOT / "tuning"


def create_cross_validation_splits(
    df: pd.DataFrame,
    n_folds: int = 5,
    random_state: int = 42,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Create stratified cross-validation splits."""
    np.random.seed(random_state)
    indices = np.random.permutation(len(df))
    fold_size = len(df) // n_folds

    splits = []
    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else len(df)

        val_indices = indices[start:end]
        train_indices = np.concatenate([indices[:start], indices[end:]])

        train_df = df.iloc[train_indices].reset_index(drop=True)
        val_df = df.iloc[val_indices].reset_index(drop=True)
        splits.append((train_df, val_df))

    return splits


def objective_neural(
    trial: optuna.Trial,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    device: str,
) -> float:
    """Optuna objective function for neural network hyperparameters."""
    # Sample hyperparameters
    embedding_dim = trial.suggest_int("embedding_dim", 16, 128, step=16)
    hidden_dim = trial.suggest_int("hidden_dim", 32, 256, step=32)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    margin = trial.suggest_float("margin", 0.1, 1.0)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    dropout = trial.suggest_float("dropout", 0.0, 0.5)

    # Create and train model
    model = NeuralRecommender(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        margin=margin,
        device=device,
    )

    # Train with reduced epochs for tuning
    X_train = train_df[TASTE_FEATURES].values
    model.fit(
        X=X_train,
        metadata=train_df,
        epochs=50,  # Reduced for faster tuning
        batch_size=batch_size,
        verbose=False,
    )

    # Evaluate on validation set
    val_data = {
        "X": val_df[TASTE_FEATURES].values,
        "metadata": val_df,
    }

    metrics = evaluate_model(model, val_data, k_values=[5])

    # Return primary metric (Precision@5)
    precision_5 = metrics.get("precision@k", {}).get(5, 0.0)

    return precision_5


def objective_classical(
    trial: optuna.Trial,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> float:
    """Optuna objective function for classical ML hyperparameters."""
    # Sample hyperparameters
    method = trial.suggest_categorical("method", ["knn", "cosine"])
    n_neighbors = trial.suggest_int("n_neighbors", 5, 100)
    normalize = trial.suggest_categorical("normalize", [True, False])

    # Create and train model
    model = ClassicalMLRecommender(
        method=method,
        n_neighbors=n_neighbors,
        normalize=normalize,
    )

    X_train = train_df[TASTE_FEATURES].values
    model.fit(X_train, train_df)

    # Evaluate on validation set
    val_data = {
        "X": val_df[TASTE_FEATURES].values,
        "metadata": val_df,
    }

    metrics = evaluate_model(model, val_data, k_values=[5])
    precision_5 = metrics.get("precision@k", {}).get(5, 0.0)

    return precision_5


def tune_neural(
    train_df: pd.DataFrame,
    n_trials: int = 50,
    n_folds: int = 3,
    device: str = "cuda",
    study_name: str = "neural_tuning",
) -> dict[str, Any]:
    """
    Tune neural network hyperparameters using Optuna.

    Args:
        train_df: Training data
        n_trials: Number of optimization trials
        n_folds: Number of cross-validation folds
        device: PyTorch device
        study_name: Name for the Optuna study

    Returns:
        Dictionary with best parameters and study results
    """
    print(f"\n{'='*60}")
    print("NEURAL NETWORK HYPERPARAMETER TUNING")
    print(f"{'='*60}")
    print(f"Trials: {n_trials}, CV Folds: {n_folds}, Device: {device}")

    # Create cross-validation splits
    splits = create_cross_validation_splits(train_df, n_folds=n_folds)

    def cv_objective(trial: optuna.Trial) -> float:
        """Cross-validated objective."""
        scores = []
        for fold_idx, (fold_train, fold_val) in enumerate(splits):
            score = objective_neural(trial, fold_train, fold_val, device)
            scores.append(score)

            # Report intermediate value for pruning
            trial.report(np.mean(scores), fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(scores)

    # Create study with TPE sampler and median pruner
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )

    study.optimize(
        cv_objective,
        n_trials=n_trials,
        show_progress_bar=True,
        gc_after_trial=True,
    )

    print(f"\nBest trial:")
    print(f"  Value (Precision@5): {study.best_trial.value:.4f}")
    print(f"  Params: {study.best_trial.params}")

    return {
        "best_params": study.best_trial.params,
        "best_value": study.best_trial.value,
        "n_trials": len(study.trials),
        "study_name": study_name,
    }


def tune_classical(
    train_df: pd.DataFrame,
    n_trials: int = 30,
    n_folds: int = 3,
    study_name: str = "classical_tuning",
) -> dict[str, Any]:
    """
    Tune classical ML hyperparameters using Optuna.

    Args:
        train_df: Training data
        n_trials: Number of optimization trials
        n_folds: Number of cross-validation folds
        study_name: Name for the Optuna study

    Returns:
        Dictionary with best parameters and study results
    """
    print(f"\n{'='*60}")
    print("CLASSICAL ML HYPERPARAMETER TUNING")
    print(f"{'='*60}")
    print(f"Trials: {n_trials}, CV Folds: {n_folds}")

    splits = create_cross_validation_splits(train_df, n_folds=n_folds)

    def cv_objective(trial: optuna.Trial) -> float:
        scores = []
        for fold_train, fold_val in splits:
            score = objective_classical(trial, fold_train, fold_val)
            scores.append(score)
        return np.mean(scores)

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=TPESampler(seed=42),
    )

    study.optimize(
        cv_objective,
        n_trials=n_trials,
        show_progress_bar=True,
    )

    print(f"\nBest trial:")
    print(f"  Value (Precision@5): {study.best_trial.value:.4f}")
    print(f"  Params: {study.best_trial.params}")

    return {
        "best_params": study.best_trial.params,
        "best_value": study.best_trial.value,
        "n_trials": len(study.trials),
        "study_name": study_name,
    }


def train_with_best_params(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    neural_params: dict[str, Any] | None,
    classical_params: dict[str, Any] | None,
    device: str,
) -> dict[str, Any]:
    """Train final models with best hyperparameters and evaluate on test set."""
    print(f"\n{'='*60}")
    print("TRAINING FINAL MODELS WITH BEST PARAMETERS")
    print(f"{'='*60}")

    results = {}

    test_data = {
        "X": test_df[TASTE_FEATURES].values,
        "metadata": test_df,
    }

    # Train neural with best params
    if neural_params:
        print("\nTraining Neural Network with tuned hyperparameters...")
        model = NeuralRecommender(
            embedding_dim=neural_params["embedding_dim"],
            hidden_dim=neural_params["hidden_dim"],
            learning_rate=neural_params["learning_rate"],
            margin=neural_params["margin"],
            device=device,
        )

        X_train = train_df[TASTE_FEATURES].values
        model.fit(
            X=X_train,
            metadata=train_df,
            epochs=100,  # Full training
            batch_size=neural_params["batch_size"],
            verbose=True,
        )

        metrics = evaluate_model(model, test_data, k_values=K_VALUES)
        results["neural"] = {
            "params": neural_params,
            "metrics": metrics,
        }

        # Save model
        CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
        model.save(CHECKPOINTS_DIR / "neural.pt")
        print(f"Saved tuned neural model to {CHECKPOINTS_DIR / 'neural.pt'}")

    # Train classical with best params
    if classical_params:
        print("\nTraining Classical ML with tuned hyperparameters...")
        model = ClassicalMLRecommender(
            method=classical_params["method"],
            n_neighbors=classical_params["n_neighbors"],
            normalize=classical_params["normalize"],
        )

        X_train = train_df[TASTE_FEATURES].values
        model.fit(X_train, train_df)

        metrics = evaluate_model(model, test_data, k_values=K_VALUES)
        results["classical"] = {
            "params": classical_params,
            "metrics": metrics,
        }

        model.save(CHECKPOINTS_DIR / "classical.pkl")
        print(f"Saved tuned classical model to {CHECKPOINTS_DIR / 'classical.pkl'}")

    # Also train baseline for comparison
    print("\nTraining Baseline for comparison...")
    baseline = NaiveBaselineRecommender(strategy="mean")
    baseline.fit(train_df[TASTE_FEATURES].values, train_df)
    baseline_metrics = evaluate_model(baseline, test_data, k_values=K_VALUES)
    results["baseline"] = {"metrics": baseline_metrics}
    baseline.save(CHECKPOINTS_DIR / "baseline.pkl")

    return results


def generate_tuning_report(
    neural_results: dict[str, Any] | None,
    classical_results: dict[str, Any] | None,
    final_results: dict[str, Any],
    output_dir: Path,
) -> str:
    """Generate a comprehensive tuning report."""
    report = []
    report.append("=" * 60)
    report.append("HYPERPARAMETER TUNING REPORT")
    report.append("=" * 60)
    report.append("")

    if neural_results:
        report.append("NEURAL NETWORK")
        report.append("-" * 40)
        report.append(f"Trials completed: {neural_results['n_trials']}")
        report.append(f"Best CV Precision@5: {neural_results['best_value']:.4f}")
        report.append("Best hyperparameters:")
        for param, value in neural_results["best_params"].items():
            report.append(f"  - {param}: {value}")
        report.append("")

    if classical_results:
        report.append("CLASSICAL ML")
        report.append("-" * 40)
        report.append(f"Trials completed: {classical_results['n_trials']}")
        report.append(f"Best CV Precision@5: {classical_results['best_value']:.4f}")
        report.append("Best hyperparameters:")
        for param, value in classical_results["best_params"].items():
            report.append(f"  - {param}: {value}")
        report.append("")

    report.append("FINAL TEST SET PERFORMANCE")
    report.append("-" * 40)
    for model_name, result in final_results.items():
        metrics = result["metrics"]
        p5 = metrics.get("precision@k", {}).get(5, 0)
        ndcg5 = metrics.get("ndcg@k", {}).get(5, 0)
        report.append(f"{model_name.upper()}:")
        report.append(f"  Precision@5: {p5:.4f}")
        report.append(f"  NDCG@5: {ndcg5:.4f}")
    report.append("")

    # Improvement analysis
    if "baseline" in final_results and "neural" in final_results:
        baseline_p5 = final_results["baseline"]["metrics"].get("precision@k", {}).get(5, 0)
        neural_p5 = final_results["neural"]["metrics"].get("precision@k", {}).get(5, 0)
        if baseline_p5 > 0:
            improvement = (neural_p5 - baseline_p5) / baseline_p5 * 100
            report.append(f"Neural improvement over baseline: {improvement:+.1f}%")

    report_text = "\n".join(report)

    # Save report
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "tuning_report.txt", "w") as f:
        f.write(report_text)

    return report_text


def main():
    """Main entry point for hyperparameter tuning."""
    parser = argparse.ArgumentParser(
        description="Tune BrewMatch model hyperparameters with Optuna"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["neural", "classical", "all"],
        default=["all"],
        help="Which models to tune",
    )
    parser.add_argument(
        "--neural-trials",
        type=int,
        default=50,
        help="Number of trials for neural network tuning",
    )
    parser.add_argument(
        "--classical-trials",
        type=int,
        default=30,
        help="Number of trials for classical ML tuning",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=3,
        help="Number of cross-validation folds",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for neural network training",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(TUNING_DIR),
        help="Directory to save tuning results",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models_to_tune = args.models
    if "all" in models_to_tune:
        models_to_tune = ["neural", "classical"]

    print("HYPERPARAMETER TUNING WITH OPTUNA")
    print("=" * 60)
    print(f"Models to tune: {models_to_tune}")
    print(f"Neural trials: {args.neural_trials}")
    print(f"Classical trials: {args.classical_trials}")
    print(f"CV folds: {args.cv_folds}")
    print(f"Device: {args.device}")
    print(f"Output: {output_dir}")

    # Load data
    print("\nLoading data...")
    data = load_processed_data()
    train_df = data["train_df"]
    test_df = data["test_df"]
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")

    # Tune models
    neural_results = None
    classical_results = None

    if "neural" in models_to_tune:
        neural_results = tune_neural(
            train_df=train_df,
            n_trials=args.neural_trials,
            n_folds=args.cv_folds,
            device=args.device,
        )

        # Save neural results
        with open(output_dir / "neural_tuning.json", "w") as f:
            json.dump(neural_results, f, indent=2)

    if "classical" in models_to_tune:
        classical_results = tune_classical(
            train_df=train_df,
            n_trials=args.classical_trials,
            n_folds=args.cv_folds,
        )

        # Save classical results
        with open(output_dir / "classical_tuning.json", "w") as f:
            json.dump(classical_results, f, indent=2)

    # Train final models with best params
    final_results = train_with_best_params(
        train_df=train_df,
        test_df=test_df,
        neural_params=neural_results["best_params"] if neural_results else None,
        classical_params=classical_results["best_params"] if classical_results else None,
        device=args.device,
    )

    # Save final results
    with open(output_dir / "final_results.json", "w") as f:
        # Convert metrics to JSON-serializable format
        json_results = {}
        for model_name, result in final_results.items():
            json_results[model_name] = {
                "params": result.get("params", {}),
                "metrics": {
                    k: {str(kk): vv for kk, vv in v.items()} if isinstance(v, dict) else v
                    for k, v in result["metrics"].items()
                },
            }
        json.dump(json_results, f, indent=2)

    # Generate report
    report = generate_tuning_report(
        neural_results=neural_results,
        classical_results=classical_results,
        final_results=final_results,
        output_dir=output_dir,
    )

    print("\n" + report)
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
