"""Training script for all BrewMatch models.

Supports hyperparameter tuning with Optuna:
- `uv run train` - Train with defaults or previously tuned hyperparameters
- `uv run train --tune` - Run Optuna tuning, save params, then train
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import pandas as pd

from brewmatch.config import (
    CHECKPOINTS_DIR,
    PROJECT_ROOT,
    TASTE_FEATURES,
)
from brewmatch.data import load_processed_data
from brewmatch.device import get_device, print_device_info
from brewmatch.models import (
    NaiveBaselineRecommender,
    ClassicalMLRecommender,
    NeuralRecommender,
)
from brewmatch.evaluation import evaluate_model


# Where tuned hyperparameters are saved
HYPERPARAMS_FILE = CHECKPOINTS_DIR / "hyperparameters.json"

# Default hyperparameters (used if no tuning has been done)
DEFAULT_NEURAL_PARAMS = {
    "embedding_dim": 32,
    "hidden_dim": 64,
    "learning_rate": 0.001,
    "margin": 0.5,
    "batch_size": 32,
    "epochs": 200,  # Max epochs (early stopping will trigger before this)
    "patience": 15,  # Early stopping patience
}

DEFAULT_CLASSICAL_PARAMS = {
    "method": "knn",
    "n_neighbors": 50,
    "normalize": True,
}


def load_hyperparameters() -> dict[str, Any]:
    """Load saved hyperparameters if they exist."""
    if HYPERPARAMS_FILE.exists():
        with open(HYPERPARAMS_FILE) as f:
            return json.load(f)
    return {}


def save_hyperparameters(params: dict[str, Any]) -> None:
    """Save hyperparameters for future runs."""
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(HYPERPARAMS_FILE, "w") as f:
        json.dump(params, f, indent=2)
    print(f"Hyperparameters saved to {HYPERPARAMS_FILE}")


def get_neural_params(saved: dict[str, Any]) -> dict[str, Any]:
    """Get neural network params (saved or defaults)."""
    if "neural" in saved:
        print("Using tuned neural hyperparameters")
        return {**DEFAULT_NEURAL_PARAMS, **saved["neural"]}
    print("Using default neural hyperparameters")
    return DEFAULT_NEURAL_PARAMS.copy()


def get_classical_params(saved: dict[str, Any]) -> dict[str, Any]:
    """Get classical ML params (saved or defaults)."""
    if "classical" in saved:
        print("Using tuned classical hyperparameters")
        return {**DEFAULT_CLASSICAL_PARAMS, **saved["classical"]}
    print("Using default classical hyperparameters")
    return DEFAULT_CLASSICAL_PARAMS.copy()


# =============================================================================
# Training Functions
# =============================================================================

def train_baseline(train_df: pd.DataFrame) -> NaiveBaselineRecommender:
    """Train the naive baseline model."""
    print("Training Naive Baseline Model...")

    X_train = train_df[TASTE_FEATURES].values
    model = NaiveBaselineRecommender(strategy="mean")
    model.fit(X_train, train_df)

    print(f"  Strategy: {model.strategy}")
    print(f"  Coffees indexed: {len(model._X)}")

    return model


def train_classical(train_df: pd.DataFrame, params: dict[str, Any]) -> ClassicalMLRecommender:
    """Train the classical ML model with given hyperparameters."""
    print("Training Classical ML Model...")
    print(f"  Params: {params}")

    X_train = train_df[TASTE_FEATURES].values
    model = ClassicalMLRecommender(
        method=params["method"],
        n_neighbors=params["n_neighbors"],
        normalize=params["normalize"],
    )
    model.fit(X_train, train_df)

    print(f"  Coffees indexed: {len(model._X)}")

    return model


def train_neural(
    train_df: pd.DataFrame,
    params: dict[str, Any],
    device: str,
) -> NeuralRecommender:
    """Train the neural network model with given hyperparameters."""
    print("Training Neural Network Model...")
    print(f"  Params: {params}")

    X_train = train_df[TASTE_FEATURES].values

    model = NeuralRecommender(
        embedding_dim=params["embedding_dim"],
        hidden_dim=params["hidden_dim"],
        learning_rate=params["learning_rate"],
        margin=params["margin"],
        device=device,
    )

    model.fit(
        X=X_train,
        metadata=train_df,
        epochs=params.get("epochs", 200),
        batch_size=params["batch_size"],
        patience=params.get("patience", 15),
        verbose=True,
    )

    return model


def save_models(
    baseline: NaiveBaselineRecommender | None,
    classical: ClassicalMLRecommender | None,
    neural: NeuralRecommender | None,
    params: dict[str, Any],
) -> None:
    """Save all trained models."""
    print("\nSaving models...")

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    if baseline:
        baseline.save(CHECKPOINTS_DIR / "baseline.pkl")
        print(f"  Baseline: {CHECKPOINTS_DIR / 'baseline.pkl'}")

    if classical:
        classical.save(CHECKPOINTS_DIR / "classical.pkl")
        print(f"  Classical: {CHECKPOINTS_DIR / 'classical.pkl'}")

    if neural:
        neural.save(CHECKPOINTS_DIR / "neural.pt")
        print(f"  Neural: {CHECKPOINTS_DIR / 'neural.pt'}")

    # Save model metadata
    model_info = {
        "models": ["baseline", "classical", "neural"],
        "taste_features": TASTE_FEATURES,
        "hyperparameters": params,
    }
    with open(CHECKPOINTS_DIR / "model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)


# =============================================================================
# Optuna Hyperparameter Tuning
# =============================================================================

def create_cv_splits(
    df: pd.DataFrame,
    n_folds: int = 3,
    seed: int = 42,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Create cross-validation splits."""
    np.random.seed(seed)
    indices = np.random.permutation(len(df))
    fold_size = len(df) // n_folds

    splits = []
    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else len(df)

        val_idx = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])

        splits.append((
            df.iloc[train_idx].reset_index(drop=True),
            df.iloc[val_idx].reset_index(drop=True),
        ))

    return splits


def tune_neural(
    train_df: pd.DataFrame,
    device: str,
    n_trials: int = 50,
    n_folds: int = 3,
) -> dict[str, Any]:
    """Tune neural network hyperparameters with Optuna."""
    print(f"\n{'='*60}")
    print("TUNING NEURAL NETWORK HYPERPARAMETERS")
    print(f"{'='*60}")
    print(f"Trials: {n_trials}, CV Folds: {n_folds}")

    splits = create_cv_splits(train_df, n_folds)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "embedding_dim": trial.suggest_int("embedding_dim", 16, 128, step=16),
            "hidden_dim": trial.suggest_int("hidden_dim", 32, 256, step=32),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "margin": trial.suggest_float("margin", 0.1, 1.0),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
        }

        scores = []
        for fold_idx, (fold_train, fold_val) in enumerate(splits):
            model = NeuralRecommender(
                embedding_dim=params["embedding_dim"],
                hidden_dim=params["hidden_dim"],
                learning_rate=params["learning_rate"],
                margin=params["margin"],
                device=device,
            )

            model.fit(
                X=fold_train[TASTE_FEATURES].values,
                metadata=fold_train,
                epochs=30,  # Reduced for tuning speed
                batch_size=params["batch_size"],
                verbose=False,
            )

            metrics = evaluate_model(
                model,
                {"X": fold_val[TASTE_FEATURES].values, "metadata": fold_val},
                k_values=[5],
            )

            score = metrics.get("precision@k", {}).get(5, 0.0)
            scores.append(score)

            trial.report(np.mean(scores), fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(scores)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )

    # Suppress Optuna logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nBest Precision@5: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    return study.best_params


def tune_classical(
    train_df: pd.DataFrame,
    n_trials: int = 30,
    n_folds: int = 3,
) -> dict[str, Any]:
    """Tune classical ML hyperparameters with Optuna."""
    print(f"\n{'='*60}")
    print("TUNING CLASSICAL ML HYPERPARAMETERS")
    print(f"{'='*60}")
    print(f"Trials: {n_trials}, CV Folds: {n_folds}")

    splits = create_cv_splits(train_df, n_folds)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "method": trial.suggest_categorical("method", ["knn", "cosine"]),
            "n_neighbors": trial.suggest_int("n_neighbors", 5, 100),
            "normalize": trial.suggest_categorical("normalize", [True, False]),
        }

        scores = []
        for fold_train, fold_val in splits:
            model = ClassicalMLRecommender(**params)
            model.fit(fold_train[TASTE_FEATURES].values, fold_train)

            metrics = evaluate_model(
                model,
                {"X": fold_val[TASTE_FEATURES].values, "metadata": fold_val},
                k_values=[5],
            )

            scores.append(metrics.get("precision@k", {}).get(5, 0.0))

        return np.mean(scores)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nBest Precision@5: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    return study.best_params


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(
        description="Train BrewMatch recommendation models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run train                    # Train with defaults or saved hyperparameters
  uv run train --tune             # Tune hyperparameters, then train
  uv run train --models neural    # Train only neural network
  uv run train --tune --neural-trials 100
        """,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["baseline", "classical", "neural", "all"],
        default=["all"],
        help="Which models to train (default: all)",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run Optuna hyperparameter tuning before training",
    )
    parser.add_argument(
        "--neural-trials",
        type=int,
        default=50,
        help="Number of Optuna trials for neural network (default: 50)",
    )
    parser.add_argument(
        "--classical-trials",
        type=int,
        default=30,
        help="Number of Optuna trials for classical ML (default: 30)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=3,
        help="Cross-validation folds for tuning (default: 3)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to train on (cuda/mps/cpu, auto-detected if not specified)",
    )
    args = parser.parse_args()

    # Device selection
    device = get_device(args.device)
    print_device_info()
    print()

    # Expand "all" to all models
    models_to_train = args.models
    if "all" in models_to_train:
        models_to_train = ["baseline", "classical", "neural"]

    print(f"Models to train: {models_to_train}")

    # Load data
    print("\nLoading processed data...")
    data = load_processed_data()
    train_df = data["train_df"]
    test_df = data["test_df"]
    print(f"  Train: {len(train_df)} samples")
    print(f"  Test: {len(test_df)} samples")

    # Load or tune hyperparameters
    saved_params = load_hyperparameters()

    if args.tune:
        print("\n" + "=" * 60)
        print("HYPERPARAMETER TUNING WITH OPTUNA")
        print("=" * 60)

        if "neural" in models_to_train:
            neural_params = tune_neural(
                train_df,
                device=str(device),
                n_trials=args.neural_trials,
                n_folds=args.cv_folds,
            )
            saved_params["neural"] = neural_params

        if "classical" in models_to_train:
            classical_params = tune_classical(
                train_df,
                n_trials=args.classical_trials,
                n_folds=args.cv_folds,
            )
            saved_params["classical"] = classical_params

        # Save tuned hyperparameters
        save_hyperparameters(saved_params)

    # Get final hyperparameters
    neural_params = get_neural_params(saved_params)
    classical_params = get_classical_params(saved_params)

    # Train models
    print("\n" + "=" * 60)
    print("TRAINING MODELS")
    print("=" * 60)

    baseline_model = None
    classical_model = None
    neural_model = None

    if "baseline" in models_to_train:
        baseline_model = train_baseline(train_df)
        print()

    if "classical" in models_to_train:
        classical_model = train_classical(train_df, classical_params)
        print()

    if "neural" in models_to_train:
        neural_model = train_neural(train_df, neural_params, str(device))
        print()

    # Save models
    all_params = {
        "neural": neural_params,
        "classical": classical_params,
    }
    save_models(baseline_model, classical_model, neural_model, all_params)

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
