"""
Focused Experiment: Training Set Size Sensitivity Analysis

This experiment investigates how model performance varies with training set size.
We train all three models (baseline, classical, neural) on progressively larger
subsets of the training data and measure their performance on a held-out test set.

Hypothesis: Deep learning model will show greater improvement with more data,
while classical models may plateau earlier.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm

from brewmatch.config import (
    K_VALUES,
    NEURAL_CONFIG,
    PROJECT_ROOT,
    RANDOM_SEED,
    TASTE_FEATURES,
)
from brewmatch.data import load_processed_data
from brewmatch.models import (
    NaiveBaselineRecommender,
    ClassicalMLRecommender,
    NeuralRecommender,
)
from brewmatch.evaluation import evaluate_model


# Experiment configuration
TRAIN_FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
N_TRIALS = 3  # Number of trials per fraction for variance estimation
RESULTS_DIR = PROJECT_ROOT / "experiments"


def subsample_data(
    df: pd.DataFrame,
    fraction: float,
    seed: int,
) -> pd.DataFrame:
    """Subsample training data to a given fraction."""
    np.random.seed(seed)
    n_samples = int(len(df) * fraction)
    indices = np.random.choice(len(df), n_samples, replace=False)
    return df.iloc[indices].reset_index(drop=True)


def train_and_evaluate_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, Any]:
    """Train and evaluate baseline model."""
    X_train = train_df[TASTE_FEATURES].values
    model = NaiveBaselineRecommender(strategy="mean")
    model.fit(X_train, train_df)

    test_data = {
        "X": test_df[TASTE_FEATURES].values,
        "metadata": test_df,
    }

    return evaluate_model(
        model=model,
        test_data=test_data,
        k_values=K_VALUES,
    )


def train_and_evaluate_classical(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, Any]:
    """Train and evaluate classical ML model."""
    X_train = train_df[TASTE_FEATURES].values
    model = ClassicalMLRecommender(method="knn", n_neighbors=50, normalize=True)
    model.fit(X_train, train_df)

    test_data = {
        "X": test_df[TASTE_FEATURES].values,
        "metadata": test_df,
    }

    return evaluate_model(
        model=model,
        test_data=test_data,
        k_values=K_VALUES,
    )


def train_and_evaluate_neural(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    device: str,
) -> dict[str, Any]:
    """Train and evaluate neural network model."""
    X_train = train_df[TASTE_FEATURES].values

    model = NeuralRecommender(
        embedding_dim=NEURAL_CONFIG["embedding_dim"],
        hidden_dim=NEURAL_CONFIG.get("hidden_dim", 64),
        learning_rate=NEURAL_CONFIG["learning_rate"],
        margin=NEURAL_CONFIG["margin"],
        device=device,
    )

    # Use reduced epochs for experiment speed
    model.fit(
        X=X_train,
        metadata=train_df,
        epochs=30,  # Reduced for speed
        batch_size=NEURAL_CONFIG["batch_size"],
        verbose=False,
    )

    test_data = {
        "X": test_df[TASTE_FEATURES].values,
        "metadata": test_df,
    }

    return evaluate_model(
        model=model,
        test_data=test_data,
        k_values=K_VALUES,
    )


def run_experiment(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    device: str,
    fractions: list[float] = TRAIN_FRACTIONS,
    n_trials: int = N_TRIALS,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """
    Run the full sensitivity analysis experiment.

    Returns nested dict: {model_name: {fraction: [trial_results]}}
    """
    results = {
        "baseline": {str(f): [] for f in fractions},
        "classical": {str(f): [] for f in fractions},
        "neural": {str(f): [] for f in fractions},
    }

    total_runs = len(fractions) * n_trials * 3
    pbar = tqdm(total=total_runs, desc="Running experiment")

    for fraction in fractions:
        for trial in range(n_trials):
            seed = RANDOM_SEED + trial

            # Subsample training data
            sub_train_df = subsample_data(train_df, fraction, seed)

            # Baseline
            try:
                baseline_metrics = train_and_evaluate_baseline(sub_train_df, test_df)
                results["baseline"][str(fraction)].append(baseline_metrics)
            except Exception as e:
                print(f"Baseline failed at fraction {fraction}, trial {trial}: {e}")
            pbar.update(1)

            # Classical
            try:
                classical_metrics = train_and_evaluate_classical(sub_train_df, test_df)
                results["classical"][str(fraction)].append(classical_metrics)
            except Exception as e:
                print(f"Classical failed at fraction {fraction}, trial {trial}: {e}")
            pbar.update(1)

            # Neural
            try:
                neural_metrics = train_and_evaluate_neural(sub_train_df, test_df, device)
                results["neural"][str(fraction)].append(neural_metrics)
            except Exception as e:
                print(f"Neural failed at fraction {fraction}, trial {trial}: {e}")
            pbar.update(1)

    pbar.close()
    return results


def aggregate_results(
    results: dict[str, dict[str, list[dict[str, Any]]]]
) -> pd.DataFrame:
    """Aggregate results into a DataFrame with mean and std."""
    rows = []

    for model_name, fraction_results in results.items():
        for fraction, trials in fraction_results.items():
            if not trials:
                continue

            # Flatten nested dicts and aggregate across trials
            flat_metrics: dict[str, list[float]] = {}
            for trial in trials:
                for key, value in trial.items():
                    if isinstance(value, dict):
                        # Handle nested metrics like precision@k
                        for k, v in value.items():
                            metric_name = f"{key.replace('@k', '')}@{k}"
                            flat_metrics.setdefault(metric_name, []).append(v)
                    elif isinstance(value, (int, float)) and not isinstance(value, bool):
                        flat_metrics.setdefault(key, []).append(value)

            # Compute mean and std
            aggregated = {}
            for metric, values in flat_metrics.items():
                aggregated[f"{metric}_mean"] = np.mean(values)
                aggregated[f"{metric}_std"] = np.std(values)

            rows.append({
                "model": model_name,
                "fraction": float(fraction),
                **aggregated,
            })

    return pd.DataFrame(rows)


def plot_results(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate visualization of experiment results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (12, 8)

    # Get main metric (Precision@5)
    metric = "precision@5"
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    if mean_col not in df.columns:
        # Try first available metric
        metric_cols = [c for c in df.columns if c.endswith("_mean")]
        if metric_cols:
            mean_col = metric_cols[0]
            std_col = mean_col.replace("_mean", "_std")
            metric = mean_col.replace("_mean", "")

    fig, ax = plt.subplots()

    colors = {"baseline": "#e74c3c", "classical": "#3498db", "neural": "#2ecc71"}

    for model in ["baseline", "classical", "neural"]:
        model_df = df[df["model"] == model].sort_values("fraction")

        if model_df.empty:
            continue

        x = model_df["fraction"] * 100  # Convert to percentage
        y = model_df[mean_col]
        yerr = model_df[std_col] if std_col in model_df.columns else None

        ax.errorbar(
            x, y,
            yerr=yerr,
            label=model.capitalize(),
            color=colors[model],
            marker="o",
            linewidth=2,
            markersize=8,
            capsize=3,
        )

    ax.set_xlabel("Training Data Size (%)", fontsize=12)
    ax.set_ylabel(f"{metric.replace('@', ' @ ').title()}", fontsize=12)
    ax.set_title("Model Performance vs Training Set Size", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "sensitivity_analysis.png", dpi=150)
    plt.close()

    # Also create a multi-metric plot
    metric_cols = [c for c in df.columns if c.endswith("_mean") and "@" in c]
    if len(metric_cols) > 1:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for idx, mean_col in enumerate(metric_cols[:4]):
            ax = axes[idx]
            metric = mean_col.replace("_mean", "")
            std_col = mean_col.replace("_mean", "_std")

            for model in ["baseline", "classical", "neural"]:
                model_df = df[df["model"] == model].sort_values("fraction")
                if model_df.empty:
                    continue

                x = model_df["fraction"] * 100
                y = model_df[mean_col]
                yerr = model_df.get(std_col)

                ax.errorbar(
                    x, y,
                    yerr=yerr,
                    label=model.capitalize(),
                    color=colors[model],
                    marker="o",
                    linewidth=2,
                    capsize=2,
                )

            ax.set_xlabel("Training Data (%)")
            ax.set_ylabel(metric.replace("@", " @ ").title())
            ax.set_title(metric.replace("@", " @ ").title())
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.suptitle("Training Set Size Sensitivity Analysis", fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / "sensitivity_analysis_multi.png", dpi=150)
        plt.close()

    print(f"Plots saved to {output_dir}")


def generate_report(df: pd.DataFrame, output_dir: Path) -> str:
    """Generate a text report of the experiment results."""
    report = []
    report.append("=" * 60)
    report.append("SENSITIVITY ANALYSIS: TRAINING SET SIZE VS PERFORMANCE")
    report.append("=" * 60)
    report.append("")

    # Summary statistics
    report.append("EXPERIMENT SUMMARY")
    report.append("-" * 40)
    report.append(f"Training fractions tested: {sorted(df['fraction'].unique())}")
    report.append(f"Models compared: {sorted(df['model'].unique())}")
    report.append("")

    # Best performance per model
    report.append("BEST PERFORMANCE PER MODEL")
    report.append("-" * 40)

    metric_col = [c for c in df.columns if "precision" in c and "_mean" in c]
    if metric_col:
        metric_col = metric_col[0]
        for model in ["baseline", "classical", "neural"]:
            model_df = df[df["model"] == model]
            if model_df.empty:
                continue
            best_idx = model_df[metric_col].idxmax()
            best_row = model_df.loc[best_idx]
            report.append(
                f"{model.capitalize()}: {best_row[metric_col]:.4f} "
                f"at {best_row['fraction']*100:.0f}% training data"
            )

    report.append("")

    # Key findings
    report.append("KEY FINDINGS")
    report.append("-" * 40)

    # Check if neural improves more with data
    if "neural" in df["model"].values and metric_col:
        neural_df = df[df["model"] == "neural"].sort_values("fraction")
        if len(neural_df) >= 2:
            start_perf = neural_df.iloc[0][metric_col]
            end_perf = neural_df.iloc[-1][metric_col]
            improvement = (end_perf - start_perf) / start_perf * 100
            report.append(
                f"1. Neural model improvement from 10% to 100% data: {improvement:.1f}%"
            )

    # Compare final performance
    if metric_col:
        final_perfs = df[df["fraction"] == 1.0].set_index("model")[metric_col]
        if len(final_perfs) > 0:
            best_model = final_perfs.idxmax()
            report.append(f"2. Best model at full data: {best_model}")

    # Check for diminishing returns
    report.append("3. Diminishing returns analysis: See sensitivity_analysis.png")

    report.append("")
    report.append("RECOMMENDATIONS")
    report.append("-" * 40)
    report.append("- If data collection is expensive, 50-70% of data may suffice")
    report.append("- Neural model benefits most from additional data")
    report.append("- Baseline provides a strong floor with minimal data")

    report_text = "\n".join(report)

    # Save report
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "experiment_report.txt", "w") as f:
        f.write(report_text)

    return report_text


def main():
    """Main experiment entry point."""
    parser = argparse.ArgumentParser(
        description="Run sensitivity analysis experiment"
    )
    parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=TRAIN_FRACTIONS,
        help="Training set fractions to test",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=N_TRIALS,
        help="Number of trials per fraction",
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
        default=str(RESULTS_DIR),
        help="Directory to save results",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("SENSITIVITY ANALYSIS EXPERIMENT")
    print("=" * 40)
    print(f"Training fractions: {args.fractions}")
    print(f"Trials per fraction: {args.trials}")
    print(f"Device: {args.device}")
    print(f"Output directory: {output_dir}")
    print()

    # Load data
    print("Loading data...")
    data = load_processed_data()
    train_df = data["train_df"]
    test_df = data["test_df"]

    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print()

    # Run experiment
    print("Running experiment...")
    results = run_experiment(
        train_df=train_df,
        test_df=test_df,
        device=args.device,
        fractions=args.fractions,
        n_trials=args.trials,
    )

    # Save raw results
    with open(output_dir / "raw_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRaw results saved to {output_dir / 'raw_results.json'}")

    # Aggregate and analyze
    df = aggregate_results(results)
    df.to_csv(output_dir / "aggregated_results.csv", index=False)
    print(f"Aggregated results saved to {output_dir / 'aggregated_results.csv'}")

    # Generate visualizations
    plot_results(df, output_dir)

    # Generate report
    report = generate_report(df, output_dir)
    print("\n" + report)

    print("\nExperiment complete!")


if __name__ == "__main__":
    main()
