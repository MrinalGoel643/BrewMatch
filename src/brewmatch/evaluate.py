"""Evaluation script for BrewMatch models."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from brewmatch.config import (
    CHECKPOINTS_DIR,
    K_VALUES,
    TASTE_FEATURES,
)
from brewmatch.data import load_processed_data
from brewmatch.models import (
    NaiveBaselineRecommender,
    ClassicalMLRecommender,
    NeuralRecommender,
)
from brewmatch.evaluation import evaluate_model, generate_error_report


def load_models() -> dict[str, Any]:
    """Load all trained models."""
    models = {}

    baseline_path = CHECKPOINTS_DIR / "baseline.pkl"
    if baseline_path.exists():
        models["baseline"] = NaiveBaselineRecommender.load(baseline_path)
        print(f"Loaded baseline model from {baseline_path}")

    classical_path = CHECKPOINTS_DIR / "classical.pkl"
    if classical_path.exists():
        models["classical"] = ClassicalMLRecommender.load(classical_path)
        print(f"Loaded classical model from {classical_path}")

    neural_path = CHECKPOINTS_DIR / "neural.pt"
    if neural_path.exists():
        models["neural"] = NeuralRecommender.load(neural_path)
        print(f"Loaded neural model from {neural_path}")

    return models


def compare_models(results: dict[str, dict[str, Any]]) -> None:
    """Print comparison table of all models."""
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    # Flatten nested dicts (precision@k, recall@k, etc.)
    flat_results = {}
    for model_name, metrics in results.items():
        flat_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    flat_metrics[f"{key.replace('@k', '')}@{k}"] = v
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                flat_metrics[key] = value
        flat_results[model_name] = flat_metrics

    if not flat_results:
        print("No results to compare.")
        return

    # Get all metric keys
    all_keys = set()
    for metrics in flat_results.values():
        all_keys.update(metrics.keys())
    all_keys = sorted(all_keys)

    # Print table
    header = f"{'Model':<12}" + "".join(f"{k:>12}" for k in all_keys)
    print(header)
    print("-" * len(header))

    for model_name, metrics in flat_results.items():
        row = f"{model_name:<12}"
        for key in all_keys:
            val = metrics.get(key, float("nan"))
            if isinstance(val, float):
                row += f"{val:>12.4f}"
            else:
                row += f"{val:>12}"
        print(row)

    # Find best model for primary metrics
    print("\nBest model per metric:")
    for key in ["precision@5", "ndcg@5", "recall@5"]:
        if key in all_keys:
            best_model = max(
                flat_results.keys(),
                key=lambda m: flat_results[m].get(key, 0)
            )
            best_value = flat_results[best_model].get(key, 0)
            print(f"  - {key}: {best_model} ({best_value:.4f})")


def main():
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(description="Evaluate BrewMatch models")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["baseline", "classical", "neural", "all"],
        default=["all"],
        help="Which models to evaluate",
    )
    parser.add_argument(
        "--error-analysis",
        action="store_true",
        help="Generate detailed error analysis",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save results to JSON file",
    )
    args = parser.parse_args()

    # Load data
    print("Loading data...")
    data = load_processed_data()
    train_df = data["train_df"]
    test_df = data["test_df"]
    print(f"Train samples (catalog): {len(train_df)}")
    print(f"Test samples (queries): {len(test_df)}")
    print()

    # Load models
    print("Loading models...")
    all_models = load_models()

    if "all" in args.models:
        models_to_eval = all_models
    else:
        models_to_eval = {k: v for k, v in all_models.items() if k in args.models}

    if not models_to_eval:
        print("No models found to evaluate!")
        return

    print(f"\nEvaluating: {list(models_to_eval.keys())}")
    print()

    # Prepare data dicts for evaluation
    # Queries come from test set, but relevance is computed against training set (model's catalog)
    test_data = {
        "X": test_df[TASTE_FEATURES].values,
        "metadata": test_df,
    }
    catalog_data = {
        "X": train_df[TASTE_FEATURES].values,
        "metadata": train_df,
    }

    # Evaluate each model
    results = {}
    for name, model in models_to_eval.items():
        print(f"\n{'=' * 40}")
        print(f"Evaluating: {name.upper()}")
        print("=" * 40)

        metrics = evaluate_model(
            model=model,
            test_data=test_data,
            catalog_data=catalog_data,
            k_values=K_VALUES,
        )

        results[name] = metrics

        print(f"\nResults for {name}:")
        for metric, value in metrics.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    print(f"  {metric}@{k}: {v:.4f}")
            elif isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")

        # Error analysis
        if args.error_analysis:
            print(f"\nError Analysis for {name}:")
            report = generate_error_report(
                model=model,
                test_data=test_data,
                catalog_data=catalog_data,
            )
            print(f"  Error rate: {report.error_rate:.1%}")
            print(f"  Total errors: {report.total_errors}/{report.total_queries}")
            print("\n  Worst errors:")
            for i, err in enumerate(report.worst_errors[:5], 1):
                print(f"    {i}. Query {err.query_idx}: magnitude={err.error_magnitude:.3f}")
                if "_root_cause" in err.query_metadata:
                    print(f"       Root cause: {err.query_metadata['_root_cause']}")
            print("\n  Patterns:")
            for pattern in report.patterns[:3]:
                print(f"    - {pattern.description} (freq: {pattern.frequency})")
            print("\n  Mitigations:")
            for mitigation in report.mitigations[:3]:
                print(f"    - {mitigation[:80]}...")

    # Compare models
    if len(results) > 1:
        compare_models(results)

    # Save results
    if args.output:
        output_path = Path(args.output)
        # Convert results to JSON-serializable format
        json_results = {}
        for model_name, metrics in results.items():
            json_results[model_name] = {}
            for key, value in metrics.items():
                if isinstance(value, dict):
                    json_results[model_name][key] = {str(k): v for k, v in value.items()}
                else:
                    json_results[model_name][key] = value
        with open(output_path, "w") as f:
            json.dump(json_results, f, indent=2)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
