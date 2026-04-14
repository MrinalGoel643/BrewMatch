"""
Statistical Experiment: Model Comparison with Cold-Start Analysis

This experiment provides rigorous statistical comparison of recommendation models:
1. Paired statistical tests (Wilcoxon signed-rank) between models
2. Bootstrap confidence intervals for all metrics
3. Effect size estimation (rank-biserial correlation)
4. Cold-start analysis: performance on rare vs common origins

Hypothesis: Classical nearest-neighbor models may struggle with cold-start (rare origins)
while neural models might generalize better through learned embeddings.
"""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from tqdm import tqdm

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
from brewmatch.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    mean_reciprocal_rank,
    _find_relevant_items_in_catalog,
)


RESULTS_DIR = PROJECT_ROOT / "experiments"
COLD_START_THRESHOLD = 10  # Origins with <= this many training samples are "rare"


@dataclass
class QueryResult:
    """Per-query evaluation result for statistical analysis."""
    query_idx: int
    origin: str
    origin_frequency: int  # How many training samples from this origin
    is_cold_start: bool
    precision_at_5: float
    ndcg_at_5: float
    mrr: float
    first_relevant_rank: int | None  # Rank of first relevant item (1-indexed), None if not found


@dataclass
class ModelResults:
    """Collection of per-query results for a model."""
    model_name: str
    query_results: list[QueryResult] = field(default_factory=list)

    def get_metric_array(self, metric: str) -> np.ndarray:
        """Extract a metric as numpy array aligned by query_idx."""
        return np.array([getattr(r, metric) for r in self.query_results])

    def get_cold_start_mask(self) -> np.ndarray:
        """Boolean mask for cold-start queries."""
        return np.array([r.is_cold_start for r in self.query_results])


def compute_origin_frequencies(train_df: pd.DataFrame) -> dict[str, int]:
    """Count training samples per origin."""
    return train_df["Country of Origin"].value_counts().to_dict()


def evaluate_per_query(
    model,
    test_df: pd.DataFrame,
    catalog_X: np.ndarray,
    catalog_metadata: list[dict],
    origin_frequencies: dict[str, int],
    model_name: str,
    k: int = 5,
) -> ModelResults:
    """Evaluate model and return per-query results for statistical analysis."""
    results = ModelResults(model_name=model_name)
    test_X = test_df[TASTE_FEATURES].values.astype(np.float32)

    for query_idx in range(len(test_df)):
        query_profile = test_X[query_idx]
        query_meta = test_df.iloc[query_idx].to_dict()
        origin = query_meta.get("Country of Origin", "Unknown")
        origin_freq = origin_frequencies.get(origin, 0)

        # Find relevant items
        relevant = _find_relevant_items_in_catalog(
            query_metadata=query_meta,
            query_profile=query_profile,
            catalog_metadata=catalog_metadata,
            catalog_profiles=catalog_X,
        )

        if not relevant:
            continue

        # Get recommendations
        recommendations = model.recommend(query_profile, k=k)
        recommended_indices = [rec["index"] for rec in recommendations]

        # Calculate metrics
        p_at_k = precision_at_k(recommended_indices, relevant, k)
        n_at_k = ndcg_at_k(recommended_indices, relevant, k)
        mrr = mean_reciprocal_rank(recommended_indices, relevant, k)

        # Find rank of first relevant item
        first_relevant_rank = None
        for rank, idx in enumerate(recommended_indices, 1):
            if idx in relevant:
                first_relevant_rank = rank
                break

        results.query_results.append(QueryResult(
            query_idx=query_idx,
            origin=origin,
            origin_frequency=origin_freq,
            is_cold_start=origin_freq <= COLD_START_THRESHOLD,
            precision_at_5=p_at_k,
            ndcg_at_5=n_at_k,
            mrr=mrr,
            first_relevant_rank=first_relevant_rank,
        ))

    return results


def paired_wilcoxon_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    alternative: str = "two-sided",
) -> dict[str, float]:
    """
    Perform Wilcoxon signed-rank test for paired samples.

    Returns test statistic, p-value, and rank-biserial correlation (effect size).
    """
    # Remove ties (where scores are equal)
    diff = scores_a - scores_b
    nonzero_mask = diff != 0

    if nonzero_mask.sum() < 10:
        return {
            "statistic": np.nan,
            "p_value": np.nan,
            "effect_size": np.nan,
            "n_pairs": int(nonzero_mask.sum()),
            "interpretation": "insufficient_data",
        }

    try:
        stat, p_value = stats.wilcoxon(
            scores_a[nonzero_mask],
            scores_b[nonzero_mask],
            alternative=alternative,
        )
    except ValueError:
        return {
            "statistic": np.nan,
            "p_value": np.nan,
            "effect_size": np.nan,
            "n_pairs": int(nonzero_mask.sum()),
            "interpretation": "test_failed",
        }

    # Rank-biserial correlation as effect size
    # r = 1 - (2W) / (n(n+1)/2) where W is the smaller of W+ and W-
    n = nonzero_mask.sum()
    r = 1 - (2 * stat) / (n * (n + 1) / 2)

    # Interpretation of effect size
    if abs(r) < 0.1:
        interpretation = "negligible"
    elif abs(r) < 0.3:
        interpretation = "small"
    elif abs(r) < 0.5:
        interpretation = "medium"
    else:
        interpretation = "large"

    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "effect_size": float(r),
        "n_pairs": int(n),
        "interpretation": interpretation,
    }


def bootstrap_ci(
    scores: np.ndarray,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Compute bootstrap confidence interval for the mean."""
    if len(scores) == 0:
        return {"mean": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "std": np.nan}

    rng = np.random.default_rng(42)
    bootstrap_means = np.array([
        np.mean(rng.choice(scores, size=len(scores), replace=True))
        for _ in range(n_bootstrap)
    ])

    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

    return {
        "mean": float(np.mean(scores)),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "std": float(np.std(scores)),
        "n": len(scores),
    }


def compute_statistical_comparison(
    results_a: ModelResults,
    results_b: ModelResults,
    metric: str = "precision_at_5",
) -> dict[str, Any]:
    """
    Comprehensive statistical comparison between two models.

    Includes overall comparison and stratified by cold-start status.
    """
    scores_a = results_a.get_metric_array(metric)
    scores_b = results_b.get_metric_array(metric)
    cold_start_mask = results_a.get_cold_start_mask()

    comparison = {
        "model_a": results_a.model_name,
        "model_b": results_b.model_name,
        "metric": metric,
        "overall": {},
        "common_origins": {},
        "rare_origins": {},
    }

    # Overall comparison
    comparison["overall"]["model_a_stats"] = bootstrap_ci(scores_a)
    comparison["overall"]["model_b_stats"] = bootstrap_ci(scores_b)
    comparison["overall"]["wilcoxon"] = paired_wilcoxon_test(scores_a, scores_b)
    comparison["overall"]["mean_diff"] = float(np.mean(scores_a) - np.mean(scores_b))

    # Common origins (warm-start)
    warm_mask = ~cold_start_mask
    if warm_mask.sum() >= 10:
        comparison["common_origins"]["model_a_stats"] = bootstrap_ci(scores_a[warm_mask])
        comparison["common_origins"]["model_b_stats"] = bootstrap_ci(scores_b[warm_mask])
        comparison["common_origins"]["wilcoxon"] = paired_wilcoxon_test(
            scores_a[warm_mask], scores_b[warm_mask]
        )
        comparison["common_origins"]["n_queries"] = int(warm_mask.sum())

    # Rare origins (cold-start)
    if cold_start_mask.sum() >= 10:
        comparison["rare_origins"]["model_a_stats"] = bootstrap_ci(scores_a[cold_start_mask])
        comparison["rare_origins"]["model_b_stats"] = bootstrap_ci(scores_b[cold_start_mask])
        comparison["rare_origins"]["wilcoxon"] = paired_wilcoxon_test(
            scores_a[cold_start_mask], scores_b[cold_start_mask]
        )
        comparison["rare_origins"]["n_queries"] = int(cold_start_mask.sum())

    return comparison


def run_cold_start_analysis(
    all_results: dict[str, ModelResults],
) -> dict[str, Any]:
    """Analyze cold-start performance across all models."""
    analysis = {
        "threshold": COLD_START_THRESHOLD,
        "models": {},
        "origin_breakdown": {},
    }

    for model_name, results in all_results.items():
        cold_mask = results.get_cold_start_mask()
        warm_mask = ~cold_mask

        p5_all = results.get_metric_array("precision_at_5")
        mrr_all = results.get_metric_array("mrr")

        model_analysis = {
            "overall": {
                "precision_at_5": bootstrap_ci(p5_all),
                "mrr": bootstrap_ci(mrr_all),
            },
            "common_origins": {
                "n_queries": int(warm_mask.sum()),
                "precision_at_5": bootstrap_ci(p5_all[warm_mask]) if warm_mask.sum() > 0 else None,
                "mrr": bootstrap_ci(mrr_all[warm_mask]) if warm_mask.sum() > 0 else None,
            },
            "rare_origins": {
                "n_queries": int(cold_mask.sum()),
                "precision_at_5": bootstrap_ci(p5_all[cold_mask]) if cold_mask.sum() > 0 else None,
                "mrr": bootstrap_ci(mrr_all[cold_mask]) if cold_mask.sum() > 0 else None,
            },
        }

        # Cold-start gap (performance drop on rare vs common)
        if warm_mask.sum() > 0 and cold_mask.sum() > 0:
            warm_p5 = np.mean(p5_all[warm_mask])
            cold_p5 = np.mean(p5_all[cold_mask])
            if warm_p5 > 0:
                model_analysis["cold_start_gap"] = {
                    "absolute": float(warm_p5 - cold_p5),
                    "relative_pct": float((warm_p5 - cold_p5) / warm_p5 * 100),
                }

        analysis["models"][model_name] = model_analysis

    # Per-origin breakdown
    origins_seen = set()
    for results in all_results.values():
        for qr in results.query_results:
            origins_seen.add((qr.origin, qr.origin_frequency, qr.is_cold_start))

    for origin, freq, is_cold in sorted(origins_seen, key=lambda x: x[1]):
        origin_data = {"frequency": freq, "is_cold_start": is_cold, "models": {}}

        for model_name, results in all_results.items():
            origin_scores = [
                qr.precision_at_5 for qr in results.query_results if qr.origin == origin
            ]
            if origin_scores:
                origin_data["models"][model_name] = {
                    "mean_p5": float(np.mean(origin_scores)),
                    "n_queries": len(origin_scores),
                }

        analysis["origin_breakdown"][origin] = origin_data

    return analysis


def plot_statistical_results(
    all_results: dict[str, ModelResults],
    cold_start_analysis: dict[str, Any],
    output_dir: Path,
) -> None:
    """Generate publication-quality visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")
    colors = {"baseline": "#e74c3c", "classical": "#3498db", "neural": "#2ecc71"}

    # 1. Overall performance with confidence intervals
    fig, ax = plt.subplots(figsize=(10, 6))

    models = list(all_results.keys())
    x_pos = np.arange(len(models))

    means = []
    ci_lowers = []
    ci_uppers = []

    for model in models:
        stats_data = cold_start_analysis["models"][model]["overall"]["precision_at_5"]
        means.append(stats_data["mean"])
        ci_lowers.append(stats_data["mean"] - stats_data["ci_lower"])
        ci_uppers.append(stats_data["ci_upper"] - stats_data["mean"])

    bars = ax.bar(x_pos, means, color=[colors[m] for m in models], alpha=0.8)
    ax.errorbar(
        x_pos, means,
        yerr=[ci_lowers, ci_uppers],
        fmt="none", color="black", capsize=5, capthick=2
    )

    ax.set_xticks(x_pos)
    ax.set_xticklabels([m.capitalize() for m in models], fontsize=12)
    ax.set_ylabel("Precision@5", fontsize=12)
    ax.set_title("Model Performance with 95% Bootstrap Confidence Intervals", fontsize=14)
    ax.set_ylim(0, 1.1)

    # Add value labels
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{mean:.3f}", ha="center", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_dir / "performance_with_ci.png", dpi=150)
    plt.close()

    # 2. Cold-start comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(models))
    width = 0.35

    common_means = []
    rare_means = []
    common_errs = []
    rare_errs = []

    for model in models:
        model_data = cold_start_analysis["models"][model]

        if model_data["common_origins"]["precision_at_5"]:
            common_stats = model_data["common_origins"]["precision_at_5"]
            common_means.append(common_stats["mean"])
            common_errs.append((common_stats["ci_upper"] - common_stats["ci_lower"]) / 2)
        else:
            common_means.append(0)
            common_errs.append(0)

        if model_data["rare_origins"]["precision_at_5"]:
            rare_stats = model_data["rare_origins"]["precision_at_5"]
            rare_means.append(rare_stats["mean"])
            rare_errs.append((rare_stats["ci_upper"] - rare_stats["ci_lower"]) / 2)
        else:
            rare_means.append(0)
            rare_errs.append(0)

    bars1 = ax.bar(x - width/2, common_means, width, label="Common Origins",
                   color="#3498db", alpha=0.8, yerr=common_errs, capsize=3)
    bars2 = ax.bar(x + width/2, rare_means, width, label="Rare Origins (Cold-Start)",
                   color="#e74c3c", alpha=0.8, yerr=rare_errs, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in models], fontsize=12)
    ax.set_ylabel("Precision@5", fontsize=12)
    ax.set_title("Cold-Start Analysis: Common vs Rare Origin Performance", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(output_dir / "cold_start_comparison.png", dpi=150)
    plt.close()

    # 3. Per-origin heatmap
    origin_data = cold_start_analysis["origin_breakdown"]
    origins = sorted(origin_data.keys(), key=lambda o: origin_data[o]["frequency"], reverse=True)

    # Take top 15 origins by frequency
    origins = origins[:15]

    heatmap_data = []
    for origin in origins:
        row = {"Origin": origin, "Frequency": origin_data[origin]["frequency"]}
        for model in models:
            if model in origin_data[origin]["models"]:
                row[model.capitalize()] = origin_data[origin]["models"][model]["mean_p5"]
            else:
                row[model.capitalize()] = np.nan
        heatmap_data.append(row)

    heatmap_df = pd.DataFrame(heatmap_data)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap matrix
    heatmap_matrix = heatmap_df[[m.capitalize() for m in models]].values

    im = ax.imshow(heatmap_matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels([m.capitalize() for m in models], fontsize=11)
    ax.set_yticks(np.arange(len(origins)))
    ax.set_yticklabels([f"{o} (n={origin_data[o]['frequency']})" for o in origins], fontsize=10)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Precision@5", fontsize=11)

    # Add text annotations
    for i in range(len(origins)):
        for j in range(len(models)):
            val = heatmap_matrix[i, j]
            if not np.isnan(val):
                text_color = "white" if val < 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color=text_color, fontsize=9)

    ax.set_title("Performance by Origin (sorted by training frequency)", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "origin_heatmap.png", dpi=150)
    plt.close()

    # 4. Cold-start gap visualization
    fig, ax = plt.subplots(figsize=(10, 6))

    gaps = []
    for model in models:
        model_data = cold_start_analysis["models"][model]
        if "cold_start_gap" in model_data:
            gaps.append(model_data["cold_start_gap"]["relative_pct"])
        else:
            gaps.append(0)

    bars = ax.bar(models, gaps, color=[colors[m] for m in models], alpha=0.8)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1)

    ax.set_ylabel("Performance Drop (%)", fontsize=12)
    ax.set_title("Cold-Start Gap: Performance Drop on Rare Origins", fontsize=14)
    ax.set_xticklabels([m.capitalize() for m in models], fontsize=12)

    for bar, gap in zip(bars, gaps):
        y_pos = bar.get_height() + 1 if bar.get_height() >= 0 else bar.get_height() - 3
        ax.text(bar.get_x() + bar.get_width()/2, y_pos, f"{gap:.1f}%",
                ha="center", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_dir / "cold_start_gap.png", dpi=150)
    plt.close()

    print(f"Plots saved to {output_dir}")


def generate_statistical_report(
    comparisons: list[dict],
    cold_start_analysis: dict[str, Any],
    output_dir: Path,
) -> str:
    """Generate comprehensive statistical report."""
    report = []
    report.append("=" * 70)
    report.append("STATISTICAL EXPERIMENT: MODEL COMPARISON WITH COLD-START ANALYSIS")
    report.append("=" * 70)
    report.append("")

    # Cold-start threshold
    report.append(f"Cold-start threshold: origins with <= {COLD_START_THRESHOLD} training samples")
    report.append("")

    # Overall performance summary
    report.append("OVERALL PERFORMANCE (Precision@5 with 95% Bootstrap CI)")
    report.append("-" * 50)

    for model_name, model_data in cold_start_analysis["models"].items():
        stats = model_data["overall"]["precision_at_5"]
        report.append(
            f"  {model_name.capitalize():12} {stats['mean']:.4f} "
            f"[{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}] (n={stats['n']})"
        )
    report.append("")

    # Pairwise comparisons
    report.append("PAIRWISE STATISTICAL COMPARISONS (Wilcoxon Signed-Rank Test)")
    report.append("-" * 50)

    for comp in comparisons:
        model_a = comp["model_a"]
        model_b = comp["model_b"]
        overall = comp["overall"]
        wilcoxon = overall["wilcoxon"]

        report.append(f"\n{model_a.capitalize()} vs {model_b.capitalize()}:")
        report.append(f"  Mean difference: {overall['mean_diff']:+.4f}")
        report.append(f"  Wilcoxon p-value: {wilcoxon['p_value']:.4f}")
        report.append(f"  Effect size (r): {wilcoxon['effect_size']:.3f} ({wilcoxon['interpretation']})")

        sig = "***" if wilcoxon["p_value"] < 0.001 else "**" if wilcoxon["p_value"] < 0.01 else "*" if wilcoxon["p_value"] < 0.05 else "ns"
        report.append(f"  Significance: {sig}")

    report.append("")

    # Cold-start analysis
    report.append("COLD-START ANALYSIS")
    report.append("-" * 50)

    for model_name, model_data in cold_start_analysis["models"].items():
        report.append(f"\n{model_name.capitalize()}:")

        common = model_data["common_origins"]
        rare = model_data["rare_origins"]

        if common["precision_at_5"]:
            report.append(f"  Common origins (n={common['n_queries']}): "
                         f"P@5 = {common['precision_at_5']['mean']:.4f}")
        if rare["precision_at_5"]:
            report.append(f"  Rare origins   (n={rare['n_queries']}): "
                         f"P@5 = {rare['precision_at_5']['mean']:.4f}")

        if "cold_start_gap" in model_data:
            gap = model_data["cold_start_gap"]
            report.append(f"  Cold-start gap: {gap['absolute']:.4f} ({gap['relative_pct']:.1f}% drop)")

    report.append("")

    # Stratified comparisons
    report.append("STRATIFIED PAIRWISE COMPARISONS")
    report.append("-" * 50)

    for comp in comparisons:
        model_a = comp["model_a"]
        model_b = comp["model_b"]

        report.append(f"\n{model_a.capitalize()} vs {model_b.capitalize()}:")

        if "wilcoxon" in comp.get("common_origins", {}):
            w = comp["common_origins"]["wilcoxon"]
            report.append(f"  Common origins: p={w['p_value']:.4f}, r={w['effect_size']:.3f} ({w['interpretation']})")

        if "wilcoxon" in comp.get("rare_origins", {}):
            w = comp["rare_origins"]["wilcoxon"]
            report.append(f"  Rare origins:   p={w['p_value']:.4f}, r={w['effect_size']:.3f} ({w['interpretation']})")

    report.append("")

    # Key findings
    report.append("KEY FINDINGS")
    report.append("-" * 50)

    # Find best overall model
    best_model = max(
        cold_start_analysis["models"].items(),
        key=lambda x: x[1]["overall"]["precision_at_5"]["mean"]
    )
    report.append(f"1. Best overall model: {best_model[0].capitalize()} "
                 f"(P@5 = {best_model[1]['overall']['precision_at_5']['mean']:.4f})")

    # Find model most robust to cold-start (smallest absolute gap)
    gaps = {
        name: abs(data.get("cold_start_gap", {}).get("relative_pct", float("inf")))
        for name, data in cold_start_analysis["models"].items()
    }
    most_robust = min(gaps.items(), key=lambda x: x[1])
    actual_gap = cold_start_analysis["models"][most_robust[0]].get("cold_start_gap", {}).get("relative_pct", 0)
    report.append(f"2. Most consistent across origins: {most_robust[0].capitalize()} "
                 f"({actual_gap:+.1f}% change on rare origins)")

    # Find best absolute performance on cold-start
    cold_start_perf = {
        name: data.get("rare_origins", {}).get("precision_at_5", {}).get("mean", 0)
        for name, data in cold_start_analysis["models"].items()
    }
    best_cold = max(cold_start_perf.items(), key=lambda x: x[1])
    report.append(f"3. Best on cold-start queries: {best_cold[0].capitalize()} "
                 f"(P@5 = {best_cold[1]:.4f})")

    # Significant differences
    sig_comps = [c for c in comparisons if c["overall"]["wilcoxon"]["p_value"] < 0.05]
    if sig_comps:
        report.append(f"4. Statistically significant differences found in {len(sig_comps)}/{len(comparisons)} comparisons")
    else:
        report.append("4. No statistically significant differences found (p < 0.05)")

    report.append("")
    report.append("INTERPRETATION GUIDE")
    report.append("-" * 50)
    report.append("  Effect size (r): < 0.1 negligible, 0.1-0.3 small, 0.3-0.5 medium, > 0.5 large")
    report.append("  Significance: * p<0.05, ** p<0.01, *** p<0.001, ns = not significant")
    report.append("  CI: 95% bootstrap confidence interval (10,000 resamples)")

    report_text = "\n".join(report)

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "statistical_report.txt", "w") as f:
        f.write(report_text)

    return report_text


def main():
    """Run the statistical experiment."""
    global COLD_START_THRESHOLD

    parser = argparse.ArgumentParser(
        description="Statistical model comparison with cold-start analysis"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(RESULTS_DIR),
        help="Directory to save results",
    )
    parser.add_argument(
        "--cold-start-threshold",
        type=int,
        default=COLD_START_THRESHOLD,
        help="Max training samples to consider an origin 'rare'",
    )
    args = parser.parse_args()

    COLD_START_THRESHOLD = args.cold_start_threshold

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("STATISTICAL EXPERIMENT: MODEL COMPARISON WITH COLD-START ANALYSIS")
    print("=" * 70)
    print(f"Cold-start threshold: {COLD_START_THRESHOLD} training samples")
    print(f"Output directory: {output_dir}")
    print()

    # Load data
    print("Loading data...")
    data = load_processed_data()
    train_df = data["train_df"]
    test_df = data["test_df"]

    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")

    origin_frequencies = compute_origin_frequencies(train_df)
    n_rare = sum(1 for f in origin_frequencies.values() if f <= COLD_START_THRESHOLD)
    print(f"Origins: {len(origin_frequencies)} total, {n_rare} rare (<= {COLD_START_THRESHOLD} samples)")
    print()

    # Load models
    print("Loading models...")
    models = {}

    baseline_path = CHECKPOINTS_DIR / "baseline.pkl"
    if baseline_path.exists():
        models["baseline"] = NaiveBaselineRecommender.load(baseline_path)
        print(f"  Loaded baseline from {baseline_path}")

    classical_path = CHECKPOINTS_DIR / "classical.pkl"
    if classical_path.exists():
        models["classical"] = ClassicalMLRecommender.load(classical_path)
        print(f"  Loaded classical from {classical_path}")

    neural_path = CHECKPOINTS_DIR / "neural.pt"
    if neural_path.exists():
        models["neural"] = NeuralRecommender.load(neural_path)
        print(f"  Loaded neural from {neural_path}")

    if len(models) < 2:
        print("ERROR: Need at least 2 models for comparison. Run `uv run train` first.")
        return

    print()

    # Prepare catalog data
    catalog_X = train_df[TASTE_FEATURES].values.astype(np.float32)
    catalog_metadata = train_df.to_dict("records")

    # Evaluate each model per-query
    print("Evaluating models per-query...")
    all_results: dict[str, ModelResults] = {}

    for model_name, model in tqdm(models.items(), desc="Evaluating"):
        results = evaluate_per_query(
            model=model,
            test_df=test_df,
            catalog_X=catalog_X,
            catalog_metadata=catalog_metadata,
            origin_frequencies=origin_frequencies,
            model_name=model_name,
        )
        all_results[model_name] = results

    print()

    # Statistical comparisons
    print("Running statistical comparisons...")
    model_names = list(all_results.keys())
    comparisons = []

    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            comp = compute_statistical_comparison(
                all_results[model_names[i]],
                all_results[model_names[j]],
                metric="precision_at_5",
            )
            comparisons.append(comp)

    # Cold-start analysis
    print("Running cold-start analysis...")
    cold_start_analysis = run_cold_start_analysis(all_results)

    # Save raw results
    raw_results = {
        "cold_start_threshold": COLD_START_THRESHOLD,
        "comparisons": comparisons,
        "cold_start_analysis": cold_start_analysis,
    }

    with open(output_dir / "statistical_results.json", "w") as f:
        json.dump(raw_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_dir / 'statistical_results.json'}")

    # Generate visualizations
    print("Generating visualizations...")
    plot_statistical_results(all_results, cold_start_analysis, output_dir)

    # Generate report
    report = generate_statistical_report(comparisons, cold_start_analysis, output_dir)
    print("\n" + report)

    print(f"\nExperiment complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
