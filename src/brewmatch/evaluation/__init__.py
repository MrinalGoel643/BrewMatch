"""Evaluation metrics and analysis modules."""

from .metrics import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    graded_ndcg_at_k,
    mean_reciprocal_rank,
    catalog_coverage,
    mean_squared_error,
    mean_absolute_error,
    evaluate_model,
)
from .error_analysis import (
    analyze_errors,
    identify_error_patterns,
    generate_error_report,
    PredictionError,
    ErrorPattern,
    ErrorReport,
)

__all__ = [
    "precision_at_k",
    "recall_at_k",
    "ndcg_at_k",
    "graded_ndcg_at_k",
    "mean_reciprocal_rank",
    "catalog_coverage",
    "mean_squared_error",
    "mean_absolute_error",
    "evaluate_model",
    "analyze_errors",
    "identify_error_patterns",
    "generate_error_report",
    "PredictionError",
    "ErrorPattern",
    "ErrorReport",
]
