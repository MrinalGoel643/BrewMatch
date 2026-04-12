"""Error analysis module for the coffee recommendation system.

This module provides tools for analyzing model errors:
- Finding worst predictions
- Identifying error patterns by origin, processing method, etc.
- Generating comprehensive error reports with mitigation strategies
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Recommender(Protocol):
    """Protocol for recommender models used in error analysis."""

    def recommend(self, preferences: np.ndarray, k: int = 5) -> list[dict[str, Any]]:
        """Recommend coffees matching user taste preferences."""
        ...


@dataclass
class PredictionError:
    """Represents a single misprediction for analysis.

    Attributes:
        query_idx: Index of the query coffee in the test set.
        query_preferences: The query taste profile used.
        query_metadata: Metadata of the query coffee.
        recommended_idx: Index of the top recommended coffee.
        recommended_metadata: Metadata of the recommended coffee.
        recommended_profile: Taste profile of the recommended coffee.
        expected_indices: Set of indices that would have been correct.
        error_magnitude: Quantified error (e.g., Euclidean distance or rank loss).
        rank_of_first_relevant: Position of first relevant item in recommendations.
    """

    query_idx: int
    query_preferences: np.ndarray
    query_metadata: dict[str, Any]
    recommended_idx: int
    recommended_metadata: dict[str, Any]
    recommended_profile: np.ndarray
    expected_indices: set[int]
    error_magnitude: float
    rank_of_first_relevant: int | None = None


@dataclass
class ErrorPattern:
    """Represents a common error pattern identified in the analysis.

    Attributes:
        pattern_type: Category of the pattern (e.g., 'origin', 'processing', 'profile').
        description: Human-readable description of the pattern.
        frequency: Number of errors exhibiting this pattern.
        affected_queries: List of query indices affected by this pattern.
        severity: Average error magnitude for this pattern.
    """

    pattern_type: str
    description: str
    frequency: int
    affected_queries: list[int] = field(default_factory=list)
    severity: float = 0.0


@dataclass
class ErrorReport:
    """Comprehensive error analysis report.

    Attributes:
        total_queries: Total number of queries evaluated.
        total_errors: Number of queries where top recommendation was incorrect.
        error_rate: Proportion of queries with errors.
        worst_errors: List of the worst prediction errors.
        patterns: Identified error patterns.
        mitigations: Suggested mitigation strategies.
    """

    total_queries: int
    total_errors: int
    error_rate: float
    worst_errors: list[PredictionError]
    patterns: list[ErrorPattern]
    mitigations: list[str]


def _compute_euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Euclidean distance between two arrays."""
    return float(np.sqrt(np.sum((a - b) ** 2)))


def _compute_similarity(profile_a: np.ndarray, profile_b: np.ndarray) -> float:
    """Compute cosine similarity between two taste profiles."""
    norm_a = np.linalg.norm(profile_a)
    norm_b = np.linalg.norm(profile_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(profile_a, profile_b) / (norm_a * norm_b))


def _find_relevant_items(
    query_metadata: dict[str, Any],
    query_profile: np.ndarray,
    all_metadata: list[dict[str, Any]],
    all_profiles: np.ndarray,
    query_idx: int,
    similarity_threshold: float = 0.95,
) -> set[int]:
    """Identify relevant items for a query coffee.

    An item is considered relevant if:
    - It shares the same country AND processing method, OR
    - It has high taste profile similarity (>= threshold)
    """
    relevant = set()
    query_country = query_metadata.get("Country of Origin", "")
    query_processing = query_metadata.get("Processing Method", "")

    for i, meta in enumerate(all_metadata):
        if i == query_idx:
            continue

        same_country = meta.get("Country of Origin", "") == query_country
        same_processing = meta.get("Processing Method", "") == query_processing

        if same_country and same_processing and query_country and query_processing:
            relevant.add(i)
            continue

        similarity = _compute_similarity(query_profile, all_profiles[i])
        if similarity >= similarity_threshold:
            relevant.add(i)

    return relevant


def analyze_errors(
    model: Recommender,
    test_data: dict[str, Any],
    n_errors: int = 5,
) -> list[PredictionError]:
    """Find the worst predictions made by the model.

    Analyzes each test coffee as a query and identifies cases where the
    model's top recommendations were most incorrect.

    Args:
        model: A fitted recommender model with a recommend() method.
        test_data: Dictionary containing:
            - 'X': Feature matrix of shape (n_samples, 9) with taste profiles.
            - 'metadata': List of metadata dicts or DataFrame.
        n_errors: Number of worst errors to return.

    Returns:
        List of PredictionError objects for the n_errors worst predictions,
        sorted by error magnitude (descending).

    Example:
        >>> errors = analyze_errors(model, test_data, n_errors=5)
        >>> for err in errors:
        ...     print(f"Query {err.query_idx}: magnitude={err.error_magnitude:.3f}")
    """
    X = np.asarray(test_data["X"], dtype=np.float32)
    metadata_raw = test_data["metadata"]

    if hasattr(metadata_raw, "to_dict"):
        all_metadata = metadata_raw.to_dict("records")
    else:
        all_metadata = list(metadata_raw)

    n_samples = len(X)
    errors: list[PredictionError] = []
    taste_features = [
        "Aroma", "Flavor", "Aftertaste", "Acidity", "Body",
        "Balance", "Uniformity", "Clean Cup", "Sweetness"
    ]

    for query_idx in range(n_samples):
        query_profile = X[query_idx]
        query_metadata = all_metadata[query_idx]

        relevant = _find_relevant_items(
            query_metadata=query_metadata,
            query_profile=query_profile,
            all_metadata=all_metadata,
            all_profiles=X,
            query_idx=query_idx,
        )

        if not relevant:
            continue

        recommendations = model.recommend(query_profile, k=max(10, n_errors))
        recommended_indices = [rec["index"] for rec in recommendations]

        if not recommendations:
            continue

        top_rec = recommendations[0]
        top_idx = top_rec["index"]

        # Check if top recommendation is relevant
        if top_idx in relevant:
            continue

        # This is an error - compute error magnitude
        rec_profile = np.array(
            [top_rec["taste_profile"][f] for f in taste_features],
            dtype=np.float32
        )
        error_magnitude = _compute_euclidean_distance(query_profile, rec_profile)

        # Find rank of first relevant item
        rank_of_first_relevant = None
        for rank, idx in enumerate(recommended_indices):
            if idx in relevant:
                rank_of_first_relevant = rank + 1  # 1-indexed
                break

        error = PredictionError(
            query_idx=query_idx,
            query_preferences=query_profile.copy(),
            query_metadata=query_metadata,
            recommended_idx=top_idx,
            recommended_metadata=all_metadata[top_idx],
            recommended_profile=rec_profile,
            expected_indices=relevant,
            error_magnitude=error_magnitude,
            rank_of_first_relevant=rank_of_first_relevant,
        )
        errors.append(error)

    # Sort by error magnitude (descending) and return top n
    errors.sort(key=lambda e: e.error_magnitude, reverse=True)
    return errors[:n_errors]


def identify_error_patterns(errors: list[PredictionError]) -> list[ErrorPattern]:
    """Analyze a list of errors to identify common failure patterns.

    Looks for patterns such as:
    - Failures on specific origins (countries)
    - Failures on specific processing methods
    - Failures on certain taste profile characteristics

    Args:
        errors: List of PredictionError objects to analyze.

    Returns:
        List of ErrorPattern objects describing common failure modes,
        sorted by frequency (descending).

    Example:
        >>> errors = analyze_errors(model, test_data, n_errors=20)
        >>> patterns = identify_error_patterns(errors)
        >>> for p in patterns:
        ...     print(f"{p.pattern_type}: {p.description} ({p.frequency} occurrences)")
    """
    if not errors:
        return []

    patterns: list[ErrorPattern] = []

    # Pattern 1: Failures by query origin
    origin_counter: Counter[str] = Counter()
    origin_errors: dict[str, list[int]] = {}
    origin_severity: dict[str, list[float]] = {}

    for err in errors:
        origin = err.query_metadata.get("Country of Origin", "Unknown")
        origin_counter[origin] += 1
        origin_errors.setdefault(origin, []).append(err.query_idx)
        origin_severity.setdefault(origin, []).append(err.error_magnitude)

    for origin, count in origin_counter.most_common():
        if count >= 2:  # Only report patterns with multiple occurrences
            avg_severity = np.mean(origin_severity[origin])
            patterns.append(ErrorPattern(
                pattern_type="origin",
                description=f"Model fails frequently on coffees from {origin}",
                frequency=count,
                affected_queries=origin_errors[origin],
                severity=float(avg_severity),
            ))

    # Pattern 2: Failures by query processing method
    processing_counter: Counter[str] = Counter()
    processing_errors: dict[str, list[int]] = {}
    processing_severity: dict[str, list[float]] = {}

    for err in errors:
        method = err.query_metadata.get("Processing Method", "Unknown")
        processing_counter[method] += 1
        processing_errors.setdefault(method, []).append(err.query_idx)
        processing_severity.setdefault(method, []).append(err.error_magnitude)

    for method, count in processing_counter.most_common():
        if count >= 2:
            avg_severity = np.mean(processing_severity[method])
            patterns.append(ErrorPattern(
                pattern_type="processing",
                description=f"Model fails frequently on {method} processed coffees",
                frequency=count,
                affected_queries=processing_errors[method],
                severity=float(avg_severity),
            ))

    # Pattern 3: Cross-origin confusion
    confusion_counter: Counter[tuple[str, str]] = Counter()
    confusion_errors: dict[tuple[str, str], list[int]] = {}
    confusion_severity: dict[tuple[str, str], list[float]] = {}

    for err in errors:
        query_origin = err.query_metadata.get("Country of Origin", "Unknown")
        rec_origin = err.recommended_metadata.get("Country of Origin", "Unknown")
        if query_origin != rec_origin:
            key = (query_origin, rec_origin)
            confusion_counter[key] += 1
            confusion_errors.setdefault(key, []).append(err.query_idx)
            confusion_severity.setdefault(key, []).append(err.error_magnitude)

    for (q_origin, r_origin), count in confusion_counter.most_common(5):
        if count >= 2:
            avg_severity = np.mean(confusion_severity[(q_origin, r_origin)])
            patterns.append(ErrorPattern(
                pattern_type="cross_origin_confusion",
                description=f"Model confuses {q_origin} with {r_origin}",
                frequency=count,
                affected_queries=confusion_errors[(q_origin, r_origin)],
                severity=float(avg_severity),
            ))

    # Pattern 4: High acidity/low body confusion (taste profile patterns)
    high_acidity_errors = []
    high_acidity_indices = []
    high_acidity_severities = []

    low_body_errors = []
    low_body_indices = []
    low_body_severities = []

    for err in errors:
        # Check for high acidity queries (above 7.5 on typical 6-10 scale)
        acidity_idx = 3  # Index of Acidity in taste features
        if err.query_preferences[acidity_idx] > 7.5:
            high_acidity_errors.append(err)
            high_acidity_indices.append(err.query_idx)
            high_acidity_severities.append(err.error_magnitude)

        # Check for low body queries (below 7.0)
        body_idx = 4  # Index of Body in taste features
        if err.query_preferences[body_idx] < 7.0:
            low_body_errors.append(err)
            low_body_indices.append(err.query_idx)
            low_body_severities.append(err.error_magnitude)

    if len(high_acidity_errors) >= 2:
        patterns.append(ErrorPattern(
            pattern_type="taste_profile",
            description="Model struggles with high-acidity coffee recommendations",
            frequency=len(high_acidity_errors),
            affected_queries=high_acidity_indices,
            severity=float(np.mean(high_acidity_severities)),
        ))

    if len(low_body_errors) >= 2:
        patterns.append(ErrorPattern(
            pattern_type="taste_profile",
            description="Model struggles with low-body coffee recommendations",
            frequency=len(low_body_errors),
            affected_queries=low_body_indices,
            severity=float(np.mean(low_body_severities)),
        ))

    # Pattern 5: Rank degradation (first relevant item is far down)
    severe_rank_errors = []
    severe_rank_indices = []
    severe_rank_severities = []

    for err in errors:
        if err.rank_of_first_relevant is not None and err.rank_of_first_relevant > 5:
            severe_rank_errors.append(err)
            severe_rank_indices.append(err.query_idx)
            severe_rank_severities.append(err.error_magnitude)

    if len(severe_rank_errors) >= 2:
        avg_rank = np.mean([e.rank_of_first_relevant for e in severe_rank_errors
                           if e.rank_of_first_relevant is not None])
        patterns.append(ErrorPattern(
            pattern_type="ranking",
            description=f"First relevant item ranked very low (avg rank: {avg_rank:.1f})",
            frequency=len(severe_rank_errors),
            affected_queries=severe_rank_indices,
            severity=float(np.mean(severe_rank_severities)),
        ))

    # Sort patterns by frequency
    patterns.sort(key=lambda p: p.frequency, reverse=True)
    return patterns


def _generate_root_cause(error: PredictionError) -> str:
    """Generate a root cause analysis for a single error."""
    query_origin = error.query_metadata.get("Country of Origin", "Unknown")
    query_processing = error.query_metadata.get("Processing Method", "Unknown")
    rec_origin = error.recommended_metadata.get("Country of Origin", "Unknown")
    rec_processing = error.recommended_metadata.get("Processing Method", "Unknown")

    causes = []

    # Check for origin mismatch
    if query_origin != rec_origin:
        causes.append(
            f"Origin mismatch: queried {query_origin}, recommended {rec_origin}"
        )

    # Check for processing mismatch
    if query_processing != rec_processing:
        causes.append(
            f"Processing mismatch: queried {query_processing}, recommended {rec_processing}"
        )

    # Check for taste profile deviation
    taste_features = [
        "Aroma", "Flavor", "Aftertaste", "Acidity", "Body",
        "Balance", "Uniformity", "Clean Cup", "Sweetness"
    ]
    large_deviations = []
    for i, feature in enumerate(taste_features):
        diff = abs(error.query_preferences[i] - error.recommended_profile[i])
        if diff > 0.5:  # Significant deviation
            large_deviations.append(f"{feature} (diff: {diff:.2f})")

    if large_deviations:
        causes.append(f"Large taste deviations: {', '.join(large_deviations[:3])}")

    if not causes:
        causes.append("Minor deviations across multiple dimensions")

    return "; ".join(causes)


def _generate_mitigations(patterns: list[ErrorPattern]) -> list[str]:
    """Generate mitigation strategies based on identified patterns."""
    mitigations = []

    pattern_types = {p.pattern_type for p in patterns}

    if "origin" in pattern_types:
        mitigations.append(
            "Consider adding origin-aware features or embeddings to better "
            "capture regional flavor characteristics."
        )

    if "processing" in pattern_types:
        mitigations.append(
            "Include processing method as an explicit feature or learn "
            "processing-specific taste profile transformations."
        )

    if "cross_origin_confusion" in pattern_types:
        mitigations.append(
            "Add contrastive learning or negative sampling to better distinguish "
            "coffees from commonly confused origins."
        )

    if "taste_profile" in pattern_types:
        mitigations.append(
            "Review feature scaling and consider non-linear transformations "
            "for extreme taste profile values (high acidity, low body)."
        )

    if "ranking" in pattern_types:
        mitigations.append(
            "Incorporate a re-ranking stage or listwise learning objective "
            "to improve early-rank precision."
        )

    # General mitigations based on error severity
    if patterns and max(p.severity for p in patterns) > 2.0:
        mitigations.append(
            "High severity errors suggest the model may benefit from ensemble "
            "methods or calibration techniques."
        )

    if not mitigations:
        mitigations.append(
            "No strong error patterns detected. Consider increasing training "
            "data or fine-tuning hyperparameters."
        )

    return mitigations


def generate_error_report(
    model: Recommender,
    test_data: dict[str, Any],
) -> ErrorReport:
    """Generate a comprehensive error analysis report.

    Analyzes the model's predictions on test data to identify:
    - The 5 worst mispredictions with root cause analysis
    - Common failure patterns across errors
    - Proposed mitigation strategies

    Args:
        model: A fitted recommender model with a recommend() method.
        test_data: Dictionary containing:
            - 'X': Feature matrix of shape (n_samples, 9) with taste profiles.
            - 'metadata': List of metadata dicts or DataFrame.

    Returns:
        ErrorReport containing detailed analysis and recommendations.

    Example:
        >>> report = generate_error_report(model, test_data)
        >>> print(f"Error rate: {report.error_rate:.1%}")
        >>> for pattern in report.patterns:
        ...     print(f"- {pattern.description}")
        >>> for mitigation in report.mitigations:
        ...     print(f"* {mitigation}")
    """
    X = np.asarray(test_data["X"], dtype=np.float32)
    metadata_raw = test_data["metadata"]

    if hasattr(metadata_raw, "to_dict"):
        all_metadata = metadata_raw.to_dict("records")
    else:
        all_metadata = list(metadata_raw)

    n_samples = len(X)

    # Get worst errors (more than 5 for pattern analysis)
    all_errors = analyze_errors(model, test_data, n_errors=50)
    worst_5_errors = all_errors[:5]

    # Count total errors (queries where top recommendation is not relevant)
    total_errors = 0
    total_valid_queries = 0

    for query_idx in range(n_samples):
        query_profile = X[query_idx]
        query_metadata = all_metadata[query_idx]

        relevant = _find_relevant_items(
            query_metadata=query_metadata,
            query_profile=query_profile,
            all_metadata=all_metadata,
            all_profiles=X,
            query_idx=query_idx,
        )

        if not relevant:
            continue

        total_valid_queries += 1

        recommendations = model.recommend(query_profile, k=1)
        if recommendations and recommendations[0]["index"] not in relevant:
            total_errors += 1

    # Identify patterns from all errors
    patterns = identify_error_patterns(all_errors)

    # Generate mitigations
    mitigations = _generate_mitigations(patterns)

    # Add root cause to worst errors (stored in metadata for reporting)
    for err in worst_5_errors:
        err.query_metadata["_root_cause"] = _generate_root_cause(err)

    error_rate = total_errors / total_valid_queries if total_valid_queries > 0 else 0.0

    return ErrorReport(
        total_queries=total_valid_queries,
        total_errors=total_errors,
        error_rate=error_rate,
        worst_errors=worst_5_errors,
        patterns=patterns,
        mitigations=mitigations,
    )
