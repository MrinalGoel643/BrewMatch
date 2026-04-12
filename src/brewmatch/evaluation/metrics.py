"""Evaluation metrics for the coffee recommendation system.

This module provides metrics for evaluating recommendation quality:
- Ranking metrics: Precision@K, Recall@K, NDCG@K
- Regression metrics: MSE, MAE for quality prediction
- Comprehensive evaluation combining all metrics
"""

from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike


@runtime_checkable
class Recommender(Protocol):
    """Protocol for recommender models used in evaluation."""

    def recommend(self, preferences: np.ndarray, k: int = 5) -> list[dict[str, Any]]:
        """Recommend coffees matching user taste preferences."""
        ...


def precision_at_k(
    recommended: list[int],
    relevant: set[int],
    k: int,
) -> float:
    """Calculate Precision@K for a recommendation list.

    Precision@K measures the proportion of recommended items in the top-K
    that are relevant.

    Args:
        recommended: List of recommended item indices, ordered by rank.
        relevant: Set of relevant item indices.
        k: Number of top recommendations to consider.

    Returns:
        Precision@K score in range [0, 1].

    Example:
        >>> recommended = [1, 2, 3, 4, 5]
        >>> relevant = {1, 3, 5, 7, 9}
        >>> precision_at_k(recommended, relevant, k=5)
        0.6
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if not recommended:
        return 0.0

    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k


def recall_at_k(
    recommended: list[int],
    relevant: set[int],
    k: int,
) -> float:
    """Calculate Recall@K for a recommendation list.

    Recall@K measures the proportion of relevant items that appear
    in the top-K recommendations.

    Args:
        recommended: List of recommended item indices, ordered by rank.
        relevant: Set of relevant item indices.
        k: Number of top recommendations to consider.

    Returns:
        Recall@K score in range [0, 1]. Returns 0.0 if there are no relevant items.

    Example:
        >>> recommended = [1, 2, 3, 4, 5]
        >>> relevant = {1, 3, 5, 7, 9}
        >>> recall_at_k(recommended, relevant, k=5)
        0.6
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if not relevant:
        return 0.0
    if not recommended:
        return 0.0

    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(
    recommended: list[int],
    relevant: set[int],
    k: int,
) -> float:
    """Calculate Normalized Discounted Cumulative Gain at K.

    NDCG@K measures ranking quality by giving higher scores when
    relevant items appear earlier in the recommendation list.

    Uses binary relevance (1 if relevant, 0 otherwise).

    Args:
        recommended: List of recommended item indices, ordered by rank.
        relevant: Set of relevant item indices.
        k: Number of top recommendations to consider.

    Returns:
        NDCG@K score in range [0, 1]. Returns 0.0 if there are no relevant items.

    Example:
        >>> recommended = [1, 2, 3, 4, 5]
        >>> relevant = {1, 3, 5}
        >>> ndcg_at_k(recommended, relevant, k=5)  # Higher because relevant items are ranked well
        0.934...
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if not relevant:
        return 0.0
    if not recommended:
        return 0.0

    top_k = recommended[:k]

    # DCG: sum of relevance / log2(position + 1)
    dcg = 0.0
    for i, item in enumerate(top_k):
        if item in relevant:
            # Position is 1-indexed for the log
            dcg += 1.0 / np.log2(i + 2)

    # Ideal DCG: all relevant items ranked first
    ideal_k = min(k, len(relevant))
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_k))

    if idcg == 0:
        return 0.0

    return dcg / idcg


def mean_squared_error(
    predicted: ArrayLike,
    actual: ArrayLike,
) -> float:
    """Calculate Mean Squared Error between predictions and actual values.

    Args:
        predicted: Predicted values.
        actual: Actual/ground truth values.

    Returns:
        Mean squared error (non-negative).

    Raises:
        ValueError: If arrays have different lengths.

    Example:
        >>> predicted = [3.0, 4.0, 5.0]
        >>> actual = [3.5, 4.0, 4.5]
        >>> mean_squared_error(predicted, actual)
        0.166...
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)

    if predicted.shape != actual.shape:
        raise ValueError(
            f"Shape mismatch: predicted {predicted.shape} vs actual {actual.shape}"
        )

    return float(np.mean((predicted - actual) ** 2))


def mean_absolute_error(
    predicted: ArrayLike,
    actual: ArrayLike,
) -> float:
    """Calculate Mean Absolute Error between predictions and actual values.

    Args:
        predicted: Predicted values.
        actual: Actual/ground truth values.

    Returns:
        Mean absolute error (non-negative).

    Raises:
        ValueError: If arrays have different lengths.

    Example:
        >>> predicted = [3.0, 4.0, 5.0]
        >>> actual = [3.5, 4.0, 4.5]
        >>> mean_absolute_error(predicted, actual)
        0.333...
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)

    if predicted.shape != actual.shape:
        raise ValueError(
            f"Shape mismatch: predicted {predicted.shape} vs actual {actual.shape}"
        )

    return float(np.mean(np.abs(predicted - actual)))


def _compute_similarity(profile_a: np.ndarray, profile_b: np.ndarray) -> float:
    """Compute cosine similarity between two taste profiles."""
    norm_a = np.linalg.norm(profile_a)
    norm_b = np.linalg.norm(profile_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(profile_a, profile_b) / (norm_a * norm_b))


def _find_relevant_items_in_catalog(
    query_metadata: dict[str, Any],
    query_profile: np.ndarray,
    catalog_metadata: list[dict[str, Any]],
    catalog_profiles: np.ndarray,
    similarity_threshold: float = 0.80,
) -> set[int]:
    """Identify relevant items in the catalog for a query.

    An item is considered relevant if:
    - It has high taste profile similarity (>= threshold), OR
    - It shares the same country of origin (with non-empty values)

    Args:
        query_metadata: Metadata dict for the query coffee.
        query_profile: Taste profile array for the query coffee.
        catalog_metadata: List of metadata dicts for catalog items.
        catalog_profiles: Array of catalog taste profiles, shape (n_catalog, n_features).
        similarity_threshold: Cosine similarity threshold for profile-based relevance.

    Returns:
        Set of indices of relevant catalog items.
    """
    relevant = set()
    query_country = query_metadata.get("Country of Origin", "")

    for i, meta in enumerate(catalog_metadata):
        # Check similarity-based relevance (primary criterion)
        similarity = _compute_similarity(query_profile, catalog_profiles[i])
        if similarity >= similarity_threshold:
            relevant.add(i)
            continue

        # Check metadata-based relevance (secondary criterion)
        same_country = meta.get("Country of Origin", "") == query_country
        if same_country and query_country:
            relevant.add(i)

    return relevant


def evaluate_model(
    model: Recommender,
    test_data: dict[str, Any],
    catalog_data: dict[str, Any] | None = None,
    k_values: list[int] | None = None,
) -> dict[str, Any]:
    """Comprehensive evaluation of a recommendation model.

    Evaluates the model using each test coffee as a query, measuring how well
    the model recommends similar coffees from its catalog.

    Args:
        model: A fitted recommender model with a recommend() method.
        test_data: Dictionary containing test queries:
            - 'X': Feature matrix of shape (n_samples, 9) with taste profiles.
            - 'metadata': List of metadata dicts or DataFrame.
        catalog_data: Dictionary containing the model's catalog (training data):
            - 'X': Feature matrix of catalog items.
            - 'metadata': Metadata for catalog items.
            If None, uses test_data (not recommended).
        k_values: List of K values for ranking metrics. Defaults to [1, 3, 5, 10].

    Returns:
        Dictionary containing:
            - 'precision@k': Dict mapping k to average Precision@K.
            - 'recall@k': Dict mapping k to average Recall@K.
            - 'ndcg@k': Dict mapping k to average NDCG@K.
            - 'mse': Mean squared error of predicted vs actual taste profiles.
            - 'mae': Mean absolute error of predicted vs actual taste profiles.
            - 'n_queries': Number of test queries evaluated.
            - 'avg_relevant_items': Average number of relevant items per query.

    Example:
        >>> results = evaluate_model(model, test_data, catalog_data, k_values=[1, 5, 10])
        >>> print(f"Precision@5: {results['precision@k'][5]:.3f}")
        >>> print(f"NDCG@10: {results['ndcg@k'][10]:.3f}")
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    # Query data (test set)
    query_X = np.asarray(test_data["X"], dtype=np.float32)
    query_metadata_raw = test_data["metadata"]
    if hasattr(query_metadata_raw, "to_dict"):
        query_metadata = query_metadata_raw.to_dict("records")
    else:
        query_metadata = list(query_metadata_raw)

    # Catalog data (training set - what the model recommends from)
    if catalog_data is None:
        catalog_data = test_data
    catalog_X = np.asarray(catalog_data["X"], dtype=np.float32)
    catalog_metadata_raw = catalog_data["metadata"]
    if hasattr(catalog_metadata_raw, "to_dict"):
        catalog_metadata = catalog_metadata_raw.to_dict("records")
    else:
        catalog_metadata = list(catalog_metadata_raw)

    n_queries = len(query_X)
    max_k = max(k_values)

    # Initialize accumulators
    precision_sums = {k: 0.0 for k in k_values}
    recall_sums = {k: 0.0 for k in k_values}
    ndcg_sums = {k: 0.0 for k in k_values}
    total_relevant = 0
    valid_queries = 0

    # For MSE/MAE: collect predicted vs actual taste profiles
    all_predicted_profiles = []
    all_actual_profiles = []

    for query_idx in range(n_queries):
        query_profile = query_X[query_idx]
        query_meta = query_metadata[query_idx]

        # Find relevant items in the CATALOG (not test set)
        relevant = _find_relevant_items_in_catalog(
            query_metadata=query_meta,
            query_profile=query_profile,
            catalog_metadata=catalog_metadata,
            catalog_profiles=catalog_X,
        )

        # Skip queries with no relevant items
        if not relevant:
            continue

        valid_queries += 1
        total_relevant += len(relevant)

        # Get recommendations
        recommendations = model.recommend(query_profile, k=max_k)
        recommended_indices = [rec["index"] for rec in recommendations]

        # Calculate ranking metrics for each k
        for k in k_values:
            precision_sums[k] += precision_at_k(recommended_indices, relevant, k)
            recall_sums[k] += recall_at_k(recommended_indices, relevant, k)
            ndcg_sums[k] += ndcg_at_k(recommended_indices, relevant, k)

        # For MSE/MAE: compare top recommendation's profile to query profile
        if recommendations:
            top_rec = recommendations[0]
            predicted_profile = np.array(
                [top_rec["taste_profile"][f] for f in [
                    "Aroma", "Flavor", "Aftertaste", "Acidity", "Body",
                    "Balance", "Uniformity", "Clean Cup", "Sweetness"
                ]],
                dtype=np.float32
            )
            all_predicted_profiles.append(predicted_profile)
            all_actual_profiles.append(query_profile)

    # Compute averages
    results: dict[str, Any] = {
        "precision@k": {},
        "recall@k": {},
        "ndcg@k": {},
        "n_queries": valid_queries,
        "avg_relevant_items": total_relevant / valid_queries if valid_queries > 0 else 0.0,
    }

    if valid_queries > 0:
        for k in k_values:
            results["precision@k"][k] = precision_sums[k] / valid_queries
            results["recall@k"][k] = recall_sums[k] / valid_queries
            results["ndcg@k"][k] = ndcg_sums[k] / valid_queries
    else:
        for k in k_values:
            results["precision@k"][k] = 0.0
            results["recall@k"][k] = 0.0
            results["ndcg@k"][k] = 0.0

    # Compute MSE and MAE
    if all_predicted_profiles:
        predicted = np.array(all_predicted_profiles)
        actual = np.array(all_actual_profiles)
        results["mse"] = mean_squared_error(predicted.flatten(), actual.flatten())
        results["mae"] = mean_absolute_error(predicted.flatten(), actual.flatten())
    else:
        results["mse"] = float("nan")
        results["mae"] = float("nan")

    return results
