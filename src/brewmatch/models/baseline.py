"""Naive baseline recommender for establishing performance floor."""

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .base import BaseRecommender


class NaiveBaselineRecommender(BaseRecommender):
    """Naive baseline recommender using global mean or weighted random.

    This establishes a performance floor for comparison with more
    sophisticated approaches. It supports two strategies:

    - 'mean': Recommends coffees closest to the global mean profile,
      ignoring user preferences entirely.
    - 'weighted_random': Randomly samples coffees weighted by their
      Total Cup Points score.

    Attributes:
        strategy: The recommendation strategy ('mean' or 'weighted_random').
    """

    def __init__(self, strategy: str = "mean") -> None:
        """Initialize the baseline recommender.

        Args:
            strategy: Recommendation strategy. One of:
                - 'mean': Recommend coffees closest to global mean profile
                - 'weighted_random': Random sampling weighted by Total Cup Points
        """
        super().__init__()
        if strategy not in ("mean", "weighted_random"):
            raise ValueError(f"Unknown strategy: {strategy}")
        self.strategy = strategy
        self._X: np.ndarray | None = None
        self._global_mean: np.ndarray | None = None
        self._weights: np.ndarray | None = None
        self._rng = np.random.default_rng()

    def fit(self, X: np.ndarray, metadata: pd.DataFrame) -> "NaiveBaselineRecommender":
        """Fit the baseline recommender.

        For 'mean' strategy, computes the global mean profile and
        distances from each coffee to it.

        For 'weighted_random' strategy, extracts Total Cup Points as
        sampling weights.

        Args:
            X: Feature matrix of shape (n_samples, 9).
            metadata: DataFrame with coffee metadata. For 'weighted_random',
                must contain 'Total Cup Points' column.

        Returns:
            self: The fitted recommender.
        """
        X = np.asarray(X, dtype=np.float32)
        if X.shape[1] != 9:
            raise ValueError(f"Expected 9 features, got {X.shape[1]}")

        self._X = X
        self._metadata = metadata.copy()
        self._global_mean = X.mean(axis=0)

        if self.strategy == "weighted_random":
            if "Total Cup Points" not in metadata.columns:
                raise ValueError(
                    "metadata must contain 'Total Cup Points' for weighted_random strategy"
                )
            scores = metadata["Total Cup Points"].values.astype(np.float32)
            # Shift to positive and normalize
            scores = scores - scores.min() + 1.0
            self._weights = scores / scores.sum()

        self.is_fitted = True
        return self

    def recommend(
        self, preferences: np.ndarray, k: int = 5
    ) -> list[dict[str, Any]]:
        """Generate recommendations.

        For 'mean' strategy, returns coffees closest to the global mean,
        ignoring the provided preferences entirely.

        For 'weighted_random' strategy, returns random coffees sampled
        proportionally to their Total Cup Points.

        Args:
            preferences: User taste preferences (ignored for baseline).
            k: Number of recommendations.

        Returns:
            List of k recommendation dictionaries.
        """
        self._validate_fitted()
        preferences = self._validate_preferences(preferences)

        n_samples = self._X.shape[0]
        k = min(k, n_samples)

        if self.strategy == "mean":
            # Find coffees closest to global mean (ignoring user preferences)
            distances = np.linalg.norm(self._X - self._global_mean, axis=1)
            indices = np.argsort(distances)[:k]
            # Convert distance to similarity score (higher is better)
            scores = 1.0 / (1.0 + distances[indices])
        else:
            # Weighted random sampling
            indices = self._rng.choice(
                n_samples, size=k, replace=False, p=self._weights
            )
            # Use weight as score
            scores = self._weights[indices]

        recommendations = []
        for idx, score in zip(indices, scores):
            rec = self._format_recommendation(idx, score, self._X[idx])
            recommendations.append(rec)

        return recommendations

    def save(self, path: str | Path) -> None:
        """Save the fitted model to disk using pickle.

        Args:
            path: File path to save the model to.
        """
        self._validate_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "strategy": self.strategy,
            "X": self._X,
            "metadata": self._metadata,
            "global_mean": self._global_mean,
            "weights": self._weights,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str | Path) -> "NaiveBaselineRecommender":
        """Load a fitted model from disk.

        Args:
            path: File path to load the model from.

        Returns:
            The loaded recommender instance.
        """
        with open(path, "rb") as f:
            state = pickle.load(f)

        model = cls(strategy=state["strategy"])
        model._X = state["X"]
        model._metadata = state["metadata"]
        model._global_mean = state["global_mean"]
        model._weights = state["weights"]
        model.is_fitted = True
        return model
