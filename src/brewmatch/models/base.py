# Code written with assistance from Claude Opus 4.5 (Anthropic)
"""Abstract base class for all recommender models."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class BaseRecommender(ABC):
    """Abstract base class for coffee recommender models.

    All recommender implementations must inherit from this class and
    implement the required methods.

    Attributes:
        TASTE_FEATURES: The 9 taste feature columns used for recommendations.
        is_fitted: Whether the model has been fitted to data.
    """

    TASTE_FEATURES = [
        "Aroma",
        "Flavor",
        "Aftertaste",
        "Acidity",
        "Body",
        "Balance",
        "Uniformity",
        "Clean Cup",
        "Sweetness",
    ]

    def __init__(self) -> None:
        self.is_fitted = False
        self._metadata: pd.DataFrame | None = None

    @abstractmethod
    def fit(self, X: np.ndarray, metadata: pd.DataFrame) -> "BaseRecommender":
        """Fit the recommender to coffee taste profiles.

        Args:
            X: Feature matrix of shape (n_samples, 9) containing taste scores.
               Columns correspond to TASTE_FEATURES in order.
            metadata: DataFrame containing coffee metadata (country, processing
                method, variety, etc.). Must have same number of rows as X.

        Returns:
            self: The fitted recommender instance.
        """
        pass

    @abstractmethod
    def recommend(
        self, preferences: np.ndarray, k: int = 5
    ) -> list[dict[str, Any]]:
        """Recommend coffees matching user taste preferences.

        Args:
            preferences: Array of shape (9,) containing desired taste scores.
                Values correspond to TASTE_FEATURES in order.
            k: Number of recommendations to return.

        Returns:
            List of k recommendation dictionaries, each containing:
                - 'index': Original index in the training data
                - 'score': Similarity/relevance score (higher is better)
                - 'metadata': Dict of coffee metadata
                - 'taste_profile': Dict of the coffee's taste scores
        """
        pass

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Save the fitted model to disk.

        Args:
            path: File path to save the model to.
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "BaseRecommender":
        """Load a fitted model from disk.

        Args:
            path: File path to load the model from.

        Returns:
            The loaded recommender instance.
        """
        pass

    def _validate_fitted(self) -> None:
        """Raise error if model is not fitted."""
        if not self.is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} must be fitted before calling this method. "
                "Call fit() first."
            )

    def _validate_preferences(self, preferences: np.ndarray) -> np.ndarray:
        """Validate and reshape preference array."""
        preferences = np.asarray(preferences, dtype=np.float32)
        if preferences.ndim == 1:
            if preferences.shape[0] != 9:
                raise ValueError(
                    f"Expected 9 taste features, got {preferences.shape[0]}"
                )
        elif preferences.ndim == 2:
            if preferences.shape[1] != 9:
                raise ValueError(
                    f"Expected 9 taste features, got {preferences.shape[1]}"
                )
            preferences = preferences.squeeze()
        else:
            raise ValueError(
                f"Preferences must be 1D or 2D array, got {preferences.ndim}D"
            )
        return preferences

    def _format_recommendation(
        self, idx: int, score: float, taste_profile: np.ndarray
    ) -> dict[str, Any]:
        """Format a single recommendation as a dictionary."""
        metadata_dict = {}
        if self._metadata is not None:
            row = self._metadata.iloc[idx]
            metadata_dict = row.to_dict()

        taste_dict = {
            feature: float(taste_profile[i])
            for i, feature in enumerate(self.TASTE_FEATURES)
        }

        return {
            "index": idx,
            "score": float(score),
            "metadata": metadata_dict,
            "taste_profile": taste_dict,
        }
