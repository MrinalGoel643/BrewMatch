"""Classical ML recommender using KNN and cosine similarity."""

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .base import BaseRecommender


class ClassicalMLRecommender(BaseRecommender):
    """Classical ML recommender using KNN or cosine similarity.

    Finds coffees with taste profiles most similar to user preferences
    using either KNN with Euclidean distance or cosine similarity ranking.

    Attributes:
        method: Similarity method ('knn' or 'cosine').
        n_neighbors: Maximum neighbors for KNN (used for internal index).
        normalize: Whether to standardize features before similarity computation.
    """

    def __init__(
        self,
        method: str = "knn",
        n_neighbors: int = 50,
        normalize: bool = True,
    ) -> None:
        """Initialize the classical ML recommender.

        Args:
            method: Similarity method. One of:
                - 'knn': K-nearest neighbors with Euclidean distance
                - 'cosine': Cosine similarity ranking
            n_neighbors: Maximum neighbors to index for KNN. Actual k in
                recommend() can be smaller.
            normalize: Whether to standardize features (recommended for
                Euclidean distance).
        """
        super().__init__()
        if method not in ("knn", "cosine"):
            raise ValueError(f"Unknown method: {method}")
        self.method = method
        self.n_neighbors = n_neighbors
        self.normalize = normalize

        self._X: np.ndarray | None = None
        self._X_normalized: np.ndarray | None = None
        self._scaler: StandardScaler | None = None
        self._knn: NearestNeighbors | None = None

    def fit(self, X: np.ndarray, metadata: pd.DataFrame) -> "ClassicalMLRecommender":
        """Fit the recommender to coffee taste profiles.

        Args:
            X: Feature matrix of shape (n_samples, 9).
            metadata: DataFrame with coffee metadata.

        Returns:
            self: The fitted recommender.
        """
        X = np.asarray(X, dtype=np.float32)
        if X.shape[1] != 9:
            raise ValueError(f"Expected 9 features, got {X.shape[1]}")

        self._X = X
        self._metadata = metadata.copy()

        if self.normalize:
            self._scaler = StandardScaler()
            self._X_normalized = self._scaler.fit_transform(X).astype(np.float32)
        else:
            self._X_normalized = X

        if self.method == "knn":
            # Build KNN index
            n_neighbors = min(self.n_neighbors, X.shape[0])
            self._knn = NearestNeighbors(
                n_neighbors=n_neighbors,
                metric="euclidean",
                algorithm="auto",
            )
            self._knn.fit(self._X_normalized)

        self.is_fitted = True
        return self

    def recommend(
        self, preferences: np.ndarray, k: int = 5
    ) -> list[dict[str, Any]]:
        """Find coffees most similar to user preferences.

        Args:
            preferences: User taste preferences of shape (9,).
            k: Number of recommendations.

        Returns:
            List of k recommendation dictionaries.
        """
        self._validate_fitted()
        preferences = self._validate_preferences(preferences)

        n_samples = self._X.shape[0]
        k = min(k, n_samples)

        # Normalize preferences if needed
        if self.normalize:
            pref_normalized = self._scaler.transform(
                preferences.reshape(1, -1)
            ).astype(np.float32)
        else:
            pref_normalized = preferences.reshape(1, -1)

        if self.method == "knn":
            indices, scores = self._recommend_knn(pref_normalized, k)
        else:
            indices, scores = self._recommend_cosine(pref_normalized, k)

        recommendations = []
        for idx, score in zip(indices, scores):
            rec = self._format_recommendation(idx, score, self._X[idx])
            recommendations.append(rec)

        return recommendations

    def _recommend_knn(
        self, pref_normalized: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Find k nearest neighbors using KNN index."""
        k = min(k, self._knn.n_neighbors)
        distances, indices = self._knn.kneighbors(pref_normalized, n_neighbors=k)

        # Convert distance to similarity (higher is better)
        # Using inverse distance with offset
        scores = 1.0 / (1.0 + distances.squeeze())

        return indices.squeeze(), scores

    def _recommend_cosine(
        self, pref_normalized: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Rank all coffees by cosine similarity."""
        # Compute cosine similarity
        pref_norm = pref_normalized / (
            np.linalg.norm(pref_normalized) + 1e-8
        )
        X_norms = np.linalg.norm(self._X_normalized, axis=1, keepdims=True) + 1e-8
        X_normalized_unit = self._X_normalized / X_norms

        similarities = (X_normalized_unit @ pref_norm.T).squeeze()

        # Get top k
        indices = np.argsort(similarities)[::-1][:k]
        scores = similarities[indices]

        # Shift to [0, 1] range (cosine similarity is in [-1, 1])
        scores = (scores + 1.0) / 2.0

        return indices, scores

    def save(self, path: str | Path) -> None:
        """Save the fitted model to disk using pickle.

        Args:
            path: File path to save the model to.
        """
        self._validate_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "method": self.method,
            "n_neighbors": self.n_neighbors,
            "normalize": self.normalize,
            "X": self._X,
            "X_normalized": self._X_normalized,
            "metadata": self._metadata,
            "scaler": self._scaler,
            "knn": self._knn,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str | Path) -> "ClassicalMLRecommender":
        """Load a fitted model from disk.

        Args:
            path: File path to load the model from.

        Returns:
            The loaded recommender instance.
        """
        with open(path, "rb") as f:
            state = pickle.load(f)

        model = cls(
            method=state["method"],
            n_neighbors=state["n_neighbors"],
            normalize=state["normalize"],
        )
        model._X = state["X"]
        model._X_normalized = state["X_normalized"]
        model._metadata = state["metadata"]
        model._scaler = state["scaler"]
        model._knn = state["knn"]
        model.is_fitted = True
        return model
