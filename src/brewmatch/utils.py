"""Utility functions for BrewMatch."""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch


def save_pickle(obj: Any, path: Union[str, Path]) -> None:
    """Save object to pickle file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Union[str, Path]) -> Any:
    """Load object from pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(obj: Dict, path: Union[str, Path], indent: int = 2) -> None:
    """Save dict to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent)


def load_json(path: Union[str, Path]) -> Dict:
    """Load dict from JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between vectors.

    Args:
        a: Query vector(s) of shape (d,) or (n, d)
        b: Reference vectors of shape (m, d)

    Returns:
        Similarity scores of shape (n, m) or (m,)
    """
    if a.ndim == 1:
        a = a.reshape(1, -1)
        squeeze = True
    else:
        squeeze = False

    # Normalize
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)

    # Compute similarity
    sim = np.dot(a_norm, b_norm.T)

    if squeeze:
        sim = sim.squeeze(0)

    return sim


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute Euclidean distance between vectors.

    Args:
        a: Query vector(s) of shape (d,) or (n, d)
        b: Reference vectors of shape (m, d)

    Returns:
        Distance scores of shape (n, m) or (m,)
    """
    if a.ndim == 1:
        a = a.reshape(1, -1)
        squeeze = True
    else:
        squeeze = False

    # Compute distances using broadcasting
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
    a_sq = np.sum(a ** 2, axis=1, keepdims=True)
    b_sq = np.sum(b ** 2, axis=1).reshape(1, -1)
    dist_sq = a_sq + b_sq - 2 * np.dot(a, b.T)
    dist = np.sqrt(np.maximum(dist_sq, 0))

    if squeeze:
        dist = dist.squeeze(0)

    return dist


def normalize_preferences(
    preferences: Dict[str, float],
    feature_names: list,
    scaler: Optional[Any] = None,
) -> np.ndarray:
    """
    Convert user preferences dict to normalized feature vector.

    Args:
        preferences: Dict mapping feature names to values (0-10 scale)
        feature_names: List of feature names in order
        scaler: Optional sklearn scaler for normalization

    Returns:
        Normalized feature vector
    """
    # Build vector in correct order
    vector = np.array([preferences.get(name, 5.0) for name in feature_names])
    vector = vector.reshape(1, -1)

    if scaler is not None:
        vector = scaler.transform(vector)

    return vector.squeeze()


def format_recommendations(
    indices: np.ndarray,
    similarities: np.ndarray,
    metadata: Any,
    feature_names: list,
    features: np.ndarray,
) -> list:
    """
    Format recommendation results for API response.

    Args:
        indices: Array of recommended coffee indices
        similarities: Similarity scores for recommendations
        metadata: DataFrame or dict with coffee metadata
        feature_names: List of taste feature names
        features: Feature matrix for coffees

    Returns:
        List of recommendation dicts
    """
    recommendations = []

    for idx, sim in zip(indices, similarities):
        rec = {
            "id": int(idx),
            "similarity": float(sim),
            "scores": {
                name: float(features[idx, i])
                for i, name in enumerate(feature_names)
            },
        }

        # Add metadata if available
        if hasattr(metadata, "iloc"):
            row = metadata.iloc[idx]
            rec["country"] = str(row.get("Country of Origin", "Unknown"))
            rec["processing_method"] = str(row.get("Processing Method", "Unknown"))
            rec["total_cup_points"] = float(row.get("Total Cup Points", 0))
        elif isinstance(metadata, dict):
            rec["country"] = metadata.get("countries", ["Unknown"] * len(indices))[idx]
            rec["processing_method"] = metadata.get(
                "processing_methods", ["Unknown"] * len(indices)
            )[idx]

        recommendations.append(rec)

    return recommendations
