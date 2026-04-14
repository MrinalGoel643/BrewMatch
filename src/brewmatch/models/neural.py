# Code written with assistance from Claude Opus 4.5 (Anthropic)
"""Neural network recommender using learned coffee embeddings."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .base import BaseRecommender


class TasteEncoder(nn.Module):
    """Neural network that encodes taste profiles into embeddings.

    Architecture: MLP with residual connections that maps 9 taste features
    to a lower-dimensional embedding space.
    """

    def __init__(
        self,
        input_dim: int = 9,
        hidden_dim: int = 64,
        embedding_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.hidden1 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, embedding_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode taste profiles to embeddings.

        Args:
            x: Taste features of shape (batch_size, 9).

        Returns:
            Embeddings of shape (batch_size, embedding_dim).
        """
        # Project to hidden dimension
        h = F.gelu(self.input_proj(x))

        # Residual block 1
        residual = h
        h = self.norm1(h)
        h = F.gelu(self.hidden1(h))
        h = self.dropout(h)
        h = h + residual

        # Residual block 2
        residual = h
        h = self.norm2(h)
        h = F.gelu(self.hidden2(h))
        h = self.dropout(h)
        h = h + residual

        # Project to embedding space
        embedding = self.output_proj(h)

        # L2 normalize for cosine similarity
        embedding = F.normalize(embedding, p=2, dim=-1)

        return embedding


class TripletDataset(Dataset):
    """Dataset that generates triplets for contrastive learning.

    For each anchor, samples a positive (similar coffee) and negative
    (dissimilar coffee) based on taste profile distance.
    """

    def __init__(
        self,
        X: np.ndarray,
        margin_quantile: float = 0.3,
    ) -> None:
        """Initialize triplet dataset.

        Args:
            X: Feature matrix of shape (n_samples, 9).
            margin_quantile: Quantile for positive/negative threshold.
                Coffees within this distance quantile are positives.
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.n_samples = X.shape[0]

        # Precompute pairwise distances
        X_tensor = self.X
        self.distances = torch.cdist(X_tensor, X_tensor, p=2)

        # Determine threshold for positive/negative
        flat_distances = self.distances.flatten()
        self.positive_threshold = torch.quantile(flat_distances, margin_quantile)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a triplet (anchor, positive, negative).

        Args:
            idx: Anchor index.

        Returns:
            Tuple of (anchor, positive, negative) taste profiles.
        """
        anchor = self.X[idx]

        # Get distances from anchor
        dists = self.distances[idx]

        # Find positives (close) and negatives (far), excluding self
        mask = torch.arange(self.n_samples) != idx
        positive_mask = mask & (dists <= self.positive_threshold)
        negative_mask = mask & (dists > self.positive_threshold)

        # Handle edge cases
        if positive_mask.sum() == 0:
            # Use closest non-self sample
            dists_masked = dists.clone()
            dists_masked[idx] = float("inf")
            positive_idx = dists_masked.argmin().item()
        else:
            positive_indices = torch.where(positive_mask)[0]
            positive_idx = positive_indices[
                torch.randint(len(positive_indices), (1,))
            ].item()

        if negative_mask.sum() == 0:
            # Use farthest sample
            negative_idx = dists.argmax().item()
        else:
            negative_indices = torch.where(negative_mask)[0]
            negative_idx = negative_indices[
                torch.randint(len(negative_indices), (1,))
            ].item()

        return anchor, self.X[positive_idx], self.X[negative_idx]


class NeuralRecommender(BaseRecommender):
    """Neural recommender using learned coffee embeddings.

    Uses contrastive learning with triplet loss to learn embeddings
    that capture taste similarity. Similar coffees have nearby embeddings.

    Attributes:
        embedding_dim: Dimension of learned embeddings.
        hidden_dim: Hidden layer dimension in encoder.
        learning_rate: Learning rate for training.
        margin: Triplet loss margin.
        device: Torch device (cuda/cpu).
    """

    def __init__(
        self,
        embedding_dim: int = 32,
        hidden_dim: int = 64,
        learning_rate: float = 1e-3,
        margin: float = 0.5,
        device: str | None = None,
    ) -> None:
        """Initialize the neural recommender.

        Args:
            embedding_dim: Embedding dimension.
            hidden_dim: Hidden layer dimension.
            learning_rate: Learning rate.
            margin: Triplet loss margin.
            device: Torch device. Auto-detects CUDA if not specified.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.margin = margin

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._encoder: TasteEncoder | None = None
        self._X: np.ndarray | None = None
        self._embeddings: np.ndarray | None = None
        self._feature_mean: np.ndarray | None = None
        self._feature_std: np.ndarray | None = None

    def fit(
        self,
        X: np.ndarray,
        metadata: pd.DataFrame,
        epochs: int = 100,
        batch_size: int = 64,
        patience: int = 15,
        min_delta: float = 1e-4,
        verbose: bool = True,
    ) -> "NeuralRecommender":
        """Fit the neural recommender using contrastive learning.

        Args:
            X: Feature matrix of shape (n_samples, 9).
            metadata: DataFrame with coffee metadata.
            epochs: Maximum number of training epochs.
            batch_size: Training batch size.
            patience: Early stopping patience (epochs without improvement).
            min_delta: Minimum loss improvement to reset patience.
            verbose: Whether to print training progress.

        Returns:
            self: The fitted recommender.
        """
        X = np.asarray(X, dtype=np.float32)
        if X.shape[1] != 9:
            raise ValueError(f"Expected 9 features, got {X.shape[1]}")

        self._X = X
        self._metadata = metadata.copy()

        # Normalize features
        self._feature_mean = X.mean(axis=0)
        self._feature_std = X.std(axis=0) + 1e-8
        X_normalized = (X - self._feature_mean) / self._feature_std

        # Create encoder
        self._encoder = TasteEncoder(
            input_dim=9,
            hidden_dim=self.hidden_dim,
            embedding_dim=self.embedding_dim,
        ).to(self.device)

        # Create dataset and dataloader
        dataset = TripletDataset(X_normalized)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )

        # Training
        optimizer = torch.optim.AdamW(
            self._encoder.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )
        triplet_loss = nn.TripletMarginLoss(margin=self.margin, p=2)

        # Early stopping state
        best_loss = float("inf")
        best_state = None
        epochs_without_improvement = 0

        self._encoder.train()
        for epoch in range(epochs):
            total_loss = 0.0
            n_batches = 0

            for anchor, positive, negative in dataloader:
                anchor = anchor.to(self.device)
                positive = positive.to(self.device)
                negative = negative.to(self.device)

                optimizer.zero_grad()

                anchor_emb = self._encoder(anchor)
                positive_emb = self._encoder(positive)
                negative_emb = self._encoder(negative)

                loss = triplet_loss(anchor_emb, positive_emb, negative_emb)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_loss = total_loss / n_batches

            # Early stopping check
            if avg_loss < best_loss - min_delta:
                best_loss = avg_loss
                best_state = {k: v.cpu().clone() for k, v in self._encoder.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

            if epochs_without_improvement >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1} (best loss: {best_loss:.4f})")
                break

        # Restore best model
        if best_state is not None:
            self._encoder.load_state_dict(best_state)
            self._encoder.to(self.device)

        # Compute embeddings for all coffees
        self._encoder.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_normalized, dtype=torch.float32).to(self.device)
            self._embeddings = self._encoder(X_tensor).cpu().numpy()

        self.is_fitted = True
        return self

    def recommend(
        self, preferences: np.ndarray, k: int = 5
    ) -> list[dict[str, Any]]:
        """Find coffees with embeddings closest to user preferences.

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

        # Normalize and encode preferences
        pref_normalized = (preferences - self._feature_mean) / self._feature_std
        pref_tensor = torch.tensor(
            pref_normalized, dtype=torch.float32
        ).unsqueeze(0).to(self.device)

        self._encoder.eval()
        with torch.no_grad():
            pref_embedding = self._encoder(pref_tensor).cpu().numpy()

        # Find nearest embeddings using cosine similarity
        # (embeddings are already L2 normalized)
        similarities = (self._embeddings @ pref_embedding.T).squeeze()

        # Get top k
        indices = np.argsort(similarities)[::-1][:k]
        scores = similarities[indices]

        # Shift to [0, 1] range
        scores = (scores + 1.0) / 2.0

        recommendations = []
        for idx, score in zip(indices, scores):
            rec = self._format_recommendation(idx, score, self._X[idx])
            recommendations.append(rec)

        return recommendations

    def get_embedding(self, preferences: np.ndarray) -> np.ndarray:
        """Get the embedding for a taste profile.

        Args:
            preferences: Taste preferences of shape (9,) or (n, 9).

        Returns:
            Embedding(s) of shape (embedding_dim,) or (n, embedding_dim).
        """
        self._validate_fitted()

        preferences = np.asarray(preferences, dtype=np.float32)
        squeeze_output = False
        if preferences.ndim == 1:
            preferences = preferences.reshape(1, -1)
            squeeze_output = True

        pref_normalized = (preferences - self._feature_mean) / self._feature_std
        pref_tensor = torch.tensor(pref_normalized, dtype=torch.float32).to(self.device)

        self._encoder.eval()
        with torch.no_grad():
            embeddings = self._encoder(pref_tensor).cpu().numpy()

        if squeeze_output:
            return embeddings.squeeze(0)
        return embeddings

    def save(self, path: str | Path) -> None:
        """Save the fitted model to disk.

        Args:
            path: File path to save the model to.
        """
        self._validate_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "learning_rate": self.learning_rate,
            "margin": self.margin,
            "encoder_state_dict": self._encoder.state_dict(),
            "X": self._X,
            "metadata": self._metadata,
            "embeddings": self._embeddings,
            "feature_mean": self._feature_mean,
            "feature_std": self._feature_std,
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "NeuralRecommender":
        """Load a fitted model from disk.

        Args:
            path: File path to load the model from.
            device: Torch device. Auto-detects if not specified.

        Returns:
            The loaded recommender instance.
        """
        state = torch.load(path, map_location="cpu", weights_only=False)

        model = cls(
            embedding_dim=state["embedding_dim"],
            hidden_dim=state["hidden_dim"],
            learning_rate=state["learning_rate"],
            margin=state["margin"],
            device=device,
        )

        model._encoder = TasteEncoder(
            input_dim=9,
            hidden_dim=state["hidden_dim"],
            embedding_dim=state["embedding_dim"],
        ).to(model.device)
        model._encoder.load_state_dict(state["encoder_state_dict"])
        model._encoder.eval()

        model._X = state["X"]
        model._metadata = state["metadata"]
        model._embeddings = state["embeddings"]
        model._feature_mean = state["feature_mean"]
        model._feature_std = state["feature_std"]
        model.is_fitted = True

        return model
