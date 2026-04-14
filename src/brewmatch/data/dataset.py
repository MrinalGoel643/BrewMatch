# Code written with assistance from Claude Opus 4.5 (Anthropic)
"""PyTorch dataset and dataloaders for coffee quality data."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .preprocess import TASTE_FEATURES, TARGET_COLUMN, load_processed_data


class CoffeeDataset(Dataset):
    """
    PyTorch Dataset for coffee quality data.

    Provides (features, target) pairs where features are the 9 taste
    profile scores and target is the total cup points.

    Attributes:
        features: Tensor of shape (n_samples, 9) with normalized taste features.
        targets: Tensor of shape (n_samples,) with total cup points.
        metadata: Optional DataFrame with metadata columns.
    """

    def __init__(
        self,
        features: np.ndarray | torch.Tensor,
        targets: np.ndarray | torch.Tensor,
        metadata: pd.DataFrame | None = None,
    ) -> None:
        """
        Initialize the dataset.

        Args:
            features: Array of shape (n_samples, n_features) with input features.
            targets: Array of shape (n_samples,) with target values.
            metadata: Optional DataFrame with metadata (not used in training).
        """
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features).float()
        if isinstance(targets, np.ndarray):
            targets = torch.from_numpy(targets).float()

        self.features = features
        self.targets = targets
        self.metadata = metadata

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample by index.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (features, target) tensors.
        """
        return self.features[idx], self.targets[idx]

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        feature_cols: list[str] | None = None,
        target_col: str | None = None,
    ) -> "CoffeeDataset":
        """
        Create a dataset from a pandas DataFrame.

        Args:
            df: DataFrame with features and target.
            feature_cols: List of feature column names (default: TASTE_FEATURES).
            target_col: Target column name (default: TARGET_COLUMN).

        Returns:
            CoffeeDataset instance.
        """
        if feature_cols is None:
            feature_cols = TASTE_FEATURES
        if target_col is None:
            target_col = TARGET_COLUMN

        features = df[feature_cols].values
        targets = df[target_col].values

        # Get metadata columns (everything that's not a feature or target)
        metadata_cols = [c for c in df.columns if c not in feature_cols and c != target_col]
        metadata = df[metadata_cols] if metadata_cols else None

        return cls(features, targets, metadata)


def create_dataloaders(
    batch_size: int = 32,
    val_split: float = 0.1,
    num_workers: int = 0,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Create train, validation, and test DataLoaders.

    Splits the training data into train/validation sets, keeps test set separate.

    Args:
        batch_size: Batch size for all loaders (default: 32).
        val_split: Fraction of training data for validation (default: 0.1).
        num_workers: Number of workers for data loading (default: 0).
        random_state: Random seed for train/val split (default: 42).

    Returns:
        Dictionary containing:
        - train_loader: DataLoader for training
        - val_loader: DataLoader for validation
        - test_loader: DataLoader for testing
        - train_dataset: Training CoffeeDataset
        - val_dataset: Validation CoffeeDataset
        - test_dataset: Test CoffeeDataset
        - n_features: Number of input features (9)
        - scaler: The fitted StandardScaler

    Raises:
        FileNotFoundError: If processed data doesn't exist.
    """
    # Load processed data
    data = load_processed_data()
    train_df = data["train_df"]
    test_df = data["test_df"]
    scaler = data["scaler"]
    feature_cols = data["taste_features"]
    target_col = data["target_column"]

    # Split training data into train/val
    n_train = len(train_df)
    n_val = int(n_train * val_split)

    # Shuffle with fixed seed
    rng = np.random.default_rng(random_state)
    indices = rng.permutation(n_train)

    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    train_subset_df = train_df.iloc[train_indices].reset_index(drop=True)
    val_subset_df = train_df.iloc[val_indices].reset_index(drop=True)

    # Create datasets
    train_dataset = CoffeeDataset.from_dataframe(
        train_subset_df, feature_cols, target_col
    )
    val_dataset = CoffeeDataset.from_dataframe(
        val_subset_df, feature_cols, target_col
    )
    test_dataset = CoffeeDataset.from_dataframe(
        test_df, feature_cols, target_col
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Created dataloaders:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Test: {len(test_dataset)} samples, {len(test_loader)} batches")
    print(f"  Batch size: {batch_size}")
    print(f"  Features: {len(feature_cols)}")

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "n_features": len(feature_cols),
        "scaler": scaler,
    }
