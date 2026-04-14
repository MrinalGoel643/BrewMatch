# Code written with assistance from Claude Opus 4.5 (Anthropic)
"""Preprocess the CQI coffee quality dataset."""

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Taste profile features (9 total) - using actual CSV column names
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

# Target column
TARGET_COLUMN = "Total Cup Points"

# Metadata columns to preserve
METADATA_COLUMNS = [
    "Country of Origin",
    "Processing Method",
    "Variety",
]


def get_project_root() -> Path:
    """Get the project root directory (where pyproject.toml is located)."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not find project root (no pyproject.toml found)")


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to use spaces instead of dots/underscores."""
    # Map common column name variations
    column_mapping = {
        "Country.of.Origin": "Country of Origin",
        "Processing.Method": "Processing Method",
        "Clean.Cup": "Clean Cup",
        "Total.Cup.Points": "Total Cup Points",
        "Cupper.Points": "Cupper Points",
        "Category.One.Defects": "Category One Defects",
        "Category.Two.Defects": "Category Two Defects",
    }

    # Apply mapping
    df = df.rename(columns=column_mapping)
    return df


def load_raw_data() -> pd.DataFrame:
    """
    Load raw CSV data from data/raw/.

    Prefers merged_data_cleaned.csv if available (has most samples).
    Falls back to combining all CSV files.

    Returns:
        Combined DataFrame from raw CSV files.

    Raises:
        FileNotFoundError: If no CSV files found in raw directory.
    """
    project_root = get_project_root()
    raw_dir = project_root / "data" / "raw"

    csv_files = list(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {raw_dir}. Run `uv run download` first."
        )

    # Prefer the merged dataset if available (has most samples)
    merged_file = raw_dir / "merged_data_cleaned.csv"
    if merged_file.exists():
        print(f"Loading merged dataset: {merged_file.name}")
        df = pd.read_csv(merged_file)
        df = normalize_column_names(df)
        return df

    print(f"Found {len(csv_files)} CSV file(s) in {raw_dir}")

    dfs = []
    for csv_file in csv_files:
        print(f"  Loading {csv_file.name}...")
        df = pd.read_csv(csv_file)
        df = normalize_column_names(df)
        dfs.append(df)

    if len(dfs) == 1:
        return dfs[0]

    # Combine multiple CSVs
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Combined {len(dfs)} files into {len(combined)} rows")
    return combined


def preprocess_data(
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Preprocess the coffee quality dataset.

    Steps:
    1. Load raw data
    2. Select relevant columns
    3. Drop rows with missing quality scores
    4. Normalize numeric features
    5. Split into train/test sets
    6. Save processed data and scaler

    Args:
        test_size: Fraction of data for test set (default 0.2).
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary with paths to saved files:
        - train_path: Path to training parquet
        - test_path: Path to test parquet
        - scaler_path: Path to scaler pickle
    """
    project_root = get_project_root()
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Load raw data
    print("Loading raw data...")
    df = load_raw_data()
    print(f"Loaded {len(df)} rows")

    # Check for required columns
    required_cols = TASTE_FEATURES + [TARGET_COLUMN]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Select columns to keep
    cols_to_keep = TASTE_FEATURES + [TARGET_COLUMN]
    for col in METADATA_COLUMNS:
        if col in df.columns:
            cols_to_keep.append(col)

    df = df[cols_to_keep].copy()
    print(f"Selected {len(cols_to_keep)} columns")

    # Report missing values before dropping
    quality_cols = TASTE_FEATURES + [TARGET_COLUMN]
    missing_before = df[quality_cols].isna().sum()
    rows_with_missing = df[quality_cols].isna().any(axis=1).sum()
    print(f"Rows with missing quality scores: {rows_with_missing}")

    if missing_before.sum() > 0:
        print("Missing values per column:")
        for col, count in missing_before[missing_before > 0].items():
            print(f"  {col}: {count}")

    # Drop rows with missing quality scores
    df_clean = df.dropna(subset=quality_cols)
    dropped = len(df) - len(df_clean)
    print(f"Dropped {dropped} rows with missing quality scores")
    print(f"Remaining: {len(df_clean)} rows")

    if len(df_clean) == 0:
        raise ValueError("No data remaining after dropping missing values")

    # Split into features and target
    X = df_clean[TASTE_FEATURES].values
    y = df_clean[TARGET_COLUMN].values
    metadata = df_clean[[c for c in METADATA_COLUMNS if c in df_clean.columns]]

    # Train/test split
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X,
        y,
        df_clean.index.values,
        test_size=test_size,
        random_state=random_state,
    )

    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    # Fit scaler on training data only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Features normalized (mean={X_train_scaled.mean():.4f}, std={X_train_scaled.std():.4f})")

    # Create DataFrames with scaled features
    train_df = pd.DataFrame(X_train_scaled, columns=TASTE_FEATURES)
    train_df[TARGET_COLUMN] = y_train

    # Add metadata using original indices
    for col in metadata.columns:
        train_df[col] = metadata.loc[idx_train, col].values

    test_df = pd.DataFrame(X_test_scaled, columns=TASTE_FEATURES)
    test_df[TARGET_COLUMN] = y_test

    for col in metadata.columns:
        test_df[col] = metadata.loc[idx_test, col].values

    # Save processed data
    train_path = processed_dir / "train.parquet"
    test_path = processed_dir / "test.parquet"
    scaler_path = processed_dir / "scaler.pkl"

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print(f"\nSaved processed data:")
    print(f"  Train: {train_path}")
    print(f"  Test: {test_path}")
    print(f"  Scaler: {scaler_path}")

    return {
        "train_path": train_path,
        "test_path": test_path,
        "scaler_path": scaler_path,
    }


def load_processed_data() -> dict[str, Any]:
    """
    Load preprocessed data from data/processed/.

    Returns:
        Dictionary containing:
        - train_df: Training DataFrame
        - test_df: Test DataFrame
        - scaler: Fitted StandardScaler
        - taste_features: List of taste feature column names
        - target_column: Name of target column

    Raises:
        FileNotFoundError: If processed data doesn't exist.
    """
    project_root = get_project_root()
    processed_dir = project_root / "data" / "processed"

    train_path = processed_dir / "train.parquet"
    test_path = processed_dir / "test.parquet"
    scaler_path = processed_dir / "scaler.pkl"

    # Check all files exist
    for path in [train_path, test_path, scaler_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Processed data not found: {path}. Run `uv run preprocess` first."
            )

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return {
        "train_df": train_df,
        "test_df": test_df,
        "scaler": scaler,
        "taste_features": TASTE_FEATURES,
        "target_column": TARGET_COLUMN,
    }


def main() -> None:
    """Entry point for `uv run preprocess`."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess the CQI coffee quality dataset"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data for test set (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    preprocess_data(test_size=args.test_size, random_state=args.seed)


if __name__ == "__main__":
    main()
