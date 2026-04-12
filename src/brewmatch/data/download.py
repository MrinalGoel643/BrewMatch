"""Download the CQI coffee quality dataset from Kaggle."""

import shutil
from pathlib import Path

import kagglehub


def get_project_root() -> Path:
    """Get the project root directory (where pyproject.toml is located)."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not find project root (no pyproject.toml found)")


def download_data(force: bool = False) -> Path:
    """
    Download the CQI coffee quality dataset from Kaggle.

    Uses kagglehub to download the dataset and copies files to data/raw/.

    Args:
        force: If True, re-download even if data already exists.

    Returns:
        Path to the raw data directory containing the downloaded files.

    Raises:
        RuntimeError: If download fails or no CSV files are found.
    """
    project_root = get_project_root()
    raw_dir = project_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Check if data already exists
    existing_csvs = list(raw_dir.glob("*.csv"))
    if existing_csvs and not force:
        print(f"Data already exists in {raw_dir} ({len(existing_csvs)} CSV files)")
        print("Use force=True to re-download")
        return raw_dir

    print("Downloading CQI coffee quality dataset from Kaggle...")
    print("Dataset: volpatto/coffee-quality-database-from-cqi")

    # kagglehub downloads to its cache directory
    # Using volpatto's dataset which has both Arabica (~1300) and Robusta (~28) samples
    cache_path = kagglehub.dataset_download("volpatto/coffee-quality-database-from-cqi")
    cache_path = Path(cache_path)

    print(f"Downloaded to cache: {cache_path}")

    # Find all CSV files in the downloaded data
    csv_files = list(cache_path.glob("**/*.csv"))
    if not csv_files:
        raise RuntimeError(f"No CSV files found in downloaded data at {cache_path}")

    # Copy CSV files to raw directory
    print(f"Copying {len(csv_files)} CSV file(s) to {raw_dir}")
    for csv_file in csv_files:
        dest = raw_dir / csv_file.name
        shutil.copy2(csv_file, dest)
        print(f"  - {csv_file.name}")

    print(f"Data saved to {raw_dir}")
    return raw_dir


def main() -> None:
    """Entry point for `uv run download`."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download the CQI coffee quality dataset from Kaggle"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if data already exists",
    )
    args = parser.parse_args()

    download_data(force=args.force)


if __name__ == "__main__":
    main()
