"""Configuration settings for BrewMatch."""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"

# Ensure directories exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, CHECKPOINTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Dataset settings
KAGGLE_DATASET = "fatihb/coffee-quality-data-cqi"
RANDOM_SEED = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1

# Feature columns (taste profile) - using actual CSV column names
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

# Metadata columns to preserve
METADATA_COLUMNS = [
    "Country of Origin",
    "Region",
    "Processing Method",
    "Variety",
    "Color",
    "Total Cup Points",
]

# Model hyperparameters
BASELINE_CONFIG = {
    "strategy": "mean_similarity",  # or "quality_weighted_random"
}

CLASSICAL_CONFIG = {
    "n_neighbors": 10,
    "metric": "cosine",
    "algorithm": "brute",
}

NEURAL_CONFIG = {
    "embedding_dim": 32,
    "hidden_dims": [64, 32],
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100,
    "margin": 0.5,  # for triplet loss
    "patience": 10,  # early stopping
}

# Evaluation settings
K_VALUES = [1, 3, 5, 10]

# API settings
API_HOST = "0.0.0.0"
API_PORT = 5000
DEBUG = False
