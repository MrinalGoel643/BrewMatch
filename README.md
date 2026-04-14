# BrewMatch

A machine learning-powered coffee recommendation system that matches users with coffee beans based on their taste
preferences. Built for the Computer Vision module project, this system implements three distinct modeling approaches and
includes a production-ready Flask API.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Data Pipeline](#data-pipeline)
- [Models](#models)
- [Evaluation](#evaluation)
- [Experiment: Sensitivity Analysis](#experiment-sensitivity-analysis)
- [API Reference](#api-reference)
- [Frontend](#frontend)
- [Deployment](#deployment)

## Overview

BrewMatch recommends coffee beans by learning taste profile similarities from the Coffee Quality Institute (CQI)dataset.
Given a user's preferred taste characteristics (aroma, flavor, acidity, body, etc.), the system finds coffees with
matching profiles.

### Key Features

- **Three modeling approaches**: Naive baseline, classical ML (KNN), and deep learning (neural embeddings)
- **Comprehensive evaluation**: Precision@K, Recall@K, NDCG@K, Graded NDCG, MRR, Coverage, MSE, MAE
- **Error analysis**: Identifies mispredictions, patterns, and mitigation strategies
- **Sensitivity analysis experiment**: Measures performance vs. training set size
- **Production-ready API**: Flask REST API with validation, error handling, and explainability
- **Web interface**: React/Vite frontend for interactive recommendations

### Taste Profile Features

The system uses 9 sensory evaluation scores (0-10 scale):

| Feature    | Description                                            |
|------------|--------------------------------------------------------|
| Aroma      | Scent/fragrance of the coffee                          |
| Flavor     | Overall taste including sweetness, bitterness, acidity |
| Aftertaste | Lingering taste after swallowing                       |
| Acidity    | Brightness and liveliness of taste                     |
| Body       | Thickness/viscosity of the coffee                      |
| Balance    | How well flavor components work together               |
| Uniformity | Consistency from cup to cup                            |
| Clean Cup  | Absence of off-flavors or defects                      |
| Sweetness  | Caramel-like, fruity, or floral notes                  |

## Installation

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- GPU (optional): NVIDIA CUDA or Apple Silicon MPS for faster training
- Kaggle account (for dataset download)

### Setup

1. **Clone the repository**

```bash
git clone https://github.com/MrinalGoel643/BrewMatch.git
cd BrewMatch
```

2. **Install dependencies**

```bash
# CPU-only or Apple Silicon (MPS)
uv sync
   
# With NVIDIA CUDA support
uv sync --extra cuda
  ```

3. **Configure Kaggle credentials**

Create `~/.kaggle/kaggle.json` with your API credentials:

```json
{
  "username": "your_username",
  "key": "your_api_key"
}
```

Get your API key from [Kaggle Account Settings](https://www.kaggle.com/settings/account).

## Quick Start

```bash
# 1. Download the CQI coffee dataset
uv run download

# 2. Preprocess the data
uv run preprocess

# 3. Train all models (with default hyperparameters)
uv run train

# 4. OR: Tune hyperparameters first, then train (recommended)
uv run train --tune

# 5. Evaluate model performance
uv run evaluate --error-analysis

# 6. Start the API server
uv run serve
```

## Project Structure

```
brewmatch/
├── pyproject.toml                 # Project config and dependencies
├── README.md                      # This file
├── data/
│   ├── raw/                       # Downloaded CSV files
│   └── processed/                 # Train/test parquet + scaler
├── models/
│   └── checkpoints/               # Saved model files
├── experiments/                   # Experiment results and plots
├── frontend/                      # React/Vite web interface
│   ├── src/
│   │   ├── components/            # React components
│   │   └── App.jsx                # Main application
│   └── package.json
├── notebooks/                     # Jupyter notebooks
└── src/brewmatch/
    ├── __init__.py
    ├── config.py                  # Configuration settings
    ├── device.py                  # Device detection (CUDA/MPS/CPU)
    ├── utils.py                   # Utility functions
    ├── train.py                   # Training script (includes Optuna tuning)
    ├── evaluate.py                # Evaluation script
    ├── experiment.py              # Sensitivity analysis
    ├── data/
    │   ├── __init__.py
    │   ├── download.py            # Kaggle dataset downloader
    │   ├── preprocess.py          # Data cleaning and splitting
    │   └── dataset.py             # PyTorch Dataset classes
    ├── models/
    │   ├── __init__.py
    │   ├── base.py                # Abstract base class
    │   ├── baseline.py            # Naive baseline recommender
    │   ├── classical.py           # KNN/cosine similarity
    │   └── neural.py              # Neural embedding model
    ├── evaluation/
    │   ├── __init__.py
    │   ├── metrics.py             # Ranking and regression metrics
    │   └── error_analysis.py      # Error pattern detection
    └── api/
        ├── __init__.py
        ├── app.py                 # Flask application
        └── schemas.py             # Request validation
```

## Data Pipeline

### Download Dataset

```bash
uv run download [--force]
```

Downloads the [CQI Coffee Quality Database](https://www.kaggle.com/datasets/volpatto/coffee-quality-database-from-cqi)
from Kaggle
to `data/raw/`. This dataset contains ~1,340 coffee samples (Arabica + Robusta) with sensory evaluations.

| Option    | Description                     |
|-----------|---------------------------------|
| `--force` | Re-download even if data exists |

### Preprocess Data

```bash
uv run preprocess [--test-size 0.2] [--seed 42]
```

Processes raw data and creates train/test splits:

1. Loads CSV files from `data/raw/`
2. Selects taste features and metadata columns
3. Drops rows with missing quality scores
4. Normalizes features using StandardScaler (fit on train only)
5. Splits data 80/20 train/test
6. Saves to `data/processed/`:
    - `train.parquet` - Training data
    - `test.parquet` - Test data
    - `scaler.pkl` - Fitted scaler for inference

| Option        | Description                                   |
|---------------|-----------------------------------------------|
| `--test-size` | Fraction for test set (default: 0.2)          |
| `--seed`      | Random seed for reproducibility (default: 42) |

## Models

### 1. Naive Baseline (`NaiveBaselineRecommender`)

Establishes a performance floor using simple heuristics.

**Strategies:**

- `mean`: Recommends coffees closest to the global mean taste profile (ignores user preferences)
- `weighted_random`: Random sampling weighted by Total Cup Points

**When to use:** Sanity check; any useful model should beat this.

### 2. Classical ML (`ClassicalMLRecommender`)

Uses traditional similarity-based methods.

**Methods:**

- `knn`: K-Nearest Neighbors with Euclidean distance (sklearn NearestNeighbors)
- `cosine`: Cosine similarity ranking

**Features:**

- Optional feature normalization via StandardScaler
- Configurable number of neighbors

**When to use:** Fast inference, interpretable results, works well with small datasets.

### 3. Neural Network (`NeuralRecommender`)

Learns taste embeddings via contrastive learning.

**Architecture:**

- MLP encoder with residual connections
- Maps 9 taste features to 32-dimensional embedding space
- L2-normalized embeddings for cosine similarity

**Training:**

- Triplet loss with margin
- AdamW optimizer with cosine annealing
- Automatic positive/negative mining based on taste distance

**When to use:** Best performance with sufficient data; captures non-linear relationships.

### Training Models

```bash
uv run train [--models baseline classical neural] [--device cuda]
```

| Option     | Description                                                  |
|------------|--------------------------------------------------------------|
| `--models` | Models to train: `baseline`, `classical`, `neural`, or `all` |
| `--device` | PyTorch device: `cuda` or `cpu` (auto-detected)              |

Models are saved to `models/checkpoints/`:

- `baseline.pkl` - Pickled baseline model
- `classical.pkl` - Pickled KNN model
- `neural.pt` - PyTorch neural model

## Hyperparameter Tuning

BrewMatch includes automated hyperparameter optimization using [Optuna](https://optuna.org/), a Bayesian optimization
framework with tree-structured Parzen estimators (TPE). Tuning is integrated into the training script.

### Training Workflow

```bash
# First run: uses default hyperparameters
uv run train

# Run with Optuna tuning (saves best params for future runs)
uv run train --tune

# Subsequent runs: automatically uses previously tuned hyperparameters
uv run train

# Re-tune anytime with --tune flag
uv run train --tune --neural-trials 100
```

| Option               | Description                                                    |
|----------------------|----------------------------------------------------------------|
| `--tune`             | Run Optuna tuning before training                              |
| `--models`           | Models to train/tune: `baseline`, `classical`, `neural`, `all` |
| `--neural-trials`    | Number of Optuna trials for neural network (default: 50)       |
| `--classical-trials` | Number of Optuna trials for classical ML (default: 30)         |
| `--cv-folds`         | Cross-validation folds for tuning (default: 3)                 |
| `--device`           | PyTorch device: `cuda`, `mps`, or `cpu` (auto-detected)        |

### Tuned Hyperparameters

**Neural Network:**

- `embedding_dim`: Embedding space dimension (16-128)
- `hidden_dim`: Hidden layer size (32-256)
- `learning_rate`: Adam learning rate (1e-4 to 1e-2, log scale)
- `margin`: Triplet loss margin (0.1-1.0)
- `batch_size`: Training batch size (16, 32, 64, 128)

**Classical ML:**

- `method`: Similarity method (`knn` or `cosine`)
- `n_neighbors`: Number of neighbors for KNN (5-100)
- `normalize`: Feature normalization (True/False)

### Outputs

Tuned hyperparameters are saved to `models/checkpoints/hyperparameters.json` and automatically loaded on subsequent
training runs

## Evaluation

### Metrics

| Metric            | Description                                                          |
|-------------------|----------------------------------------------------------------------|
| **Precision@K**   | Proportion of top-K recommendations that are relevant                |
| **Recall@K**      | Proportion of relevant items found in top-K                          |
| **NDCG@K**        | Normalized Discounted Cumulative Gain (rewards early relevant items) |
| **Graded NDCG@K** | NDCG using continuous cosine similarity as relevance (not binary)    |
| **MRR**           | Mean Reciprocal Rank (rank position of first relevant item)          |
| **Coverage**      | Fraction of catalog recommended at least once (measures diversity)   |
| **MSE**           | Mean Squared Error of taste profile predictions                      |
| **MAE**           | Mean Absolute Error of taste profile predictions                     |

**Relevance definition:** A coffee is relevant if it has cosine similarity >= 0.80, OR shares the same country of
origin.

### Running Evaluation

```bash
uv run evaluate [--models all] [--error-analysis] [--output results.json]
```

| Option             | Description                                                     |
|--------------------|-----------------------------------------------------------------|
| `--models`         | Models to evaluate: `baseline`, `classical`, `neural`, or `all` |
| `--error-analysis` | Generate detailed error analysis                                |
| `--output`         | Save results to JSON file                                       |

### Error Analysis

The error analysis module identifies:

1. **5 Worst Mispredictions** with root cause analysis:
    - Origin mismatch
    - Processing method mismatch
    - Large taste profile deviations

2. **Common Error Patterns**:
    - Failures by country of origin
    - Failures by processing method
    - Cross-origin confusion (e.g., confusing Ethiopia with Kenya)
    - Taste profile edge cases (high acidity, low body)

3. **Mitigation Strategies**:
    - Origin-aware embeddings
    - Processing method features
    - Contrastive learning for confused origins
    - Re-ranking stages

## Experiment: Sensitivity Analysis

Investigates how model performance varies with training set size.

### Hypothesis

Deep learning models benefit more from additional data, while classical models plateau earlier.

### Running the Experiment

```bash
uv run experiment [--fractions 0.1 0.2 ... 1.0] [--trials 3] [--device cuda]
```

| Option         | Description                                              |
|----------------|----------------------------------------------------------|
| `--fractions`  | Training set fractions to test (default: 0.1 to 1.0)     |
| `--trials`     | Trials per fraction for variance estimation (default: 3) |
| `--device`     | PyTorch device                                           |
| `--output-dir` | Directory for results (default: `experiments/`)          |

### Outputs

- `raw_results.json` - Per-trial metrics
- `aggregated_results.csv` - Mean and std per model/fraction
- `sensitivity_analysis.png` - Performance vs. training size plot
- `sensitivity_analysis_multi.png` - Multi-metric comparison
- `experiment_report.txt` - Text summary with findings

## API Reference

### Starting the Server

```bash
uv run serve
```

Or with environment variables:

```bash
FLASK_HOST=0.0.0.0 FLASK_PORT=8000 FLASK_DEBUG=true uv run serve
```

### Endpoints

#### Health Check

```http
GET /health
```

**Response:**

```json
{
  "status": "healthy",
  "models_loaded": 3,
  "available_models": [
    "baseline",
    "classical",
    "neural"
  ]
}
```

#### List Models

```http
GET /api/models
```

**Response:**

```json
{
  "models": [
    {
      "name": "baseline",
      "available": true,
      "is_fitted": true
    },
    {
      "name": "classical",
      "available": true,
      "is_fitted": true
    },
    {
      "name": "neural",
      "available": true,
      "is_fitted": true
    }
  ]
}
```

#### Get Recommendations

```http
POST /api/recommend
Content-Type: application/json

{
  "preferences": {
    "aroma": 8.0,
    "flavor": 7.5,
    "aftertaste": 7.0,
    "acidity": 7.5,
    "body": 8.0,
    "balance": 7.5,
    "uniformity": 10.0,
    "clean_cup": 10.0,
    "sweetness": 10.0
  },
  "model": "neural",
  "k": 5
}
```

**Response:**

```json
{
  "recommendations": [
    {
      "id": 42,
      "similarity": 0.95,
      "scores": {
        "aroma": 7.92,
        "flavor": 7.58
      },
      "country": "Ethiopia",
      "metadata": {}
    }
  ],
  "model_used": "neural",
  "k": 5
}
```

| Field         | Type    | Description                                                        |
|---------------|---------|--------------------------------------------------------------------|
| `preferences` | object  | Required. All 9 taste features (0-10 scale)                        |
| `model`       | string  | Optional. `baseline`, `classical`, or `neural` (default: `neural`) |
| `k`           | integer | Optional. Number of recommendations (1-100, default: 5)            |

#### Explain Recommendation

```http
POST /api/explain
Content-Type: application/json

{
  "coffee_id": 42,
  "preferences": {
    "aroma": 8.0,
    "flavor": 7.5,
    "aftertaste": 7.0,
    "acidity": 7.5,
    "body": 8.0,
    "balance": 7.5,
    "uniformity": 10.0,
    "clean_cup": 10.0,
    "sweetness": 10.0
  }
}
```

**Response:**

```json
{
  "coffee_id": 42,
  "coffee_info": {
    "country": "Ethiopia",
    "processing_method": "Washed / Wet",
    "variety": "Bourbon"
  },
  "overall_similarity": 0.9521,
  "feature_breakdown": [
    {
      "feature": "Aroma",
      "your_preference": 8.0,
      "coffee_score": 7.92,
      "match_pct": 99.2,
      "gap": 0.08
    }
  ],
  "best_matches": [
    "Sweetness",
    "Uniformity",
    "Clean Cup"
  ],
  "biggest_gaps": [
    "Acidity",
    "Body",
    "Balance"
  ]
}
```

This endpoint explains why a specific coffee was (or would be) recommended by showing a per-feature breakdown of how
closely it matches your preferences.

#### Get Coffee Details

```http
GET /api/coffee/{id}
```

**Response:**

```json
{
  "id": 42,
  "metadata": {
    "Country.of.Origin": "Ethiopia",
    "Processing.Method": "Washed / Wet"
  },
  "taste_profile": {
    "aroma": 7.92
  }
}
```

#### Get Statistics

```http
GET /api/stats
```

**Response:**

```json
{
  "total_coffees": 1200,
  "models": {
    "baseline": {
      "is_fitted": true,
      "training_samples": 960
    },
    "classical": {
      "is_fitted": true,
      "training_samples": 960
    },
    "neural": {
      "is_fitted": true,
      "training_samples": 960
    }
  }
}
```

### Error Responses

| Status | Description                               |
|--------|-------------------------------------------|
| 400    | Validation error (missing/invalid fields) |
| 404    | Resource not found                        |
| 503    | No models loaded                          |
| 500    | Internal server error                     |

## Frontend

A React-based web interface for interacting with the BrewMatch API.

### Running Locally

```bash
cd frontend
npm install
npm run dev
```

The development server runs at `http://localhost:5173` by default.

### Building for Production

```bash
cd frontend
npm run build
npm run preview  # Preview the production build
```

The frontend is configured for deployment on Vercel via `vercel.json`.

## Deployment

### Production with Gunicorn

```bash
uv run gunicorn "brewmatch.api.app:create_app()" \
  --bind 0.0.0.0:8000 \
  --workers 4 \
  --timeout 120
```

### Docker

```dockerfile
FROM python:3.13-slim

WORKDIR /app
COPY . .

RUN pip install uv && uv sync --frozen

# Download and preprocess data, train models
RUN uv run download && uv run preprocess && uv run train

EXPOSE 8000
CMD ["uv", "run", "gunicorn", "brewmatch.api.app:create_app()", "--bind", "0.0.0.0:8000"]
```

### Environment Variables

| Variable      | Description         | Default     |
|---------------|---------------------|-------------|
| `FLASK_HOST`  | Server bind address | `127.0.0.1` |
| `FLASK_PORT`  | Server port         | `5000`      |
| `FLASK_DEBUG` | Enable debug mode   | `false`     |

---

**Dataset:** [Coffee Quality Database (CQI)](https://www.kaggle.com/datasets/volpatto/coffee-quality-database-from-cqi)
by Diego Volpatto
