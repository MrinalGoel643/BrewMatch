# Code written with assistance from Claude Opus 4.5 (Anthropic)
"""Flask API for BrewMatch coffee recommendations."""

import logging
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler

from brewmatch.models import (
    ClassicalMLRecommender,
    NaiveBaselineRecommender,
    NeuralRecommender,
)
from brewmatch.models.base import BaseRecommender

from .schemas import (
    TASTE_FEATURES,
    VALID_MODELS,
    ValidationError,
    validate_coffee_id,
    validate_recommend_request,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Model type mapping
MODEL_CLASSES: dict[str, type[BaseRecommender]] = {
    "baseline": NaiveBaselineRecommender,
    "classical": ClassicalMLRecommender,
    "neural": NeuralRecommender,
}

# File extensions for each model type
MODEL_EXTENSIONS: dict[str, str] = {
    "baseline": ".pkl",
    "classical": ".pkl",
    "neural": ".pt",
}

# Ordered taste feature names matching BaseRecommender.TASTE_FEATURES
_TASTE_FEATURE_NAMES = [
    "Aroma", "Flavor", "Aftertaste", "Acidity", "Body",
    "Balance", "Uniformity", "Clean Cup", "Sweetness",
]

# API field names → model feature names
_API_TO_FEATURE = {
    "aroma": "Aroma",
    "flavor": "Flavor",
    "aftertaste": "Aftertaste",
    "acidity": "Acidity",
    "body": "Body",
    "balance": "Balance",
    "uniformity": "Uniformity",
    "clean_cup": "Clean Cup",
    "sweetness": "Sweetness",
}


def load_models(checkpoint_dir: Path) -> dict[str, BaseRecommender]:
    """Load all available models from the checkpoint directory.

    Args:
        checkpoint_dir: Path to the directory containing model checkpoints.

    Returns:
        Dictionary mapping model names to loaded model instances.
    """
    models: dict[str, BaseRecommender] = {}

    if not checkpoint_dir.exists():
        logger.warning(f"Checkpoint directory does not exist: {checkpoint_dir}")
        return models

    for model_name, model_class in MODEL_CLASSES.items():
        extension = MODEL_EXTENSIONS[model_name]
        model_path = checkpoint_dir / f"{model_name}{extension}"

        if model_path.exists():
            try:
                logger.info(f"Loading {model_name} model from {model_path}")
                models[model_name] = model_class.load(model_path)
                logger.info(f"Successfully loaded {model_name} model")
            except Exception as e:
                logger.error(f"Failed to load {model_name} model: {e}")
        else:
            logger.info(f"No checkpoint found for {model_name} at {model_path}")

    return models


def load_scaler(checkpoint_dir: Path) -> StandardScaler | None:
    """Load the fitted StandardScaler from the processed data directory.

    The scaler is fit on the original 0-10 raw taste scores and must be
    applied to user-provided preferences before passing them to any model,
    since all models were trained on scaled data.

    Args:
        checkpoint_dir: Path to the model checkpoints directory. The scaler
            is looked up relative to the project root at data/processed/scaler.pkl.

    Returns:
        Fitted StandardScaler, or None if not found.
    """
    # scaler lives at project_root/data/processed/scaler.pkl
    # checkpoint_dir is project_root/models/checkpoints
    project_root = checkpoint_dir.parent.parent
    scaler_path = project_root / "data" / "processed" / "scaler.pkl"

    if not scaler_path.exists():
        logger.warning(
            f"Scaler not found at {scaler_path}. User preference scaling will be "
            "skipped — run `uv run preprocess` to generate it."
        )
        return None

    try:
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        logger.info(f"Loaded preference scaler from {scaler_path}")
        return scaler
    except Exception as e:
        logger.error(f"Failed to load scaler: {e}")
        return None


def scale_preferences(
    preferences_array: np.ndarray,
    scaler: StandardScaler | None,
) -> np.ndarray:
    """Apply the preprocessing scaler to raw user preference values.

    All models were trained on StandardScaler-transformed data. User inputs
    arrive on the original 0-10 scale and must be transformed to match the
    training distribution before calling model.recommend().

    Args:
        preferences_array: Raw user preferences of shape (9,) on 0-10 scale.
        scaler: Fitted StandardScaler. If None, preferences are returned unchanged
            (degraded mode — recommendations will be less accurate).

    Returns:
        Scaled preferences array of shape (9,).
    """
    if scaler is None:
        return preferences_array

    return scaler.transform(preferences_array.reshape(1, -1)).squeeze().astype(np.float32)


def create_app(config: dict[str, Any] | None = None) -> Flask:
    """Create and configure the Flask application.

    Args:
        config: Optional configuration dictionary. Supported keys:
            - CHECKPOINT_DIR: Path to model checkpoints directory.
            - TESTING: Enable testing mode.
            - DEBUG: Enable debug mode.

    Returns:
        Configured Flask application instance.
    """
    app = Flask(__name__)

    # Apply configuration
    if config:
        app.config.update(config)

    # Enable CORS for all routes
    CORS(app)

    # Determine checkpoint directory
    checkpoint_dir = app.config.get("CHECKPOINT_DIR")
    if checkpoint_dir:
        checkpoint_dir = Path(checkpoint_dir)
    else:
        # Default to models/checkpoints relative to project root
        # Path: app.py -> api -> brewmatch -> src -> project_root
        checkpoint_dir = Path(__file__).parent.parent.parent.parent / "models" / "checkpoints"

    # Load models on startup
    app.models: dict[str, BaseRecommender] = load_models(checkpoint_dir)

    # Load the preprocessing scaler so user inputs can be correctly scaled
    app.scaler: StandardScaler | None = load_scaler(checkpoint_dir)

    # Store coffee data reference (populated when first model is loaded)
    app.coffee_data: dict[int, dict[str, Any]] = {}

    # Helper to convert numpy types to Python native for JSON serialization
    def to_python_native(obj):
        """Convert numpy types to Python native types."""
        if hasattr(obj, "item"):  # numpy scalar
            return obj.item()
        if isinstance(obj, dict):
            return {k: to_python_native(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [to_python_native(v) for v in obj]
        return obj

    # Build coffee data index from loaded models
    if app.models:
        first_model = next(iter(app.models.values()))
        if hasattr(first_model, "_metadata") and first_model._metadata is not None:
            for idx in range(len(first_model._metadata)):
                row = first_model._metadata.iloc[idx]
                app.coffee_data[idx] = {
                    "id": idx,
                    "metadata": to_python_native(row.to_dict()),
                }
                # Add taste profile if available
                if hasattr(first_model, "_X") and first_model._X is not None:
                    taste_profile = first_model._X[idx]
                    app.coffee_data[idx]["taste_profile"] = {
                        feature.lower().replace(" ", "_"): float(taste_profile[i])
                        for i, feature in enumerate(BaseRecommender.TASTE_FEATURES)
                    }

    @app.errorhandler(ValidationError)
    def handle_validation_error(error: ValidationError):
        """Handle validation errors with proper JSON response."""
        response = {"error": error.message}
        if error.field:
            response["field"] = error.field
        return jsonify(response), 400

    @app.errorhandler(404)
    def handle_not_found(error):
        """Handle 404 errors."""
        return jsonify({"error": "Resource not found"}), 404

    @app.errorhandler(500)
    def handle_internal_error(error):
        """Handle internal server errors."""
        logger.exception("Internal server error")
        return jsonify({"error": "Internal server error"}), 500

    @app.route("/health", methods=["GET"])
    def health_check():
        """Health check endpoint.

        Returns:
            JSON response with status and loaded models count.
        """
        return jsonify({
            "status": "healthy",
            "models_loaded": len(app.models),
            "available_models": list(app.models.keys()),
            "scaler_loaded": app.scaler is not None,
        })

    @app.route("/api/models", methods=["GET"])
    def list_models():
        """List available recommendation models.

        Returns:
            JSON response with list of available models and their status.
        """
        models_info = []
        for model_name in VALID_MODELS:
            model_info = {
                "name": model_name,
                "available": model_name in app.models,
            }
            if model_name in app.models:
                model = app.models[model_name]
                model_info["is_fitted"] = model.is_fitted
            models_info.append(model_info)

        return jsonify({"models": models_info})

    @app.route("/api/recommend", methods=["POST"])
    def get_recommendations():
        """Get coffee recommendations based on taste preferences.

        User preferences are provided on the 0-10 scale. They are internally
        scaled to match the training data distribution before inference.

        Request body:
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

        Returns:
            JSON response with list of recommended coffees.
        """
        data = request.get_json(silent=True)
        validated = validate_recommend_request(data)

        model_name = validated["model"]
        preferences = validated["preferences"]
        k = validated["k"]

        # Check if requested model is available
        if model_name not in app.models:
            available = list(app.models.keys())
            if not available:
                return jsonify({
                    "error": "No models are currently loaded",
                }), 503
            return jsonify({
                "error": f"Model '{model_name}' is not available",
                "available_models": available,
            }), 400

        model = app.models[model_name]

        # Build raw preference array in the order models expect (9 features)
        raw_preferences = np.array([
            preferences[feature.lower().replace(" ", "_")]
            for feature in BaseRecommender.TASTE_FEATURES
        ], dtype=np.float32)

        # Scale preferences to match training data distribution.
        # Models were trained on StandardScaler-transformed data (mean≈0, std≈1).
        # Without this step, raw 0-10 user inputs are far outside the training
        # distribution, producing incorrect recommendations.
        scaled_preferences = scale_preferences(raw_preferences, app.scaler)

        try:
            recommendations = model.recommend(scaled_preferences, k=k)
        except Exception as e:
            logger.exception("Error generating recommendations")
            return jsonify({"error": f"Failed to generate recommendations: {str(e)}"}), 500

        # Format response (convert numpy types to Python native for JSON serialization)
        formatted_recommendations = []
        for rec in recommendations:
            formatted_rec = {
                "id": int(rec["index"]),
                "similarity": float(rec["score"]),
                "scores": {
                    key.lower().replace(" ", "_"): float(value)
                    for key, value in rec["taste_profile"].items()
                },
            }
            # Add metadata fields at top level for convenience
            if rec.get("metadata"):
                formatted_rec["country"] = rec["metadata"].get("Country of Origin", "Unknown")
                formatted_rec["metadata"] = {
                    k: (v.item() if hasattr(v, "item") else v)
                    for k, v in rec["metadata"].items()
                }

            formatted_recommendations.append(formatted_rec)

        return jsonify({
            "recommendations": formatted_recommendations,
            "model_used": model_name,
            "k": k,
        })

    @app.route("/api/explain", methods=["POST"])
    def explain_recommendation():
        """Explain why a specific coffee was (or would be) recommended.

        Given a user's taste preferences and a coffee ID, returns a per-feature
        breakdown showing how closely the coffee matches each preference dimension.
        This is model-agnostic — it works on raw taste profiles.

        Request body:
            {
                "coffee_id": 42,
                "preferences": {
                    "aroma": 8.0,
                    "flavor": 7.5,
                    ...
                }
            }

        Returns:
            JSON response with:
            - overall_similarity: Cosine similarity between preferences and coffee
            - feature_breakdown: Per-feature match details
            - best_matches: Features with smallest gap (closest match)
            - biggest_gaps: Features with largest gap (worst match)
        """
        data = request.get_json(silent=True)

        if not data or not isinstance(data, dict):
            return jsonify({"error": "Request body must be a JSON object"}), 400

        # Validate coffee_id
        coffee_id_raw = data.get("coffee_id")
        if coffee_id_raw is None:
            return jsonify({"error": "coffee_id is required", "field": "coffee_id"}), 400
        try:
            coffee_id = validate_coffee_id(coffee_id_raw)
        except ValidationError as e:
            return jsonify({"error": e.message, "field": e.field}), 400

        if coffee_id not in app.coffee_data:
            return jsonify({"error": f"Coffee with id {coffee_id} not found"}), 404

        # Validate preferences using existing schema validation
        try:
            validated = validate_recommend_request({
                "preferences": data.get("preferences"),
                "model": "neural",  # dummy — not used here
                "k": 1,
            })
        except ValidationError as e:
            return jsonify({"error": e.message, "field": e.field}), 400

        preferences = validated["preferences"]
        coffee = app.coffee_data[coffee_id]

        if "taste_profile" not in coffee:
            return jsonify({"error": "Taste profile not available for this coffee"}), 404

        coffee_profile = coffee["taste_profile"]

        # Build ordered arrays for vector operations
        feature_keys = [f.lower().replace(" ", "_") for f in _TASTE_FEATURE_NAMES]
        user_vec = np.array([preferences[k] for k in feature_keys], dtype=np.float64)
        coffee_vec = np.array([coffee_profile[k] for k in feature_keys], dtype=np.float64)

        # Cosine similarity on raw 0-10 profiles
        user_norm = np.linalg.norm(user_vec)
        coffee_norm = np.linalg.norm(coffee_vec)
        if user_norm > 0 and coffee_norm > 0:
            overall_similarity = float(np.dot(user_vec, coffee_vec) / (user_norm * coffee_norm))
        else:
            overall_similarity = 0.0

        # Per-feature breakdown
        feature_breakdown = []
        gaps = []
        for key, display_name in zip(feature_keys, _TASTE_FEATURE_NAMES):
            user_val = preferences[key]
            coffee_val = coffee_profile[key]
            gap = abs(user_val - coffee_val)
            # Match percentage: 100% when gap=0, 0% when gap=10
            match_pct = round(max(0.0, (1.0 - gap / 10.0)) * 100, 1)
            feature_breakdown.append({
                "feature": display_name,
                "your_preference": round(user_val, 2),
                "coffee_score": round(coffee_val, 2),
                "match_pct": match_pct,
                "gap": round(gap, 2),
            })
            gaps.append((display_name, gap))

        # Sort features by gap to identify best/worst matches
        gaps.sort(key=lambda x: x[1])
        best_matches = [name for name, _ in gaps[:3]]
        biggest_gaps = [name for name, _ in gaps[-3:][::-1]]

        return jsonify({
            "coffee_id": coffee_id,
            "coffee_info": {
                "country": coffee["metadata"].get("Country of Origin", "Unknown"),
                "processing_method": coffee["metadata"].get("Processing Method", "Unknown"),
                "variety": coffee["metadata"].get("Variety", "Unknown"),
            },
            "overall_similarity": round(overall_similarity, 4),
            "feature_breakdown": feature_breakdown,
            "best_matches": best_matches,
            "biggest_gaps": biggest_gaps,
        })

    @app.route("/api/coffee/<int:coffee_id>", methods=["GET"])
    def get_coffee(coffee_id: int):
        """Get details for a specific coffee by ID.

        Args:
            coffee_id: The ID of the coffee to retrieve.

        Returns:
            JSON response with coffee details.
        """
        validated_id = validate_coffee_id(coffee_id)

        if validated_id not in app.coffee_data:
            return jsonify({"error": f"Coffee with id {validated_id} not found"}), 404

        coffee = app.coffee_data[validated_id]
        return jsonify(coffee)

    @app.route("/api/stats", methods=["GET"])
    def get_stats():
        """Get model performance statistics.

        Returns:
            JSON response with statistics about loaded models and data.
        """
        stats: dict[str, Any] = {
            "total_coffees": len(app.coffee_data),
            "models": {},
        }

        for model_name, model in app.models.items():
            model_stats: dict[str, Any] = {
                "is_fitted": model.is_fitted,
            }
            if hasattr(model, "_X") and model._X is not None:
                model_stats["training_samples"] = len(model._X)
            if hasattr(model, "_metadata") and model._metadata is not None:
                model_stats["metadata_columns"] = list(model._metadata.columns)
            stats["models"][model_name] = model_stats

        return jsonify(stats)

    return app


def main() -> None:
    """Entry point for running the BrewMatch API server.

    By default, runs with gunicorn (production mode).
    Use --dev for Flask's built-in development server with auto-reload.

    Usage:
        uv run serve           # Production mode (gunicorn)
        uv run serve --dev     # Development mode (Flask built-in)
        uv run serve --port 5000 --dev  # Dev mode on custom port
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="BrewMatch API server")
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Run Flask's built-in development server (auto-reload, debug mode)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.environ.get("FLASK_HOST", "127.0.0.1"),
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("FLASK_PORT", "8000")),
        help="Port to bind to (default: 8000)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("GUNICORN_WORKERS", "4")),
        help="Number of gunicorn workers (default: 4, production only)",
    )
    args = parser.parse_args()

    if args.dev:
        # Development mode: Flask's built-in server with auto-reload
        logger.info(f"Starting BrewMatch API (dev mode) on {args.host}:{args.port}")
        app = create_app()
        app.run(host=args.host, port=args.port, debug=True)
    else:
        # Production mode: gunicorn
        try:
            from gunicorn.app.base import BaseApplication
        except ImportError:
            logger.error("gunicorn not installed. Use --dev for development server.")
            sys.exit(1)

        class StandaloneApplication(BaseApplication):
            def __init__(self, app, options=None):
                self.options = options or {}
                self.application = app
                super().__init__()

            def load_config(self):
                for key, value in self.options.items():
                    if key in self.cfg.settings and value is not None:
                        self.cfg.set(key.lower(), value)

            def load(self):
                return self.application

        app = create_app()
        options = {
            "bind": f"{args.host}:{args.port}",
            "workers": args.workers,
            "accesslog": "-",
            "errorlog": "-",
            "loglevel": "info",
        }

        logger.info(
            f"Starting BrewMatch API (gunicorn) on {args.host}:{args.port} "
            f"with {args.workers} workers"
        )
        StandaloneApplication(app, options).run()


if __name__ == "__main__":
    main()
