"""Flask API for BrewMatch coffee recommendations."""

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

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

    # Store coffee data reference (populated when first model is loaded)
    app.coffee_data: dict[int, dict[str, Any]] = {}

    # Build coffee data index from loaded models
    if app.models:
        first_model = next(iter(app.models.values()))
        if hasattr(first_model, "_metadata") and first_model._metadata is not None:
            for idx in range(len(first_model._metadata)):
                row = first_model._metadata.iloc[idx]
                app.coffee_data[idx] = {
                    "id": idx,
                    "metadata": row.to_dict(),
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

        # Convert preferences dict to numpy array in correct order
        # Map API field names to model feature names
        feature_mapping = {
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

        preferences_array = np.array([
            preferences[feature.lower().replace(" ", "_")]
            for feature in BaseRecommender.TASTE_FEATURES
        ], dtype=np.float32)

        try:
            recommendations = model.recommend(preferences_array, k=k)
        except Exception as e:
            logger.exception("Error generating recommendations")
            return jsonify({"error": f"Failed to generate recommendations: {str(e)}"}), 500

        # Format response
        formatted_recommendations = []
        for rec in recommendations:
            formatted_rec = {
                "id": rec["index"],
                "similarity": rec["score"],
                "scores": {
                    key.lower().replace(" ", "_"): value
                    for key, value in rec["taste_profile"].items()
                },
            }
            # Add metadata fields at top level for convenience
            if rec.get("metadata"):
                formatted_rec["country"] = rec["metadata"].get("Country of Origin", "Unknown")
                formatted_rec["metadata"] = rec["metadata"]

            formatted_recommendations.append(formatted_rec)

        return jsonify({
            "recommendations": formatted_recommendations,
            "model_used": model_name,
            "k": k,
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
    """Entry point for running the Flask development server.

    This function is called by `uv run serve`. For production deployments,
    use a WSGI server like gunicorn instead.
    """
    host = os.environ.get("FLASK_HOST", "127.0.0.1")
    port = int(os.environ.get("FLASK_PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"

    logger.info(f"Starting BrewMatch API server on {host}:{port}")

    app = create_app()
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()
