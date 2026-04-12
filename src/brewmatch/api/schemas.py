"""Request/response validation schemas for the BrewMatch API."""

from typing import Any

# The 9 taste preference features
TASTE_FEATURES = [
    "aroma",
    "flavor",
    "aftertaste",
    "acidity",
    "body",
    "balance",
    "uniformity",
    "clean_cup",
    "sweetness",
]

# Valid model names
VALID_MODELS = ["baseline", "classical", "neural"]


class ValidationError(Exception):
    """Raised when request validation fails."""

    def __init__(self, message: str, field: str | None = None) -> None:
        self.message = message
        self.field = field
        super().__init__(message)


def validate_preferences(preferences: dict[str, Any] | None) -> dict[str, float]:
    """Validate taste preferences from API request.

    Args:
        preferences: Dictionary of taste preferences with feature names as keys
            and scores as values. All 9 features must be present.

    Returns:
        Validated preferences dictionary with float values.

    Raises:
        ValidationError: If preferences are invalid.
    """
    if preferences is None:
        raise ValidationError("preferences is required", field="preferences")

    if not isinstance(preferences, dict):
        raise ValidationError(
            "preferences must be an object", field="preferences"
        )

    validated: dict[str, float] = {}
    missing_fields = []

    for feature in TASTE_FEATURES:
        if feature not in preferences:
            missing_fields.append(feature)
            continue

        value = preferences[feature]

        if value is None:
            raise ValidationError(
                f"{feature} cannot be null", field=f"preferences.{feature}"
            )

        try:
            float_value = float(value)
        except (TypeError, ValueError):
            raise ValidationError(
                f"{feature} must be a number", field=f"preferences.{feature}"
            )

        if float_value < 0.0 or float_value > 10.0:
            raise ValidationError(
                f"{feature} must be between 0 and 10", field=f"preferences.{feature}"
            )

        validated[feature] = float_value

    if missing_fields:
        raise ValidationError(
            f"Missing required fields: {', '.join(missing_fields)}",
            field="preferences",
        )

    return validated


def validate_model_name(model: str | None) -> str:
    """Validate model name from API request.

    Args:
        model: Name of the model to use. Must be one of 'baseline', 'classical',
            or 'neural'. Defaults to 'neural' if not provided.

    Returns:
        Validated model name.

    Raises:
        ValidationError: If model name is invalid.
    """
    if model is None:
        return "neural"

    if not isinstance(model, str):
        raise ValidationError("model must be a string", field="model")

    model = model.lower().strip()

    if model not in VALID_MODELS:
        raise ValidationError(
            f"model must be one of: {', '.join(VALID_MODELS)}", field="model"
        )

    return model


def validate_k(k: Any | None) -> int:
    """Validate k (number of recommendations) from API request.

    Args:
        k: Number of recommendations to return. Must be a positive integer
            between 1 and 100. Defaults to 5 if not provided.

    Returns:
        Validated k value.

    Raises:
        ValidationError: If k is invalid.
    """
    if k is None:
        return 5

    try:
        k_int = int(k)
    except (TypeError, ValueError):
        raise ValidationError("k must be an integer", field="k")

    if k_int < 1:
        raise ValidationError("k must be at least 1", field="k")

    if k_int > 100:
        raise ValidationError("k must be at most 100", field="k")

    return k_int


def validate_coffee_id(coffee_id: Any) -> int:
    """Validate coffee ID from API request.

    Args:
        coffee_id: ID of the coffee to retrieve.

    Returns:
        Validated coffee ID as integer.

    Raises:
        ValidationError: If coffee ID is invalid.
    """
    if coffee_id is None:
        raise ValidationError("coffee id is required", field="id")

    try:
        id_int = int(coffee_id)
    except (TypeError, ValueError):
        raise ValidationError("coffee id must be an integer", field="id")

    if id_int < 0:
        raise ValidationError("coffee id must be non-negative", field="id")

    return id_int


def validate_recommend_request(data: dict[str, Any] | None) -> dict[str, Any]:
    """Validate the full recommendation request body.

    Args:
        data: Request body dictionary containing preferences, model, and k.

    Returns:
        Validated request data with all fields populated.

    Raises:
        ValidationError: If request data is invalid.
    """
    if data is None:
        raise ValidationError("Request body is required")

    if not isinstance(data, dict):
        raise ValidationError("Request body must be a JSON object")

    return {
        "preferences": validate_preferences(data.get("preferences")),
        "model": validate_model_name(data.get("model")),
        "k": validate_k(data.get("k")),
    }
