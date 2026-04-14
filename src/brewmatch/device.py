# Code written with assistance from Claude Opus 4.5 (Anthropic)
"""Device detection and selection utilities.

Provides automatic device selection with fallback:
CUDA (if available) > MPS (Apple Silicon) > CPU
"""

import torch


def get_device(preferred: str | None = None) -> torch.device:
    """
    Get the best available device for PyTorch operations.

    Priority: CUDA > MPS > CPU (unless preferred is specified)

    Args:
        preferred: Optional preferred device ("cuda", "mps", "cpu").
                   If specified and available, uses that device.
                   If not available, falls back to best available.

    Returns:
        torch.device for the selected device.
    """
    if preferred:
        preferred = preferred.lower()
        if preferred == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif preferred == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        elif preferred == "cpu":
            return torch.device("cpu")
        # Fall through to auto-detection if preferred not available

    # Auto-detect best available
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_device_info() -> dict[str, bool | str]:
    """
    Get information about available devices.

    Returns:
        Dictionary with device availability and selected device.
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "selected": str(get_device()),
    }

    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_device_count"] = torch.cuda.device_count()

    return info


def print_device_info() -> None:
    """Print device information to console."""
    info = get_device_info()
    print(f"Device: {info['selected']}")

    if info["cuda_available"]:
        print(f"  CUDA: {info.get('cuda_device_name', 'Unknown')} "
              f"(x{info.get('cuda_device_count', 1)})")
    elif info["mps_available"]:
        print("  MPS: Apple Silicon GPU")
    else:
        print("  CPU: No GPU acceleration available")
