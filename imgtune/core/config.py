"""Centralised configuration. No magic numbers anywhere else."""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """All tunables live here, driven by environment variables."""

    # Model names
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"
    blip_model: str = "Salesforce/blip2-opt-2.7b"
    musicgen_model: str = "facebook/musicgen-small"

    # Generation parameters
    guidance_scale: float = 3.0
    top_k: int = 250
    temperature: float = 1.0
    default_duration_s: int = 30
    max_duration_s: int = 90

    # Thresholds
    min_axis_confidence: float = 0.15
    degenerate_threshold: float = 0.10

    # Post-processing
    enforce_tempo: bool = False
    target_lufs: float = -14.0
    fade_ms: int = 200

    # Paths
    output_dir: str = "outputs"

    # Device
    device: str = "cuda"

    model_config = {"env_prefix": "CINESCORE_", "env_file": ".env"}


settings = Settings()
