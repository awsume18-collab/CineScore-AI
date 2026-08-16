"""Typed exception hierarchy for Cinescore."""


class CinescoreError(Exception):
    """Base exception for all Cinescore errors."""


class ExtractionError(CinescoreError):
    """Raised when vision extraction fails."""


class FusionError(CinescoreError):
    """Raised when feature-to-spec fusion fails."""


class GenerationError(CinescoreError):
    """Raised when MusicGen audio generation fails."""


class PostProcessError(CinescoreError):
    """Raised when audio post-processing fails."""
