"""Deterministic RawFeatures → MusicSpec fusion (rule-based).

Implements the heuristic table from ARCHITECTURE.md §5.2.
This is the default fuser and must always work offline.
"""
from __future__ import annotations

import random

from imgtune.core.config import settings
from imgtune.core.schemas import MusicSpec, RawFeatures

# Neutral defaults for low-confidence axes

_DEFAULTS: dict[str, str] = {
    "energy": "moderate",
    "valence": "neutral",
    "time_of_day": "afternoon",
    "setting": "nature",
    "era_feel": "modern",
    "weather": "clear",
}


def fuse(features: RawFeatures, seed: int | None = None) -> MusicSpec:
    """Convert validated *features* into a MusicSpec."""
    if seed is None:
        seed = random.randint(0, 2**31)

    color = features.color
    axes = features.axes
    conf: dict[str, float] = {}

    def axis_top(name: str) -> str:
        ax = axes.get(name)
        if ax is None or ax.confidence < settings.min_axis_confidence:
            conf[name] = ax.confidence if ax else 0.0
            return _DEFAULTS[name]
        conf[name] = ax.confidence
        return ax.top

    energy = axis_top("energy")
    valence = axis_top("valence")
    time_of_day = axis_top("time_of_day")
    setting_val = axis_top("setting")
    era_feel = axis_top("era_feel")
    weather = axis_top("weather")

    # Tempo from energy
    tempo_map = {
        "calm": 60, "relaxed": 75, "moderate": 100,
        "energetic": 125, "intense": 145,
    }
    tempo_bpm = tempo_map.get(energy, 100)

    # Mode from valence + lightness
    mode = _derive_mode(valence, color.mean_lightness)

    # Genre from setting × era × energy (never from CLIP directly)
    genre = _derive_genre(setting_val, era_feel, energy)

    # Mood adjectives (1-3)
    mood = _derive_mood(valence, energy, weather)

    # Key (deterministic from image hash)
    key_options = ["C", "D", "E", "F", "G", "A", "Bb", "Eb"]
    key = key_options[hash(features.image_hash) % len(key_options)]

    # Instrumentation from saturation + warm_ratio
    instrumentation = _derive_instruments(color, era_feel)

    # Texture from edge_density
    if color.edge_density < 0.15:
        texture = "sparse"
    elif color.edge_density < 0.30:
        texture = "moderate"
    elif color.edge_density < 0.50:
        texture = "dense"
    else:
        texture = "wall_of_sound"

    # Dynamics from contrast
    if color.contrast < 0.3:
        dynamics = "soft and steady"
    elif color.contrast < 0.6:
        dynamics = "gentle swells"
    else:
        dynamics = "wide dynamic range"

    return MusicSpec(
        genre=genre,
        mood=mood,
        tempo_bpm=tempo_bpm,
        key=key,
        mode=mode,
        instrumentation=instrumentation,
        texture=texture,
        dynamics=dynamics,
        duration_s=settings.default_duration_s,
        seed=seed,
        source_confidence=conf,
    )


# Private helpers


def _derive_mode(valence: str, lightness: float) -> str:
    if valence == "melancholic" or (valence == "somber" and lightness < 0.3):
        return "minor"
    if valence == "somber":
        return "dorian"
    if valence == "joyful" or lightness > 0.7:
        return "major"
    if valence == "hopeful":
        return "mixolydian"
    if lightness > 0.6:
        return "lydian"
    return "major"


def _derive_genre(setting: str, era: str, energy: str) -> str:
    if era in ("ancient", "vintage"):
        return "classical" if energy in ("calm", "relaxed") else "folk"
    if era == "retro":
        return "funk" if setting == "urban" else "classic rock"
    if setting == "urban" and energy in ("energetic", "intense"):
        return "electronic"
    if setting == "nature" and energy in ("calm", "relaxed"):
        return "ambient"
    if setting == "coastal":
        return "indie folk"
    if setting == "indoor" and energy == "calm":
        return "jazz"
    if energy == "intense":
        return "cinematic"
    return "indie"


def _derive_mood(valence: str, energy: str, weather: str) -> list[str]:
    valence_moods = {
        "melancholic": "melancholic", "somber": "dark",
        "neutral": "contemplative", "hopeful": "uplifting",
        "joyful": "bright",
    }
    moods = [valence_moods.get(valence, "contemplative")]

    if energy in ("energetic", "intense"):
        moods.append("driving")
    elif energy == "calm":
        moods.append("peaceful")

    if weather in ("rainy", "foggy"):
        moods.append("atmospheric")
    elif weather == "stormy":
        moods.append("dramatic")

    return moods[:3]


def _derive_instruments(color, era: str) -> list[str]:
    instruments: list[str] = []

    # warm_ratio → acoustic vs synthetic
    if color.warm_ratio > 0.5:
        instruments.extend(["strings", "acoustic guitar"])
    else:
        instruments.extend(["synthesizer", "electric piano"])

    # saturation → timbre character
    if color.mean_saturation < 0.3:
        instruments.append("felt piano")
    elif color.mean_saturation > 0.7:
        instruments.append("brass")
    else:
        instruments.append("pad")

    # era → rhythm source
    instruments.append("drum machine" if era in ("modern", "futuristic") else "drums")

    return instruments[:5]
