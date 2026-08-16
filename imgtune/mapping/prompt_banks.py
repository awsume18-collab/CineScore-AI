"""CLIP class vocabularies for zero-shot scoring.

Each axis has a set of mutually exclusive classes and shared prompt
templates.  BANK_VERSION must be bumped on any edit so that cached
features are automatically invalidated.
"""

BANK_VERSION: str = "1.0.0"

# Axis class sets (mutually exclusive within each axis)

AXES: dict[str, list[str]] = {
    "time_of_day": [
        "dawn",
        "morning",
        "afternoon",
        "golden hour",
        "dusk",
        "night",
    ],
    "weather": [
        "clear",
        "cloudy",
        "rainy",
        "foggy",
        "stormy",
        "snowy",
    ],
    "setting": [
        "urban",
        "rural",
        "nature",
        "indoor",
        "coastal",
        "mountain",
    ],
    "energy": [
        "calm",
        "relaxed",
        "moderate",
        "energetic",
        "intense",
    ],
    "valence": [
        "melancholic",
        "somber",
        "neutral",
        "hopeful",
        "joyful",
    ],
    "era_feel": [
        "ancient",
        "vintage",
        "retro",
        "modern",
        "futuristic",
    ],
}

# Prompt templates (5-8 per class, embedding-level ensembled)
# {cls} is substituted with the class label before encoding.

TEMPLATES: list[str] = [
    "a photo that feels {cls}",
    "an image with a {cls} atmosphere",
    "a {cls} scene",
    "this picture looks {cls}",
    "a photograph depicting something {cls}",
    "a {cls} visual mood",
    "a picture conveying a sense of {cls}",
    "an image that evokes {cls}",
]
