"""Orchestrates vision extractors → validated RawFeatures."""
from __future__ import annotations

import logging

import cv2
import imagehash
import numpy as np
from PIL import Image

from imgtune.core.config import settings
from imgtune.core.schemas import AxisScores, RawFeatures
from imgtune.mapping.prompt_banks import AXES, BANK_VERSION
from imgtune.vision.captioner import caption_image
from imgtune.vision.clip_scorer import score_axes
from imgtune.vision.color_stats import compute_color_stats

logger = logging.getLogger(__name__)


def extract_features(
    image: Image.Image,
    clip_model=None,
    clip_preprocess=None,
    clip_tokenizer=None,
    blip_model=None,
    blip_processor=None,
    device: str | None = None,
) -> RawFeatures:
    """Run all extractors on *image* and return validated RawFeatures."""
    device = device or settings.device

    # pHash
    image_hash = str(imagehash.phash(image))

    # Colour stats (always succeeds, no ML)
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    color = compute_color_stats(img_bgr)

    # CLIP scoring
    try:
        axes = score_axes(
            image, clip_model, clip_preprocess, clip_tokenizer, device,
        )
    except Exception as exc:
        logger.warning("CLIP extraction failed, using neutral axes: %s", exc)
        axes = _neutral_axes()

    # BLIP-2 captioning
    try:
        caption = caption_image(image, blip_model, blip_processor, device)
    except Exception as exc:
        logger.warning("BLIP captioning failed: %s", exc)
        caption = "an image"

    return RawFeatures(
        image_hash=image_hash,
        caption=caption,
        axes=axes,
        color=color,
        extractor_versions={
            "clip": settings.clip_model,
            "blip": settings.blip_model,
            "prompt_bank": BANK_VERSION,
            "color_stats": "1.0",
        },
    )


def _neutral_axes() -> dict[str, AxisScores]:
    """Uniform distributions — used when CLIP fails."""
    result: dict[str, AxisScores] = {}
    for axis_name, classes in AXES.items():
        n = len(classes)
        dist = {cls: 1.0 / n for cls in classes}
        result[axis_name] = AxisScores(
            distribution=dist,
            top=classes[0],
            confidence=0.0,
        )
    return result
