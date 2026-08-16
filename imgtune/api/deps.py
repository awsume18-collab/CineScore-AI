"""Dependency injection: model singletons and in-memory job store."""
from __future__ import annotations

import logging

from imgtune.core.config import settings
from imgtune.core.schemas import JobRecord

logger = logging.getLogger(__name__)

# In-memory job store (v1)
jobs: dict[str, JobRecord] = {}

# Model singletons (lazy-loaded once)
_models: dict[str, object] = {}


def get_clip_models():
    """Return (model, preprocess, tokenizer, device) — loaded once."""
    if "clip" not in _models:
        logger.info("Loading CLIP model …")
        from imgtune.vision.clip_scorer import load_clip_model
        _models["clip"] = load_clip_model()
    return _models["clip"]


def get_blip_models():
    """Return (model, processor, device) — loaded once."""
    if "blip" not in _models:
        logger.info("Loading BLIP-2 model …")
        from imgtune.vision.captioner import load_blip_model
        _models["blip"] = load_blip_model()
    return _models["blip"]


def get_musicgen_model():
    """Return MusicGen model — loaded once."""
    if "musicgen" not in _models:
        logger.info("Loading MusicGen model …")
        from imgtune.audio.musicgen import load_musicgen
        _models["musicgen"] = load_musicgen()
    return _models["musicgen"]
