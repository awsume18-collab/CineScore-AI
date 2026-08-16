"""BLIP-2 image captioner wrapper.  Greedy decode, max 40 tokens."""
from __future__ import annotations

import logging

import torch
from PIL import Image
from transformers import Blip2ForConditionalGeneration, Blip2Processor

from imgtune.core.config import settings

logger = logging.getLogger(__name__)


def load_blip_model(device: str | None = None):
    """Load BLIP-2 model and processor.  Returns (model, processor, device)."""
    device = device or settings.device
    dtype = torch.float16 if "cuda" in device else torch.float32
    processor = Blip2Processor.from_pretrained(settings.blip_model)
    model = Blip2ForConditionalGeneration.from_pretrained(
        settings.blip_model, torch_dtype=dtype,
    )
    model = model.to(device).eval()
    return model, processor, device


def caption_image(
    image: Image.Image,
    model=None,
    processor=None,
    device: str | None = None,
) -> str:
    """Return a short caption (≤40 tokens, greedy)."""
    if model is None:
        model, processor, device = load_blip_model(device)

    inputs = processor(image, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=False,  # greedy
        )
    caption = processor.decode(output_ids[0], skip_special_tokens=True).strip()
    return caption
