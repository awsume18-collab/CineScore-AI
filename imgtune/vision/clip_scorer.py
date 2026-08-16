"""CLIP zero-shot axis scoring with embedding-level prompt ensembling."""
from __future__ import annotations

import logging

import open_clip
import torch
from PIL import Image

from imgtune.core.config import settings
from imgtune.core.schemas import AxisScores
from imgtune.mapping.prompt_banks import AXES, TEMPLATES

logger = logging.getLogger(__name__)


def load_clip_model(device: str | None = None):
    """Load CLIP model, transforms, and tokenizer.  Returns (model, preprocess, tokenizer, device)."""
    device = device or settings.device
    model, _, preprocess = open_clip.create_model_and_transforms(
        settings.clip_model, pretrained=settings.clip_pretrained,
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(settings.clip_model)
    return model, preprocess, tokenizer, device


def score_axes(
    image: Image.Image,
    model=None,
    preprocess=None,
    tokenizer=None,
    device: str | None = None,
) -> dict[str, AxisScores]:
    """Run per-axis softmax classification on *image*.

    Each axis is scored independently with its own class set,
    using averaged text embeddings per class (prompt ensembling).
    """
    if model is None:
        model, preprocess, tokenizer, device = load_clip_model(device)

    img_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(img_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    results: dict[str, AxisScores] = {}

    for axis_name, classes in AXES.items():
        class_embeddings = []
        for cls in classes:
            texts = [t.format(cls=cls) for t in TEMPLATES]
            tokens = tokenizer(texts).to(device)
            with torch.no_grad():
                text_feats = model.encode_text(tokens)
                text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            # Average *embeddings*, not scores (per §5.1)
            mean_feat = text_feats.mean(dim=0)
            mean_feat = mean_feat / mean_feat.norm()
            class_embeddings.append(mean_feat)

        class_embeddings_t = torch.stack(class_embeddings)
        similarities = (image_features @ class_embeddings_t.T).squeeze(0)
        # Temperature-scaled softmax (CLIP default logit scale ≈ 100)
        probs = torch.softmax(similarities * 100.0, dim=-1).cpu().numpy()

        distribution = {cls: float(p) for cls, p in zip(classes, probs)}
        sorted_probs = sorted(probs, reverse=True)
        top_cls = classes[int(probs.argmax())]
        confidence = (
            float(sorted_probs[0] - sorted_probs[1])
            if len(sorted_probs) > 1
            else 1.0
        )

        results[axis_name] = AxisScores(
            distribution=distribution,
            top=top_cls,
            confidence=confidence,
        )

    return results
