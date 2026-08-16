"""MusicGen model wrapper with continuation for long clips."""
from __future__ import annotations

import logging

import torch
from audiocraft.models import MusicGen

from imgtune.core.config import settings

logger = logging.getLogger(__name__)


def load_musicgen(device: str | None = None) -> MusicGen:
    """Load and return a MusicGen model singleton."""
    device = device or settings.device
    logger.info("Loading MusicGen model: %s on %s", settings.musicgen_model, device)
    model = MusicGen.get_pretrained(settings.musicgen_model, device=device)
    return model


def generate_audio(
    prompt: str,
    duration_s: int,
    seed: int,
    model: MusicGen | None = None,
    device: str | None = None,
) -> tuple[torch.Tensor, int]:
    """Generate audio from a text prompt.

    Returns (waveform_tensor, sample_rate).
    The tensor shape is ``(channels, samples)``.
    """
    if model is None:
        model = load_musicgen(device)

    torch.manual_seed(seed)

    if duration_s <= 30:
        model.set_generation_params(
            duration=duration_s,
            use_sampling=True,
            top_k=settings.top_k,
            temperature=settings.temperature,
            cfg_coef=settings.guidance_scale,
        )
        wav = model.generate([prompt])
    else:
        wav = _generate_with_continuation(model, prompt, duration_s, seed)

    sample_rate: int = model.sample_rate
    return wav[0], sample_rate


def _generate_with_continuation(
    model: MusicGen, prompt: str, total_duration: int, seed: int,
) -> torch.Tensor:
    """Continuation with 20 s stride / 10 s overlap for clips > 30 s."""
    stride = 20
    overlap = 10

    model.set_generation_params(
        duration=30,
        use_sampling=True,
        top_k=settings.top_k,
        temperature=settings.temperature,
        cfg_coef=settings.guidance_scale,
    )
    torch.manual_seed(seed)
    wav = model.generate([prompt])

    generated = 30
    while generated < total_duration:
        remaining = total_duration - generated + overlap
        next_dur = min(30, remaining)

        prompt_wav = wav[:, :, -overlap * model.sample_rate :]
        model.set_generation_params(
            duration=next_dur,
            use_sampling=True,
            top_k=settings.top_k,
            temperature=settings.temperature,
            cfg_coef=settings.guidance_scale,
        )
        continuation = model.generate_continuation(
            prompt_wav, model.sample_rate, [prompt],
        )
        new_part = continuation[:, :, overlap * model.sample_rate :]
        wav = torch.cat([wav, new_part], dim=-1)
        generated += stride

    # Trim to exact duration
    target_samples = total_duration * model.sample_rate
    wav = wav[:, :, :target_samples]
    return wav
