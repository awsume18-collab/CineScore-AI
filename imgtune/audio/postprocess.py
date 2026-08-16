"""Audio post-processing chain.

Fixed order:
  peak-safety limiter → loudness normalise to −14 LUFS →
  clip guard → 200 ms in/out fades → encode WAV.
"""
from __future__ import annotations

import logging

import numpy as np
import pyloudnorm as pyln
import soundfile as sf

from imgtune.core.config import settings

logger = logging.getLogger(__name__)


def postprocess(
    audio: np.ndarray,
    sample_rate: int,
    output_path: str,
    tempo_bpm: int | None = None,
) -> str:
    """Apply the full post-processing chain and write to *output_path*.

    *audio* shape: ``(channels, samples)`` or ``(samples,)``.
    Returns the output path.
    """
    # Ensure float32
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    is_mono = audio.ndim == 1

    # 1. Peak-safety limiter (soft-clip to prevent harsh digital distortion)
    peak = float(np.max(np.abs(audio)))
    if peak > 0.0:
        # Normalize to 0.9 peak first to give headroom
        audio = audio * (0.9 / max(peak, 0.9))

    # 2. Soft clipping (tanh) to remove any remaining harsh transients
    # This prevents the crackling/breaking sounds
    audio = np.tanh(audio * 1.1) * 0.9

    # 3. Loudness normalise to target LUFS
    audio_for_meter = audio if is_mono else audio.T
    meter = pyln.Meter(sample_rate)
    try:
        loudness = meter.integrated_loudness(audio_for_meter)
        if np.isfinite(loudness) and loudness < 0:
            audio_for_meter = pyln.normalize.loudness(
                audio_for_meter, loudness, settings.target_lufs,
            )
            audio = audio_for_meter if is_mono else audio_for_meter.T
    except Exception as exc:
        logger.warning("Loudness normalisation skipped: %s", exc)

    # 4. Hard clip guard (absolute safety net after LUFS normalisation)
    audio = np.clip(audio, -0.95, 0.95)

    # 5. Fades
    fade_samples = int(settings.fade_ms / 1000.0 * sample_rate)
    if fade_samples > 0:
        fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
        fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)

        if is_mono:
            audio[:fade_samples] *= fade_in
            audio[-fade_samples:] *= fade_out
        else:
            for ch in range(audio.shape[0]):
                audio[ch, :fade_samples] *= fade_in
                audio[ch, -fade_samples:] *= fade_out

    # 6. Write WAV
    write_data = audio if is_mono else audio.T
    sf.write(output_path, write_data, sample_rate, subtype="FLOAT")
    logger.info("Wrote %s", output_path)

    return output_path
