"""MusicSpec → natural-language prompt string.

Output is 15-30 words of natural description.
Key and mode are NOT included (MusicGen ignores them).
Tempo is a soft nudge, not a hard instruction.
"""
from __future__ import annotations

from imgtune.core.schemas import MusicSpec


def build_prompt(spec: MusicSpec) -> str:
    """Render a human-readable generation prompt from *spec*."""
    mood_str = ", ".join(spec.mood)
    instr_str = " and ".join(spec.instrumentation[:3])

    # Soft tempo nudge
    if spec.tempo_bpm < 80:
        tempo_hint = f"slow, around {spec.tempo_bpm} BPM"
    elif spec.tempo_bpm < 110:
        tempo_hint = f"mid-tempo, around {spec.tempo_bpm} BPM"
    else:
        tempo_hint = f"upbeat, around {spec.tempo_bpm} BPM"

    parts = [
        f"{mood_str} {spec.genre}",
        f"with {instr_str}",
        tempo_hint,
        f"{spec.texture} texture",
        spec.dynamics,
    ]

    prompt = ", ".join(parts)

    # Enforce 15-30 word limit
    words = prompt.split()
    if len(words) > 30:
        prompt = " ".join(words[:30])

    return prompt
