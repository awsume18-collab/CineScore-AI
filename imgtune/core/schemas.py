"""Pydantic models: RawFeatures, MusicSpec, JobRecord."""
from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class AxisScores(BaseModel):
    """Per-axis CLIP zero-shot classification result."""
    distribution: dict[str, float]   # sums to 1.0
    top: str
    confidence: float                # top prob minus second prob, 0..1

    @field_validator("distribution")
    @classmethod
    def distribution_sums_to_one(cls, v: dict[str, float]) -> dict[str, float]:
        total = sum(v.values())
        if abs(total - 1.0) > 1e-3:
            # Renormalize instead of crashing
            factor = 1.0 / total if total > 0 else 1.0
            v = {k: val * factor for k, val in v.items()}
        return v


class ColorStats(BaseModel):
    """Pure-CV colour statistics extracted from the image."""
    mean_lightness: float = Field(ge=0.0, le=1.0)
    mean_saturation: float = Field(ge=0.0, le=1.0)
    contrast: float = Field(ge=0.0, le=1.0)
    edge_density: float = Field(ge=0.0, le=1.0)
    warm_ratio: float = Field(ge=0.0, le=1.0)
    color_entropy: float = Field(ge=0.0, le=1.0)
    dominant_colors: list[tuple[int, int, int]]   # 5 RGB triples


class RawFeatures(BaseModel):
    """Combined output of all vision extractors."""
    schema_version: Literal["1.0"] = "1.0"
    image_hash: str                  # perceptual hash (pHash)
    caption: str
    axes: dict[str, AxisScores]      # keys: time_of_day, weather, setting,
                                     # energy, valence, era_feel
    color: ColorStats
    extractor_versions: dict[str, str]


class MusicSpec(BaseModel):
    """The contract between perception and synthesis."""
    schema_version: Literal["1.0"] = "1.0"
    genre: str
    mood: list[str] = Field(min_length=1, max_length=3)
    tempo_bpm: int = Field(ge=40, le=180)
    key: str
    mode: Literal["major", "minor", "dorian", "lydian", "mixolydian", "phrygian"]
    instrumentation: list[str] = Field(min_length=2, max_length=5)
    texture: Literal["sparse", "moderate", "dense", "wall_of_sound"]
    dynamics: str
    duration_s: int = Field(ge=10, le=120)
    seed: int
    source_confidence: dict[str, float]
    spec_hash: str = ""

    def compute_spec_hash(self) -> str:
        """SHA-256 of all fields except seed and spec_hash."""
        data = self.model_dump(exclude={"seed", "spec_hash"})
        raw = json.dumps(data, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    def model_post_init(self, __context: object) -> None:
        if not self.spec_hash:
            object.__setattr__(self, "spec_hash", self.compute_spec_hash())


class JobStatus(str, Enum):
    """Lifecycle states for an async generation job."""
    PENDING = "pending"
    EXTRACTING = "extracting"
    GENERATING = "generating"
    POSTPROCESSING = "postprocessing"
    COMPLETE = "complete"
    FAILED = "failed"


class JobRecord(BaseModel):
    """Tracks a single generation job end-to-end."""
    job_id: str
    status: JobStatus = JobStatus.PENDING
    features: RawFeatures | None = None
    spec: MusicSpec | None = None
    prompt: str | None = None
    audio_url: str | None = None
    error: str | None = None
    degraded: bool = False
