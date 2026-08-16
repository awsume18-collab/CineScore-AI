# ARCHITECTURE.md

Reference document for the image-to-music generation system. Read before writing code.
Update this file when a decision changes — it is the source of truth, not the code.

---

## 1. What this system does

Takes an image, extracts fuzzy perceptual attributes (mood, genre lean, time of day,
energy, colour character), converts them into a structured musical description, and
generates an audio clip that matches.

---

## 2. Core architectural principle

> **Text and structured JSON are the only bridge between vision and audio.**

There is no learned embedding path from CLIP space to MusicGen's conditioning space in
v1. Every stage communicates through an inspectable, serialisable artifact.

The pipeline is:

```
Image
  → RawFeatures     (three independent extractors, JSON)
  → MusicSpec       (fused, canonical, versioned JSON)  ← THE CONTRACT
  → PromptText      (rendered string, user-editable)
  → RawAudio        (MusicGen output, 32kHz mono/stereo)
  → FinalAudio      (normalised, faded, encoded)
```

**Why this matters:** any stage can be swapped, mocked, or debugged in isolation.
If output sounds wrong, you can point at exactly which artifact is wrong.

### The MusicSpec is the contract

Everything upstream of `MusicSpec` is "perception". Everything downstream is
"synthesis". Neither side may reach across. Specifically:

- The prompt builder **never** sees CLIP logits or the raw image.
- The extractors **never** know that MusicGen exists.
- If you need a new musical control, it goes in `MusicSpec` first, then both sides
  are updated. Never smuggle data through a side channel.

---

## 3. Repository layout

```
imgtune/
├── core/
│   ├── schemas.py          # Pydantic models: RawFeatures, MusicSpec, JobRecord
│   ├── config.py           # Pydantic Settings, env-driven. No literals elsewhere.
│   └── errors.py           # Typed exception hierarchy
├── vision/
│   ├── clip_scorer.py      # Zero-shot axis scoring
│   ├── captioner.py        # BLIP-2 wrapper
│   ├── color_stats.py      # Pure numpy/OpenCV, no ML, no GPU
│   └── extract.py          # Orchestrates the three, returns RawFeatures
├── mapping/
│   ├── prompt_banks.py     # CLIP class vocabularies, versioned
│   ├── rules.py            # Deterministic RawFeatures → MusicSpec
│   ├── llm_fuser.py        # Optional LLM fuser, same signature as rules.py
│   └── prompt_builder.py   # MusicSpec → prompt string
├── audio/
│   ├── musicgen.py         # Model wrapper, generation, continuation
│   └── postprocess.py      # Loudness, fades, time-stretch, encoding
├── api/
│   ├── main.py             # FastAPI app, routes only
│   ├── routes/
│   └── deps.py
├── worker/
│   ├── runner.py           # Queue consumer, model warm-up
│   └── tasks.py
├── eval/
│   ├── golden/             # 50 labelled images + expected attribute ranges
│   ├── run_golden.py
│   └── metrics.py          # ImageBind / CLAP scoring
├── tests/
├── web/                    # Frontend
└── ARCHITECTURE.md, OBJECTIVES.md
```

**Rule:** `vision/`, `mapping/`, and `audio/` must be importable and runnable without
FastAPI, Redis, or any network. They are libraries. `api/` and `worker/` are the only
places that know about infrastructure.

---

## 4. Canonical schemas

### RawFeatures

```python
class AxisScores(BaseModel):
    distribution: dict[str, float]   # sums to 1.0
    top: str
    confidence: float                # top prob minus second prob, 0..1

class ColorStats(BaseModel):
    mean_lightness: float            # 0..1
    mean_saturation: float           # 0..1
    contrast: float                  # 0..1, std of L channel normalised
    edge_density: float              # 0..1, Canny edge pixel ratio
    warm_ratio: float                # 0..1, warm-hue pixel share
    color_entropy: float             # 0..1, normalised Shannon entropy
    dominant_colors: list[tuple[int, int, int]]   # 5 RGB triples

class RawFeatures(BaseModel):
    schema_version: Literal["1.0"]
    image_hash: str                  # perceptual hash (pHash)
    caption: str
    axes: dict[str, AxisScores]      # keys: time_of_day, weather, setting,
                                     # energy, valence, era_feel
    color: ColorStats
    extractor_versions: dict[str, str]
```

### MusicSpec

```python
class MusicSpec(BaseModel):
    schema_version: Literal["1.0"]
    genre: str
    mood: list[str]                  # 1-3 adjectives
    tempo_bpm: int                   # 40..180
    key: str                         # "C".."B", with optional "#"/"b"
    mode: Literal["major","minor","dorian","lydian","mixolydian","phrygian"]
    instrumentation: list[str]       # 2-5 items, concrete instrument names
    texture: Literal["sparse","moderate","dense","wall_of_sound"]
    dynamics: str                    # short phrase
    duration_s: int                  # 10..120
    seed: int
    source_confidence: dict[str, float]
    spec_hash: str                   # sha256 of the above, minus seed
```

**`spec_hash` is the audio cache key.** `seed` is excluded so a re-roll produces a new
track; include it if you want re-rolls cached too.

---

## 5. Stage rules

### 5.1 Vision extractors

- **CLIP scoring:** one softmax **per axis**, over a mutually exclusive class set.
  Never pool classes from different axes into one similarity comparison — the scores
  are uncalibrated and cross-axis comparison is meaningless.
- Use prompt ensembling: 5–8 templates per class, average the *text embeddings*
  (not the scores) before comparison.
- Prompt banks live in `mapping/prompt_banks.py` with a `BANK_VERSION` string. Bump it
  on any edit; it goes into `extractor_versions` so cached features are invalidated.
- **BLIP-2:** greedy decode, max 40 tokens. Long captions add noise, not signal.
- **Colour stats:** pure function, deterministic, no model. Must run in <100ms on a
  1024px image. Resize to 512px longest edge first.

### 5.2 Fusion

- `rules.fuse(features) -> MusicSpec` and `llm_fuser.fuse(features) -> MusicSpec` share
  an identical signature. Selected by config, not by import.
- Rule-based fuser is the default and must always work offline.
- **Low-confidence handling:** if an axis confidence is below `MIN_AXIS_CONFIDENCE`
  (default 0.15), do not use its top label. Fall back to the neutral default for that
  axis and record it in `source_confidence`.
- The LLM fuser must validate its output against the `MusicSpec` Pydantic model and
  retry once on failure, then fall back to the rule-based fuser. Never ship an
  unvalidated LLM JSON blob downstream.

**Baseline mapping heuristics** (starting point, tune against the golden set):

| Image signal | Musical target |
|---|---|
| `mean_lightness` low | lower register, minor/dorian lean |
| `mean_lightness` high | higher register, major/lydian lean |
| `mean_saturation` low | acoustic, muted, felt/brushed timbres |
| `mean_saturation` high | synthetic, bright, saturated timbres |
| `contrast` | dynamic range descriptor |
| `edge_density` | rhythmic density, `texture` field |
| `warm_ratio` high | strings, brass, wood, tape warmth |
| `warm_ratio` low | pads, synths, glass, bells |
| `color_entropy` | harmonic complexity, extended chords |
| `energy` axis | `tempo_bpm` primary driver |
| `valence` axis | `mode` primary driver |
| `setting` × `era_feel` × `energy` | `genre` |

Genre is **never** classified by CLIP directly. Derive it.

### 5.3 Prompt building

- Output 15–30 words of natural description. MusicGen degrades on longer prompts.
- Never serialise the JSON into the prompt.
- Tempo goes in as a soft nudge ("slow, around 75 BPM"), not a hard instruction.
- Key and mode are **not** included in the prompt string — MusicGen ignores them.
  They exist in the spec for the symbolic/MIDI path and for UI display.
- The rendered prompt is returned to the client and is user-editable. An edited prompt
  bypasses stages 1–3 on regeneration.

### 5.4 Generation

- Default model: `musicgen-small` in dev, `musicgen-stereo-medium` in prod. Set by
  config; never hardcode a checkpoint name outside `config.py`.
- Fixed defaults: `guidance_scale=3.0`, `top_k=250`, `temperature=1.0`. Any sweep
  results that change these must be recorded in this file with the date and the golden
  set score that justified it.
- `duration_s > 30` uses continuation with a 20s stride and 10s overlap. Document
  expected drift in the UI; do not offer >90s in v1.
- Seed is always explicit and always logged.

### 5.5 Post-processing

Fixed chain, in order: peak-safety limiter → loudness normalise to −14 LUFS →
optional time-stretch to `tempo_bpm` → 200ms in/out fades → encode.

Encode to Opus 96kbps for streaming and keep the WAV in object storage for download.

---

## 6. Runtime architecture

```
Client → FastAPI → Redis queue → GPU worker → Object storage → CDN
                ↑                     │
                └── SSE progress ─────┘
```

- **Generation is never synchronous.** `POST /generate` returns a job id immediately.
- Vision extraction runs on the API side (or a cheap CPU worker) and returns within
  ~1–2s so the client can display features while audio renders.
- Worker loads all models once at boot into module-level singletons. Cold-loading per
  job is a bug, not a performance issue.
- One worker process per GPU. No forking after model load.

### Caching

| Cache | Key | TTL |
|---|---|---|
| Features | `pHash(image) + extractor_versions` | 30 days |
| Audio | `spec_hash` | 7 days |

Store both in Redis; audio bytes in object storage with the Redis entry holding the URL.

### Failure policy

- Extractor failure → fall back to colour-stats-only spec, flag `degraded: true`.
- Fusion failure → rule-based fuser.
- Generation failure → retry once with a new seed, then fail the job with a typed error.
- Never return silence or a placeholder track as if it were a real result.

---

## 7. Development rules

1. **Config only in `core/config.py`.** No magic numbers, model names, thresholds, or
   URLs anywhere else. If you need a constant, it goes in config with a default.
2. **Every stage is a pure function where possible.** `f(input_artifact) -> output_artifact`.
   Side effects (S3, Redis, logging) live in `api/` and `worker/` only.
3. **Determinism is testable.** Given the same image, same config, same seed, the
   pipeline must produce byte-identical audio. Add a test that asserts this.
4. **Schema changes bump `schema_version` and invalidate caches.** No silent migrations.
5. **Log the full artifact chain per job** (RawFeatures, MusicSpec, prompt, seed,
   model version) as structured JSON. You cannot debug audio quality without this.
6. **No model loading in tests.** Vision and audio wrappers must accept an injectable
   model object; tests use fakes. Only `tests/integration/` touches real weights, and
   it is opt-in via a marker.
7. **Type hints everywhere, Pydantic at every boundary.** `mypy --strict` on `core/`
   and `mapping/` at minimum.
8. **No `print`.** Use structured logging with the job id in context.
9. **Pin every dependency.** `audiocraft`, `transformers`, and `torch` versions
   interact badly; a lockfile is not optional.

---

## 8. Constraints and known limits

Record these in the UI copy so expectations are set:

- MusicGen produces **no usable vocals**. Do not promise songs with singing.
- Long-form output drifts. Clips beyond ~45s lose thematic coherence.
- Exact key and BPM are **not** controllable via text conditioning. BPM is enforced in
  post via time-stretch when `ENFORCE_TEMPO=true`; key is display-only.
- Abstract or flat images produce low-confidence features. Detect via mean axis
  confidence below `DEGENERATE_THRESHOLD` and prompt the user for a text hint.

### Licensing — resolve before any commercial launch

- `audiocraft` **code** is MIT.
- MusicGen **pretrained weights** are CC-BY-NC 4.0 — **non-commercial only**.
- This blocks monetisation. Options: license negotiation, Stable Audio Open (check its
  own terms), or training/fine-tuning owned weights.
- **Do not build billing until this is resolved.** Track it as a hard blocker.

### Safety

- NSFW classifier on every upload before any model sees it.
- Decide and document the policy on images of identifiable people before launch.
- Strip EXIF (including GPS) on ingest. Never persist original uploads longer than the
  job TTL unless the user explicitly opts in.

---

## 9. Deferred to later phases

Listed here so they are not accidentally designed out:

- Learned CLIP→T5 projection bridge (needs paired training data).
- Melody conditioning via `musicgen-melody` chromagram input.
- Symbolic/MIDI generation path for exact key, tempo, and structure control.
- Multi-image sequences → evolving track sections.
- User-adjustable spec sliders with regeneration.
