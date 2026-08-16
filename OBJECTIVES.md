# OBJECTIVES.md

Ordered milestones. Each has a goal, tasks, and a **verification block** that must pass
before moving on. Do not start milestone N+1 with N's verification unchecked.

Legend: `[ ]` not started · `[~]` in progress · `[x]` verified

---

## M0 — Environment and skeleton

**Goal:** every dependency installs, every model loads, nothing is built yet.

### Tasks
- [ ] Repo scaffolded per `ARCHITECTURE.md` §3
- [ ] Pinned lockfile; `torch`, `transformers`, `audiocraft` versions compatible
- [ ] `core/config.py` with Pydantic Settings, `.env.example` committed
- [ ] Smoke script loads CLIP, BLIP-2, MusicGen-small and prints VRAM usage
- [ ] Pre-commit: ruff, black, mypy

### Verify
- [ ] Fresh clone → install → smoke script runs on target GPU with no manual fixes
- [ ] Peak VRAM recorded in README; all three models fit simultaneously
- [ ] Model load time measured and recorded (this becomes the worker cold-start budget)
- [ ] `mypy --strict core/` passes
- [ ] CI runs on push and is green

**Exit criterion:** a second developer can clone and run the smoke script in under 20 min.

---

## M1 — Vision extraction

**Goal:** image in, `RawFeatures` out, deterministic and fast.

### Tasks
- [ ] `color_stats.py` — all six scalars plus dominant colours
- [ ] `prompt_banks.py` — six axes, mutually exclusive classes, 5–8 templates each,
      `BANK_VERSION` constant
- [ ] `clip_scorer.py` — per-axis softmax, embedding-level prompt ensembling,
      returns full distributions plus confidence margin
- [ ] `captioner.py` — BLIP-2, greedy, 40-token cap
- [ ] `extract.py` — orchestrates, returns validated `RawFeatures`
- [ ] pHash computation for cache keys

### Verify
- [ ] Each axis distribution sums to 1.0 ± 1e-5 (unit test)
- [ ] Colour stats are pure: same image → identical output across 10 runs
- [ ] Colour stats run in <100ms on a 1024px image
- [ ] Full extraction completes in <2s on target hardware
- [ ] **Manual sanity pass:** run on 10 obviously-different images (bright beach, night
      city, foggy forest, dim interior, snow, neon, desert, rain, studio white, sunset).
      For each, the top `time_of_day` and `energy` labels are ones a human would pick.
      Record results in `eval/m1_sanity.md`. **Any obviously wrong label means the
      prompt bank is wrong — fix before proceeding.**
- [ ] Confidence margins are meaningfully lower on a deliberately ambiguous image
      (grey wall, close-up texture) than on a clear one
- [ ] `extractor_versions` populated and changes when `BANK_VERSION` changes

**Exit criterion:** you trust the features enough to build mapping on top of them.

---

## M2 — Golden set

**Goal:** a regression harness. Do this *before* tuning anything.

### Tasks
- [ ] 50 images collected, licence-clear, spanning the attribute space
- [ ] Each labelled by hand with expected axis values and acceptable alternates
- [ ] `eval/run_golden.py` — runs extraction, scores against labels, prints a report
- [ ] Scores committed as a baseline JSON

### Verify
- [ ] Coverage check: every class in every axis appears as the expected label for at
      least 2 images. Gaps mean untested classes.
- [ ] Harness runs end to end and produces a single top-line accuracy number per axis
- [ ] Baseline recorded with date, `BANK_VERSION`, and model versions
- [ ] Deliberately corrupting one prompt bank entry causes the score to drop —
      proving the harness actually detects regressions

**Exit criterion:** you can answer "did that change help?" with a number, not a vibe.

---

## M3 — Mapping and prompt building

**Goal:** `RawFeatures` → `MusicSpec` → prompt string.

### Tasks
- [ ] `MusicSpec` Pydantic model with all validators (BPM range, mode enum, list lengths)
- [ ] `rules.py` implementing the §5.2 heuristic table
- [ ] Low-confidence fallback per axis with `MIN_AXIS_CONFIDENCE`
- [ ] `prompt_builder.py` — template rendering, 15–30 words, no JSON leakage
- [ ] `spec_hash` computation

### Verify
- [ ] Every golden-set image produces a spec that passes validation — zero exceptions
- [ ] BPM correlates with the `energy` axis across the golden set (spot-check the
      extremes: calmest image is slower than the most energetic)
- [ ] Dark images skew minor/dorian, bright images skew major/lydian — check counts
      across the golden set, not one example
- [ ] All 50 prompts fall within 15–30 words
- [ ] No prompt contains `{`, `}`, `:`, or a key/mode name
- [ ] Same features → identical spec (excluding seed) across runs
- [ ] Artificially setting all confidences to zero produces the neutral default spec
      without crashing
- [ ] **Manual read-through:** print all 50 image/prompt pairs side by side. Would a
      musician given only the prompt produce something that fits the image? Record
      pass/fail per image; target ≥40/50 pass.

**Exit criterion:** the prompts read like something you'd brief a composer with.

---

## M4 — Generation and post-processing

**Goal:** end-to-end image → audio file, offline, in a script.

### Tasks
- [ ] `musicgen.py` wrapper with injectable model, explicit seed, config-driven params
- [ ] Continuation logic for `duration_s > 30`
- [ ] `postprocess.py` — limiter, −14 LUFS, optional time-stretch, fades, Opus + WAV
- [ ] CLI: `python -m imgtune.run <image> --out track.wav`

### Verify
- [ ] Same image + same seed → byte-identical WAV (determinism test)
- [ ] Different seeds → audibly different tracks
- [ ] Output measured at −14 LUFS ± 0.5 (verify with `pyloudnorm`, not by ear)
- [ ] No clipping: peak ≤ −1.0 dBTP on all 50 golden outputs
- [ ] Fades present — first and last 200ms are not abrupt
- [ ] With `ENFORCE_TEMPO=true`, detected BPM of the output is within ±2 of
      `spec.tempo_bpm` (verify with `librosa.beat.tempo`)
- [ ] 60s continuation produces a continuous file with no seam clicks at the 20s and
      40s boundaries
- [ ] Generation latency recorded for small and medium at 30s and 60s
- [ ] **Listening pass:** generate all 50, listen to each with the image on screen.
      Score fit 1–5. Record mean. This is the baseline audio quality number.

**Exit criterion:** mean fit score ≥3.0 and no track is offensively wrong for its image.

---

## M5 — Service layer

**Goal:** the pipeline behind an API with async jobs.

### Tasks
- [ ] FastAPI: `POST /generate`, `GET /jobs/{id}`, `GET /jobs/{id}/events` (SSE)
- [ ] Redis queue, worker with model warm-up at boot
- [ ] Feature and audio caches per §6
- [ ] Object storage upload, signed URLs
- [ ] Typed error hierarchy and the §6 failure policy
- [ ] Structured per-job logging of the full artifact chain

### Verify
- [ ] `POST /generate` returns in <2s with a job id **and** the extracted features
- [ ] SSE stream emits progress and terminates on completion or failure
- [ ] Same image submitted twice → second request hits the feature cache
      (verify via cache-hit counter, not timing)
- [ ] Same spec twice → audio cache hit, no GPU work
- [ ] Killing the worker mid-job leaves the job in a recoverable state, not stuck
- [ ] Forcing an extractor exception yields a `degraded: true` result, not a 500
- [ ] Forcing a generation failure retries once then fails cleanly with a typed error
- [ ] Worker cold start does not load models per job — assert model object identity
      across two consecutive jobs
- [ ] 10 concurrent submissions queue correctly; no OOM
- [ ] A full job's log contains features, spec, prompt, seed, and model version

**Exit criterion:** you can leave it running overnight under light load without babysitting.

---

## M6 — Frontend

**Goal:** upload, watch, listen, edit, regenerate.

### Tasks
- [ ] Upload with client-side resize and type validation
- [ ] Feature panel appears immediately (mood, time of day, palette swatches)
- [ ] Progress indicator driven by SSE
- [ ] Audio player with waveform and download
- [ ] Editable prompt field with "regenerate from prompt"
- [ ] Re-roll button (new seed, same spec)
- [ ] Copy stating no vocals, ≤90s, experimental output

### Verify
- [ ] Features render before audio is ready (the perceived-latency win — confirm visually)
- [ ] Editing the prompt and regenerating bypasses vision entirely (confirm in logs)
- [ ] Re-roll produces a different track and does **not** hit the audio cache
- [ ] Error states render as readable messages, never a raw stack trace or spinner-forever
- [ ] Works on mobile viewport
- [ ] Upload of a non-image, a 50MB file, and a 1×1 pixel image each fail gracefully

**Exit criterion:** hand it to someone who has never seen it; they get a track without
asking you a question.

---

## M7 — Safety, legal, quality gate

**Goal:** everything required before showing it to people outside the team.

### Tasks
- [ ] NSFW classifier on ingest, before any model call
- [ ] EXIF stripping including GPS
- [ ] Upload retention policy implemented and documented
- [ ] Identifiable-persons policy written and enforced
- [ ] Rate limiting per IP/user
- [ ] `eval/metrics.py` — ImageBind image↔audio cosine, CLAP prompt adherence
- [ ] Guidance-scale sweep (2.0–4.5) scored on the golden set
- [ ] **Licensing decision recorded** (see `ARCHITECTURE.md` §8)

### Verify
- [ ] NSFW classifier blocks a known-positive test image and passes a known-negative
- [ ] Uploaded file with GPS EXIF → stored file has none (verify with `exiftool`)
- [ ] Rate limit triggers at the configured threshold and returns 429
- [ ] ImageBind score on real image/audio pairs is measurably higher than on shuffled
      pairs — if not, the metric is not working and must be fixed before it is trusted
- [ ] Guidance sweep results committed with the chosen value and its score
- [ ] Licensing status is written down with a named owner and a decision, not "TBD"
- [ ] Second listening pass on the golden set post-tuning; mean fit score improved
      over the M4 baseline

**Exit criterion:** no open item that would embarrass you if a stranger used this today.

---

## Standing checks (run before every merge to main)

- [ ] `pytest` green, including the determinism test
- [ ] `mypy --strict core/ mapping/` clean
- [ ] Golden set score not lower than the committed baseline
- [ ] No new hardcoded constants outside `core/config.py`
- [ ] `ARCHITECTURE.md` updated if any contract or default changed

---

## Deferred — do not start until M7 is verified

- Learned CLIP→T5 embedding bridge
- Melody conditioning
- Symbolic/MIDI path for exact key and tempo
- Spec slider UI with live regeneration
- Multi-image evolving tracks
- Batching and cost optimisation
