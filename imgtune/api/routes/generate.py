"""Generation API routes.

POST /generate  — upload image, get features + kick off audio generation
GET  /jobs/{id} — poll job status
POST /regenerate — re-run with edited prompt or new seed
"""
from __future__ import annotations

import logging
import os
import random
import uuid
from io import BytesIO

import numpy as np
from fastapi import APIRouter, BackgroundTasks, File, Form, UploadFile
from PIL import Image

from imgtune.api.deps import get_blip_models, get_clip_models, get_musicgen_model, jobs
from imgtune.core.config import settings
from imgtune.core.schemas import JobRecord, JobStatus
from imgtune.mapping.prompt_builder import build_prompt
from imgtune.mapping.rules import fuse
from imgtune.vision.extract import extract_features

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/generate")
async def generate(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Accept an image, extract features (sync), queue audio generation (async)."""
    job_id = str(uuid.uuid4())
    job = JobRecord(job_id=job_id, status=JobStatus.EXTRACTING)
    jobs[job_id] = job

    # Read image
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")

    # Extraction (fast, ~1-2 s)
    try:
        clip_model, clip_preprocess, clip_tokenizer, device = get_clip_models()
        blip_model, blip_processor, blip_device = get_blip_models()
        features = extract_features(
            image,
            clip_model=clip_model,
            clip_preprocess=clip_preprocess,
            clip_tokenizer=clip_tokenizer,
            blip_model=blip_model,
            blip_processor=blip_processor,
            device=device,
        )
    except Exception as exc:
        logger.warning("Extraction degraded: %s", exc)
        features = extract_features(image, device="cpu")
        job.degraded = True

    job.features = features

    # Fusion + prompt
    spec = fuse(features)
    job.spec = spec
    prompt = build_prompt(spec)
    job.prompt = prompt

    # Kick off generation
    background_tasks.add_task(_run_generation, job_id, prompt, spec.duration_s, spec.seed)

    return {
        "job_id": job_id,
        "status": job.status,
        "features": job.features.model_dump(),
        "spec": job.spec.model_dump(),
        "prompt": job.prompt,
    }


@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """Poll job status and results."""
    job = jobs.get(job_id)
    if not job:
        return {"error": "Job not found"}, 404
    return job.model_dump()


@router.post("/regenerate")
async def regenerate(
    background_tasks: BackgroundTasks,
    job_id: str = Form(...),
    prompt: str = Form(None),
    new_seed: bool = Form(False),
):
    """Re-generate with an edited prompt or a new seed."""
    job = jobs.get(job_id)
    if not job or not job.spec:
        return {"error": "Job not found"}

    if prompt:
        job.prompt = prompt
    seed = job.spec.seed
    if new_seed:
        seed = random.randint(0, 2**31)
        job.spec.seed = seed

    job.status = JobStatus.GENERATING
    job.audio_url = None

    background_tasks.add_task(_run_generation, job_id, job.prompt, job.spec.duration_s, seed)
    return {"job_id": job_id, "status": "generating", "prompt": job.prompt, "seed": seed}


# Background task


def _run_generation(job_id: str, prompt: str, duration_s: int, seed: int) -> None:
    job = jobs.get(job_id)
    if not job:
        return

    try:
        job.status = JobStatus.GENERATING
        model = get_musicgen_model()

        from imgtune.audio.musicgen import generate_audio
        wav_tensor, sample_rate = generate_audio(prompt, duration_s, seed, model=model)

        job.status = JobStatus.POSTPROCESSING
        audio_np: np.ndarray = wav_tensor.cpu().numpy()
        output_path = os.path.join(settings.output_dir, f"{job_id}.wav")

        from imgtune.audio.postprocess import postprocess
        postprocess(audio_np, sample_rate, output_path)

        job.audio_url = f"/outputs/{job_id}.wav"
        job.status = JobStatus.COMPLETE

    except Exception as exc:
        logger.error("Generation failed for %s: %s", job_id, exc)
        # Retry once with a new seed (per §6 failure policy)
        try:
            new_seed = random.randint(0, 2**31)
            from imgtune.audio.musicgen import generate_audio

            model = get_musicgen_model()
            wav_tensor, sample_rate = generate_audio(prompt, duration_s, new_seed, model=model)
            audio_np = wav_tensor.cpu().numpy()
            output_path = os.path.join(settings.output_dir, f"{job_id}.wav")

            from imgtune.audio.postprocess import postprocess
            postprocess(audio_np, sample_rate, output_path)

            job.audio_url = f"/outputs/{job_id}.wav"
            job.status = JobStatus.COMPLETE
        except Exception as retry_exc:
            job.status = JobStatus.FAILED
            job.error = str(retry_exc)
