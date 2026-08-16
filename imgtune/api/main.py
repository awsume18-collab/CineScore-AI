"""FastAPI application entry point."""
from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from imgtune.core.config import settings
from imgtune.api.routes.generate import router as generate_router

app = FastAPI(title="Cinescore", description="Image → Music Generation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(settings.output_dir, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=settings.output_dir), name="outputs")
app.include_router(generate_router, prefix="/api")


@app.get("/health")
async def health():
    return {"status": "ok"}
