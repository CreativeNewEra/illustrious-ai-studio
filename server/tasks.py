from __future__ import annotations
import os
import base64
import io
from typing import Any
from celery import Celery
from core.image_generator import ImageGenerator
from core.sdxl import GenerationParams
from app import app_state

# Celery configuration using Redis as broker and backend
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", CELERY_BROKER_URL)

celery_app = Celery(
    "illustrious_tasks",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
)

@celery_app.task(bind=True)
def generate_image_task(self, params: GenerationParams) -> dict[str, Any]:
    """Background task to generate an image using SDXL."""
    state = app_state
    if state.sdxl_pipe is None:
        return {"success": False, "error": "SDXL model not loaded"}

    generator = ImageGenerator(state)
    image, status = generator.generate(params)
    if image is None:
        return {"success": False, "error": status}

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return {"success": True, "image_base64": img_b64, "message": status}
