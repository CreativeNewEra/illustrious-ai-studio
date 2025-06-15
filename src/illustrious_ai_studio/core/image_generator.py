"""Utility class for SDXL image generation."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

from PIL import Image
import torch

from .state import AppState
from .config import CONFIG
from .memory import clear_gpu_memory
from .memory_guardian import get_memory_guardian
from .sdxl import (
    GenerationParams,
    _estimate_generation_memory,
    _get_adaptive_settings,
    _reduce_settings_for_retry,
    save_to_gallery,
)

logger = logging.getLogger(__name__)


class ImageGenerator:
    """Encapsulates the SDXL generation pipeline."""

    def __init__(self, state: AppState) -> None:
        self.state = state
        self.guardian = get_memory_guardian(state)

    # --------------------------------------------------------------
    def validate_params(self, params: GenerationParams) -> GenerationParams:
        """Validate and sanitize incoming parameters."""
        prompt = str(params.get("prompt", "") or "").strip()
        if not prompt:
            raise ValueError("Prompt cannot be empty")

        negative_prompt = str(params.get("negative_prompt", "") or "").strip()

        def _int(name: str, value: int, default: int, low: int, high: int) -> int:
            try:
                if value is None:
                    return default
                value = int(value)
                if value < low or value > high:
                    logger.warning("%s %s out of range, using %s", name, value, default)
                    return default
                return value
            except Exception:  # pragma: no cover - validation failures
                logger.warning("Invalid %s value %s, using %s", name, value, default)
                return default

        def _float(name: str, value: float, default: float, low: float, high: float) -> float:
            try:
                if value is None:
                    return default
                value = float(value)
                if value < low or value > high:
                    logger.warning("%s %s out of range, using %s", name, value, default)
                    return default
                return value
            except Exception:
                logger.warning("Invalid %s value %s, using %s", name, value, default)
                return default

        steps = _int("steps", params.get("steps", 30), 30, 1, 200)
        guidance = _float("guidance", params.get("guidance", 7.5), 7.5, 0.1, 50)
        width = _int("width", params.get("width", 1024), 1024, 64, 2048)
        height = _int("height", params.get("height", 1024), 1024, 64, 2048)

        seed = params.get("seed", -1)
        try:
            seed = int(seed)
            if seed != -1 and (seed < 0 or seed >= 2**32):
                logger.warning("Seed %s out of range, using random", seed)
                seed = -1
        except Exception:
            logger.warning("Invalid seed %s, using random", seed)
            seed = -1

        return {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": steps,
            "guidance": guidance,
            "seed": seed,
            "save_to_gallery_flag": bool(params.get("save_to_gallery_flag", True)),
            "width": width,
            "height": height,
            "progress_callback": params.get("progress_callback"),
        }

    # --------------------------------------------------------------
    def check_resources(self, width: int, height: int, steps: int) -> Tuple[int, int, int]:
        """Validate memory availability and adapt settings if needed."""
        est_mem = _estimate_generation_memory(width, height, steps)
        if not self.guardian.check_memory_requirements("image_generation", est_mem):
            width, height, steps = _get_adaptive_settings(width, height, steps, self.guardian)
        return width, height, steps

    # --------------------------------------------------------------
    def save(self, image: Image.Image, prompt: str, metadata: dict) -> str:
        """Delegate saving to the existing gallery helper."""
        return save_to_gallery(self.state, image, prompt, metadata)

    # --------------------------------------------------------------
    def generate(self, params: GenerationParams) -> Tuple[Optional[Image.Image], str]:
        """Run the SDXL pipeline with retries and saving."""
        try:
            params = self.validate_params(params)
        except ValueError as e:
            return None, f"❌ Generation failed: {e}"

        if not self.state.sdxl_pipe:
            return None, "❌ SDXL model not loaded. Please check your model path."

        prompt = params["prompt"]
        negative_prompt = params["negative_prompt"]
        steps = params["steps"]
        guidance = params["guidance"]
        seed = params["seed"]
        width = params["width"]
        height = params["height"]
        save_flag = params["save_to_gallery_flag"]
        callback = params.get("progress_callback")

        width, height, steps = self.check_resources(width, height, steps)

        strategy = getattr(self.state, "degradation_strategy", None)

        clear_gpu_memory()
        if callback:
            callback(0, steps)

        device = "cuda" if CONFIG.gpu_backend in ("cuda", "rocm") and torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=device)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                if seed == -1:
                    actual_seed = generator.initial_seed()
                else:
                    generator.manual_seed(seed)
                    actual_seed = seed

                def _cb(step: int, total, _latents):
                    if callback:
                        callback(step + 1, steps)

                result = self.state.sdxl_pipe.generate(
                    prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    generator=generator,
                    width=width,
                    height=height,
                    callback=_cb if callback else None,
                    callback_steps=1,
                )

                image = None
                if hasattr(result, "images") and result.images:
                    image = result.images[0]
                elif isinstance(result, (list, tuple)) and result:
                    image = result[0]
                elif isinstance(result, Image.Image):
                    image = result

                if image is None or not isinstance(image, Image.Image):
                    raise RuntimeError("Pipeline returned no image")

                if callback:
                    callback(steps, steps)

                if save_flag:
                    self.save(
                        image,
                        prompt,
                        {
                            "negative_prompt": negative_prompt,
                            "steps": steps,
                            "guidance": guidance,
                            "seed": actual_seed,
                            "width": width,
                            "height": height,
                        },
                    )
                if strategy:
                    strategy.restore()
                clear_gpu_memory()
                return image, f"✅ Image generated successfully! Seed: {actual_seed}"
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and attempt < max_retries - 1:
                    logger.warning(
                        "Out of memory detected during attempt %d. Exception: %s. Retrying with reduced settings: width=%d, height=%d, steps=%d",
                        attempt + 1, str(e), width, height, steps
                    )
                    clear_gpu_memory()
                    if strategy:
                        width, height, steps = strategy.degrade(width, height, steps)
                    width, height, steps = _reduce_settings_for_retry(width, height, steps, attempt)
                    continue
                if callback:
                    callback(steps, steps)
                return None, f"❌ Generation failed: {e}"
            except Exception as e:  # pragma: no cover - unexpected errors
                if callback:
                    callback(steps, steps)
                return None, f"❌ Generation failed: {e}"

        if callback:
            callback(steps, steps)
        return None, "❌ Generation failed after all retries."
