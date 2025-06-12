import json
import logging
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image
import torch
from diffusers import StableDiffusionXLPipeline

from .memory import clear_gpu_memory
from .state import AppState
from .config import CONFIG

logger = logging.getLogger(__name__)


TEMP_DIR = Path(tempfile.gettempdir()) / "illustrious_ai"
TEMP_DIR.mkdir(exist_ok=True)
GALLERY_DIR = TEMP_DIR / "gallery"
GALLERY_DIR.mkdir(exist_ok=True)



def init_sdxl(state: AppState) -> Optional[StableDiffusionXLPipeline]:
    """Load the Stable Diffusion XL model."""
    try:
        if not os.path.exists(CONFIG.sd_model):
            logger.error("SDXL model not found: %s", CONFIG.sd_model)
            return None
        logger.info("Loading Stable Diffusion XL model...")
        pipe = StableDiffusionXLPipeline.from_single_file(
            CONFIG.sd_model,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )
        if CONFIG.gpu_backend in ("cuda", "rocm") and torch.cuda.is_available():
            pipe.to("cuda")
            logger.info("SDXL loaded on GPU")
        else:
            logger.warning("GPU not available, using CPU")
        state.sdxl_pipe = pipe
        state.model_status["sdxl"] = True
        return pipe
    except Exception as e:
        logger.error("Failed to initialize SDXL: %s", e)
        state.model_status["sdxl"] = False
        return None


def save_to_gallery(image: Image.Image, prompt: str, metadata: dict | None = None) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{uuid.uuid4().hex[:8]}.png"
    filepath = GALLERY_DIR / filename
    image.save(filepath)
    metadata_file = filepath.with_suffix('.json')
    metadata_info = {
        "prompt": prompt,
        "timestamp": timestamp,
        "filename": filename,
        **(metadata or {}),
    }
    with open(metadata_file, "w") as f:
        json.dump(metadata_info, f, indent=2)
    return str(filepath)


def generate_image(
    state: AppState,
    prompt: str,
    negative_prompt: str = "",
    steps: int = 30,
    guidance: float = 7.5,
    seed: int = -1,
    save_to_gallery_flag: bool = True,
) -> Tuple[Optional[Image.Image], str]:
    """Generate an image using SDXL."""
    if not state.sdxl_pipe:
        return None, "❌ SDXL model not loaded. Please check your model path."
    clear_gpu_memory()
    max_retries = 2
    for attempt in range(max_retries):
        try:
            device = "cuda" if CONFIG.gpu_backend in ("cuda", "rocm") and torch.cuda.is_available() else "cpu"
            generator = torch.Generator(device=device)
            if seed != -1:
                generator.manual_seed(seed)
            else:
                seed = generator.initial_seed()
            image = state.sdxl_pipe(
                prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
                width=1024,
                height=1024,
            ).images[0]
            state.latest_generated_image = image
            if save_to_gallery_flag:
                save_to_gallery(
                    image,
                    prompt,
                    {
                        "negative_prompt": negative_prompt,
                        "steps": steps,
                        "guidance": guidance,
                        "seed": seed,
                    },
                )
            clear_gpu_memory()
            return image, f"✅ Image generated successfully! Seed: {seed}"
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and attempt < max_retries - 1:
                logger.warning("CUDA OOM on attempt %s: %s", attempt + 1, e)
                clear_gpu_memory()
                continue
            return None, f"❌ Generation failed: {e}"
        except Exception as e:
            logger.error("Image generation failed: %s", e)
            return None, f"❌ Generation failed: {e}"
    return None, "❌ Generation failed after retries"

def get_latest_image(state: AppState) -> Optional[Image.Image]:
    """Return the last generated image."""
    return state.latest_generated_image


def switch_sdxl_model(state: AppState, path: str) -> bool:
    """Switch to a different SDXL checkpoint."""
    if not os.path.exists(path):
        logger.error("SDXL model not found: %s", path)
        return False
    state.sdxl_pipe = None
    clear_gpu_memory()
    CONFIG.sd_model = path
    logger.info("Switching SDXL model to %s", path)
    return init_sdxl(state) is not None
