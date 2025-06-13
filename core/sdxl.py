import json
import logging
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List, Dict

from PIL import Image
import torch
from diffusers import StableDiffusionXLPipeline

from .memory import clear_gpu_memory
from .memory_guardian import get_memory_guardian
from .state import AppState
from .config import CONFIG

logger = logging.getLogger(__name__)


TEMP_DIR = Path(tempfile.gettempdir()) / "illustrious_ai"
TEMP_DIR.mkdir(exist_ok=True)
GALLERY_DIR = Path(CONFIG.gallery_dir)
PROJECTS_DIR = Path("projects")
PROJECTS_DIR.mkdir(parents=True, exist_ok=True)



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


def _get_active_gallery_dir(state: AppState) -> Path:
    """Return gallery directory for the current project or default gallery."""
    if state.current_project:
        return PROJECTS_DIR / state.current_project / "gallery"
    return GALLERY_DIR


def save_to_gallery(state: AppState, image: Image.Image, prompt: str, metadata: dict | None = None) -> str:
    """Save image and metadata to the active project's gallery."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{uuid.uuid4().hex[:8]}.png"
    gallery_dir = _get_active_gallery_dir(state)
    gallery_dir.mkdir(parents=True, exist_ok=True)
    filepath = gallery_dir / filename
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
    width: int = 1024,
    height: int = 1024,
    progress_callback: Optional[callable] = None,
) -> Tuple[Optional[Image.Image], str]:
    """Generate an image using SDXL with automatic OOM prevention."""
    if not state.sdxl_pipe:
        return None, "❌ SDXL model not loaded. Please check your model path."
    
    # Get memory guardian instance
    guardian = get_memory_guardian(state)
    
    # Check memory requirements before generation
    estimated_memory_gb = _estimate_generation_memory(width, height, steps)
    if not guardian.check_memory_requirements("image_generation", estimated_memory_gb):
        # Try adaptive settings if memory is insufficient
        original_width, original_height, original_steps = width, height, steps
        width, height, steps = _get_adaptive_settings(width, height, steps, guardian)
        
        if width != original_width or height != original_height or steps != original_steps:
            logger.warning(
                f"Adapted generation settings due to memory constraints: "
                f"{original_width}x{original_height}@{original_steps} -> {width}x{height}@{steps}"
            )
    
    clear_gpu_memory()
    max_retries = 3  # Increased retries with memory guardian
    if progress_callback:
        progress_callback(0, steps)
    
    for attempt in range(max_retries):
        try:
            device = "cuda" if CONFIG.gpu_backend in ("cuda", "rocm") and torch.cuda.is_available() else "cpu"
            generator = torch.Generator(device=device)
            if seed != -1:
                generator.manual_seed(seed)
            else:
                seed = generator.initial_seed()
            
            # Pre-generation memory check
            stats = guardian.get_memory_stats()
            if stats and stats.pressure_level.value in ["high", "critical"]:
                logger.warning(f"Starting generation with {stats.pressure_level.value} memory pressure")
            
            def _progress_cb(step: int, t, latents):
                if progress_callback:
                    progress_callback(step + 1, steps)

            result = state.sdxl_pipe(
                prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
                width=width,
                height=height,
                callback=_progress_cb if progress_callback else None,
                callback_steps=1,
            )
            
            # Extract image from result - handle StableDiffusionXLPipelineOutput
            if hasattr(result, 'images') and result.images and len(result.images) > 0:
                image = result.images[0]
            elif isinstance(result, (list, tuple)) and len(result) > 0:
                image = result[0]
            else:
                image = result
            
            # Ensure we have a PIL Image
            if not isinstance(image, Image.Image):
                logger.error(f"Expected PIL Image, got {type(image)}")
                return None, (
                    f"❌ Invalid image type returned: {type(image)}. "
                    "Ensure you are using a compatible SDXL pipeline."
                )
            
            state.latest_generated_image = image
            if progress_callback:
                progress_callback(steps, steps)
            if save_to_gallery_flag:
                save_to_gallery(
                    state,
                    image,
                    prompt,
                    {
                        "negative_prompt": negative_prompt,
                        "steps": steps,
                        "guidance": guidance,
                        "seed": seed,
                        "width": width,
                        "height": height,
                        "adapted_settings": (width != 1024 or height != 1024 or steps != 30)
                    },
                )
            clear_gpu_memory()
            
            success_msg = f"✅ Image generated successfully! Seed: {seed}"
            if width != 1024 or height != 1024:
                success_msg += f" (Resolution: {width}x{height})"
            if steps != 30:
                success_msg += f" (Steps: {steps})"
                
            return image, success_msg
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"CUDA OOM on attempt {attempt + 1}: {e}")
                
                # Let memory guardian handle the OOM
                clear_gpu_memory()
                
                # Progressive degradation for next attempt
                if attempt < max_retries - 1:
                    width, height, steps = _reduce_settings_for_retry(width, height, steps, attempt)
                    logger.info(f"Retrying with reduced settings: {width}x{height}@{steps}")
                    continue
                    
            if progress_callback:
                progress_callback(steps, steps)
            return None, (
                f"❌ Generation failed: {e}. "
                "Check your model path and GPU memory usage. "
                "See the README's Logging and Debugging section."
            )
        except Exception as e:
            logger.error("Image generation failed: %s", e)
            if progress_callback:
                progress_callback(steps, steps)
            return None, (
                f"❌ Generation failed: {e}. "
                "Check your model path and GPU memory usage. "
                "See the README's Logging and Debugging section."
            )
    
    if progress_callback:
        progress_callback(steps, steps)
    return None, (
        "❌ Generation failed after all retries. "
        "Check GPU memory or model path and review the README's Logging and Debugging section."
    )

def _estimate_generation_memory(width: int, height: int, steps: int) -> float:
    """Estimate memory requirements for image generation in GB"""
    # Base memory for SDXL model (~6GB)
    base_memory = 6.0
    
    # Additional memory based on resolution and steps
    pixels = width * height
    # Rough estimate: more pixels and steps = more memory
    additional_memory = (pixels / (1024 * 1024)) * 0.001 * (steps / 30)
    
    return base_memory + additional_memory

def _get_adaptive_settings(width: int, height: int, steps: int, guardian) -> Tuple[int, int, int]:
    """Get adaptive settings based on available memory"""
    stats = guardian.get_memory_stats()
    if not stats:
        return width, height, steps
    
    # Available memory in GB
    available_gb = stats.gpu_free_gb
    
    # Adaptive scaling based on available memory
    if available_gb < 4.0:  # Very low memory
        return 512, 512, max(10, steps // 3)
    elif available_gb < 6.0:  # Low memory
        return 768, 768, max(15, steps // 2)
    elif available_gb < 8.0:  # Medium memory
        return 1024, 1024, max(20, int(steps * 0.7))
    else:  # Sufficient memory
        return width, height, steps

def _reduce_settings_for_retry(width: int, height: int, steps: int, attempt: int) -> Tuple[int, int, int]:
    """Progressively reduce settings for retry attempts"""
    if attempt == 0:  # First retry - reduce resolution
        if width > 512:
            return max(512, width // 2), max(512, height // 2), steps
    elif attempt == 1:  # Second retry - reduce steps
        return width, height, max(10, steps // 2)
    else:  # Final retry - minimal settings
        return 512, 512, 10
    
    return width, height, steps

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

def get_available_models() -> List[Dict[str, str]]:
    """Scan models directory for available SDXL checkpoints."""
    models_dir = Path("models")
    if not models_dir.exists():
        logger.warning("Models directory not found: %s", models_dir)
        return []
    
    models = []
    for model_file in models_dir.glob("*.safetensors"):
        if model_file.is_file():
            # Get file size in MB
            size_mb = model_file.stat().st_size / (1024 * 1024)
            
            models.append({
                "path": str(model_file),
                "display_name": get_model_display_name(model_file.name),
                "filename": model_file.name,
                "size_mb": round(size_mb, 1),
                "is_current": str(model_file) == CONFIG.sd_model
            })
    
    return sorted(models, key=lambda x: x["display_name"])

def get_model_display_name(filename: str) -> str:
    """Convert model filename to user-friendly display name."""
    # Remove extension
    name = filename.replace(".safetensors", "")
    
    # Handle common naming patterns
    name = name.replace("_", " ")
    name = name.replace("-", " ")
    
    # Capitalize words
    words = []
    for word in name.split():
        if word.upper() in ["XL", "SDXL", "V1", "V2", "V3"]:
            words.append(word.upper())
        else:
            words.append(word.capitalize())
    
    return " ".join(words)

def validate_model_file(model_path: str) -> Tuple[bool, str]:
    """Validate that a model file exists and is accessible."""
    try:
        path = Path(model_path)
        if not path.exists():
            return False, "File does not exist"
        
        if not path.is_file():
            return False, "Path is not a file"
        
        if path.suffix.lower() != ".safetensors":
            return False, "Not a .safetensors file"
        
        # Check file size (should be at least 1GB for SDXL)
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb < 1000:
            return False, f"File too small ({size_mb:.1f}MB), possibly corrupted"
        
        return True, f"Valid model file ({size_mb:.1f}MB)"
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def test_model_generation(state: AppState, model_path: str) -> Tuple[bool, str, Optional[Image.Image]]:
    """Test model by generating a simple image."""
    try:
        # Store current model
        original_model = CONFIG.sd_model
        original_pipe = state.sdxl_pipe
        
        # Try to switch to test model
        if not switch_sdxl_model(state, model_path):
            return False, "Failed to load model", None
        
        # Generate test image
        test_prompt = "a red apple on a white background, simple, clean"
        image, status = generate_image(
            state=state,
            prompt=test_prompt,
            negative_prompt="blurry, complex",
            steps=10,  # Quick test
            guidance=7.5,
            seed=42,  # Fixed seed for consistency
            save_to_gallery_flag=False  # Don't save test images
        )
        
        if image is not None:
            # Restore original model if different
            if original_model != model_path:
                CONFIG.sd_model = original_model
                state.sdxl_pipe = original_pipe
            return True, "✅ Model test successful", image
        else:
            return False, (
                f"Generation failed: {status}. "
                "Verify the model path and check GPU memory usage."
            ), None
            
    except Exception as e:
        logger.error("Model test failed: %s", e)
        return False, f"Test failed: {str(e)}", None

def get_current_model_info(state: AppState) -> Dict[str, str]:
    """Get information about the currently loaded model."""
    current_path = CONFIG.sd_model
    
    if not current_path or not os.path.exists(current_path):
        return {
            "path": current_path or "None",
            "display_name": "No model loaded",
            "status": "❌ Not found",
            "size": "Unknown"
        }
    
    try:
        path = Path(current_path)
        size_mb = path.stat().st_size / (1024 * 1024)
        
        return {
            "path": current_path,
            "display_name": get_model_display_name(path.name),
            "status": "✅ Loaded" if state.sdxl_pipe else "⚠️ Not initialized",
            "size": f"{size_mb:.1f}MB"
        }
    except Exception as e:
        return {
            "path": current_path,
            "display_name": "Error reading model",
            "status": f"❌ Error: {str(e)}",
            "size": "Unknown"
        }
