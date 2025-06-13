"""
Illustrious AI Studio - Stable Diffusion XL (SDXL) Integration

This module provides comprehensive SDXL model management and image generation
capabilities for the AI Studio. It handles:

CORE FEATURES:
- SDXL model loading and initialization
- High-quality image generation with advanced parameters
- Model switching and management
- Gallery integration and metadata handling
- Project-based image organization
- Memory optimization and cleanup
- Batch generation support

IMAGE GENERATION:
- Text-to-image generation with prompts
- Negative prompting for content exclusion
- Adjustable generation parameters (steps, guidance, etc.)
- Multiple resolution presets
- Seed control for reproducible results
- Automatic metadata embedding

MODEL MANAGEMENT:
- Support for multiple SDXL models
- Hot-swapping between models
- Model validation and health checks
- Performance testing and benchmarking
- Memory usage optimization

GALLERY FEATURES:
- Automatic image saving with metadata
- Project-based organization
- Thumbnail generation
- Batch operations
- Export/import functionality

The module is designed to be robust, handling errors gracefully and
providing detailed logging for troubleshooting model issues.
"""

import json
import logging
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List, Dict, TypedDict, Protocol, Callable, Any
import asyncio

from PIL import Image
import torch
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline

from .memory import clear_gpu_memory
from .memory_guardian import get_memory_guardian
from .state import AppState
from .config import CONFIG

logger = logging.getLogger(__name__)


class GenerationParams(TypedDict, total=False):
    """Parameters for image generation."""
    prompt: str
    negative_prompt: str
    steps: int
    guidance: float
    seed: int
    save_to_gallery_flag: bool
    width: int
    height: int
    progress_callback: Optional[Callable[[int, int], Any]]


class ModelProtocol(Protocol):
    """Protocol describing the expected SDXL pipeline interface."""

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        generator: Optional[torch.Generator] = None,
        width: int = 1024,
        height: int = 1024,
        callback: Optional[Callable[[int, int, Any], Any]] = None,
        callback_steps: int = 1,
    ) -> Any:
        ...


class DiffusersPipelineAdapter:
    """Adapter to expose a StableDiffusionXLPipeline with a generate method."""

    def __init__(self, pipe: StableDiffusionXLPipeline):
        self.pipe = pipe

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        generator: Optional[torch.Generator] = None,
        width: int = 1024,
        height: int = 1024,
        callback: Optional[Callable[[int, int, Any], Any]] = None,
        callback_steps: int = 1,
    ) -> Any:
        return self.pipe(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            width=width,
            height=height,
            callback=callback,
            callback_steps=callback_steps,
        )


# ==================================================================
# DIRECTORY STRUCTURE AND PATHS
# ==================================================================

# Temporary directory for processing and cache
TEMP_DIR = Path(tempfile.gettempdir()) / "illustrious_ai"
TEMP_DIR.mkdir(exist_ok=True)

# Gallery directory for saved images
GALLERY_DIR = Path(CONFIG.gallery_dir)

# Projects directory for workspace organization
PROJECTS_DIR = Path("projects")
PROJECTS_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================================
# MODEL INITIALIZATION AND MANAGEMENT
# ==================================================================

def init_sdxl(state: AppState) -> Optional[StableDiffusionXLPipeline]:
    """
    Initialize and load the Stable Diffusion XL model.
    
    This function handles the complete SDXL model loading process:
    - Validates model file existence and accessibility
    - Loads the model with appropriate precision settings
    - Configures GPU acceleration if available
    - Updates application state with model instance
    - Handles errors gracefully with detailed logging
    
    Args:
        state: Application state to store the loaded model
        
    Returns:
        Optional[StableDiffusionXLPipeline]: Loaded pipeline instance,
                                            or None if loading failed
    
    Notes:
        - Model loading can take 1-2 minutes depending on hardware
        - Requires ~6GB of GPU memory for optimal performance
        - Falls back to CPU if GPU unavailable (much slower)
        - Uses float16 precision for memory efficiency
    """
    try:
        # Validate model file exists
        if not os.path.exists(CONFIG.sd_model):
            logger.error("SDXL model file not found: %s", CONFIG.sd_model)
            return None
            
        logger.info("Loading Stable Diffusion XL model from: %s", CONFIG.sd_model)
        
        # Load model with optimized settings
        pipe = StableDiffusionXLPipeline.from_single_file(
            CONFIG.sd_model,
            torch_dtype=torch.float16,    # Use half precision for memory efficiency
            variant="fp16",               # Load FP16 variant if available
            use_safetensors=True,        # Use safetensors format for security
        )
        
        # Configure device placement
        if CONFIG.gpu_backend in ("cuda", "rocm") and torch.cuda.is_available():
            pipe.to("cuda")
            logger.info("✅ SDXL model loaded successfully on GPU")
        else:
            logger.warning("⚠️ GPU not available, SDXL will use CPU (significantly slower)")
            
        # Update application state with adapter implementing ModelProtocol
        state.sdxl_pipe = DiffusersPipelineAdapter(pipe)
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


def generate_image(state: AppState, params: GenerationParams) -> Tuple[Optional[Image.Image], str]:
    """Generate an image using SDXL with automatic OOM prevention."""
    prompt = params.get("prompt", "")
    negative_prompt = params.get("negative_prompt", "")
    steps = params.get("steps", 30)
    guidance = params.get("guidance", 7.5)
    seed = params.get("seed", -1)
    save_to_gallery_flag = params.get("save_to_gallery_flag", True)
    width = params.get("width", 1024)
    height = params.get("height", 1024)
    progress_callback = params.get("progress_callback")
    # Comprehensive parameter validation and sanitization FIRST
    try:
        # Validate and sanitize prompt - this is critical!
        if prompt is None:
            return None, "❌ Generation failed: Prompt cannot be None. Please provide a valid text prompt."
        
        if not isinstance(prompt, str):
            try:
                prompt = str(prompt)
            except Exception as e:
                return None, f"❌ Generation failed: Cannot convert prompt to string: {e}"
        
        prompt = prompt.strip()
        if not prompt:
            return None, "❌ Generation failed: Prompt cannot be empty. Please provide a descriptive text prompt."
        
        # Validate and sanitize negative prompt
        if negative_prompt is None:
            negative_prompt = ""
        elif not isinstance(negative_prompt, str):
            try:
                negative_prompt = str(negative_prompt)
            except Exception:
                logger.warning(f"Invalid negative_prompt '{negative_prompt}', using empty string")
                negative_prompt = ""
        negative_prompt = negative_prompt.strip()
        
        # Validate and sanitize steps parameter
        try:
            if steps is None:
                steps = 30
            steps = int(steps)
            if steps <= 0 or steps > 200:
                logger.warning(f"Steps {steps} out of valid range (1-200), using 30")
                steps = 30
        except (ValueError, TypeError) as e:
            logger.warning(f"Invalid steps parameter '{steps}': {e}, using 30")
            steps = 30
        
        # Validate and sanitize guidance parameter
        try:
            if guidance is None:
                guidance = 7.5
            guidance = float(guidance)
            if guidance <= 0 or guidance > 50:
                logger.warning(f"Guidance {guidance} out of valid range (0-50), using 7.5")
                guidance = 7.5
        except (ValueError, TypeError) as e:
            logger.warning(f"Invalid guidance parameter '{guidance}': {e}, using 7.5")
            guidance = 7.5
        
        # Validate and sanitize dimensions
        try:
            if width is None:
                width = 1024
            if height is None:
                height = 1024
            width = int(width)
            height = int(height)
            if width <= 0 or width > 2048 or height <= 0 or height > 2048:
                logger.warning(f"Dimensions {width}x{height} out of valid range, using 1024x1024")
                width, height = 1024, 1024
        except (ValueError, TypeError) as e:
            logger.warning(f"Invalid dimensions '{width}x{height}': {e}, using 1024x1024")
            width, height = 1024, 1024
        
        # Validate and sanitize seed parameter
        try:
            if seed is None:
                seed = -1  # Use random seed
            elif not isinstance(seed, (int, float)):
                # Try to convert to int, fallback to random if invalid
                try:
                    seed = int(seed)
                except (ValueError, TypeError):
                    logger.warning(f"Invalid seed value '{seed}', using random seed instead")
                    seed = -1
            else:
                seed = int(seed)  # Ensure it's an integer
            
            # Clamp seed to valid range for PyTorch
            if seed != -1 and (seed < 0 or seed >= 2**32):
                logger.warning(f"Seed {seed} out of valid range, using random seed instead")
                seed = -1
        except Exception as e:
            logger.warning(f"Error processing seed parameter: {e}, using random seed")
            seed = -1
        
        # Log validated parameters for debugging
        logger.info(f"Generating image with validated parameters: prompt='{prompt[:50]}...', "
                   f"negative_prompt='{negative_prompt[:30]}...', steps={steps}, guidance={guidance}, "
                   f"seed={seed}, dimensions={width}x{height}")
        
    except Exception as e:
        logger.error(f"Parameter validation failed: {e}")
        return None, f"❌ Generation failed: Parameter validation error: {e}"
    
    # Check if model is loaded AFTER parameter validation
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
            
            # Handle seed generation with proper validation
            if seed == -1:
                # Generate random seed
                actual_seed = generator.initial_seed()
            else:
                # Use provided seed
                generator.manual_seed(seed)
                actual_seed = seed
            
            # Pre-generation memory check
            stats = guardian.get_memory_stats()
            if stats and stats.pressure_level.value in ["high", "critical"]:
                logger.warning(f"Starting generation with {stats.pressure_level.value} memory pressure")
            
            def _progress_cb(step: int, t, latents):
                if progress_callback:
                    progress_callback(step + 1, steps)

            result = state.sdxl_pipe.generate(
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
                        "seed": actual_seed,
                        "width": width,
                        "height": height,
                        "adapted_settings": (width != 1024 or height != 1024 or steps != 30)
                    },
                )
            clear_gpu_memory()
            
            success_msg = f"✅ Image generated successfully! Seed: {actual_seed}"
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


async def generate_image_async(state: AppState, params: GenerationParams) -> Tuple[Optional[Image.Image], str]:
    """Asynchronous wrapper for ``generate_image`` running in a thread."""
    return await asyncio.to_thread(generate_image, state, params)


async def generate_with_notifications(
    state: AppState, params: GenerationParams, progress
) -> Tuple[Optional[Image.Image], str]:
    """Generate an image while emitting high level progress notifications."""
    # Initializing
    if progress:
        progress(0.1, desc="Initializing model…")
    if state.sdxl_pipe is None:
        await asyncio.to_thread(init_sdxl, state)

    # Processing prompt
    if progress:
        progress(0.3, desc="Processing prompt…")

    # Prepare progress callback for generation phase
    if progress:
        progress(0.6, desc="Generating image…")

        def cb(step: int, total: int):
            progress(0.6 + (step / total) * 0.4, desc=f"{step}/{total}")
        params["progress_callback"] = cb

    # Run the actual generation
    image, status = await generate_image_async(state, params)

    if progress:
        progress(1.0, desc="Complete!")

    return image, status

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
            state,
            {
                "prompt": test_prompt,
                "negative_prompt": "blurry, complex",
                "steps": 10,
                "guidance": 7.5,
                "seed": 42,
                "save_to_gallery_flag": False,
            },
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


def batch_generate(
    state: AppState,
    prompts: List[str],
    shared_settings: dict,
    progress: Optional[Callable[[float, str], Any]] = None,
) -> List[Optional[Image.Image]]:
    """Generate multiple images with shared settings."""
    results: List[Optional[Image.Image]] = []
    total = len(prompts)
    for i, prm in enumerate(prompts):
        if progress:
            progress(i / total, f"Generating {i+1}/{total}")
        params = dict(shared_settings)
        params["prompt"] = prm
        img, _ = generate_image(state, params)
        results.append(img)
    return results


def create_variations(base_image: Image.Image, num_variations: int = 4) -> List[Image.Image]:
    """Create simple variations of an existing image."""
    variations: List[Image.Image] = []
    for i in range(num_variations):
        variations.append(base_image.rotate(i * 5))
    return variations
