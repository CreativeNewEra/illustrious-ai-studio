"""
Illustrious AI Studio - Configuration Management

This module handles all configuration aspects of the AI Studio including:
- Application settings and defaults
- Model paths and parameters
- Performance and memory settings
- UI preferences and behavior
- Environment variable overrides

The configuration system supports:
- YAML file-based configuration
- Environment variable overrides
- Runtime configuration updates
- Type-safe configuration with Pydantic

Configuration Priority (highest to lowest):
1. Environment variables
2. YAML configuration file
3. Default values in SDXLConfig model
"""

import os
from pathlib import Path
import tempfile
import warnings
import logging
import yaml
from pydantic import BaseModel, validator
from .hardware_profiler import HardwareProfiler

logger = logging.getLogger(__name__)

_GALLERY_BASE = Path(tempfile.gettempdir()) / "illustrious_ai"

# ==================================================================
# CONSTANTS AND PATHS
# ==================================================================

# Default directory for AI model files
# Can be overridden with MODELS_DIR environment variable
MODEL_DIR = Path(os.getenv("MODELS_DIR", "models"))


# ==================================================================
# MAIN CONFIGURATION CLASS
# ==================================================================


class SDXLConfig(BaseModel):
    """Configuration model for Illustrious AI Studio."""

    # ==============================================================
    # MODEL CONFIGURATION
    # ==============================================================

    sd_model: str = str(MODEL_DIR / "Illustrious.safetensors")

    ollama_model: str = "goekdenizguelmez/JOSIEFIED-Qwen3:8b-q6_k"

    ollama_vision_model: str = "qwen2.5vl:7b"

    ollama_base_url: str = "http://localhost:11434"

    # ==============================================================
    # PERFORMANCE AND GPU SETTINGS
    # ==============================================================

    cuda_settings: dict | None = None

    generation_defaults: dict | None = None

    gpu_backend: str = "cuda"

    load_models_on_startup: bool = True

    # ==============================================================
    # STORAGE AND UI SETTINGS
    # ==============================================================

    gallery_dir: str = str(_GALLERY_BASE / "gallery")

    memory_guardian: dict | None = None

    memory_stats_refresh_interval: float = 2.0

    @validator("sd_model")
    def _check_sd_model(cls, v: str) -> str:
        if v and not Path(v).exists():
            warnings.warn(f"SDXL model file not found: {v}", stacklevel=2)
        return v

    @validator("gallery_dir")
    def _ensure_gallery_dir(cls, v: str) -> str:
        try:
            Path(v).mkdir(parents=True, exist_ok=True)
        except Exception as e:  # pragma: no cover - unlikely to fail in tests
            msg = f"Unable to create gallery directory {v}: {e}"
            warnings.warn(msg, stacklevel=2)
        return v

    @validator("cuda_settings", pre=True, always=True)
    def _default_cuda(cls, v):
        return v or {
            "device": "cuda:0",
            "dtype": "float16",
            "enable_tf32": True,
            "memory_fraction": 0.95,
        }

    @validator("generation_defaults", pre=True, always=True)
    def _default_generation(cls, v):
        return v or {
            "steps": 30,
            "guidance_scale": 7.5,
            "width": 1024,
            "height": 1024,
            "batch_size": 1,
        }

    def as_dict(self) -> dict:
        return self.dict()

    def apply_hardware_profile(self) -> "SDXLConfig":
        """Detect hardware and apply optimal settings automatically."""
        logger = logging.getLogger(__name__)
        profiler = HardwareProfiler()
        profile = profiler.detect_hardware()

        self.cuda_settings = {
            "device": "cuda:0" if profile.vram_gb > 0 else "cpu",
            "dtype": "float16" if profile.vram_gb > 0 else "float32",
            "enable_tf32": profile.enable_tf32,
            "memory_fraction": 0.95,
        }

        gen = self.generation_defaults or {}
        gen.update(
            {
                "steps": profile.recommended_steps,
                "width": profile.recommended_resolution[0],
                "height": profile.recommended_resolution[1],
                "batch_size": profile.recommended_batch_size,
            }
        )
        self.generation_defaults = gen

        logger.info(
            "Applied '%s' hardware profile: %dx%d, %d steps",
            profile.profile_name,
            profile.recommended_resolution[0],
            profile.recommended_resolution[1],
            profile.recommended_steps,
        )

        return self


# ==================================================================
# CONFIGURATION LOADING AND MANAGEMENT
# ==================================================================


def load_config(path: str | None = None) -> SDXLConfig:
    """Load configuration from YAML with environment variable overrides.

    The priority order is:
    1. SDXLConfig defaults
    2. Values from the YAML file if present
    3. Environment variable overrides

    Args:
        path: Optional path to YAML config. Defaults to ``config.yaml`` or the
            ``CONFIG_FILE`` environment variable.

    Returns:
        SDXLConfig: Fully configured configuration instance.

    Environment Variables:
        - ``CONFIG_FILE``: Path to configuration file
        - ``SD_MODEL``: Path to Stable Diffusion model
        - ``OLLAMA_MODEL``: Name of Ollama language model
        - ``OLLAMA_BASE_URL``: Ollama server URL
        - ``GPU_BACKEND``: GPU backend type
        - ``GALLERY_DIR``: Gallery directory path
        - ``MEMORY_STATS_REFRESH_INTERVAL``: UI refresh interval
        - ``LOAD_MODELS_ON_STARTUP``: Whether to load models on startup
        - ``MODELS_DIR``: Base directory for model files
    """
    # Start with defaults
    cfg_data: dict = {}

    # Load from YAML file if it exists
    cfg_path = Path(path or os.getenv("CONFIG_FILE", "config.yaml"))
    if cfg_path.exists():
        try:
            with open(cfg_path, "r") as f:
                file_data = yaml.safe_load(f) or {}
            if isinstance(file_data, dict):
                cfg_data.update(file_data)
        except Exception as e:
            logger.warning("Error loading config file %s: %s", cfg_path, e)

    # Environment variable overrides
    env_map = {
        "sd_model": "SD_MODEL",
        "ollama_model": "OLLAMA_MODEL",
        "ollama_base_url": "OLLAMA_BASE_URL",
        "gpu_backend": "GPU_BACKEND",
        "gallery_dir": "GALLERY_DIR",
    }
    for key, env_var in env_map.items():
        val = os.getenv(env_var)
        if val is not None:
            cfg_data[key] = val

    refresh_val = os.getenv("MEMORY_STATS_REFRESH_INTERVAL")
    if refresh_val is not None:
        try:
            cfg_data["memory_stats_refresh_interval"] = float(refresh_val)
        except ValueError:
            logger.warning(
                "Invalid MEMORY_STATS_REFRESH_INTERVAL value: %s", refresh_val
            )

    env_lazy = os.getenv("LOAD_MODELS_ON_STARTUP")
    if env_lazy is not None:
        cfg_data["load_models_on_startup"] = env_lazy.lower() not in (
            "false",
            "0",
            "no",
        )

    return SDXLConfig(**cfg_data)


# ==================================================================
# GLOBAL CONFIGURATION INSTANCE
# ==================================================================

# Global configuration instance loaded at module import
# This is the primary configuration object used throughout the application
CONFIG = load_config()
