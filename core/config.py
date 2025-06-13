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
- Type-safe configuration with dataclasses

Configuration Priority (highest to lowest):
1. Environment variables
2. YAML configuration file
3. Default values in AppConfig dataclass
"""

from dataclasses import dataclass, fields, asdict
import os
from pathlib import Path
import tempfile
import yaml

# ==================================================================
# CONSTANTS AND PATHS
# ==================================================================

# Default directory for AI model files
# Can be overridden with MODELS_DIR environment variable
MODEL_DIR = Path(os.getenv("MODELS_DIR", "models"))


# ==================================================================
# MAIN CONFIGURATION CLASS
# ==================================================================

@dataclass
class AppConfig:
    """
    Main configuration class for Illustrious AI Studio.
    
    This dataclass defines all configurable aspects of the application
    with sensible defaults. All fields can be overridden via YAML
    configuration file or environment variables.
    
    Attributes:
        sd_model: Path to the Stable Diffusion XL model file
        ollama_model: Name of the default Ollama language model
        ollama_vision_model: Name of the Ollama vision/multimodal model
        ollama_base_url: Base URL for Ollama API server
        cuda_settings: GPU/CUDA configuration dictionary
        generation_defaults: Default parameters for image generation
        gpu_backend: GPU backend to use ('cuda', 'mps', 'cpu')
        load_models_on_startup: Whether to load models during startup
        gallery_dir: Directory for storing generated images
        memory_guardian: Memory management configuration
        memory_stats_refresh_interval: UI refresh rate for memory stats
    """
    
    # ==============================================================
    # MODEL CONFIGURATION
    # ==============================================================
    
    sd_model: str = str(MODEL_DIR / "Illustrious.safetensors")
    """Path to the Stable Diffusion XL model file"""
    
    ollama_model: str = "goekdenizguelmez/JOSIEFIED-Qwen3:8b-q6_k"
    """Default Ollama language model for text generation and chat"""
    
    ollama_vision_model: str = "qwen2.5vl:7b"
    """Ollama vision model for image analysis and multimodal tasks"""
    
    ollama_base_url: str = "http://localhost:11434"
    """Base URL for the Ollama API server"""
    
    # ==============================================================
    # PERFORMANCE AND GPU SETTINGS
    # ==============================================================
    
    cuda_settings: dict | None = None
    """GPU/CUDA configuration - set in __post_init__ if None"""
    
    generation_defaults: dict | None = None
    """Default image generation parameters - set in __post_init__ if None"""
    
    gpu_backend: str = "cuda"
    """GPU backend: 'cuda' for NVIDIA, 'mps' for Apple Silicon, 'cpu' for CPU-only"""
    
    load_models_on_startup: bool = True
    """Whether to initialize models during application startup"""

    # ==============================================================
    # STORAGE AND UI SETTINGS
    # ==============================================================
    
    gallery_dir: str = str(Path(tempfile.gettempdir()) / "illustrious_ai" / "gallery")
    """Directory for storing generated images in the gallery"""
    
    memory_guardian: dict | None = None
    """Memory management system configuration"""

    memory_stats_refresh_interval: float = 2.0
    """Refresh interval (in seconds) for memory statistics in the UI"""

    def __post_init__(self):
        """
        Initialize default values for complex dictionary fields.
        
        This method is called automatically after dataclass initialization
        to set up default values for fields that couldn't be set directly
        in the field definitions.
        """
        # Set default CUDA/GPU settings if not provided
        if self.cuda_settings is None:
            self.cuda_settings = {
                "device": "cuda:0",           # Primary GPU device
                "dtype": "float16",           # Use half precision for memory efficiency
                "enable_tf32": True,          # Enable TensorFloat-32 for performance
                "memory_fraction": 0.95       # Use 95% of available GPU memory
            }
            
        # Set default image generation parameters if not provided
        if self.generation_defaults is None:
            self.generation_defaults = {
                "steps": 30,                  # Number of denoising steps
                "guidance_scale": 7.5,        # How closely to follow the prompt
                "width": 1024,                # Image width in pixels
                "height": 1024,               # Image height in pixels
                "batch_size": 1               # Number of images per generation
            }

    def as_dict(self) -> dict:
        """
        Convert configuration to dictionary format.
        
        Returns:
            dict: Configuration as a dictionary suitable for serialization
        """
        return asdict(self)

# ==================================================================
# CONFIGURATION LOADING AND MANAGEMENT
# ==================================================================

def load_config(path: str | None = None) -> AppConfig:
    """
    Load configuration from YAML file with environment variable overrides.
    
    This function implements a multi-layer configuration system:
    1. Start with AppConfig defaults
    2. Override with values from YAML file (if exists)
    3. Apply environment variable overrides
    
    Args:
        path: Optional path to YAML config file. If None, uses CONFIG_FILE 
              environment variable or defaults to 'config.yaml'
    
    Returns:
        AppConfig: Fully configured AppConfig instance
        
    Environment Variables (override YAML and defaults):
        CONFIG_FILE: Path to configuration file
        SD_MODEL: Path to Stable Diffusion model
        OLLAMA_MODEL: Name of Ollama language model
        OLLAMA_BASE_URL: Ollama server URL
        GPU_BACKEND: GPU backend type
        GALLERY_DIR: Gallery directory path
        MEMORY_STATS_REFRESH_INTERVAL: UI refresh interval
        LOAD_MODELS_ON_STARTUP: Whether to load models on startup
        MODELS_DIR: Base directory for model files
    """
    # Start with default configuration
    config = AppConfig()
    
    # Load from YAML file if it exists
    cfg_path = Path(path or os.getenv("CONFIG_FILE", "config.yaml"))
    if cfg_path.exists():
        try:
            with open(cfg_path, "r") as f:
                data = yaml.safe_load(f) or {}
            
            # Apply YAML values to config fields
            for field in fields(AppConfig):
                if field.name in data:
                    setattr(config, field.name, data[field.name])
                    
        except Exception as e:
            # Log error but continue with defaults
            print(f"Warning: Error loading config file {cfg_path}: {e}")
    
    # Apply environment variable overrides (highest priority)
    config.sd_model = os.getenv("SD_MODEL", config.sd_model)
    config.ollama_model = os.getenv("OLLAMA_MODEL", config.ollama_model)
    config.ollama_base_url = os.getenv("OLLAMA_BASE_URL", config.ollama_base_url)
    config.gpu_backend = os.getenv("GPU_BACKEND", config.gpu_backend)
    config.gallery_dir = os.getenv("GALLERY_DIR", config.gallery_dir)
    
    # Handle numeric environment variables with error checking
    refresh_val = os.getenv("MEMORY_STATS_REFRESH_INTERVAL")
    if refresh_val is not None:
        try:
            config.memory_stats_refresh_interval = float(refresh_val)
        except ValueError:
            print(f"Warning: Invalid MEMORY_STATS_REFRESH_INTERVAL value: {refresh_val}")
    
    # Handle boolean environment variable
    env_lazy = os.getenv("LOAD_MODELS_ON_STARTUP")
    if env_lazy is not None:
        config.load_models_on_startup = env_lazy.lower() not in ("false", "0", "no")
    
    return config


# ==================================================================
# GLOBAL CONFIGURATION INSTANCE
# ==================================================================

# Global configuration instance loaded at module import
# This is the primary configuration object used throughout the application
CONFIG = load_config()
