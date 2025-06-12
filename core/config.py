from dataclasses import dataclass, fields, asdict
import os
from pathlib import Path
import yaml

MODEL_DIR = Path(os.getenv("MODELS_DIR", "models"))


@dataclass
class AppConfig:
    sd_model: str = str(MODEL_DIR / "Illustrious.safetensors")
    ollama_model: str = "goekdenizguelmez/JOSIEFIED-Qwen3:8b-q6_k"
    ollama_vision_model: str = "qwen2.5vl:7b"
    ollama_base_url: str = "http://localhost:11434"
    
    # Performance settings
    cuda_settings: dict = None
    generation_defaults: dict = None
    gpu_backend: str = "cuda"
    load_models_on_startup: bool = True

    def __post_init__(self):
        if self.cuda_settings is None:
            self.cuda_settings = {
                "device": "cuda:0",
                "dtype": "float16",
                "enable_tf32": True,
                "memory_fraction": 0.95
            }
        if self.generation_defaults is None:
            self.generation_defaults = {
                "steps": 30,
                "guidance_scale": 7.5,
                "width": 1024,
                "height": 1024,
                "batch_size": 1
            }

    def as_dict(self) -> dict:
        return asdict(self)


def load_config(path: str | None = None) -> AppConfig:
    """Load configuration from YAML file and environment variables."""
    config = AppConfig()
    cfg_path = Path(path or os.getenv("CONFIG_FILE", "config.yaml"))
    if cfg_path.exists():
        with open(cfg_path, "r") as f:
            data = yaml.safe_load(f) or {}
        for field in fields(AppConfig):
            if field.name in data:
                setattr(config, field.name, data[field.name])
    # environment overrides
    config.sd_model = os.getenv("SD_MODEL", config.sd_model)
    config.ollama_model = os.getenv("OLLAMA_MODEL", config.ollama_model)
    config.ollama_base_url = os.getenv("OLLAMA_BASE_URL", config.ollama_base_url)
    config.gpu_backend = os.getenv("GPU_BACKEND", config.gpu_backend)
    env_lazy = os.getenv("LOAD_MODELS_ON_STARTUP")
    if env_lazy is not None:
        config.load_models_on_startup = env_lazy.lower() not in ("false", "0", "no")
    return config


CONFIG = load_config()
