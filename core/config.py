from dataclasses import dataclass, fields, asdict
import os
from pathlib import Path
import yaml


@dataclass
class AppConfig:
    sd_model: str = "/home/ant/AI/Project/SDXL Models/waiNSFWIllustrious_v140.safetensors"
    ollama_model: str = "goekdenizguelmez/JOSIEFIED-Qwen3:8b-q6_k"
    ollama_base_url: str = "http://localhost:11434"

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
    return config


CONFIG = load_config()
