from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from PIL import Image
from diffusers import StableDiffusionXLPipeline

@dataclass
class AppState:
    """Holds runtime objects for the application."""

    sdxl_pipe: Optional[StableDiffusionXLPipeline] = None
    ollama_model: Optional[str] = None
    model_status: Dict[str, bool] = field(
        default_factory=lambda: {"sdxl": False, "ollama": False, "multimodal": False}
    )
    chat_history_store: Dict[str, List[Tuple[str, str]]] = field(default_factory=dict)
    latest_generated_image: Optional[Image.Image] = None
