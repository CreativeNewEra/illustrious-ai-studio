import gc
import logging
import torch

from .state import AppState
from .config import CONFIG

logger = logging.getLogger(__name__)


def clear_gpu_memory():
    """Clear GPU cache and run garbage collection based on backend."""
    if CONFIG.gpu_backend in ("cuda", "rocm") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    logger.info("GPU memory cleared and garbage collection performed")


def get_model_status(state: AppState) -> str:
    """Return formatted Markdown describing current model availability."""
    status_text = "🤖 **Model Status:**\n"
    status_text += f"• SDXL: {'✅ Loaded' if state.model_status['sdxl'] else '❌ Not loaded'}\n"
    status_text += f"• Ollama: {'✅ Connected' if state.model_status['ollama'] else '❌ Not connected'}\n"
    status_text += f"• Vision: {'✅ Available' if state.model_status['multimodal'] else '❌ Not available'}\n"
    status_text += f"• Backend: {CONFIG.gpu_backend}\n"
    status_text += f"• GPU: {'✅ Available' if torch.cuda.is_available() else '❌ Not available'}"
    return status_text
