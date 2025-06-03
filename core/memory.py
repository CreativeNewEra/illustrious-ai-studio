import gc
import logging
import torch

from .state import AppState

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def clear_cuda_memory():
    """Clear CUDA cache and run garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    logger.info("CUDA memory cleared and garbage collection performed")


def get_model_status(state: AppState) -> str:
    """Return formatted Markdown describing current model availability."""
    status_text = "🤖 **Model Status:**\n"
    status_text += f"• SDXL: {'✅ Loaded' if state.model_status['sdxl'] else '❌ Not loaded'}\n"
    status_text += f"• Ollama: {'✅ Connected' if state.model_status['ollama'] else '❌ Not connected'}\n"
    status_text += f"• Vision: {'✅ Available' if state.model_status['multimodal'] else '❌ Not available'}\n"
    status_text += f"• CUDA: {'✅ Available' if torch.cuda.is_available() else '❌ Not available'}"
    return status_text
