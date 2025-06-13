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


def get_memory_stats_markdown(state: AppState) -> str:
    """Return formatted Markdown for current memory usage."""
    from .memory_guardian import get_memory_guardian  # Lazy import to avoid circular dependency
    guardian = get_memory_guardian(state)
    stats = guardian.get_memory_stats()
    if not stats:
        return "🚫 GPU not available"

    text = "🧠 **Memory Usage:**\n"
    text += f"• GPU: {stats.gpu_reserved_gb:.1f}/{stats.gpu_total_gb:.1f} GB ({stats.gpu_usage_percent:.1f}%)\n"
    text += f"• RAM: {stats.system_ram_usage_percent:.1f}% used\n"
    text += f"• Pressure: {stats.pressure_level.value}"
    return text
