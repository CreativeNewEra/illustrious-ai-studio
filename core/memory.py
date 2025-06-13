"""
Illustrious AI Studio - Memory Management

This module provides utilities for managing GPU and system memory usage
in AI applications. It includes functions for:

KEY FEATURES:
- GPU memory cleanup and cache clearing
- Model status monitoring and reporting
- Memory usage statistics and visualization
- Cross-platform GPU backend support (CUDA, ROCm, MPS, CPU)
- Integration with memory guardian system

SUPPORTED BACKENDS:
- CUDA: NVIDIA GPUs (primary support)
- ROCm: AMD GPUs (experimental)
- MPS: Apple Silicon (Metal Performance Shaders)
- CPU: Fallback for systems without GPU acceleration

The module handles missing PyTorch gracefully for testing environments
and provides consistent interfaces regardless of hardware availability.
"""

import gc
import logging

# Optional PyTorch import with graceful fallback for testing
try:
    import torch
except Exception:  # pragma: no cover - allow missing torch in tests
    torch = None  # type: ignore

from .state import AppState
from .config import CONFIG

logger = logging.getLogger(__name__)


# ==================================================================
# GPU MEMORY MANAGEMENT
# ==================================================================

def clear_gpu_memory():
    """
    Clear GPU memory cache and perform garbage collection.
    
    This function performs memory cleanup operations appropriate for
    the configured GPU backend:
    - CUDA/ROCm: Empties CUDA cache and synchronizes device
    - CPU/MPS: Only performs garbage collection
    
    Should be called:
    - After model unloading
    - Before loading new models  
    - During memory pressure situations
    - On application shutdown
    """
    # Clear GPU-specific caches if available
    if torch is not None and CONFIG.gpu_backend in ("cuda", "rocm") and torch.cuda.is_available():
        torch.cuda.empty_cache()    # Free unused memory in cache
        torch.cuda.synchronize()    # Wait for all operations to complete
        
    # Always perform Python garbage collection
    gc.collect()
    
    logger.info("GPU memory cleared and garbage collection performed")


# ==================================================================
# MODEL STATUS REPORTING  
# ==================================================================

def get_model_status(state: AppState) -> str:
    """
    Generate formatted status report for all AI models.
    
    Args:
        state: Application state containing model status information
        
    Returns:
        str: Markdown-formatted status report showing:
             - SDXL model loading status
             - Ollama connection status  
             - Vision model availability
             - GPU backend configuration
             - Hardware availability
    """
    status_text = "🤖 **Model Status:**\n"
    
    # Model availability indicators
    status_text += f"• SDXL: {'✅ Loaded' if state.model_status['sdxl'] else '❌ Not loaded'}\n"
    status_text += f"• Ollama: {'✅ Connected' if state.model_status['ollama'] else '❌ Not connected'}\n"
    status_text += f"• Vision: {'✅ Available' if state.model_status['multimodal'] else '❌ Not available'}\n"
    
    # System configuration
    status_text += f"• Backend: {CONFIG.gpu_backend}\n"
    
    # Hardware availability check
    gpu_available = torch.cuda.is_available() if torch is not None else False
    status_text += f"• GPU: {'✅ Available' if gpu_available else '❌ Not available'}"
    
    return status_text


# ==================================================================
# MEMORY STATISTICS AND MONITORING
# ==================================================================

def get_memory_stats_markdown(state: AppState) -> str:
    """
    Generate formatted memory usage statistics.
    
    Args:
        state: Application state for memory guardian access
        
    Returns:
        str: Markdown-formatted memory statistics including:
             - GPU memory usage and total capacity
             - System RAM usage percentage
             - Memory pressure level indicator
             - Error message if GPU unavailable
    """
    # Import here to avoid circular dependencies
    from .memory_guardian import get_memory_guardian
    
    guardian = get_memory_guardian(state)
    stats = guardian.get_memory_stats()
    
    # Handle case where GPU statistics are unavailable
    if not stats:
        return "🚫 GPU not available"

    # Format memory statistics for display
    text = "🧠 **Memory Usage:**\n"
    text += f"• GPU: {stats.gpu_reserved_gb:.1f}/{stats.gpu_total_gb:.1f} GB ({stats.gpu_usage_percent:.1f}%)\n"
    text += f"• RAM: {stats.system_ram_usage_percent:.1f}% used\n"
    text += f"• Pressure: {stats.pressure_level.value}"
    
    return text


def get_memory_stats_wrapper(state: AppState) -> str:
    """
    Compatibility wrapper for memory statistics.
    
    This function exists to maintain backward compatibility with
    existing code that expects this specific function name.
    
    Args:
        state: Application state for memory access
        
    Returns:
        str: Formatted memory statistics (same as get_memory_stats_markdown)
    """
    return get_memory_stats_markdown(state)
