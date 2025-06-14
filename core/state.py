"""
Illustrious AI Studio - Application State Management

This module defines the central application state container that holds
all runtime objects and state information shared across the application.

The AppState class serves as a singleton-like container for:
- Loaded AI model instances (SDXL, Ollama)
- Model status and availability flags
- Chat conversation history
- Generated images and parameters
- Current project context

This centralized state management approach ensures:
- Consistent state across UI and API components
- Easy sharing of expensive model instances
- Simplified state debugging and monitoring
- Clean separation of concerns
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any, TYPE_CHECKING
from collections import defaultdict, deque
import threading
from contextlib import contextmanager

from .metrics import Metrics
from .degradation import DegradationStrategy

if TYPE_CHECKING:
    from .sdxl import ModelProtocol
else:
    ModelProtocol = Any
from PIL import Image

# Import pipeline type only for type checking to avoid runtime import issues
if TYPE_CHECKING:
    from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline


# ==================================================================
# MAIN APPLICATION STATE CLASS
# ==================================================================

@dataclass
class AppState:
    """
    Central container for all runtime application state.
    
    This class holds all mutable state that needs to be shared between
    different components of the application (UI, API, background tasks).
    Using a dataclass ensures type safety and provides clear documentation
    of what state is available.
    
    The state is designed to be thread-safe for read operations, but
    modifications should be coordinated through appropriate locking
    mechanisms if accessed from multiple threads.
    
    Attributes:
        sdxl_pipe: Loaded Stable Diffusion XL pipeline instance
        ollama_model: Name of the currently loaded Ollama language model
        ollama_vision_model: Name of the currently loaded Ollama vision model
        model_status: Dictionary tracking which models are loaded and ready
        chat_history_store: Per-session chat conversation history
        latest_generated_image: Most recently generated image
        last_generation_params: Parameters used for the last image generation
        current_project: Currently active project identifier
    """
    
    # ==============================================================
    # AI MODEL INSTANCES
    # ==============================================================
    
    sdxl_pipe: Optional[ModelProtocol] = None
    """
    Loaded Stable Diffusion XL pipeline instance.
    
    This is the main image generation model. When None, SDXL functionality
    is not available. Loading this model typically requires several GB of
    GPU memory and can take 1-2 minutes.
    """
    
    ollama_model: Optional[str] = None
    """
    Name of the currently loaded Ollama language model.
    
    This string identifies which Ollama model is active for text generation
    and chat functionality. When None, language model features are disabled.
    """
    
    ollama_vision_model: Optional[str] = None
    """
    Name of the currently loaded Ollama vision/multimodal model.
    
    This model handles image analysis and multimodal tasks. When None,
    image analysis features are not available.
    """
    
    # ==============================================================
    # MODEL STATUS TRACKING
    # ==============================================================
    
    model_status: Dict[str, bool] = field(
        default_factory=lambda: {"sdxl": False, "ollama": False, "multimodal": False}
    )
    """
    Dictionary tracking the loading status of each AI model.
    
    Keys:
        - "sdxl": True if SDXL pipeline is loaded and ready
        - "ollama": True if Ollama language model is available
        - "multimodal": True if Ollama vision model is available
    
    This status dict is used by the UI to show model availability
    and enable/disable relevant features.
    """
    
    # ==============================================================
    # CONVERSATION AND INTERACTION STATE
    # ==============================================================
    
    chat_history_store: Dict[str, deque] = field(
        default_factory=lambda: defaultdict(lambda: deque(maxlen=100))
    )
    """
    Storage for chat conversation history organized by session.
    
    Structure: {session_id: [(user_message, assistant_response), ...]}
    
    Each session maintains its own conversation history, allowing
    multiple concurrent chat sessions with proper context isolation.
    The history is kept in memory and lost on application restart.
    """
    
    # ==============================================================
    # IMAGE GENERATION STATE
    # ==============================================================
    
    latest_generated_image: Optional[Image.Image] = None
    """
    The most recently generated image from SDXL.
    
    This PIL Image object is kept in memory for quick access by the UI
    and API. Used for displaying the latest result and enabling quick
    operations like saving to gallery.
    """
    
    last_generation_params: Optional[Dict[str, Any]] = None
    """
    Parameters used to generate the latest image.
    
    Includes settings like:
        - prompt: Text prompt used
        - steps: Number of denoising steps
        - guidance_scale: How closely to follow the prompt
        - width/height: Image dimensions
        - seed: Random seed for reproducibility
        - model: Model variant used
    
    Used for the "regenerate" feature and parameter display in UI.
    """
    
    # ==============================================================
    # PROJECT AND WORKSPACE STATE
    # ==============================================================
    
    current_project: Optional[str] = None
    """
    Identifier for the currently active project.
    
    Projects provide workspace isolation for different creative
    endeavors. When set, generated images and settings are
    associated with this project context.
    """

    simple_mode: bool = True
    """Flag to indicate if the UI should run in beginner-friendly simple mode."""

    metrics: Metrics = field(default_factory=Metrics)
    """Runtime metrics for performance monitoring."""

    degradation_strategy: DegradationStrategy = field(default_factory=DegradationStrategy)
    """Strategy for adaptive degradation when OOM occurs."""

    # ==============================================================
    # INTERNAL SYNCHRONIZATION
    # ==============================================================

    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    # ==============================================================
    # INTERNAL MEMORY GUARDIAN
    # ==============================================================

    _memory_guardian: Optional["MemoryGuardian"] = field(default=None, init=False, repr=False)

    @contextmanager
    def atomic_operation(self):
        """Context manager for thread-safe state mutations."""
        with self._lock:
            yield self

    def update_chat_history(self, session_id: str, message: tuple) -> None:
        """Thread-safe update of chat history for a session."""
        with self._lock:
            self.chat_history_store[session_id].append(message)

    # ------------------------------------------------------------------
    # Memory guardian management
    # ------------------------------------------------------------------

    @property
    def memory_guardian(self):
        """Lazy-initialize and return the MemoryGuardian instance."""
        if self._memory_guardian is None:
            from .memory_guardian import MemoryGuardian
            self._memory_guardian = MemoryGuardian(self)
        return self._memory_guardian
