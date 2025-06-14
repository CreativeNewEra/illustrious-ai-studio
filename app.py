"""
Illustrious AI Studio - Compatibility Wrapper & Quick Access Module

This module serves as a compatibility wrapper and convenience layer that provides:
- Ready-to-use FastAPI app instance for external integrations
- Quick access to core functionality without complex initialization
- Backward compatibility for existing integrations
- Simplified API for common operations

This is particularly useful for:
- External applications that need to integrate with the AI Studio
- Quick scripts that need access to image generation or chat functionality
- Testing and development scenarios
- MCP (Model Context Protocol) server implementations

The module exposes all major components and provides a pre-configured
application state and FastAPI app instance.
"""

# ==================================================================
# CORE API AND UI COMPONENTS
# ==================================================================

from server.api import create_api_app, GenerateImageRequest, ChatRequest, AnalyzeImageRequest, run_mcp_server  # noqa: F401
# These imports are exposed for external use as part of the module's convenience layer.
# They allow external scripts and integrations to access core API functionality directly.
from ui.web import create_gradio_app  # noqa: F401
from core.state import AppState

# ==================================================================
# IMAGE GENERATION FUNCTIONALITY (SDXL)
# ==================================================================

from core.sdxl import (  # noqa: F401
    generate_image,        # Main image generation function
    init_sdxl,            # Initialize SDXL model pipeline
    save_to_gallery,      # Save generated images to gallery
    get_latest_image,     # Retrieve the most recently generated image
    TEMP_DIR,             # Temporary directory for processing
    GALLERY_DIR,          # Gallery directory for saved images
)

# ==================================================================
# CHAT AND LANGUAGE MODEL FUNCTIONALITY (OLLAMA)
# ==================================================================

from core.ollama import (  # noqa: F401
    chat_completion,      # Direct chat completion API
    handle_chat,          # High-level chat handling with history
    generate_prompt,      # AI-assisted prompt generation
    analyze_image,        # Multi-modal image analysis
    init_ollama,          # Initialize Ollama connection and models
)

# ==================================================================
# MEMORY AND SYSTEM UTILITIES
# ==================================================================

from core.memory import clear_gpu_memory, get_model_status  # noqa: F401

# ==================================================================
# GLOBAL APPLICATION STATE AND INSTANCES
# ==================================================================

# Single shared application state used when this module is imported
# This ensures consistency across all components using this wrapper
app_state = AppState()

# Pre-configured FastAPI application instance
# Ready to use for external integrations or direct deployment
app = create_api_app(app_state)


# ==================================================================
# CONVENIENCE FUNCTIONS
# ==================================================================

def initialize_models():
    """
    Initialize all AI models (SDXL and Ollama).
    
    This is a convenience function that initializes both the SDXL
    image generation pipeline and Ollama language models using the
    shared application state.
    
    Note: This function blocks until all models are loaded, which
    may take several minutes depending on your hardware and the
    size of the models.
    
    Usage:
        import app
        app.initialize_models()
        # Models are now ready for use
    """
    init_sdxl(app_state)
    init_ollama(app_state)
