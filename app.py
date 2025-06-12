"""Compatibility wrapper that provides a ready-to-use FastAPI app and helpers."""
from server.api import create_api_app, GenerateImageRequest, ChatRequest, AnalyzeImageRequest, run_mcp_server
from ui.web import create_gradio_app
from core.state import AppState
from core.sdxl import (
    generate_image,
    init_sdxl,
    save_to_gallery,
    get_latest_image,
    TEMP_DIR,
    GALLERY_DIR,
)
from core.ollama import (
    chat_completion,
    handle_chat,
    generate_prompt,
    analyze_image,
    init_ollama,
)
from core.memory import clear_gpu_memory, get_model_status

# Single application state used when this module is imported
app_state = AppState()
app = create_api_app(app_state)


def initialize_models():
    init_sdxl(app_state)
    init_ollama(app_state)
