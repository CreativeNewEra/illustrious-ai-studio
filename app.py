"""Compatibility wrapper importing the main application components."""
from server.api import app, GenerateImageRequest, ChatRequest, AnalyzeImageRequest, run_mcp_server
from ui.web import create_gradio_app
import core.sdxl as sdxl
import core.ollama as ollama
from core.sdxl import (
    generate_image,
    init_sdxl,
    save_to_gallery,
    get_latest_image,
    sdxl_pipe,
    MODEL_PATHS,
    TEMP_DIR,
    GALLERY_DIR,
)
from core.ollama import (
    chat_completion,
    handle_chat,
    generate_prompt,
    analyze_image,
    init_ollama,
    ollama_model,
    chat_history_store,
)
from core.memory import clear_cuda_memory, get_model_status, model_status, latest_generated_image


def initialize_models():
    init_sdxl()
    init_ollama()
