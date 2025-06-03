import logging
import threading

from ui.web import create_gradio_app
from server.api import run_mcp_server
from core.sdxl import init_sdxl
from core.ollama import init_ollama

logger = logging.getLogger(__name__)


def initialize_models():
    init_sdxl()
    init_ollama()


if __name__ == "__main__":
    initialize_models()
    mcp_thread = threading.Thread(target=run_mcp_server, daemon=True)
    mcp_thread.start()
    logger.info("MCP Server started on http://localhost:8000")
    gradio_app = create_gradio_app()
    gradio_app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        auth=None,
        show_error=True,
    )
