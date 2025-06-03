import logging
import threading

from ui.web import create_gradio_app
from server.api import run_mcp_server
from core.sdxl import init_sdxl
from core.ollama import init_ollama
from core.state import AppState

logger = logging.getLogger(__name__)


app_state = AppState()


def initialize_models() -> None:
    init_sdxl(app_state)
    init_ollama(app_state)


if __name__ == "__main__":
    initialize_models()
    mcp_thread = threading.Thread(target=run_mcp_server, args=(app_state,), daemon=True)
    mcp_thread.start()
    logger.info("MCP Server started on http://localhost:8000")
    gradio_app = create_gradio_app(app_state)
    gradio_app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        auth=None,
        show_error=True,
    )
