"""Alternate entry point for Illustrious AI Studio."""
import logging
import threading

from ui.web import create_gradio_app
from server.api import run_mcp_server
from core.state import AppState

logger = logging.getLogger(__name__)


app_state = AppState()


if __name__ == "__main__":
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
