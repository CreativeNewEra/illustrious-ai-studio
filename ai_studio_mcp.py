"""Alternate entry point for Illustrious AI Studio.

This thin wrapper simply reuses the application logic defined in
:mod:`app` so we don't maintain two divergent copies.
"""

from app import create_gradio_app, run_mcp_server
import logging
import threading

logger = logging.getLogger(__name__)


if __name__ == "__main__":
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
