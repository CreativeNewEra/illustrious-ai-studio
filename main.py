import argparse
import logging
import logging.handlers
import os
import signal
import threading
from pathlib import Path

from ui.web import create_gradio_app
from server.api import create_api_app
from core.sdxl import init_sdxl
from core.ollama import init_ollama
from core.state import AppState
from core.memory import clear_gpu_memory

import uvicorn


class IllustriousAIStudio:
    """Application runner with CLI support."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self._configure_environment()
        self.logger = self._setup_logging(args.log_level)
        self.app_state = AppState()
        self.api_server: uvicorn.Server | None = None
        self.api_thread: threading.Thread | None = None
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

    # ------------------------------------------------------------------
    @staticmethod
    def parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="Illustrious AI Studio")
        parser.add_argument("--lazy-load", action="store_true", help="Defer model initialization")
        parser.add_argument("--no-sdxl", action="store_true", help="Skip SDXL initialization")
        parser.add_argument("--no-ollama", action="store_true", help="Skip Ollama initialization")
        parser.add_argument("--web-port", type=int, default=7860, help="Gradio port")
        parser.add_argument("--api-port", type=int, default=8000, help="API server port")
        parser.add_argument("--no-api", action="store_true", help="Do not start API server")
        parser.add_argument("--auth", help="Gradio auth in user:pass or u1:p1,u2:p2 format")
        parser.add_argument("--open-browser", action="store_true", help="Open browser on launch")
        parser.add_argument("--optimize-memory", action="store_true", help="Enable memory optimizations")
        parser.add_argument("--log-level", default="INFO", help="Logging level")
        return parser.parse_args()

    # ------------------------------------------------------------------
    def _configure_environment(self) -> None:
        os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "false")
        if self.args.optimize_memory:
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            os.environ.setdefault("OLLAMA_KEEP_ALIVE", "0")

    # ------------------------------------------------------------------
    @staticmethod
    def _setup_logging(level: str) -> logging.Logger:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "illustrious_ai_studio.log", maxBytes=10 * 1024 * 1024, backupCount=5
        )
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

        root = logging.getLogger()
        root.setLevel(level.upper())
        root.addHandler(file_handler)
        root.addHandler(console_handler)
        logging.getLogger("gradio").setLevel(logging.INFO)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        return logging.getLogger(__name__)

    # ------------------------------------------------------------------
    def _start_api_server(self) -> None:
        app = create_api_app(self.app_state, auto_load=not self.args.lazy_load)
        config = uvicorn.Config(app, host="0.0.0.0", port=self.args.api_port, log_level="info")
        self.api_server = uvicorn.Server(config)
        self.api_thread = threading.Thread(target=self.api_server.run, daemon=True)
        self.api_thread.start()
        self.logger.info("✓ MCP Server started on http://localhost:%s", self.args.api_port)

    # ------------------------------------------------------------------
    def _launch_gradio(self) -> None:
        gradio_app = create_gradio_app(self.app_state)
        auth = None
        if self.args.auth:
            creds = [tuple(pair.split(":", 1)) for pair in self.args.auth.split(",")]
            auth = creds[0] if len(creds) == 1 else creds
        gradio_app.launch(
            server_name="0.0.0.0",
            server_port=self.args.web_port,
            share=False,
            auth=auth,
            show_error=True,
            inbrowser=self.args.open_browser,
        )

    # ------------------------------------------------------------------
    def run(self) -> None:
        self.logger.info("%s", "=" * 50)
        self.logger.info("🎨 Starting Illustrious AI Studio")
        self.logger.info("%s", "=" * 50)

        if not self.args.lazy_load:
            if not self.args.no_sdxl:
                init_sdxl(self.app_state)
            if not self.args.no_ollama:
                init_ollama(self.app_state)

        if not self.args.no_api:
            self._start_api_server()

        self._launch_gradio()
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)
        signal.pause()  # Wait for termination signals
    # ------------------------------------------------------------------
    def _shutdown(self, *args) -> None:
        if self.api_server:
            self.api_server.should_exit = True
        clear_gpu_memory()
        if self.api_thread and self.api_thread.is_alive():
            self.api_thread.join(timeout=5)


if __name__ == "__main__":
    studio = IllustriousAIStudio(IllustriousAIStudio.parse_args())
    studio.run()
