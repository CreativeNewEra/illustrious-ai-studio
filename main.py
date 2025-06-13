import argparse
import logging
import logging.handlers
import os
import signal
import threading
import sys
from pathlib import Path

from ui.web import create_gradio_app
from server.api import create_api_app
from core.sdxl import init_sdxl
from core.ollama import init_ollama
from core.state import AppState
from core.memory import clear_gpu_memory
from core.memory_guardian import start_memory_guardian, stop_memory_guardian

import uvicorn


def force_exit_after_timeout() -> threading.Timer:
    """Force terminate the process if graceful shutdown hangs."""
    timer = threading.Timer(10.0, lambda: os._exit(1))
    timer.start()
    return timer

def create_parser() -> argparse.ArgumentParser:
    """Return argument parser for the CLI."""
    parser = argparse.ArgumentParser(description="Illustrious AI Studio")
    parser.add_argument("--quick-start", action="store_true", help="Skip all model initialization for fastest startup")
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
    return parser

class IllustriousAIStudio:
    """Application runner with CLI support."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self._configure_environment()
        self.logger = self._setup_logging(args.log_level)
        self.app_state = AppState()
        self.api_server: uvicorn.Server | None = None
        self.api_thread: threading.Thread | None = None
        # Moved signal handling to run() to ensure it's set up before pause()
        # and after Gradio might fork or manage signals.

    # ------------------------------------------------------------------
    @staticmethod
    def parse_args() -> argparse.Namespace:
        return create_parser().parse_args()

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
        
        # Set up signal handlers here, just before Gradio launch,
        # as Gradio might interfere with signal handling if set too early.
        signal.signal(signal.SIGINT, self._shutdown_handler)
        signal.signal(signal.SIGTERM, self._shutdown_handler)
        
        gradio_app.launch(
            server_name="127.0.0.1",
            server_port=self.args.web_port,
            share=True,
            auth=auth,
            show_error=True,
            inbrowser=self.args.open_browser,
            allowed_paths=["."],
        )

    # ------------------------------------------------------------------
    def run(self) -> None:
        self.logger.info("%s", "=" * 50)
        self.logger.info("🎨 Starting Illustrious AI Studio")
        self.logger.info("%s", "=" * 50)

        # Start Memory Guardian
        self.logger.info("🛡️ Starting Memory Guardian...")
        start_memory_guardian(self.app_state)
        self.logger.info("✅ Memory Guardian started")
        
        # Model initialization with progress feedback
        if self.args.quick_start:
            self.logger.info("🚀 Quick start mode - skipping all model initialization")
            self.logger.info("💡 Models can be loaded later through the web interface")
        elif not self.args.lazy_load:
            self.logger.info("🔄 Initializing models...")
            self._initialize_models_with_progress()
        else:
            self.logger.info("⏳ Lazy load mode - models will be initialized on first use")

        # Start API server
        if not self.args.no_api:
            self.logger.info("🌐 Starting API server...")
            self._start_api_server()

        # Launch Gradio interface
        self.logger.info("🖥️ Launching web interface...")
        self._launch_gradio()

    # ------------------------------------------------------------------
    def _initialize_models_with_progress(self) -> None:
        """Initialize models with detailed progress feedback and error handling."""
        import time
        
        # Pre-flight checks
        self.logger.info("🔍 Running pre-flight checks...")
        
        if not self.args.no_sdxl:
            self.logger.info("📋 Checking SDXL model...")
            if self._check_sdxl_model():
                self.logger.info("✅ SDXL model file validated")
                self.logger.info("🔄 Loading SDXL model (this may take a few minutes)...")
                start_time = time.time()
                try:
                    result = init_sdxl(self.app_state)
                    elapsed = time.time() - start_time
                    if result:
                        self.logger.info(f"✅ SDXL model loaded successfully in {elapsed:.1f}s")
                    else:
                        self.logger.warning("⚠️ SDXL model failed to load - continuing without it")
                except Exception as e:
                    elapsed = time.time() - start_time
                    self.logger.error(f"❌ SDXL initialization failed after {elapsed:.1f}s: {e}")
                    self.logger.info("💡 Try using --no-sdxl flag to skip SDXL initialization")
            else:
                self.logger.warning("⚠️ SDXL model check failed - skipping SDXL initialization")
        
        if not self.args.no_ollama:
            self.logger.info("📋 Checking Ollama connection...")
            if self._check_ollama_connection():
                self.logger.info("✅ Ollama server is accessible")
                self.logger.info("🔄 Initializing Ollama models...")
                start_time = time.time()
                try:
                    result = init_ollama(self.app_state)
                    elapsed = time.time() - start_time
                    if result:
                        self.logger.info(f"✅ Ollama initialized successfully in {elapsed:.1f}s")
                    else:
                        self.logger.warning("⚠️ Ollama initialization failed - continuing without it")
                except Exception as e:
                    elapsed = time.time() - start_time
                    self.logger.error(f"❌ Ollama initialization failed after {elapsed:.1f}s: {e}")
                    self.logger.info("💡 Try using --no-ollama flag to skip Ollama initialization")
            else:
                self.logger.warning("⚠️ Ollama connection check failed - skipping Ollama initialization")
        
        self.logger.info("🎉 Model initialization completed!")

    # ------------------------------------------------------------------
    def _check_sdxl_model(self) -> bool:
        """Check if SDXL model file exists and is valid."""
        from core.config import CONFIG
        import os
        
        model_path = CONFIG.sd_model
        if not os.path.exists(model_path):
            self.logger.error(f"❌ SDXL model file not found: {model_path}")
            self.logger.info("💡 Please check your model path in config.yaml or use --no-sdxl flag")
            return False
        
        # Check file size (SDXL models should be at least 1GB)
        try:
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            if size_mb < 1000:
                self.logger.error(f"❌ SDXL model file too small ({size_mb:.1f}MB) - possibly corrupted")
                return False
            self.logger.info(f"📁 Model file: {model_path} ({size_mb:.1f}MB)")
            return True
        except Exception as e:
            self.logger.error(f"❌ Error checking model file: {e}")
            return False

    # ------------------------------------------------------------------
    def _check_ollama_connection(self) -> bool:
        """Check if Ollama server is accessible."""
        from core.config import CONFIG
        import requests
        
        try:
            self.logger.info(f"🔗 Checking connection to {CONFIG.ollama_base_url}")
            response = requests.get(f"{CONFIG.ollama_base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                self.logger.info(f"📚 Found {len(models)} Ollama models available")
                return True
            else:
                self.logger.error(f"❌ Ollama server responded with status {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            self.logger.error("❌ Cannot connect to Ollama server - is it running?")
            self.logger.info("💡 Start Ollama with: ollama serve")
            return False
        except requests.exceptions.Timeout:
            self.logger.error("❌ Ollama server connection timeout")
            return False
        except Exception as e:
            self.logger.error(f"❌ Ollama connection check failed: {e}")
            return False

    # ------------------------------------------------------------------
    def _shutdown_handler(self, signum, frame) -> None: # Renamed to avoid conflict if imported elsewhere
        self.logger.info(f"Shutdown initiated by signal {signum}")
        # Gradio's launch() should handle SIGINT/SIGTERM to close itself.
        # This custom handler is for additional cleanup.
        
        # Stop Memory Guardian
        self.logger.info("🛡️ Stopping Memory Guardian")
        stop_memory_guardian()
        
        if self.api_server and hasattr(self.api_server, 'should_exit'):
            self.api_server.should_exit = True
        
        clear_gpu_memory()
        
        if self.api_thread and self.api_thread.is_alive():
            self.logger.debug("Joining API thread...")
            self.api_thread.join(timeout=5)
            self.logger.debug("API thread joined")

        self.logger.info("Application exiting")
        timer = force_exit_after_timeout()
        timer.cancel()  # Cancel the timer to prevent forced termination
        sys.exit(0)  # Allow proper cleanup and exit handlers

if __name__ == "__main__":
    args = IllustriousAIStudio.parse_args()
    studio = IllustriousAIStudio(args)
    studio.run()
