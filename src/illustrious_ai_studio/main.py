"""
Illustrious AI Studio - Main Application Entry Point

This is the primary entry point for the Illustrious AI Studio application.
It handles:
- CLI argument parsing and configuration
- Model initialization (SDXL and Ollama)
- Memory management and monitoring
- Web interface (Gradio) and API server (FastAPI) startup
- Graceful shutdown handling

The application can run in various modes:
- Quick Start: Skip model initialization for fastest startup
- Lazy Load: Defer model initialization until first use
- Development: Full initialization with debug logging
"""

import argparse
import logging
import logging.handlers
import os
import signal
import threading
import sys
from pathlib import Path

# UI and server components
from .ui.web import create_gradio_app
from .server.api import create_api_app
from .server.logging_utils import RequestIdFilter

# Core functionality
from .core.sdxl import init_sdxl
from .core.ollama import init_ollama
from .core.state import AppState
from .core.memory import clear_gpu_memory
from .core.memory_guardian import start_memory_guardian, stop_memory_guardian
from .core.hardware_profiler import HardwareProfiler

# Web server
import uvicorn


def force_exit_after_timeout() -> threading.Timer:
    """
    Emergency shutdown mechanism.
    
    Creates a timer that will forcefully terminate the process if graceful 
    shutdown takes longer than 10 seconds. This prevents the application 
    from hanging indefinitely during shutdown.
    
    Returns:
        threading.Timer: Timer object that can be cancelled if shutdown completes normally
    """
    timer = threading.Timer(10.0, lambda: os._exit(1))
    timer.start()
    return timer

def create_parser() -> argparse.ArgumentParser:
    """
    Create and configure the command-line argument parser.
    
    Defines all available CLI options for controlling application behavior,
    including model initialization, server ports, authentication, and 
    memory management settings.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(description="Illustrious AI Studio")
    
    # Startup mode options
    parser.add_argument("--quick-start", action="store_true", 
                       help="Skip all model initialization for fastest startup")
    parser.add_argument("--lazy-load", action="store_true", 
                       help="Defer model initialization until first use")
    
    # Model control options
    parser.add_argument("--no-sdxl", action="store_true", 
                       help="Skip SDXL initialization")
    parser.add_argument("--no-ollama", action="store_true", 
                       help="Skip Ollama initialization")
    
    # Server configuration
    parser.add_argument("--web-port", type=int, default=7860, 
                       help="Gradio web interface port")
    parser.add_argument("--api-port", type=int, default=8000, 
                       help="FastAPI server port")
    parser.add_argument("--no-api", action="store_true", 
                       help="Do not start API server")
    
    # Security and UI options
    parser.add_argument("--auth",
                       help="Gradio auth in user:pass or u1:p1,u2:p2 format")
    parser.add_argument("--open-browser", action="store_true",
                       help="Open browser automatically on launch")
    parser.add_argument("--share", action="store_true",
                       help="Create a public shareable link")
    
    # Performance and debugging
    parser.add_argument("--optimize-memory", action="store_true", 
                       help="Enable memory optimizations")
    parser.add_argument("--log-level", default="INFO", 
                       help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    
    # Memory management options
    parser.add_argument(
        "--memory-profile",
        choices=["conservative", "balanced", "aggressive"],
        help="Set memory guardian profile for automatic memory management",
    )
    parser.add_argument(
        "--memory-threshold",
        action="append",
        metavar="LEVEL:PERCENT",
        help="Override memory threshold (format: warning:80, critical:90)",
    )
    return parser

class IllustriousAIStudio:
    """
    Main application class for Illustrious AI Studio.
    
    Manages the complete application lifecycle including:
    - Environment configuration and logging setup
    - Model initialization and health checks
    - Web interface and API server management
    - Memory monitoring and cleanup
    - Graceful shutdown handling
    
    Attributes:
        args: Parsed command-line arguments
        logger: Application logger instance
        app_state: Shared application state container
        api_server: FastAPI server instance (if enabled)
        api_thread: Thread running the API server
    """

    def __init__(self, args: argparse.Namespace) -> None:
        """
        Initialize the application with parsed CLI arguments.
        
        Args:
            args: Parsed command-line arguments from argparse
        """
        self.args = args
        self._configure_environment()
        self.logger = self._setup_logging(args.log_level)
        self.app_state = AppState()

        profiler = HardwareProfiler()
        profile = profiler.detect_hardware()
        from .core.config import CONFIG
        CONFIG.apply_hardware_profile()
        self.app_state.hardware_profile = profile
        self.logger.info(
            "🖥️ Detected: %s (%0.0fGB VRAM, %0.0fGB RAM)",
            profiler.gpu_name or "CPU",
            profile.vram_gb,
            profile.ram_gb,
        )
        self.logger.info(
            "✅ Applied '%s' profile: %dx%d, %d steps",
            profile.profile_name,
            profile.recommended_resolution[0],
            profile.recommended_resolution[1],
            profile.recommended_steps,
        )
        self.api_server: uvicorn.Server | None = None
        self.api_thread: threading.Thread | None = None
        # Signal handling is set up in run() to ensure proper timing with Gradio

    # ==================================================================
    # STATIC METHODS
    # ==================================================================
    
    @staticmethod
    def parse_args() -> argparse.Namespace:
        """
        Parse command-line arguments.
        
        Returns:
            argparse.Namespace: Parsed command-line arguments
        """
        return create_parser().parse_args()

    # ==================================================================
    # PRIVATE METHODS - INITIALIZATION & CONFIGURATION
    # ==================================================================
    
    def _configure_environment(self) -> None:
        """
        Configure system environment variables for optimal performance.
        
        Sets up environment variables for:
        - Gradio analytics (disabled for privacy)
        - PyTorch CUDA memory management (if memory optimization enabled)
        - Ollama keep-alive settings (if memory optimization enabled)
        """
        # Disable Gradio analytics for privacy
        os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "false")
        
        # Apply memory optimizations if requested
        if self.args.optimize_memory:
            # Enable PyTorch CUDA expandable segments for better memory management
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            # Disable Ollama model persistence to save memory
            os.environ.setdefault("OLLAMA_KEEP_ALIVE", "0")

    # ------------------------------------------------------------------
    @staticmethod
    def _setup_logging(level: str) -> logging.Logger:
        """
        Configure application logging with both file and console output.
        
        Creates a rotating file handler for persistent logs and a console
        handler for immediate feedback. Also configures log levels for
        various third-party libraries to reduce noise.
        
        Args:
            level: Logging level (DEBUG, INFO, WARNING, ERROR)
            
        Returns:
            logging.Logger: Configured logger instance
        """
        # Create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Set up rotating file handler (10MB max, 5 backup files)
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "illustrious_ai_studio.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=5
        )
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(request_id)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        file_handler.addFilter(RequestIdFilter())
        
        # Set up console handler with simpler format
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter("%(levelname)s: %(message)s [%(request_id)s]")
        )
        console_handler.addFilter(RequestIdFilter())

        # Configure root logger and add handlers
        root = logging.getLogger()
        root.setLevel(level.upper())
        root.addHandler(file_handler)
        root.addHandler(console_handler)
        
        # Adjust log levels for third-party libraries to reduce noise
        logging.getLogger("gradio").setLevel(logging.INFO)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        
        return logging.getLogger(__name__)

    # ==================================================================
    # PRIVATE METHODS - SERVER MANAGEMENT
    # ==================================================================

    def _start_api_server(self) -> None:
        """
        Start the FastAPI server in a separate thread.
        
        Creates and configures the FastAPI application with the current
        app state, then starts it in a daemon thread so it doesn't
        prevent application shutdown.
        """
        # Create FastAPI app with current state
        app = create_api_app(self.app_state, auto_load=not self.args.lazy_load)
        
        # Configure uvicorn server
        config = uvicorn.Config(
            app, 
            host="0.0.0.0", 
            port=self.args.api_port, 
            log_level="info"
        )
        
        # Start server in daemon thread
        self.api_server = uvicorn.Server(config)
        self.api_thread = threading.Thread(target=self.api_server.run, daemon=True)
        self.api_thread.start()
        
        self.logger.info("✓ API Server started on http://localhost:%s", self.args.api_port)

    # ------------------------------------------------------------------
    def _launch_gradio(self) -> None:
        """
        Launch the Gradio web interface.
        
        Creates and configures the Gradio app with authentication if specified,
        sets up signal handlers for graceful shutdown, and launches the interface
        with appropriate settings for sharing and browser opening.
        """
        # Create Gradio application
        gradio_app = create_gradio_app(self.app_state)
        
        # Configure authentication if provided
        auth = None
        if self.args.auth:
            # Parse authentication string (supports multiple user:pass pairs)
            creds = [tuple(pair.split(":", 1)) for pair in self.args.auth.split(",")]
            auth = creds[0] if len(creds) == 1 else creds
        
        # Set up signal handlers just before launch
        # (Gradio might interfere with signal handling if set too early)
        signal.signal(signal.SIGINT, self._shutdown_handler)
        signal.signal(signal.SIGTERM, self._shutdown_handler)
        
        # Launch Gradio interface
        gradio_app.launch(
            server_name="127.0.0.1",  # Local access only for security
            server_port=self.args.web_port,
            share=self.args.share,  # Enable Gradio sharing if --share is set
            auth=auth,
            show_error=True,  # Show detailed error messages
            inbrowser=self.args.open_browser,  # Auto-open browser if requested
            allowed_paths=["."],  # Allow access to current directory for file serving
        )

    # ==================================================================
    # PUBLIC METHODS - APPLICATION LIFECYCLE
    # ==================================================================

    def run(self) -> None:
        """
        Main application entry point and orchestrator.
        
        Coordinates the complete application startup sequence:
        1. Start memory guardian for resource monitoring
        2. Initialize AI models (SDXL and Ollama) based on CLI flags
        3. Start API server (if not disabled)
        4. Launch Gradio web interface
        
        The method handles different startup modes:
        - Quick start: Skip all model loading for immediate UI access
        - Lazy load: Defer model loading until first use
        - Full load: Initialize all models during startup (default)
        """
        # Application startup banner
        self.logger.info("%s", "=" * 50)
        self.logger.info("🎨 Starting Illustrious AI Studio")
        self.logger.info("%s", "=" * 50)

        # Phase 1: Start Memory Guardian for resource monitoring
        self.logger.info("🛡️ Starting Memory Guardian...")
        guardian = start_memory_guardian(self.app_state)
        
        # Configure memory guardian profile if specified
        if self.args.memory_profile:
            try:
                guardian.set_profile(self.args.memory_profile)
                self.logger.info("Memory profile set to %s", self.args.memory_profile)
            except ValueError as e:
                self.logger.error("%s", e)
                
        # Apply custom memory thresholds if provided
        if self.args.memory_threshold:
            for th in self.args.memory_threshold:
                try:
                    level, val = th.split(":")
                    guardian.set_threshold(level, float(val))
                except Exception as e:
                    self.logger.error("Invalid threshold '%s': %s", th, e)
                    
        self.logger.info("✅ Memory Guardian started")
        
        # Phase 2: Model initialization based on startup mode
        if self.args.quick_start:
            self.logger.info("🚀 Quick start mode - skipping all model initialization")
            self.logger.info("💡 Models can be loaded later through the web interface")
        elif not self.args.lazy_load:
            self.logger.info("🔄 Initializing models...")
            self._initialize_models_with_progress()
        else:
            self.logger.info("⏳ Lazy load mode - models will be initialized on first use")

        # Phase 3: Start API server (if enabled)
        if not self.args.no_api:
            self.logger.info("🌐 Starting API server...")
            self._start_api_server()

        # Phase 4: Launch main web interface
        self.logger.info("🖥️ Launching web interface...")
        self._launch_gradio()

    # ==================================================================
    # PRIVATE METHODS - MODEL INITIALIZATION & HEALTH CHECKS
    # ==================================================================

    def _initialize_models_with_progress(self) -> None:
        """
        Initialize AI models with comprehensive progress reporting and error handling.
        
        Performs pre-flight checks before attempting to load models, provides
        detailed timing information, and continues gracefully if individual
        models fail to load. This method is called during full startup mode.
        """
        import time
        
        # Pre-flight system checks
        self.logger.info("🔍 Running pre-flight checks...")
        
        # SDXL Model Initialization
        if not self.args.no_sdxl:
            self.logger.info("📋 Checking SDXL model...")
            if self._check_sdxl_model():
                self.logger.info("✅ SDXL model file validated")
                self.logger.info("🔄 Loading SDXL model (this may take a few minutes)...")
                start_time = time.time()
                try:
                    sdxl_result = init_sdxl(self.app_state)
                    elapsed = time.time() - start_time
                    if sdxl_result:
                        self.logger.info(f"✅ SDXL model loaded successfully in {elapsed:.1f}s")
                    else:
                        self.logger.warning("⚠️ SDXL model failed to load - continuing without it")
                except Exception as e:
                    elapsed = time.time() - start_time
                    self.logger.error(f"❌ SDXL initialization failed after {elapsed:.1f}s: {e}")
                    self.logger.info("💡 Try using --no-sdxl flag to skip SDXL initialization")
            else:
                self.logger.warning("⚠️ SDXL model check failed - skipping SDXL initialization")
        
        # Ollama Model Initialization
        if not self.args.no_ollama:
            self.logger.info("📋 Checking Ollama connection...")
            if self._check_ollama_connection():
                self.logger.info("✅ Ollama server is accessible")
                self.logger.info("🔄 Initializing Ollama models...")
                start_time = time.time()
                try:
                    ollama_result = init_ollama(self.app_state)
                    elapsed = time.time() - start_time
                    if ollama_result:
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
        """
        Validate SDXL model file existence and basic integrity.
        
        Performs several checks on the SDXL model file:
        - File existence at configured path
        - File size validation (SDXL models should be > 1GB)
        - Basic file accessibility
        
        Returns:
            bool: True if model file passes all checks, False otherwise
        """
        from .core.config import CONFIG
        import os
        
        model_path = CONFIG.sd_model
        
        # Check if model file exists
        if not os.path.exists(model_path):
            self.logger.error(f"❌ SDXL model file not found: {model_path}")
            self.logger.info("💡 Please check your model path in config.yaml or use --no-sdxl flag")
            return False
        
        # Validate file size (SDXL models should be substantial in size)
        try:
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            if size_mb < 1000:  # Less than 1GB is suspicious for SDXL
                self.logger.error(f"❌ SDXL model file too small ({size_mb:.1f}MB) - possibly corrupted")
                return False
            self.logger.info(f"📁 Model file: {model_path} ({size_mb:.1f}MB)")
            return True
        except Exception as e:
            self.logger.error(f"❌ Error checking model file: {e}")
            return False

    # ------------------------------------------------------------------
    def _check_ollama_connection(self) -> bool:
        """
        Verify Ollama server accessibility and model availability.
        
        Attempts to connect to the Ollama server and retrieve the list
        of available models. Provides detailed feedback about connection
        issues and available models.
        
        Returns:
            bool: True if Ollama server is accessible, False otherwise
        """
        from .core.config import CONFIG
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

    # ==================================================================
    # PRIVATE METHODS - SHUTDOWN HANDLING
    # ==================================================================

    def _shutdown_handler(self, signum, frame) -> None:
        """
        Handle graceful application shutdown on SIGINT/SIGTERM signals.
        
        Coordinates the shutdown sequence:
        1. Stop memory guardian
        2. Shutdown API server if running
        3. Clear GPU memory
        4. Clean up threads and resources
        5. Force exit if cleanup takes too long
        
        Args:
            signum: Signal number that triggered shutdown
            frame: Current stack frame (not used)
        """
        self.logger.info(f"🔄 Shutdown initiated by signal {signum}")
        
        # Gradio's launch() handles SIGINT/SIGTERM automatically for UI shutdown
        # This handler provides additional cleanup for our components
        
        # Step 1: Stop Memory Guardian
        self.logger.info("🛡️ Stopping Memory Guardian...")
        stop_memory_guardian(self.app_state)
        
        # Step 2: Stop API server if running
        if self.api_server and hasattr(self.api_server, 'should_exit'):
            self.logger.info("🌐 Stopping API server...")
            self.api_server.should_exit = True
        
        # Step 3: Clear GPU memory to free resources
        self.logger.info("🧹 Clearing GPU memory...")
        clear_gpu_memory()
        
        # Step 4: Clean up API thread
        if self.api_thread and self.api_thread.is_alive():
            self.logger.debug("⏳ Waiting for API thread to finish...")
            self.api_thread.join(timeout=5)
            if self.api_thread.is_alive():
                self.logger.warning("⚠️ API thread did not finish gracefully")
            else:
                self.logger.debug("✅ API thread finished cleanly")

        # Step 5: Successful shutdown
        self.logger.info("✅ Application shutdown completed")
        
        # Set up emergency timeout, then cancel it since we completed successfully
        timer = force_exit_after_timeout()
        timer.cancel()  # Cancel the emergency timer
        
        sys.exit(0)  # Clean exit


# ==================================================================
# APPLICATION ENTRY POINT
# ==================================================================

if __name__ == "__main__":
    """
    Application entry point.
    
    Parses command-line arguments, creates the main application instance,
    and starts the application. This is the standard entry point when
    running the application directly.
    """
    args = IllustriousAIStudio.parse_args()
    studio = IllustriousAIStudio(args)
    studio.run()
