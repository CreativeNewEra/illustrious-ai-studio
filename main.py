import logging
import logging.handlers
import threading
from pathlib import Path
import sys
import traceback

from ui.web import create_gradio_app
from server.api import run_mcp_server
from core.sdxl import init_sdxl
from core.ollama import init_ollama
from core.state import AppState
from core.config import load_config

def setup_logging():
    """Setup centralized logging for the application."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    
    # Create handlers
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "illustrious_ai_studio.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Suppress verbose third-party logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("gradio").setLevel(logging.INFO)
    
    return logging.getLogger(__name__)

logger = setup_logging()
app_state = AppState()


def initialize_models() -> None:
    """Initialize AI models with proper error handling."""
    try:
        logger.info("Initializing AI models...")
        
        # Load and validate configuration
        config = load_config()
        logger.info(f"Configuration loaded successfully")
        
        # Initialize SDXL
        logger.info("Initializing SDXL model...")
        init_sdxl(app_state)
        logger.info("✓ SDXL model initialized")
        
        # Initialize Ollama
        logger.info("Initializing Ollama models...")
        init_ollama(app_state)
        logger.info("✓ Ollama models initialized")
        
        logger.info("All models initialized successfully!")
        
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def start_mcp_server() -> threading.Thread:
    """Start the MCP server in a separate thread."""
    def run_server():
        try:
            logger.info("Starting MCP Server...")
            run_mcp_server(app_state)
        except Exception as e:
            logger.error(f"MCP Server error: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    mcp_thread = threading.Thread(target=run_server, daemon=True)
    mcp_thread.start()
    logger.info("✓ MCP Server started on http://localhost:8000")
    return mcp_thread

def start_web_interface():
    """Start the Gradio web interface."""
    try:
        logger.info("Starting Gradio web interface...")
        gradio_app = create_gradio_app(app_state)
        logger.info("✓ Gradio app created successfully")
        
        gradio_app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            auth=None,
            show_error=True,
        )
    except Exception as e:
        logger.error(f"Failed to start web interface: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


if __name__ == "__main__":
    try:
        logger.info("=" * 50)
        logger.info("🎨 Starting Illustrious AI Studio")
        logger.info("=" * 50)
        
        # Initialize models
        initialize_models()
        
        # Start MCP server
        mcp_thread = start_mcp_server()
        
        # Start web interface
        logger.info("🌐 Starting web interface on http://localhost:7860")
        start_web_interface()
        
    except KeyboardInterrupt:
        logger.info("Received shutdown signal, stopping application...")
    except Exception as e:
        logger.critical(f"Critical error starting application: {e}")
        logger.critical(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)
