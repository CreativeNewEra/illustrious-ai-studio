import logging
import logging.handlers
from pathlib import Path

from ..server.logging_utils import RequestIdFilter


def configure_logging(level: str) -> None:
    """Configure root logging handlers and level."""
    root = logging.getLogger()
    root.setLevel(level.upper())

    has_file_handler = any(isinstance(h, logging.handlers.RotatingFileHandler) for h in root.handlers)
    if not has_file_handler:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "illustrious_ai_studio.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
        )
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(request_id)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        file_handler.addFilter(RequestIdFilter())

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter("%(levelname)s: %(message)s [%(request_id)s]")
        )
        console_handler.addFilter(RequestIdFilter())

        root.addHandler(file_handler)
        root.addHandler(console_handler)

    # Reduce noise from third-party libraries
    logging.getLogger("gradio").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
