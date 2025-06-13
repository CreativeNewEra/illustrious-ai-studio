import json
import logging
from pathlib import Path

from .sdxl import TEMP_DIR

logger = logging.getLogger(__name__)

GALLERY_FILTER_FILE = TEMP_DIR / "gallery_filters.json"


def save_gallery_filter(filters: dict) -> None:
    """Save gallery filter settings to disk."""
    try:
        GALLERY_FILTER_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(GALLERY_FILTER_FILE, "w", encoding="utf-8") as f:
            json.dump(filters, f, indent=2, ensure_ascii=False)
    except Exception as e:  # pragma: no cover - log and continue
        logger.error("Failed to save gallery filter: %s", e)


def load_gallery_filter() -> dict:
    """Load gallery filter settings from disk."""
    try:
        if GALLERY_FILTER_FILE.exists():
            with open(GALLERY_FILTER_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:  # pragma: no cover - log and return default
        logger.error("Failed to load gallery filter: %s", e)
    return {}

