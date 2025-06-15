"""Preset configurations for image generation."""

DEFAULT_PRESET = "balanced"
GENERATION_PRESETS = {
    "draft": {
        "steps": 15,
        "guidance": 6.0,
        "width": 512,
        "height": 512,
    },
    "balanced": {
        "steps": 25,
        "guidance": 7.5,
        "width": 768,
        "height": 768,
    },
    "quality": {
        "steps": 40,
        "guidance": 8.0,
        "width": 1024,
        "height": 1024,
    },
}
