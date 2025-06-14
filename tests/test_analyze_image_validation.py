import os
import sys
from PIL import Image

if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

from core.state import AppState
import types


def load_analyze_image():
    if 'core.sdxl' not in sys.modules:
        dummy = types.ModuleType('core.sdxl')
        dummy.generate_image = lambda *a, **k: None
        dummy.save_to_gallery = lambda *a, **k: None
        dummy.TEMP_DIR = '/tmp'
        sys.modules['core.sdxl'] = dummy
    return __import__('core.ollama', fromlist=['analyze_image']).analyze_image


def test_analyze_image_rejects_large_image():
    analyze_image = load_analyze_image()

    state = AppState()
    state.model_status["multimodal"] = True
    state.ollama_vision_model = "dummy"

    img = Image.new("RGB", (5000, 4000))  # 20MP > 16MP
    result = analyze_image(state, img)
    assert result.startswith("❌") and "16MP" in result
