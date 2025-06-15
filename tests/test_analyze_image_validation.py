import sys
from PIL import Image

from core.state import AppState
import types


def load_analyze_image():
    if 'core.sdxl' in sys.modules:
        del sys.modules['core.sdxl']
    dummy = types.ModuleType('core.sdxl')
    dummy.generate_image = lambda *a, **k: None
    dummy.save_to_gallery = lambda *a, **k: None
    dummy.TEMP_DIR = '/tmp'
    sys.modules['core.sdxl'] = dummy
    if 'core.ollama' in sys.modules:
        del sys.modules['core.ollama']
    module = __import__('core.ollama', fromlist=['analyze_image', 'MAX_IMAGE_PIXELS'])
    return module.analyze_image, module.MAX_IMAGE_PIXELS


def test_analyze_image_rejects_large_image():
    analyze_image, MAX_IMAGE_PIXELS = load_analyze_image()

    state = AppState()
    state.model_status["multimodal"] = True
    state.ollama_vision_model = "dummy"

    width = 4000
    height = MAX_IMAGE_PIXELS // width + 1
    img = Image.new("RGB", (width, height))
    result = analyze_image(state, img)
    expected = f"{MAX_IMAGE_PIXELS // 1_000_000}MP"
    assert result.startswith("❌") and expected in result
