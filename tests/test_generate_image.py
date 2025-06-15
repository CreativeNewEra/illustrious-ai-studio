from PIL import Image
import types
import importlib
import sys
import pytest

def load_app():
    if 'gradio' not in sys.modules:
        sys.modules['gradio'] = types.ModuleType('gradio')
    if 'diffusers' not in sys.modules:
        diff = types.ModuleType('diffusers')
        diff.StableDiffusionXLPipeline = object
        sys.modules['diffusers'] = diff
    if 'torch' not in sys.modules:
        torch = types.ModuleType('torch')
        class DummyCuda:
            @staticmethod
            def is_available():
                return False
            @staticmethod
            def empty_cache():
                pass
            @staticmethod
            def synchronize():
                pass
        torch.cuda = DummyCuda()
        class DummyGenerator:
            def __init__(self, device=None):
                pass
            def manual_seed(self, seed):
                pass
            def initial_seed(self):
                return 0
        torch.Generator = DummyGenerator
        torch.float16 = 'float16'
        sys.modules['torch'] = torch
    return importlib.import_module('illustrious_ai_studio.app')

class DummyPipe:
    def generate(self, *args, **kwargs):
        return types.SimpleNamespace(images=[Image.new('RGB', (64, 64), color='white')])

# Ensure clear_gpu_memory is patched to avoid torch calls
@pytest.fixture(autouse=True)
def patch_clear_cuda(monkeypatch):
    app = load_app()
    monkeypatch.setattr(app, 'clear_gpu_memory', lambda: None)


def test_generate_image_no_model(monkeypatch):
    app = load_app()
    app.app_state.sdxl_pipe = None
    image, status = app.generate_image(app.app_state, {"prompt": 'test'})
    assert image is None
    assert 'model not loaded' in status.lower()


def test_generate_image_success(monkeypatch):
    app = load_app()
    app.app_state.sdxl_pipe = DummyPipe()
    img, status = app.generate_image(
        app.app_state,
        {"prompt": 'test prompt', "save_to_gallery_flag": False}
    )
    assert isinstance(img, Image.Image)
    assert 'successfully' in status.lower()
