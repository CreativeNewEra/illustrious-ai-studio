import sys
import types
import pytest

@pytest.fixture(scope="session", autouse=True)
def stub_dependencies():
    # Provide minimal stubs for heavy optional dependencies
    if 'gradio' not in sys.modules:
        sys.modules['gradio'] = types.ModuleType('gradio')
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
    if 'diffusers' not in sys.modules:
        diff = types.ModuleType('diffusers')
        class DummyPipe:
            def __init__(self, *a, **k):
                pass
            def to(self, device):
                pass
            def __call__(self, *args, **kwargs):
                from PIL import Image
                return types.SimpleNamespace(images=[Image.new('RGB', (64,64), 'white')])
        diff.StableDiffusionXLPipeline = DummyPipe
        sys.modules['diffusers'] = diff
    if 'uvicorn' not in sys.modules:
        sys.modules['uvicorn'] = types.ModuleType('uvicorn')
    if 'requests' not in sys.modules:
        requests = types.ModuleType('requests')
        class DummyResponse:
            status_code = 200
            text = ''
            def json(self):
                return {'models': []}
        def dummy_get(*args, **kwargs):
            return DummyResponse()
        def dummy_post(*args, **kwargs):
            return DummyResponse()
        requests.get = dummy_get
        requests.post = dummy_post
        sys.modules['requests'] = requests
