import sys
import types
from pathlib import Path
import pytest


@pytest.fixture(scope="session", autouse=True)
def add_project_root_to_path():
    """Prepend the project root directory to ``sys.path`` for tests."""
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        yield
    finally:
        if str(root) in sys.path:
            sys.path.remove(str(root))

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
            def generate(self, *args, **kwargs):
                from PIL import Image
                return types.SimpleNamespace(images=[Image.new('RGB', (64,64), 'white')])
        diff.StableDiffusionXLPipeline = DummyPipe
        sys.modules['diffusers'] = diff
        pipelines = types.ModuleType('diffusers.pipelines')
        sdxl_mod = types.ModuleType('diffusers.pipelines.stable_diffusion_xl')
        submod = types.ModuleType('diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl')
        submod.StableDiffusionXLPipeline = DummyPipe
        sys.modules['diffusers.pipelines'] = pipelines
        sys.modules['diffusers.pipelines.stable_diffusion_xl'] = sdxl_mod
        sys.modules['diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl'] = submod
    if 'ui' not in sys.modules:
        sys.modules['ui'] = types.ModuleType('ui')
    if 'ui.web' not in sys.modules:
        web = types.ModuleType('ui.web')
        def create_gradio_app(state):
            class DummyApp:
                def launch(self, *a, **k):
                    pass
            return DummyApp()
        web.create_gradio_app = create_gradio_app
        sys.modules['ui.web'] = web
    if 'server' not in sys.modules:
        sys.modules['server'] = types.ModuleType('server')
    if 'server.api' not in sys.modules:
        api = types.ModuleType('server.api')
        def create_api_app(state):
            class Dummy:
                pass
            return Dummy()
        api.create_api_app = create_api_app
        sys.modules['server.api'] = api
    if 'server.logging_utils' not in sys.modules:
        log_mod = types.ModuleType('server.logging_utils')
        class RequestIdFilter:
            def filter(self, record):
                return True
        log_mod.RequestIdFilter = RequestIdFilter
        log_mod.request_id_var = None
        sys.modules['server.logging_utils'] = log_mod
    if 'core' not in sys.modules:
        sys.modules['core'] = types.ModuleType('core')
    for mod_name in ['core.sdxl', 'core.ollama', 'core.state', 'core.memory',
                     'core.memory_guardian', 'core.hardware_profiler']:
        if mod_name not in sys.modules:
            module = types.ModuleType(mod_name)
            if mod_name == 'core.sdxl':
                def init_sdxl(*a, **k):
                    pass
                module.init_sdxl = init_sdxl
            if mod_name == 'core.ollama':
                def init_ollama(*a, **k):
                    pass
                module.init_ollama = init_ollama
            if mod_name == 'core.state':
                class AppState:
                    pass
                module.AppState = AppState
            if mod_name == 'core.memory':
                def clear_gpu_memory():
                    pass
                module.clear_gpu_memory = clear_gpu_memory
            if mod_name == 'core.memory_guardian':
                def start_memory_guardian(*a, **k):
                    pass
                def stop_memory_guardian():
                    pass
                module.start_memory_guardian = start_memory_guardian
                module.stop_memory_guardian = stop_memory_guardian
            if mod_name == 'core.hardware_profiler':
                class HardwareProfiler:
                    def start(self):
                        pass
                    def stop(self):
                        pass
                module.HardwareProfiler = HardwareProfiler
            sys.modules[mod_name] = module
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
