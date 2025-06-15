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
    if 'illustrious_ai_studio.ui' not in sys.modules:
        sys.modules['illustrious_ai_studio.ui'] = types.ModuleType('illustrious_ai_studio.ui')
    if 'illustrious_ai_studio.ui.web' not in sys.modules:
        web = types.ModuleType('illustrious_ai_studio.ui.web')
        def create_gradio_app(state):
            class DummyApp:
                def launch(self, *a, **k):
                    pass
            return DummyApp()
        web.create_gradio_app = create_gradio_app
        sys.modules['illustrious_ai_studio.ui.web'] = web
    if 'illustrious_ai_studio.server' not in sys.modules:
        sys.modules['illustrious_ai_studio.server'] = types.ModuleType('illustrious_ai_studio.server')
    if 'illustrious_ai_studio.server.api' not in sys.modules:
        api = types.ModuleType('illustrious_ai_studio.server.api')
        def create_api_app(state):
            class Dummy:
                pass
            return Dummy()
        api.create_api_app = create_api_app
        sys.modules['illustrious_ai_studio.server.api'] = api
    if 'illustrious_ai_studio.server.logging_utils' not in sys.modules:
        log_mod = types.ModuleType('illustrious_ai_studio.server.logging_utils')
        class RequestIdFilter:
            def filter(self, record):
                return True
        log_mod.RequestIdFilter = RequestIdFilter
        log_mod.request_id_var = None
        sys.modules['illustrious_ai_studio.server.logging_utils'] = log_mod
    if 'illustrious_ai_studio.core' not in sys.modules:
        sys.modules['illustrious_ai_studio.core'] = types.ModuleType('illustrious_ai_studio.core')
    for mod_name in [
        'illustrious_ai_studio.core.sdxl',
        'illustrious_ai_studio.core.ollama',
        'illustrious_ai_studio.core.state',
        'illustrious_ai_studio.core.memory',
        'illustrious_ai_studio.core.memory_guardian',
        'illustrious_ai_studio.core.hardware_profiler',
    ]:
        if mod_name not in sys.modules:
            module = types.ModuleType(mod_name)
            if mod_name == 'illustrious_ai_studio.core.sdxl':
                def init_sdxl(*a, **k):
                    pass
                module.init_sdxl = init_sdxl
            if mod_name == 'illustrious_ai_studio.core.ollama':
                def init_ollama(*a, **k):
                    pass
                module.init_ollama = init_ollama
            if mod_name == 'illustrious_ai_studio.core.state':
                class AppState:
                    pass
                module.AppState = AppState
            if mod_name == 'illustrious_ai_studio.core.memory':
                def clear_gpu_memory():
                    pass
                module.clear_gpu_memory = clear_gpu_memory
            if mod_name == 'illustrious_ai_studio.core.memory_guardian':
                def start_memory_guardian(*a, **k):
                    pass
                def stop_memory_guardian():
                    pass
                module.start_memory_guardian = start_memory_guardian
                module.stop_memory_guardian = stop_memory_guardian
            if mod_name == 'illustrious_ai_studio.core.hardware_profiler':
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


def stub_core_modules():
    """Provide stubs for core modules used in tests."""
    # Ensure any placeholder ui modules from stub_dependencies do not interfere
    for mod in ['illustrious_ai_studio.ui.web', 'illustrious_ai_studio.ui']:
        if mod in sys.modules:
            del sys.modules[mod]
    core_pkg = types.ModuleType('illustrious_ai_studio.core')
    sys.modules.setdefault('illustrious_ai_studio.core', core_pkg)

    sdxl = types.ModuleType('illustrious_ai_studio.core.sdxl')
    sdxl.generate_image = lambda *a, **k: None
    sdxl.generate_with_notifications = lambda *a, **k: None
    sdxl.TEMP_DIR = Path('/tmp')
    sdxl.get_latest_image = lambda *a, **k: None
    sdxl.init_sdxl = lambda *a, **k: None
    sdxl.get_available_models = lambda *a, **k: []
    sdxl.get_current_model_info = lambda *a, **k: {}
    sdxl.test_model_generation = lambda *a, **k: None
    sdxl.switch_sdxl_model = lambda *a, **k: None
    sdxl.save_to_gallery = lambda *a, **k: None
    sdxl.export_gallery = lambda *a, **k: None
    sdxl.PROJECTS_DIR = Path('/tmp')
    sys.modules['illustrious_ai_studio.core.sdxl'] = sdxl
    setattr(core_pkg, 'sdxl', sdxl)

    config = types.ModuleType('illustrious_ai_studio.core.config')
    config.CONFIG = {}
    sys.modules['illustrious_ai_studio.core.config'] = config
    setattr(core_pkg, 'config', config)

    state = types.ModuleType('illustrious_ai_studio.core.state')
    class DummyState:
        ...
    state.AppState = DummyState
    sys.modules['illustrious_ai_studio.core.state'] = state
    setattr(core_pkg, 'state', state)

    ollama = types.ModuleType('illustrious_ai_studio.core.ollama')
    ollama.generate_prompt = lambda *a, **k: None
    ollama.handle_chat = lambda *a, **k: None
    ollama.analyze_image = lambda *a, **k: None
    ollama.init_ollama = lambda *a, **k: None
    sys.modules['illustrious_ai_studio.core.ollama'] = ollama
    setattr(core_pkg, 'ollama', ollama)

    ig = types.ModuleType('illustrious_ai_studio.core.image_generator')
    class ImageGenerator:
        pass
    ig.ImageGenerator = ImageGenerator
    sys.modules['illustrious_ai_studio.core.image_generator'] = ig
    setattr(core_pkg, 'image_generator', ig)

    gp = types.ModuleType('illustrious_ai_studio.core.generation_presets')
    gp.GENERATION_PRESETS = {}
    gp.DEFAULT_PRESET = {}
    sys.modules['illustrious_ai_studio.core.generation_presets'] = gp
    setattr(core_pkg, 'generation_presets', gp)

    memory = types.ModuleType('illustrious_ai_studio.core.memory')
    memory.get_model_status = lambda *a, **k: None
    memory.get_memory_stats_markdown = lambda *a, **k: None
    memory.get_memory_stats_wrapper = lambda *a, **k: None
    sys.modules['illustrious_ai_studio.core.memory'] = memory
    setattr(core_pkg, 'memory', memory)

    mg = types.ModuleType('illustrious_ai_studio.core.memory_guardian')
    mg.start_memory_guardian = lambda *a, **k: None
    mg.stop_memory_guardian = lambda *a, **k: None
    mg.get_memory_guardian = lambda *a, **k: None
    sys.modules['illustrious_ai_studio.core.memory_guardian'] = mg
    setattr(core_pkg, 'memory_guardian', mg)

    pt = types.ModuleType('illustrious_ai_studio.core.prompt_templates')
    pt.template_manager = None
    sys.modules['illustrious_ai_studio.core.prompt_templates'] = pt
    setattr(core_pkg, 'prompt_templates', pt)

    pa = types.ModuleType('illustrious_ai_studio.core.prompt_analyzer')
    pa.analyze_prompt = lambda *a, **k: None
    class PromptAnalyzer:
        pass
    pa.PromptAnalyzer = PromptAnalyzer
    pa.auto_enhance_prompt = lambda *a, **k: None
    sys.modules['illustrious_ai_studio.core.prompt_analyzer'] = pa
    setattr(core_pkg, 'prompt_analyzer', pa)

    gf = types.ModuleType('illustrious_ai_studio.core.gallery_filters')
    gf.load_gallery_filter = lambda *a, **k: None
    gf.save_gallery_filter = lambda *a, **k: None
    sys.modules['illustrious_ai_studio.core.gallery_filters'] = gf
    setattr(core_pkg, 'gallery_filters', gf)


@pytest.fixture
def stub_core_modules_fixture():
    stub_core_modules()
    yield

