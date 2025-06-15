import types


def test_model_loader_button(monkeypatch):
    events = {}
    import os, sys, importlib
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    from tests.test_parse_resolution import stub_core_modules
    stub_core_modules()
    import types
    sys.modules['core.sdxl'].save_to_gallery = lambda *a, **k: None
    image_gen_mod = types.ModuleType('core.image_generator')
    image_gen_mod.ImageGenerator = object
    sys.modules['core.image_generator'] = image_gen_mod
    tm = types.SimpleNamespace(templates={"categories": []})
    sys.modules['core.prompt_templates'].template_manager = tm
    sys.modules['core.config'].CONFIG = types.SimpleNamespace(
        sd_model='', ollama_model='', memory_stats_refresh_interval=1, generation_defaults={}, gallery_dir='g', as_dict=lambda: {}
    )
    sys.modules['core.memory_guardian'].get_memory_guardian = lambda st: types.SimpleNamespace(config={}, thresholds=types.SimpleNamespace(low_threshold=0.5, high_threshold=0.9))
    class DummyState:
        def __init__(self):
            self.simple_mode = True
            self.model_status = {"sdxl": False, "ollama": False, "multimodal": False}
            self.current_project = None
        def __getattr__(self, name):
            return None
    sys.modules['core.state'].AppState = DummyState
    sys.modules['core.prompt_analyzer'].PromptAnalyzer = object
    sys.modules['core.prompt_analyzer'].auto_enhance_prompt = lambda *a, **k: None
    if 'ui.web' in sys.modules:
        del sys.modules['ui.web']
    if 'ui' in sys.modules:
        del sys.modules['ui']
    web = importlib.import_module('ui.web')
    from core.state import AppState

    class DummyComp:
        def __init__(self, *a, label=None, **k):
            self.label = label or (a[0] if a else None)
        def __call__(self, *a, **k):
            return self
        def click(self, fn=None, inputs=None, outputs=None):
            if self.label == "⚡ Load Selected":
                events['fn'] = fn
            if self.label == "Rename":
                events['rename_fn'] = fn
            return self
        def tick(self, fn=None, inputs=None, outputs=None):
            events['tick_fn'] = fn
            return self
        def change(self, *a, **k):
            return self
        def submit(self, *a, **k):
            return self
        def then(self, *a, **k):
            return self
        def load(self, *a, **k):
            return self
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, tb):
            pass

    class DummyBlocks(DummyComp):
        def Row(self, *a, **k):
            return self
        def Column(self, *a, **k):
            return self
        def Tab(self, *a, **k):
            return self
        def Accordion(self, *a, **k):
            return self
        def Group(self, *a, **k):
            return self

    class DummyModule(types.ModuleType):
        def __getattr__(self, name):
            if name in ["Blocks", "Row", "Column", "Tab", "Accordion", "Group"]:
                return DummyBlocks
            if name in ["Button", "DownloadButton", "Checkbox", "Textbox", "Slider", "Dropdown", "Image", "Chatbot", "File", "JSON", "Code", "Timer"]:
                return DummyComp
            if name == "Markdown":
                return lambda *a, **k: None
            if name == "themes":
                return types.SimpleNamespace(Base=lambda *a, **k: None)
            if name == "update":
                return lambda *a, **k: None
            return DummyComp

    dummy_gr = DummyModule('gr')
    monkeypatch.setattr(web, 'gr', dummy_gr)

    calls = {"sdxl": 0, "ollama": 0}
    monkeypatch.setattr(web.sdxl, 'init_sdxl', lambda st: calls.__setitem__('sdxl', calls['sdxl'] + 1))
    monkeypatch.setattr(web.ollama, 'init_ollama', lambda st: calls.__setitem__('ollama', calls['ollama'] + 1))
    monkeypatch.setattr(web, 'get_model_status', lambda st: 'ok')

    state = AppState()
    assert state.ollama_vision_model is None
    web.create_gradio_app(state)
    fn = events.get('fn')
    assert fn is not None
    fn(True, False, True)
    assert calls["sdxl"] == 1
    assert calls["ollama"] == 1


def test_project_rename(monkeypatch, tmp_path):
    events = {}
    import os, sys, importlib
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    from tests.test_parse_resolution import stub_core_modules
    stub_core_modules()
    sys.modules['core.sdxl'].save_to_gallery = lambda *a, **k: None
    image_gen_mod = types.ModuleType('core.image_generator')
    image_gen_mod.ImageGenerator = object
    sys.modules['core.image_generator'] = image_gen_mod
    tm = types.SimpleNamespace(templates={"categories": []})
    sys.modules['core.prompt_templates'].template_manager = tm
    sys.modules['core.config'].CONFIG = types.SimpleNamespace(
        sd_model='', ollama_model='', memory_stats_refresh_interval=1, generation_defaults={}, gallery_dir='g', as_dict=lambda: {}
    )
    sys.modules['core.memory_guardian'].get_memory_guardian = lambda st: types.SimpleNamespace(config={}, thresholds=types.SimpleNamespace(low_threshold=0.5, high_threshold=0.9))
    class DummyState:
        def __init__(self):
            self.simple_mode = True
            self.model_status = {"sdxl": False, "ollama": False, "multimodal": False}
            self.current_project = None
        def __getattr__(self, name):
            return None
    sys.modules['core.state'].AppState = DummyState
    sys.modules['core.prompt_analyzer'].PromptAnalyzer = object
    sys.modules['core.prompt_analyzer'].auto_enhance_prompt = lambda *a, **k: None
    if 'ui.web' in sys.modules:
        del sys.modules['ui.web']
    if 'ui' in sys.modules:
        del sys.modules['ui']
    web = importlib.import_module('ui.web')
    from core.state import AppState
    dummy_gr = types.ModuleType('gr')
    dummy_gr.update = lambda *a, **k: None
    monkeypatch.setattr(web, 'gr', dummy_gr)
    monkeypatch.setattr(web.sdxl, 'PROJECTS_DIR', tmp_path)
    monkeypatch.setattr(web, 'PROJECTS_DIR', tmp_path)

    (tmp_path / 'old' / 'gallery').mkdir(parents=True)

    state = AppState()
    state.current_project = 'old'

    web.rename_project(state, 'old', 'new')
    assert not (tmp_path / 'old').exists()
    assert (tmp_path / 'new').exists()
    assert state.current_project == 'new'
