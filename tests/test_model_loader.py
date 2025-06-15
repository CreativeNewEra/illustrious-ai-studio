import types


def test_model_loader_button(monkeypatch):
    events = {}
    import illustrious_ai_studio.ui.web as web
    from illustrious_ai_studio.core.state import AppState

    class DummyComp:
        def __init__(self, *a, label=None, **k):
            self.label = label or (a[0] if a else None)
        def __call__(self, *a, **k):
            return self
        def click(self, fn=None, inputs=None, outputs=None):
            if self.label == "⚡ Load Selected":
                events['fn'] = fn
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
