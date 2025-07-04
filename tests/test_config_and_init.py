import pytest

from illustrious_ai_studio.core import config


def test_load_config_file_and_env(tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "sd_model: '/tmp/model.pt'\nollama_model: 'test-model'\n"
        "ollama_base_url: 'http://example.com'\nmemory_stats_refresh_interval: 3.0"
    )
    monkeypatch.setenv("SD_MODEL", "/env/model.pt")
    monkeypatch.setenv("GALLERY_DIR", "/tmp/gallery_env")
    monkeypatch.setenv("MEMORY_STATS_REFRESH_INTERVAL", "4.0")
    conf = config.load_config(str(cfg))
    assert conf.sd_model == "/env/model.pt"
    assert conf.ollama_model == "test-model"
    assert conf.ollama_base_url == "http://example.com"
    assert conf.gallery_dir == "/tmp/gallery_env"
    assert conf.memory_stats_refresh_interval == 4.0


def test_init_sdxl_missing_model(tmp_path, monkeypatch):
    from illustrious_ai_studio.core import sdxl
    from illustrious_ai_studio.core.state import AppState

    state = AppState()
    assert state.ollama_vision_model is None
    assert state.ollama_vision_model is None
    custom_cfg = config.SDXLConfig(sd_model=str(tmp_path / "missing.safetensors"))
    result = sdxl.init_sdxl(state, custom_cfg)
    assert result is None
    assert state.sdxl_pipe is None
    assert state.model_status["sdxl"] is False


def test_init_ollama_no_model(monkeypatch):
    from illustrious_ai_studio.core import ollama
    from illustrious_ai_studio.core.state import AppState

    state = AppState()

    class Resp:
        status_code = 200

        def json(self):
            return {"models": [{"name": "other"}]}

    class DummyClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        async def get(self, *a, **k):
            return Resp()

        async def post(self, *a, **k):
            return Resp()

    monkeypatch.setattr(ollama.httpx, "AsyncClient", lambda *a, **k: DummyClient())
    monkeypatch.setattr(ollama.CONFIG, "ollama_model", "missing")
    result = ollama.init_ollama_sync(state)
    assert result is None
    assert state.model_status["ollama"] is False


def test_init_ollama_no_name_error(monkeypatch):
    """Ensure init_ollama does not raise NameError when dependencies are patched."""
    from illustrious_ai_studio.core import ollama
    from illustrious_ai_studio.core.state import AppState

    state = AppState()

    class Resp:
        status_code = 200

        def json(self):
            return {"models": [{"name": ollama.CONFIG.ollama_model}]}

    class DummyClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        async def get(self, *a, **k):
            return Resp()

        async def post(self, *a, **k):
            return Resp()

    monkeypatch.setattr(ollama.httpx, "AsyncClient", lambda *a, **k: DummyClient())
    monkeypatch.setattr(ollama, "load_chat_history", lambda st: None)

    try:
        ollama.init_ollama_sync(state)
    except NameError as e:
        pytest.fail(f"init_ollama raised NameError: {e}")
