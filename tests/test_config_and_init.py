import pytest

from core import config


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
    from core import sdxl
    from core.state import AppState
    state = AppState()
    assert state.ollama_vision_model is None
    assert state.ollama_vision_model is None
    custom_cfg = config.SDXLConfig(sd_model=str(tmp_path / "missing.safetensors"))
    result = sdxl.init_sdxl(state, custom_cfg)
    assert result is None
    assert state.sdxl_pipe is None
    assert state.model_status["sdxl"] is False


def test_init_ollama_no_model(monkeypatch):
    from core import ollama
    from core.state import AppState
    state = AppState()

    class Resp:
        status_code = 200
        def json(self):
            return {"models": [{"name": "other"}]}

    monkeypatch.setattr(ollama.requests, "get", lambda *a, **k: Resp())
    monkeypatch.setattr(ollama.requests, "post", lambda *a, **k: Resp())
    monkeypatch.setattr(ollama.CONFIG, "ollama_model", "missing")
    result = ollama.init_ollama(state)
    assert result is None
    assert state.model_status["ollama"] is False


def test_init_ollama_circuit_breaker(monkeypatch):
    import types, sys, importlib
    from core.circuit import CircuitBreaker

    sys.modules.pop('core.state', None)
    from core.state import AppState

    sdxl_stub = types.ModuleType("core.sdxl")
    sdxl_stub.generate_image = lambda *a, **k: None
    sdxl_stub.save_to_gallery = lambda *a, **k: None
    sdxl_stub.TEMP_DIR = "/tmp"
    sys.modules['core.sdxl'] = sdxl_stub

    sys.modules.pop('core.ollama', None)
    ollama = importlib.import_module('core.ollama')

    state = AppState()

    # Use a fresh circuit breaker to avoid interference with other tests
    breaker = CircuitBreaker()
    monkeypatch.setattr(ollama, "breaker", breaker, raising=False)

    def fail(*a, **k):
        raise Exception("connection failed")

    monkeypatch.setattr(ollama.requests, "get", fail, raising=False)

    # Trigger failures to open the circuit
    for _ in range(3):
        assert ollama.init_ollama(state) is None
        assert state.model_status["ollama"] is False

    assert breaker.state == "OPEN"

    # Subsequent call should be short-circuited
    assert ollama.init_ollama(state) is None
