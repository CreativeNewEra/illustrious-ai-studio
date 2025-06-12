import os
import sys

if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())


def test_load_sdxl_button(monkeypatch):
    from ui import web
    from core.state import AppState
    calls = {}
    monkeypatch.setattr(web.sdxl, "init_sdxl", lambda st: calls.setdefault("sdxl", True))
    monkeypatch.setattr(web, "get_model_status", lambda st: "STATUS")
    state = AppState()
    result = web.load_sdxl_model_fn(state)
    assert calls.get("sdxl")
    assert result == "STATUS"


def test_load_ollama_text_button(monkeypatch):
    from ui import web
    from core.state import AppState
    calls = {}
    monkeypatch.setattr(web.ollama, "init_ollama", lambda st: calls.setdefault("ollama_text", True))
    monkeypatch.setattr(web, "get_model_status", lambda st: "STATUS")
    state = AppState()
    result = web.load_ollama_text_model_fn(state)
    assert calls.get("ollama_text")
    assert result == "STATUS"


def test_load_ollama_vision_button(monkeypatch):
    from ui import web
    from core.state import AppState
    calls = {}
    monkeypatch.setattr(web.ollama, "init_ollama", lambda st: calls.setdefault("vision", True))
    monkeypatch.setattr(web, "get_model_status", lambda st: "STATUS")
    state = AppState()
    result = web.load_ollama_vision_model_fn(state)
    assert calls.get("vision")
    assert result == "STATUS"
