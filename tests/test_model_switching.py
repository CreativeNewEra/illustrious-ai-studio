import os
import sys
from illustrious_ai_studio.core.config import CONFIG
from collections import deque


def test_switch_sdxl_model_missing(monkeypatch):
    from illustrious_ai_studio.core.state import AppState
    state = AppState()
    assert state.ollama_vision_model is None
    from illustrious_ai_studio.core import sdxl
    monkeypatch.setattr(os.path, "exists", lambda p: False)
    old = CONFIG.sd_model
    result = sdxl.switch_sdxl_model(state, "/new/path")
    assert result is False
    assert CONFIG.sd_model == old
    assert state.sdxl_pipe is None


def test_switch_sdxl_model_success(monkeypatch):
    from illustrious_ai_studio.core.state import AppState
    state = AppState()
    assert state.ollama_vision_model is None
    from illustrious_ai_studio.core import sdxl
    monkeypatch.setattr(os.path, "exists", lambda p: True)

    def dummy_init(st):
        st.sdxl_pipe = "pipe"
        return "pipe"
    monkeypatch.setattr(sdxl, "init_sdxl", dummy_init)

    result = sdxl.switch_sdxl_model(state, "/good/path")
    assert result is True
    assert CONFIG.sd_model == "/good/path"
    assert state.sdxl_pipe == "pipe"


def test_switch_ollama_model(monkeypatch):
    from illustrious_ai_studio.core.state import AppState
    state = AppState()
    assert state.ollama_vision_model is None
    state.chat_history_store = {"s": deque([("hi", "there")], maxlen=100)}
    from illustrious_ai_studio.core import ollama

    def dummy_init(st):
        st.ollama_model = CONFIG.ollama_model
        return CONFIG.ollama_model
    monkeypatch.setattr(ollama, "init_ollama_sync", dummy_init)

    res = ollama.switch_ollama_model(state, "new-model")
    assert res is True
    assert CONFIG.ollama_model == "new-model"
    assert state.ollama_model == "new-model"
    assert state.chat_history_store == {}

