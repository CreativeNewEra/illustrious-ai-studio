import json

import pytest





def test_history_saved_and_loaded(tmp_path, monkeypatch):
    from core.state import AppState
    import core.ollama as ollama

    state = AppState()
    assert state.ollama_vision_model is None
    hist_file = tmp_path / "history.json"
    monkeypatch.setattr(ollama, "CHAT_HISTORY_FILE", hist_file)
    monkeypatch.setattr(ollama, "chat_completion", lambda *a, **k: "ai")

    # Should load nothing
    ollama.load_chat_history(state)
    assert state.chat_history_store == {}

    history, _ = ollama.handle_chat(state, "hi", session_id="s1", chat_history=[])
    assert history == [["hi", "ai"]]

    # File written
    data = json.loads(hist_file.read_text())
    assert data["s1"] == [["hi", "ai"]]

    # Load into new state
    new_state = AppState()
    assert new_state.ollama_vision_model is None
    monkeypatch.setattr(ollama, "CHAT_HISTORY_FILE", hist_file)
    ollama.load_chat_history(new_state)
    assert list(new_state.chat_history_store["s1"]) == [("hi", "ai")]

