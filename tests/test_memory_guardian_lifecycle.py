def test_guardian_recreated_after_stop():
    from illustrious_ai_studio.core.state import AppState
    from illustrious_ai_studio.core.memory_guardian import (
        start_memory_guardian,
        stop_memory_guardian,
        get_memory_guardian,
    )

    app_state = AppState()
    assert app_state.ollama_vision_model is None
    guardian1 = start_memory_guardian(app_state)
    stop_memory_guardian(app_state)
    guardian2 = start_memory_guardian(app_state)
    # guardian instance is stored on the state and reused
    assert guardian1 is guardian2
    assert get_memory_guardian(app_state) is guardian2
    stop_memory_guardian(app_state)
