import os
import sys

if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

def test_guardian_recreated_after_stop():
    from core.state import AppState
    from core.memory_guardian import start_memory_guardian, stop_memory_guardian
    from core.memory_guardian import get_memory_guardian

    app_state = AppState()
    guardian1 = start_memory_guardian(app_state)
    stop_memory_guardian()
    guardian2 = start_memory_guardian(app_state)
    assert guardian1 is not guardian2
    # ensure global instance matches returned guardian2
    assert get_memory_guardian() is guardian2
    stop_memory_guardian()
