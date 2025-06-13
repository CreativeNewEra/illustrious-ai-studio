import os
import sys
import types

if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())


def test_guardian_active_on_server_start(monkeypatch):
    from core.state import AppState
    import server.api as api
    from core.memory_guardian import get_memory_guardian

    app_state = AppState()
    started = {}

    def dummy_run(app, host="0.0.0.0", port=8000, log_level="info"):
        started['active'] = get_memory_guardian(app_state).is_monitoring

    monkeypatch.setitem(sys.modules, 'uvicorn', types.SimpleNamespace(run=dummy_run))
    api.run_mcp_server(app_state, auto_load=False)

    assert started.get('active') is True

