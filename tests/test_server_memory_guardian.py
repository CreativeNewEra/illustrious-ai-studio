import sys
import types


def test_guardian_active_on_server_start(monkeypatch):
    from illustrious_ai_studio.core.state import AppState
    import illustrious_ai_studio.server.api as api
    from illustrious_ai_studio.core.memory_guardian import get_memory_guardian

    app_state = AppState()
    started = {}

    def dummy_run(app, host="0.0.0.0", port=8000, log_level="info"):
        started['active'] = get_memory_guardian(app_state).is_monitoring

    monkeypatch.setitem(sys.modules, 'uvicorn', types.SimpleNamespace(run=dummy_run))
    api.run_mcp_server(app_state, auto_load=False)

    assert started.get('active') is True

