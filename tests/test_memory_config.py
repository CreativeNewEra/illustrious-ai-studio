from core.config import CONFIG


def test_config_overrides(monkeypatch):
    custom = {
        "thresholds": {"low": 60, "medium": 80},
        "safety_margins": {"generation_reserve": 1.0},
        "monitoring": {"normal_interval": 3.0, "aggressive_interval": 1.0},
    }
    monkeypatch.setattr(CONFIG, "memory_guardian", custom, raising=False)
    from core.state import AppState
    from core.memory_guardian import MemoryGuardian
    state = AppState()
    assert state.ollama_vision_model is None
    guardian = MemoryGuardian(state)
    th = guardian.thresholds
    assert th.low_threshold == 0.60
    assert th.medium_threshold == 0.80
    assert th.high_threshold == 0.95
    assert th.critical_threshold == 0.98
    assert th.generation_reserve_gb == 1.0
    assert th.llm_reserve_gb == 1.5
    assert th.monitoring_interval == 3.0
    assert th.aggressive_interval == 1.0


def test_defaults_preserved(monkeypatch):
    partial = {
        "thresholds": {"high": 90},
        "monitoring": {"aggressive_interval": 0.3},
    }
    monkeypatch.setattr(CONFIG, "memory_guardian", partial, raising=False)
    from core.state import AppState
    from core.memory_guardian import MemoryGuardian
    state2 = AppState()
    assert state2.ollama_vision_model is None
    guardian = MemoryGuardian(state2)
    th = guardian.thresholds
    assert th.low_threshold == 0.70
    assert th.medium_threshold == 0.85
    assert th.high_threshold == 0.90
    assert th.critical_threshold == 0.98
    assert th.monitoring_interval == 2.0
    assert th.aggressive_interval == 0.3
