from core.state import AppState
from core.memory_guardian import MemoryGuardian, PROFILE_PRESETS

def test_profile_update_runtime():
    guardian = MemoryGuardian(AppState())
    guardian.set_profile("aggressive")
    assert guardian.config["profile"] == "aggressive"
    assert guardian.thresholds.low_threshold == PROFILE_PRESETS["aggressive"]["low"]


def test_set_threshold_runtime():
    guardian = MemoryGuardian(AppState())
    guardian.set_threshold("medium", 75)
    assert guardian.thresholds.medium_threshold == 0.75
