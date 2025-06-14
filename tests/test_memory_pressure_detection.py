import os
import sys
from datetime import datetime

if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

from core.state import AppState
from core.memory_guardian import MemoryGuardian, MemoryStats, MemoryPressureLevel


def _dummy_intervention():
    return True


def make_stats(level: MemoryPressureLevel) -> MemoryStats:
    return MemoryStats(
        timestamp=datetime.now(),
        gpu_total_gb=10,
        gpu_allocated_gb=5,
        gpu_reserved_gb=8,
        gpu_free_gb=2,
        gpu_usage_percent=80,
        system_ram_gb=16,
        system_ram_usage_percent=40,
        pressure_level=level,
    )


def test_pressure_reduction_detection(monkeypatch):
    guardian = MemoryGuardian(AppState())
    guardian.intervention_callbacks[MemoryPressureLevel.HIGH] = [_dummy_intervention]

    start_stats = make_stats(MemoryPressureLevel.HIGH)
    reduced_stats = make_stats(MemoryPressureLevel.MEDIUM)

    monkeypatch.setattr(guardian, "get_memory_stats", lambda: reduced_stats)
    monkeypatch.setattr("core.memory_guardian.time", "sleep", lambda _: None)

    guardian._handle_memory_pressure(start_stats)
    assert guardian.oom_prevented_count == 1

    monkeypatch.setattr(guardian, "get_memory_stats", lambda: start_stats)
    guardian._handle_memory_pressure(start_stats)
    assert guardian.oom_prevented_count == 1
