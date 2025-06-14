from __future__ import annotations

from dataclasses import dataclass, field
from typing import List
import math

@dataclass
class Metrics:
    """Simple runtime metrics collection."""

    generation_durations: List[float] = field(default_factory=list)
    sdxl_load_times: List[float] = field(default_factory=list)
    ollama_load_times: List[float] = field(default_factory=list)
    _generation_lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _sdxl_lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _ollama_lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def record_generation(self, duration: float) -> None:
        with self._generation_lock:
            self.generation_durations.append(duration)

    def record_sdxl_load(self, duration: float) -> None:
        with self._sdxl_lock:
            self.sdxl_load_times.append(duration)

    def record_ollama_load(self, duration: float) -> None:
        with self._ollama_lock:
            self.ollama_load_times.append(duration)

    def p95_generation_time(self) -> float:
        if not self.generation_durations:
            return 0.0
        data = sorted(self.generation_durations)
        k = int(math.ceil(0.95 * len(data))) - 1
        return data[max(k, 0)]

    def average_generation_time(self) -> float:
        if not self.generation_durations:
            return 0.0
        return sum(self.generation_durations) / len(self.generation_durations)

    def to_dict(self) -> dict:
        return {
            "generation_count": len(self.generation_durations),
            "avg_generation_time": self.average_generation_time(),
            "p95_generation_time": self.p95_generation_time(),
            "sdxl_load_times": self.sdxl_load_times,
            "ollama_load_times": self.ollama_load_times,
        }
