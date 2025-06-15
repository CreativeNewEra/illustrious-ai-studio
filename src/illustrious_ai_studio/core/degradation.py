from __future__ import annotations

from dataclasses import dataclass

@dataclass
class DegradationStrategy:
    """Simple degradation strategy for generation parameters."""

    level: int = 0
    max_level: int = 3

    def degrade(self, width: int, height: int, steps: int) -> tuple[int, int, int]:
        """Increase degradation level and return adjusted parameters."""
        if self.level < self.max_level:
            self.level += 1

        factor = 0.8 ** self.level  # reduce parameters by 20% per level
        new_width = max(64, int(width * factor))
        new_height = max(64, int(height * factor))
        new_steps = max(1, int(steps * factor))
        return new_width, new_height, new_steps

    def restore(self) -> None:
        """Reset degradation level to zero."""
        self.level = 0
