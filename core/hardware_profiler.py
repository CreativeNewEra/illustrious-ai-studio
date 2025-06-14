"""Hardware capability detection and optimization profiles."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Tuple

try:
    import torch
except Exception:  # pragma: no cover - allow missing torch
    torch = None  # type: ignore

try:
    import psutil
except Exception:  # pragma: no cover - allow missing psutil
    psutil = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class HardwareProfile:
    """Recommended settings based on detected hardware."""

    profile_name: str
    vram_gb: float
    ram_gb: float
    recommended_resolution: Tuple[int, int]
    recommended_steps: int
    recommended_batch_size: int
    enable_memory_efficient_attention: bool
    enable_cpu_offload: bool
    enable_tf32: bool
    max_concurrent_generations: int
    description: str
    warnings: List[str] = field(default_factory=list)


class HardwareProfiler:
    """Detect GPU/CPU capabilities and recommend optimal settings."""

    _cached_profile: HardwareProfile | None = None

    def __init__(self) -> None:
        self.logger = logger
        self.gpu_name: str | None = None

    # ------------------------------------------------------------------
    def detect_hardware(self) -> HardwareProfile:
        """Detect hardware capabilities and return the recommended profile."""
        if HardwareProfiler._cached_profile is not None:
            return HardwareProfiler._cached_profile

        vram_gb = 0.0
        ram_gb = 0.0
        warnings: List[str] = []
        enable_memory_efficient = False
        enable_cpu_offload = False
        enable_tf32 = False
        batch_size = 1
        resolution = (512, 512)
        steps = 15
        profile_name = "CPU"
        max_concurrent = 1
        description = "Safe defaults for CPU mode"

        # Detect GPU
        if torch is not None and torch.cuda.is_available():
            try:
                device = torch.cuda.get_device_properties(0)
                vram_gb = device.total_memory / (1024 ** 3)
                self.gpu_name = device.name
                compute_capability = torch.cuda.get_device_capability(0)
            except Exception as e:  # pragma: no cover - runtime protection
                self.logger.warning("GPU detection failed: %s", e)
                compute_capability = (0, 0)
        else:
            enable_cpu_offload = True
            compute_capability = (0, 0)
            warnings.append("GPU not detected - running on CPU")

        # System RAM
        if psutil is not None:
            try:
                ram_gb = psutil.virtual_memory().total / (1024 ** 3)
            except Exception as e:  # pragma: no cover - runtime protection
                self.logger.warning("RAM detection failed: %s", e)
        else:
            self.logger.warning("psutil not available - cannot detect RAM")

        # Determine profile by VRAM
        if vram_gb >= 16:
            profile_name = "Enthusiast"
            resolution = (1024, 1024)
            steps = 50
            batch_size = 4
            max_concurrent = 4
            description = "Top-tier GPU detected"
        elif vram_gb >= 12:
            profile_name = "High-end"
            resolution = (1024, 1024)
            steps = 30
            batch_size = 2
            max_concurrent = 2
            description = "Optimized for quality"
        elif vram_gb >= 8:
            profile_name = "Good"
            resolution = (1024, 1024)
            steps = 25
            description = "Balanced performance and quality"
        elif vram_gb >= 6:
            profile_name = "Mid-range"
            resolution = (768, 768)
            steps = 20
            description = "Moderate GPU detected"
        elif vram_gb >= 4:
            profile_name = "Low-end"
            resolution = (512, 512)
            steps = 15
            enable_memory_efficient = True
            description = "Limited VRAM detected"
        else:
            warnings.append("Very low VRAM - using CPU safe defaults")

        # Compute capability optimizations
        if compute_capability >= (8, 0):
            enable_memory_efficient = True
            enable_tf32 = True
        elif compute_capability == (0, 0) and vram_gb > 0:
            warnings.append("Unable to determine compute capability")

        profile = HardwareProfile(
            profile_name=profile_name,
            vram_gb=round(vram_gb, 2),
            ram_gb=round(ram_gb, 2),
            recommended_resolution=resolution,
            recommended_steps=steps,
            recommended_batch_size=batch_size,
            enable_memory_efficient_attention=enable_memory_efficient,
            enable_cpu_offload=enable_cpu_offload,
            enable_tf32=enable_tf32,
            max_concurrent_generations=max_concurrent,
            description=description,
            warnings=warnings,
        )

        HardwareProfiler._cached_profile = profile
        return profile

    # ------------------------------------------------------------------
    def get_dynamic_settings(self, current_memory_usage: float) -> dict:
        """Return settings adjusted for current GPU memory usage."""
        adjustments: dict = {}
        message = ""

        if current_memory_usage > 95:
            adjustments = {"width": 512, "height": 512, "steps": 10}
            message = "🚨 Memory critically high - emergency low settings applied"
        elif current_memory_usage > 90:
            adjustments = {"width": 512, "height": 512, "steps": int(self.detect_hardware().recommended_steps * 0.5)}
            message = "⚠️ Very high memory usage - reducing resolution and steps"
        elif current_memory_usage > 80:
            adjustments = {"width": 768, "height": 768}
            message = "⚠️ High memory usage - lowering resolution"

        if message:
            adjustments["message"] = message
        return adjustments
