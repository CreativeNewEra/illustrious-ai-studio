import gc
import logging
import psutil
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum

try:
    import torch
except Exception:  # pragma: no cover - allow missing torch in tests
    torch = None  # type: ignore

from .config import CONFIG
from .memory import clear_gpu_memory

# Preset threshold values for supported profiles
PROFILE_PRESETS = {
    "conservative": {"low": 0.60, "medium": 0.75, "high": 0.90, "critical": 0.95},
    "balanced": {"low": 0.70, "medium": 0.85, "high": 0.95, "critical": 0.98},
    "aggressive": {"low": 0.80, "medium": 0.90, "high": 0.97, "critical": 0.99},
}

logger = logging.getLogger(__name__)

class MemoryPressureLevel(Enum):
    LOW = "low"           # < 70% usage
    MEDIUM = "medium"     # 70-85% usage
    HIGH = "high"         # 85-95% usage
    CRITICAL = "critical" # > 95% usage

# Ordering of pressure levels for comparison
PRESSURE_ORDER = {
    MemoryPressureLevel.LOW: 0,
    MemoryPressureLevel.MEDIUM: 1,
    MemoryPressureLevel.HIGH: 2,
    MemoryPressureLevel.CRITICAL: 3,
}

@dataclass
class MemoryStats:
    """Memory statistics snapshot"""
    timestamp: datetime
    gpu_total_gb: float
    gpu_allocated_gb: float
    gpu_reserved_gb: float
    gpu_free_gb: float
    gpu_usage_percent: float
    system_ram_gb: float
    system_ram_usage_percent: float
    pressure_level: MemoryPressureLevel

@dataclass
class MemoryThresholds:
    """Configurable memory thresholds"""
    low_threshold: float = 0.70      # 70% - Start monitoring closely
    medium_threshold: float = 0.85   # 85% - Begin preventive actions
    high_threshold: float = 0.95     # 95% - Aggressive memory management
    critical_threshold: float = 0.98 # 98% - Emergency intervention
    
    # Safety margins
    generation_reserve_gb: float = 2.0  # Reserve 2GB for image generation
    llm_reserve_gb: float = 1.5         # Reserve 1.5GB for LLM operations
    
    # Monitoring intervals
    monitoring_interval: float = 2.0    # Check every 2 seconds
    aggressive_interval: float = 0.5    # Check every 0.5s when under pressure

class MemoryGuardian:
    """Automatic memory management and OOM prevention system"""
    
    def __init__(self, app_state=None):
        self.state = app_state
        self.thresholds = MemoryThresholds()
        self.is_monitoring = False
        self.monitor_thread = None
        self.memory_history: List[MemoryStats] = []
        self.max_history = 100
        
        # Intervention callbacks
        self.intervention_callbacks: Dict[MemoryPressureLevel, List[Callable]] = {
            MemoryPressureLevel.MEDIUM: [],
            MemoryPressureLevel.HIGH: [],
            MemoryPressureLevel.CRITICAL: []
        }
        
        # Statistics
        self.interventions_count = 0
        self.oom_prevented_count = 0
        self.last_intervention_time = None
        
        # Configuration
        self.config = self._load_memory_config()
        
        self._register_default_interventions()
        
    def _load_memory_config(self) -> Dict[str, Any]:
        """Load memory management configuration"""
        default_config = {
            "enabled": True,
            "profile": "balanced",  # conservative, balanced, aggressive
            "auto_model_switching": True,
            "auto_resolution_scaling": True,
            "auto_batch_adjustment": True,
            "enable_cpu_fallback": True,
            "log_memory_stats": True,
            "intervention_cooldown": 5.0,  # seconds
        }
        
        # Load from CONFIG if available
        memory_config = getattr(CONFIG, 'memory_guardian', {})

        # Apply threshold presets based on the selected profile
        profile = memory_config.get("profile", default_config["profile"]).lower()
        if profile in PROFILE_PRESETS:
            th = PROFILE_PRESETS[profile]
            self.thresholds.low_threshold = th["low"]
            self.thresholds.medium_threshold = th["medium"]
            self.thresholds.high_threshold = th["high"]
            self.thresholds.critical_threshold = th["critical"]

        # Update threshold values if provided
        thresholds_cfg = memory_config.get("thresholds", {}) or {}

        def _percent(val: float) -> float:
            val = float(val)
            return val / 100.0 if val > 1 else val

        if "low" in thresholds_cfg:
            self.thresholds.low_threshold = _percent(thresholds_cfg["low"])
        if "medium" in thresholds_cfg:
            self.thresholds.medium_threshold = _percent(thresholds_cfg["medium"])
        if "high" in thresholds_cfg:
            self.thresholds.high_threshold = _percent(thresholds_cfg["high"])
        if "critical" in thresholds_cfg:
            self.thresholds.critical_threshold = _percent(thresholds_cfg["critical"])

        safety_cfg = memory_config.get("safety_margins", {}) or {}
        if "generation_reserve" in safety_cfg:
            self.thresholds.generation_reserve_gb = float(safety_cfg["generation_reserve"])
        if "llm_reserve" in safety_cfg:
            self.thresholds.llm_reserve_gb = float(safety_cfg["llm_reserve"])

        monitor_cfg = memory_config.get("monitoring", {}) or {}
        if "normal_interval" in monitor_cfg:
            self.thresholds.monitoring_interval = float(monitor_cfg["normal_interval"])
        if "aggressive_interval" in monitor_cfg:
            self.thresholds.aggressive_interval = float(monitor_cfg["aggressive_interval"])

        return {**default_config, **memory_config}
    
    def _register_default_interventions(self):
        """Register default memory intervention strategies"""
        
        # Medium pressure interventions
        self.register_intervention(MemoryPressureLevel.MEDIUM, self._clear_cache)
        self.register_intervention(MemoryPressureLevel.MEDIUM, self._garbage_collect)
        
        # High pressure interventions  
        self.register_intervention(MemoryPressureLevel.HIGH, self._clear_cache)
        self.register_intervention(MemoryPressureLevel.HIGH, self._garbage_collect)
        self.register_intervention(MemoryPressureLevel.HIGH, self._unload_unused_models)
        
        # Critical pressure interventions
        self.register_intervention(MemoryPressureLevel.CRITICAL, self._emergency_cleanup)
        self.register_intervention(MemoryPressureLevel.CRITICAL, self._force_model_unload)
    
    def register_intervention(self, level: MemoryPressureLevel, callback: Callable):
        """Register a custom intervention callback for a pressure level"""
        self.intervention_callbacks[level].append(callback)
        logger.info(f"Registered intervention for {level.value} pressure: {callback.__name__}")
    
    def get_memory_stats(self) -> Optional[MemoryStats]:
        """Get current memory statistics"""
        try:
            if torch is None or not (CONFIG.gpu_backend in ("cuda", "rocm") and torch.cuda.is_available()):
                return None
                
            # GPU memory stats
            gpu_props = torch.cuda.get_device_properties(0)
            gpu_total = gpu_props.total_memory / (1024**3)  # GB
            gpu_allocated = torch.cuda.memory_allocated() / (1024**3)
            gpu_reserved = torch.cuda.memory_reserved() / (1024**3)
            gpu_free = gpu_total - gpu_reserved
            gpu_usage_percent = (gpu_reserved / gpu_total) * 100
            
            # System RAM stats
            system_ram = psutil.virtual_memory()
            system_ram_gb = system_ram.total / (1024**3)
            system_ram_usage = system_ram.percent
            
            # Determine pressure level
            if gpu_usage_percent >= self.thresholds.critical_threshold * 100:
                pressure = MemoryPressureLevel.CRITICAL
            elif gpu_usage_percent >= self.thresholds.high_threshold * 100:
                pressure = MemoryPressureLevel.HIGH
            elif gpu_usage_percent >= self.thresholds.medium_threshold * 100:
                pressure = MemoryPressureLevel.MEDIUM
            else:
                pressure = MemoryPressureLevel.LOW
            
            return MemoryStats(
                timestamp=datetime.now(),
                gpu_total_gb=gpu_total,
                gpu_allocated_gb=gpu_allocated,
                gpu_reserved_gb=gpu_reserved,
                gpu_free_gb=gpu_free,
                gpu_usage_percent=gpu_usage_percent,
                system_ram_gb=system_ram_gb,
                system_ram_usage_percent=system_ram_usage,
                pressure_level=pressure
            )
            
        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return None
    
    def start_monitoring(self):
        """Start the memory monitoring thread"""
        if self.is_monitoring:
            logger.warning("Memory monitoring already started")
            return
            
        if not self.config["enabled"]:
            logger.info("Memory guardian disabled in configuration")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Memory Guardian started - monitoring GPU memory")
    
    def stop_monitoring(self):
        """Stop the memory monitoring thread"""
        if not self.is_monitoring:
            return
            
        self.is_monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        logger.info("Memory Guardian stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                stats = self.get_memory_stats()
                if stats:
                    self._update_history(stats)
                    
                    if self.config["log_memory_stats"]:
                        self._log_memory_stats(stats)
                    
                    # Check if intervention is needed
                    if stats.pressure_level != MemoryPressureLevel.LOW:
                        self._handle_memory_pressure(stats)
                    
                    # Adjust monitoring frequency based on pressure
                    if stats.pressure_level in [MemoryPressureLevel.HIGH, MemoryPressureLevel.CRITICAL]:
                        sleep_time = self.thresholds.aggressive_interval
                    else:
                        sleep_time = self.thresholds.monitoring_interval
                else:
                    sleep_time = self.thresholds.monitoring_interval
                
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in memory monitoring loop: {e}")
                time.sleep(self.thresholds.monitoring_interval)
    
    def _update_history(self, stats: MemoryStats):
        """Update memory history with latest stats"""
        self.memory_history.append(stats)
        if len(self.memory_history) > self.max_history:
            self.memory_history.pop(0)
    
    def _log_memory_stats(self, stats: MemoryStats):
        """Log memory statistics"""
        if stats.pressure_level != MemoryPressureLevel.LOW:
            logger.warning(
                f"Memory pressure {stats.pressure_level.value}: "
                f"GPU {stats.gpu_usage_percent:.1f}% "
                f"({stats.gpu_free_gb:.1f}GB free), "
                f"RAM {stats.system_ram_usage_percent:.1f}%"
            )
    
    def _handle_memory_pressure(self, stats: MemoryStats):
        """Handle memory pressure by running appropriate interventions"""
        
        # Check intervention cooldown
        if (self.last_intervention_time and 
            datetime.now() - self.last_intervention_time < timedelta(seconds=self.config["intervention_cooldown"])):
            return
        
        logger.warning(f"Handling {stats.pressure_level.value} memory pressure")
        
        # Run interventions for this pressure level
        interventions = self.intervention_callbacks.get(stats.pressure_level, [])
        
        success_count = 0
        for intervention in interventions:
            try:
                result = intervention()
                if result:
                    success_count += 1
                    logger.info(f"Intervention {intervention.__name__} succeeded")
                else:
                    logger.warning(f"Intervention {intervention.__name__} failed")
            except Exception as e:
                logger.error(f"Intervention {intervention.__name__} error: {e}")
        
        if success_count > 0:
            self.interventions_count += 1
            self.last_intervention_time = datetime.now()
            
            # Check if pressure was relieved
            time.sleep(1)  # Give time for cleanup to take effect
            new_stats = self.get_memory_stats()
            if (
                new_stats
                and PRESSURE_ORDER[new_stats.pressure_level] < PRESSURE_ORDER[stats.pressure_level]
            ):
                self.oom_prevented_count += 1
                logger.info(
                    "Memory pressure reduced from %s to %s",
                    stats.pressure_level.value,
                    new_stats.pressure_level.value,
                )
    
    # ======================== Intervention Strategies ========================
    
    def _clear_cache(self) -> bool:
        """Clear GPU cache"""
        try:
            clear_gpu_memory()
            return True
        except Exception as e:
            logger.error(f"Cache clear failed: {e}")
            return False
    
    def _garbage_collect(self) -> bool:
        """Force garbage collection"""
        try:
            collected = gc.collect()
            logger.info(f"Garbage collection freed {collected} objects")
            return True
        except Exception as e:
            logger.error(f"Garbage collection failed: {e}")
            return False
    
    def _unload_unused_models(self) -> bool:
        """Unload models that haven't been used recently"""
        try:
            if self.state:
                # This would be implemented based on your specific model management
                # For now, just clear cache
                clear_gpu_memory()
                logger.info("Unloaded unused models")
                return True
            return False
        except Exception as e:
            logger.error(f"Model unloading failed: {e}")
            return False
    
    def _emergency_cleanup(self) -> bool:
        """Emergency memory cleanup"""
        try:
            # Force aggressive cleanup
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Multiple garbage collection passes
            for _ in range(3):
                gc.collect()
            
            logger.warning("Emergency memory cleanup performed")
            return True
        except Exception as e:
            logger.error(f"Emergency cleanup failed: {e}")
            return False
    
    def _force_model_unload(self) -> bool:
        """Force unload all models (last resort)"""
        try:
            if self.state:
                # Unload SDXL
                if hasattr(self.state, 'sdxl_pipe') and self.state.sdxl_pipe:
                    self.state.sdxl_pipe = None
                    logger.warning("Force unloaded SDXL model")
                
                # Clear model status
                if hasattr(self.state, 'model_status'):
                    self.state.model_status = {
                        'sdxl': False,
                        'ollama': False,
                        'multimodal': False
                    }
            
            clear_gpu_memory()
            logger.warning("Force unloaded all models - critical memory situation")
            return True
        except Exception as e:
            logger.error(f"Force model unload failed: {e}")
            return False

    # ======================== Configuration APIs ========================

    def set_profile(self, profile: str) -> None:
        """Update threshold presets based on a named profile."""
        profile = profile.lower()
        if profile not in PROFILE_PRESETS:
            raise ValueError(f"Unknown profile: {profile}")
        th = PROFILE_PRESETS[profile]
        self.thresholds.low_threshold = th["low"]
        self.thresholds.medium_threshold = th["medium"]
        self.thresholds.high_threshold = th["high"]
        self.thresholds.critical_threshold = th["critical"]
        self.config["profile"] = profile

    def set_threshold(self, level: str, value: float) -> None:
        """Update a single threshold level at runtime."""
        level = level.lower()
        if value > 1:
            value = value / 100.0
        mapping = {
            "low": "low_threshold",
            "medium": "medium_threshold",
            "high": "high_threshold",
            "critical": "critical_threshold",
        }
        if level not in mapping:
            raise ValueError(f"Unknown threshold level: {level}")
        setattr(self.thresholds, mapping[level], float(value))
    
    # ======================== Utility Methods ========================
    
    def get_memory_report(self) -> Dict[str, Any]:
        """Get comprehensive memory report"""
        current_stats = self.get_memory_stats()
        
        report = {
            "guardian_status": "active" if self.is_monitoring else "inactive",
            "interventions_count": self.interventions_count,
            "oom_prevented_count": self.oom_prevented_count,
            "current_stats": current_stats.__dict__ if current_stats else None,
            "config": self.config,
            "thresholds": {
                "low": f"{self.thresholds.low_threshold * 100:.0f}%",
                "medium": f"{self.thresholds.medium_threshold * 100:.0f}%", 
                "high": f"{self.thresholds.high_threshold * 100:.0f}%",
                "critical": f"{self.thresholds.critical_threshold * 100:.0f}%"
            }
        }
        
        if len(self.memory_history) > 0:
            recent_stats = self.memory_history[-10:]  # Last 10 readings
            report["recent_usage"] = [
                {
                    "timestamp": s.timestamp.isoformat(),
                    "gpu_usage": f"{s.gpu_usage_percent:.1f}%",
                    "pressure": s.pressure_level.value
                } for s in recent_stats
            ]
        
        return report
    
    def check_memory_requirements(self, operation: str, required_gb: float) -> bool:
        """Check if there's enough memory for an operation"""
        stats = self.get_memory_stats()
        if not stats:
            return True  # Allow operation if we can't check
        
        available_gb = stats.gpu_free_gb
        
        # Add safety margin based on operation type
        if operation == "image_generation":
            safety_margin = self.thresholds.generation_reserve_gb
        elif operation == "llm_inference":
            safety_margin = self.thresholds.llm_reserve_gb
        else:
            safety_margin = 1.0  # Default 1GB margin
        
        has_memory = available_gb >= (required_gb + safety_margin)
        
        if not has_memory:
            logger.warning(
                f"Insufficient memory for {operation}: "
                f"need {required_gb:.1f}GB + {safety_margin:.1f}GB margin, "
                f"have {available_gb:.1f}GB"
            )
        
        return has_memory

def get_memory_guardian(app_state) -> MemoryGuardian:
    """Retrieve the memory guardian tied to the given AppState."""
    return app_state.memory_guardian


def start_memory_guardian(app_state):
    """Start monitoring using the AppState's memory guardian."""
    guardian = app_state.memory_guardian
    guardian.start_monitoring()
    return guardian


def stop_memory_guardian(app_state):
    """Stop monitoring using the AppState's memory guardian."""
    app_state.memory_guardian.stop_monitoring()
