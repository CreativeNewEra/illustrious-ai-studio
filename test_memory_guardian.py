#!/usr/bin/env python3
"""
Test script for Memory Guardian functionality
Tests automatic OOM prevention and memory management
"""

import sys
import time
import torch
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from core.memory_guardian import get_memory_guardian, MemoryPressureLevel
from core.state import AppState
from core.config import CONFIG
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_memory_stats():
    """Test basic memory statistics functionality"""
    print("🧪 Testing Memory Statistics")
    print("-" * 40)
    
    app_state = AppState()
    guardian = get_memory_guardian(app_state)
    
    stats = guardian.get_memory_stats()
    if stats:
        print(f"✅ GPU Total: {stats.gpu_total_gb:.1f} GB")
        print(f"✅ GPU Used: {stats.gpu_reserved_gb:.1f} GB ({stats.gpu_usage_percent:.1f}%)")
        print(f"✅ GPU Free: {stats.gpu_free_gb:.1f} GB")
        print(f"✅ System RAM: {stats.system_ram_usage_percent:.1f}%")
        print(f"✅ Pressure Level: {stats.pressure_level.value}")
        return True
    else:
        print("❌ Failed to get memory statistics")
        return False

def test_memory_requirements_check():
    """Test memory requirements checking"""
    print("\n🧪 Testing Memory Requirements Check")
    print("-" * 40)
    
    app_state = AppState()
    guardian = get_memory_guardian(app_state)
    
    # Test different operation types
    operations = [
        ("image_generation", 4.0),  # 4GB for image generation
        ("llm_inference", 2.0),     # 2GB for LLM
        ("light_operation", 0.5),   # 0.5GB for light operation
    ]
    
    for operation, required_gb in operations:
        has_memory = guardian.check_memory_requirements(operation, required_gb)
        status = "✅" if has_memory else "⚠️"
        print(f"{status} {operation}: {required_gb}GB -> {'Available' if has_memory else 'Insufficient'}")
    
    return True

def test_intervention_registration():
    """Test intervention callback registration"""
    print("\n🧪 Testing Intervention Registration")
    print("-" * 40)
    
    app_state = AppState()
    guardian = get_memory_guardian(app_state)
    
    # Test callback
    def test_intervention():
        print("  📢 Test intervention called!")
        return True
    
    # Register test intervention
    guardian.register_intervention(MemoryPressureLevel.MEDIUM, test_intervention)
    
    # Check if registered
    medium_interventions = guardian.intervention_callbacks[MemoryPressureLevel.MEDIUM]
    if test_intervention in medium_interventions:
        print("✅ Intervention successfully registered")
        return True
    else:
        print("❌ Intervention registration failed")
        return False

def test_memory_monitoring():
    """Test memory monitoring functionality"""
    print("\n🧪 Testing Memory Monitoring")
    print("-" * 40)
    
    app_state = AppState()
    guardian = get_memory_guardian(app_state)
    
    # Start monitoring
    print("Starting memory monitoring...")
    guardian.start_monitoring()
    
    if guardian.is_monitoring:
        print("✅ Memory monitoring started")
        
        # Monitor for a few seconds
        print("Monitoring for 5 seconds...")
        for i in range(5):
            time.sleep(1)
            stats = guardian.get_memory_stats()
            if stats:
                print(f"  📊 GPU: {stats.gpu_usage_percent:.1f}% | Pressure: {stats.pressure_level.value}")
        
        # Stop monitoring
        guardian.stop_monitoring()
        print("✅ Memory monitoring stopped")
        return True
    else:
        print("❌ Failed to start memory monitoring")
        return False

def test_adaptive_settings():
    """Test adaptive settings calculation"""
    print("\n🧪 Testing Adaptive Settings")
    print("-" * 40)
    
    app_state = AppState()
    guardian = get_memory_guardian(app_state)
    
    # Import the adaptive settings function
    from core.sdxl import _get_adaptive_settings
    
    # Test with different scenarios
    scenarios = [
        (1024, 1024, 30),  # Standard settings
        (2048, 2048, 50),  # High settings
    ]
    
    for width, height, steps in scenarios:
        adapted_width, adapted_height, adapted_steps = _get_adaptive_settings(width, height, steps, guardian)
        
        adapted = (adapted_width != width or adapted_height != height or adapted_steps != steps)
        status = "⚠️ Adapted" if adapted else "✅ No change"
        
        print(f"{status} {width}x{height}@{steps} -> {adapted_width}x{adapted_height}@{adapted_steps}")
    
    return True

def test_memory_report():
    """Test memory report generation"""
    print("\n🧪 Testing Memory Report")
    print("-" * 40)
    
    app_state = AppState()
    guardian = get_memory_guardian(app_state)
    
    try:
        report = guardian.get_memory_report()
        
        required_keys = ['guardian_status', 'interventions_count', 'oom_prevented_count', 'config', 'thresholds']
        
        for key in required_keys:
            if key in report:
                print(f"✅ Report contains {key}: {report[key]}")
            else:
                print(f"❌ Report missing {key}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error generating report: {e}")
        return False

def test_configuration():
    """Test configuration loading"""
    print("\n🧪 Testing Configuration")
    print("-" * 40)
    
    app_state = AppState()
    guardian = get_memory_guardian(app_state)
    
    config = guardian.config
    print(f"✅ Enabled: {config['enabled']}")
    print(f"✅ Profile: {config['profile']}")
    print(f"✅ Auto model switching: {config['auto_model_switching']}")
    print(f"✅ Log memory stats: {config['log_memory_stats']}")
    
    thresholds = guardian.thresholds
    print(f"✅ Medium threshold: {thresholds.medium_threshold * 100:.0f}%")
    print(f"✅ High threshold: {thresholds.high_threshold * 100:.0f}%")
    print(f"✅ Critical threshold: {thresholds.critical_threshold * 100:.0f}%")
    
    return True

def test_memory_allocation_simulation():
    """Simulate memory allocation to test pressure detection"""
    print("\n🧪 Testing Memory Allocation Simulation")
    print("-" * 40)
    
    if not torch.cuda.is_available():
        print("⚠️ CUDA not available, skipping allocation test")
        return True
    
    app_state = AppState()
    guardian = get_memory_guardian(app_state)
    
    initial_stats = guardian.get_memory_stats()
    if not initial_stats:
        print("❌ Cannot get initial memory stats")
        return False
    
    print(f"Initial GPU usage: {initial_stats.gpu_usage_percent:.1f}%")
    
    try:
        # Allocate some memory to increase pressure
        print("Allocating test tensors...")
        tensors = []
        
        # Allocate in chunks until we see pressure increase
        for i in range(3):
            # Allocate 500MB tensor
            tensor = torch.randn(64, 1024, 1024, device='cuda', dtype=torch.float32)
            tensors.append(tensor)
            
            current_stats = guardian.get_memory_stats()
            if current_stats:
                print(f"  Allocation {i+1}: {current_stats.gpu_usage_percent:.1f}% "
                      f"(pressure: {current_stats.pressure_level.value})")
                
                # If we reach medium pressure, stop allocating
                if current_stats.pressure_level != MemoryPressureLevel.LOW:
                    break
        
        # Clean up
        print("Cleaning up test tensors...")
        for tensor in tensors:
            del tensor
        torch.cuda.empty_cache()
        
        final_stats = guardian.get_memory_stats()
        if final_stats:
            print(f"Final GPU usage: {final_stats.gpu_usage_percent:.1f}%")
        
        print("✅ Memory allocation simulation completed")
        return True
        
    except Exception as e:
        print(f"❌ Error in memory allocation test: {e}")
        # Clean up on error
        try:
            for tensor in tensors:
                del tensor
            torch.cuda.empty_cache()
        except:
            pass
        return False

def run_all_tests():
    """Run all memory guardian tests"""
    print("🛡️ Memory Guardian Test Suite")
    print("=" * 50)
    
    tests = [
        ("Memory Statistics", test_memory_stats),
        ("Memory Requirements Check", test_memory_requirements_check),
        ("Intervention Registration", test_intervention_registration),
        ("Memory Monitoring", test_memory_monitoring),
        ("Adaptive Settings", test_adaptive_settings),
        ("Memory Report", test_memory_report),
        ("Configuration", test_configuration),
        ("Memory Allocation Simulation", test_memory_allocation_simulation),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"\n✅ {test_name} PASSED")
            else:
                print(f"\n❌ {test_name} FAILED")
        except Exception as e:
            print(f"\n💥 {test_name} ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Memory Guardian is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
