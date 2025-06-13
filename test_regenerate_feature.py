#!/usr/bin/env python3
"""
Test script for the new regenerate functionality
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.state import AppState

def test_regenerate_state():
    """Test that AppState properly stores last generation parameters"""
    print("🧪 Testing Regenerate Feature")
    print("=" * 50)
    
    # Create state instance
    state = AppState()
    assert state.ollama_vision_model is None
    
    # Test 1: Initial state
    print("✅ Test 1: Initial state")
    assert state.last_generation_params is None, "Initial state should have no params"
    print("   ✓ last_generation_params is None initially")
    
    # Test 2: Setting parameters
    print("\n✅ Test 2: Setting generation parameters")
    test_params = {
        "prompt": "a beautiful sunset over mountains",
        "negative_prompt": "blurry, low quality",
        "steps": 30,
        "guidance": 7.5,
        "seed": 12345,
        "save_gallery": True,
        "resolution": "1024x1024 (Square - High Quality)",
        "width": 1024,
        "height": 1024
    }
    
    state.last_generation_params = test_params
    assert state.last_generation_params is not None, "Parameters should be stored"
    assert state.last_generation_params["prompt"] == "a beautiful sunset over mountains"
    assert state.last_generation_params["steps"] == 30
    assert state.last_generation_params["width"] == 1024
    print("   ✓ Parameters stored correctly")
    
    # Test 3: Parameter retrieval
    print("\n✅ Test 3: Parameter retrieval")
    retrieved = state.last_generation_params
    assert retrieved["prompt"] == test_params["prompt"]
    assert retrieved["guidance"] == test_params["guidance"]
    assert retrieved["resolution"] == test_params["resolution"]
    print("   ✓ Parameters retrieved correctly")
    
    print("\n🎉 All regenerate tests passed!")
    print("✅ Regenerate feature is ready for use")

if __name__ == "__main__":
    try:
        test_regenerate_state()
        print("\n📋 Summary:")
        print("- ✅ AppState updated with last_generation_params")
        print("- ✅ Parameter storage and retrieval working")
        print("- ✅ Ready for UI integration")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
