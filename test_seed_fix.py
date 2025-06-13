#!/usr/bin/env python3
"""Test script to verify seed handling fix for the manual_seed NoneType error."""

import sys
import torch
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from core.state import AppState
from core.sdxl import generate_image, init_sdxl
from core.config import CONFIG

def test_seed_validation():
    """Test that seed validation handles None and invalid values correctly."""
    print("🧪 Testing seed validation fix...")
    
    # Initialize app state
    state = AppState()
    
    # Initialize SDXL if model exists
    if not Path(CONFIG.sd_model).exists():
        print("⚠️  SDXL model not found, skipping image generation test")
        return True
    
    # Load the model
    pipe = init_sdxl(state)
    if not pipe:
        print("❌ Failed to load SDXL model")
        return False
    
    test_prompt = "a simple red apple on white background"
    
    # Test cases for seed validation
    test_cases = [
        {"name": "None seed", "seed": None, "should_work": True},
        {"name": "Random seed (-1)", "seed": -1, "should_work": True},
        {"name": "Valid seed (42)", "seed": 42, "should_work": True},
        {"name": "String seed ('123')", "seed": "123", "should_work": True},
        {"name": "Float seed (42.5)", "seed": 42.5, "should_work": True},
        {"name": "Invalid string seed ('abc')", "seed": "abc", "should_work": True},  # Should fallback to random
        {"name": "Very large seed", "seed": 2**33, "should_work": True},  # Should fallback to random
    ]
    
    print("\n📋 Testing different seed values:")
    
    all_passed = True
    for test_case in test_cases:
        print(f"\n   Testing {test_case['name']}: {test_case['seed']}")
        
        try:
            image, status = generate_image(
                state=state,
                prompt=test_prompt,
                negative_prompt="blurry",
                steps=5,  # Very fast generation for testing
                guidance=7.5,
                seed=test_case['seed'],
                save_to_gallery_flag=False,  # Don't save test images
                width=512,  # Small size for speed
                height=512
            )
            
            if image is not None:
                print(f"   ✅ {test_case['name']}: SUCCESS - {status}")
            else:
                print(f"   ❌ {test_case['name']}: FAILED - {status}")
                if test_case['should_work']:
                    all_passed = False
                    
        except Exception as e:
            print(f"   ❌ {test_case['name']}: EXCEPTION - {str(e)}")
            if "manual_seed expected a long, but got NoneType" in str(e):
                print("   🚨 ORIGINAL BUG DETECTED! Seed validation failed!")
                all_passed = False
            elif test_case['should_work']:
                all_passed = False
    
    return all_passed

def main():
    """Run the seed validation test."""
    print("🔧 Testing seed handling fix for manual_seed NoneType error")
    print("=" * 60)
    
    try:
        success = test_seed_validation()
        
        print("\n" + "=" * 60)
        if success:
            print("✅ All seed validation tests passed!")
            print("🎉 The manual_seed NoneType error has been fixed!")
        else:
            print("❌ Some seed validation tests failed!")
            print("🔧 The seed handling may need additional work.")
            
        return 0 if success else 1
        
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
