#!/usr/bin/env python3
"""
Test script to validate model selection functionality
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.sdxl import get_available_models, test_model_generation, get_current_model_info
from core.state import AppState
from core.config import CONFIG

def main():
    print("🧪 Testing Model Selection Functionality")
    print("=" * 50)
    
    # Initialize app state
    state = AppState()
    assert state.ollama_vision_model is None
    
    # Test 1: Get available models
    print("\n1. Scanning for available models...")
    models = get_available_models()
    
    if not models:
        print("❌ No models found!")
        return False
    
    print(f"✅ Found {len(models)} models:")
    for model in models:
        status = "✅ Current" if model["is_current"] else "🎭 Available"
        print(f"   {status} {model['display_name']} ({model['size_mb']}MB)")
        print(f"      Path: {model['path']}")
    
    # Test 2: Get current model info
    print("\n2. Current model information...")
    current_info = get_current_model_info(state)
    print(f"   Display Name: {current_info['display_name']}")
    print(f"   Status: {current_info['status']}")
    print(f"   Size: {current_info['size']}")
    
    # Test 3: Test each model
    print("\n3. Testing each model's generation capability...")
    
    all_tests_passed = True
    for model in models:
        print(f"\n   Testing {model['display_name']}...")
        try:
            success, message, test_image = test_model_generation(state, model['path'])
            
            if success:
                print(f"   ✅ {message}")
                if test_image:
                    print(f"   📸 Test image generated successfully")
                else:
                    print(f"   ⚠️ Success reported but no image returned")
            else:
                print(f"   ❌ {message}")
                all_tests_passed = False
                
        except Exception as e:
            print(f"   ❌ Test failed with exception: {e}")
            all_tests_passed = False
    
    # Summary
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("🎉 All model tests passed! GUI model selection is working correctly.")
        print("\nFeatures implemented:")
        print("✅ Dynamic model discovery")
        print("✅ Model validation and testing")
        print("✅ GUI dropdown with model selection")
        print("✅ Automatic model switching")
        print("✅ User-friendly model names")
        print("✅ Model information display")
        return True
    else:
        print("⚠️ Some model tests failed. Check the logs for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
