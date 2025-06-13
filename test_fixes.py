#!/usr/bin/env python3
"""
Test script to verify recent fixes are working
Tests the new features and improvements made to address the issues.
"""

import sys
import json
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

def test_recent_prompts_functionality():
    """Test the recent prompts functionality."""
    print("🧪 Testing Recent Prompts Functionality")
    print("-" * 40)
    
    try:
        # Import the functions from web.py
        from ui.web import load_recent_prompts, save_recent_prompts, add_to_recent_prompts, clear_recent_prompts
        
        # Test saving and loading
        test_prompts = [
            "a beautiful sunset over mountains",
            "anime girl with blue hair",
            "cyberpunk cityscape at night"
        ]
        
        # Clear any existing prompts
        clear_recent_prompts()
        
        # Add test prompts
        for prompt in test_prompts:
            add_to_recent_prompts(prompt)
        
        # Load and verify
        loaded_prompts = load_recent_prompts()
        
        print(f"✅ Added {len(test_prompts)} prompts")
        print(f"✅ Loaded {len(loaded_prompts)} prompts")
        
        # Verify order (most recent first)
        assert loaded_prompts[0] == test_prompts[-1], "❌ Incorrect order: Most recent prompt is not first"
        print("✅ Correct order (most recent first)")
        
        # Test clearing
        clear_recent_prompts()
        cleared_prompts = load_recent_prompts()
        
        if len(cleared_prompts) == 0:
            print("✅ Clear functionality works")
        else:
            print("❌ Clear functionality failed")
        
        return True
        
    except Exception as e:
        print(f"❌ Recent prompts test failed: {e}")
        return False

def test_resolution_parsing():
    """Test the resolution parsing functionality."""
    print("\n🧪 Testing Resolution Parsing")
    print("-" * 40)
    
    try:
        from ui.web import parse_resolution
        
        test_cases = [
            ("512x512 (Square - Fast)", (512, 512)),
            ("1024x768 (Landscape HD)", (1024, 768)),
            ("768x1024 (Portrait HD)", (768, 1024)),
            ("invalid_resolution", (1024, 1024))  # Should default
        ]
        
        all_passed = True
        for resolution_string, expected in test_cases:
            result = parse_resolution(resolution_string)
            if result == expected:
                print(f"✅ {resolution_string} -> {result}")
            else:
                print(f"❌ {resolution_string} -> {result} (expected {expected})")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Resolution parsing test failed: {e}")
        return False

# Removed duplicate definition of test_api_error_handling
def test_api_error_handling():
    """Test the improved API error handling."""
    print("\n🧪 Testing API Error Handling")
    print("-" * 40)

    try:
        from server.api import create_api_app
        from core.state import AppState
        from fastapi.testclient import TestClient

        # Create a test app state and FastAPI app
        state = AppState()
        app = create_api_app(state, auto_load=False)
        client = TestClient(app)

        # Test missing required fields
        response = client.post("/generate-image", json={})
        if response.status_code != 422 and response.status_code != 400:
            print(f"❌ Expected 400/422 for missing fields, got {response.status_code}")
            return False
        else:
            print("✅ Proper error for missing required fields")

        # Test invalid data types
        response = client.post("/generate-image", json={"prompt": 12345})
        if response.status_code != 422 and response.status_code != 400:
            print(f"❌ Expected 400/422 for invalid data type, got {response.status_code}")
            return False
        else:
            print("✅ Proper error for invalid data type")

        # Test valid request if possible (optional)
        # response = client.post("/generate-image", json={"prompt": "test prompt"})
        # if response.status_code == 200:
        #     print("✅ Valid request succeeded")
        # else:
        #     print(f"⚠️ Valid request failed with status {response.status_code}")

        print("✅ API app creation and error handling tested")
        return True

    except Exception as e:
        print(f"❌ API error handling test failed: {e}")
        return False
    """Test that all necessary imports are working."""
    print("\n🧪 Testing Core Imports")
    print("-" * 40)
    
    try:
        from core.state import AppState
        from core.config import CONFIG
        print("✅ Core imports successful")
        
        from ui.web import create_gradio_app
        print("✅ Web UI imports successful")
        
        from server.api import create_api_app
        print("✅ API imports successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🎨 Illustrious AI Studio - Fix Verification Tests")
    print("=" * 50)
    
    tests = [
        ("Core Imports", test_imports),
        ("Recent Prompts", test_recent_prompts_functionality),
        ("Resolution Parsing", test_resolution_parsing),
        ("API Error Handling", test_api_error_handling)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All fixes are working correctly!")
        return True
    else:
        print("⚠️ Some issues remain - check the failed tests above")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
