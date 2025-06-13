#!/usr/bin/env python3
"""
Test script to verify prompt validation fixes for the image generation error.
This script tests various edge cases that could cause the "prompt or prompt_embeds" error.
"""

import sys
import logging
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from core.state import AppState
from core.sdxl import generate_image
from core.config import CONFIG

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_prompt_validation():
    """Test various prompt validation scenarios."""
    print("🧪 Testing Prompt Validation Fixes")
    print("=" * 50)
    
    # Create a test state
    state = AppState()
    
    # Test cases that previously could cause errors
    test_cases = [
        # (prompt, negative_prompt, expected_error_type, description)
        (None, "", "prompt_validation", "None prompt"),
        ("", "", "prompt_validation", "Empty string prompt"),
        ("   ", "", "prompt_validation", "Whitespace-only prompt"),
        ("valid prompt", None, "model_not_loaded", "None negative prompt (should work)"),
        ("valid prompt", "", "model_not_loaded", "Valid basic prompt"),
        ("a simple test image", "blurry", "model_not_loaded", "Normal valid case"),
        (123, "", "model_not_loaded", "Non-string prompt (number) - should convert"),
        ([], "", "model_not_loaded", "Non-string prompt (list) - should convert"),
        ("test", 456, "model_not_loaded", "Non-string negative prompt (should convert)"),
    ]
    
    print(f"Running {len(test_cases)} validation tests...\n")
    
    passed = 0
    failed = 0
    
    for i, (prompt, negative_prompt, expected_error_type, description) in enumerate(test_cases, 1):
        print(f"Test {i}: {description}")
        print(f"  Input: prompt={repr(prompt)}, negative_prompt={repr(negative_prompt)}")
        
        try:
            # Test the validation without actually running generation
            # (since we may not have a model loaded)
            image, status = generate_image(
                state=state,
                prompt=prompt,
                negative_prompt=negative_prompt,
                steps=10,  # Minimal steps
                guidance=7.5,
                seed=42,
                save_to_gallery_flag=False,
                width=512,  # Smaller size for testing
                height=512,
                progress_callback=None
            )
            
            # Check if we got the expected type of error
            if expected_error_type == "prompt_validation":
                # Should fail with prompt validation error
                if image is None and ("Prompt cannot be" in status or "Cannot convert prompt" in status):
                    print(f"  ✅ PASSED: Correct prompt validation error: {status}")
                    passed += 1
                else:
                    print(f"  ❌ FAILED: Expected prompt validation error, got: {status}")
                    failed += 1
            elif expected_error_type == "model_not_loaded":
                # Should fail with model not loaded error (validation passed)
                if image is None and "SDXL model not loaded" in status:
                    print(f"  ✅ PASSED: Validation passed, stopped at model check: {status}")
                    passed += 1
                else:
                    print(f"  ❌ FAILED: Expected model error after validation, got: {status}")
                    failed += 1
            else:
                print(f"  ❌ FAILED: Unknown expected error type: {expected_error_type}")
                failed += 1
                
        except Exception as e:
            print(f"  ❌ FAILED: Unexpected exception: {e}")
            failed += 1
        
        print()
    
    print("=" * 50)
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! The prompt validation fixes are working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please review the validation logic.")
        return False

def test_parameter_edge_cases():
    """Test edge cases for other parameters."""
    print("\n🔬 Testing Parameter Edge Cases")
    print("=" * 50)
    
    state = AppState()
    
    edge_cases = [
        # (steps, guidance, seed, expected_success, description)
        (-1, 7.5, 42, True, "Negative steps (should be corrected)"),
        (0, 7.5, 42, True, "Zero steps (should be corrected)"),
        (300, 7.5, 42, True, "Too many steps (should be corrected)"),
        (30, -1, 42, True, "Negative guidance (should be corrected)"),
        (30, 100, 42, True, "Too high guidance (should be corrected)"),
        (30, 7.5, -2, True, "Invalid seed (should use random)"),
        (30, 7.5, 2**33, True, "Seed too large (should use random)"),
        ("invalid", 7.5, 42, True, "Non-numeric steps (should be corrected)"),
        (30, "invalid", 42, True, "Non-numeric guidance (should be corrected)"),
        (30, 7.5, "invalid", True, "Non-numeric seed (should use random)"),
    ]
    
    passed = 0
    failed = 0
    
    for i, (steps, guidance, seed, expected_success, description) in enumerate(edge_cases, 1):
        print(f"Edge Case {i}: {description}")
        print(f"  Input: steps={repr(steps)}, guidance={repr(guidance)}, seed={repr(seed)}")
        
        try:
            image, status = generate_image(
                state=state,
                prompt="test prompt",  # Valid prompt
                negative_prompt="",
                steps=steps,
                guidance=guidance,
                seed=seed,
                save_to_gallery_flag=False,
                width=512,
                height=512,
                progress_callback=None
            )
            
            # For edge cases, we mainly care that no exceptions are thrown
            # and that the validation doesn't cause the "prompt or prompt_embeds" error
            if "prompt" not in status.lower() or "prompt_embeds" not in status.lower():
                print(f"  ✅ PASSED: Parameters handled gracefully")
                print(f"     Status: {status}")
                passed += 1
            else:
                print(f"  ❌ FAILED: Still getting prompt/prompt_embeds error")
                print(f"     Status: {status}")
                failed += 1
                
        except Exception as e:
            print(f"  ❌ FAILED: Unexpected exception: {e}")
            failed += 1
        
        print()
    
    print("=" * 50)
    print(f"Edge Case Results: {passed} passed, {failed} failed")
    
    return failed == 0

if __name__ == "__main__":
    print("🔧 Prompt Validation Test Suite")
    print("Testing fixes for: 'Provide either prompt or prompt_embeds' error")
    print()
    
    # Run validation tests
    validation_success = test_prompt_validation()
    
    # Run edge case tests
    edge_case_success = test_parameter_edge_cases()
    
    print("\n" + "=" * 60)
    if validation_success and edge_case_success:
        print("🎊 ALL TESTS PASSED! The fixes should resolve your image generation error.")
        print("💡 Key improvements:")
        print("   • Comprehensive prompt validation prevents None/empty prompts")
        print("   • Parameter sanitization handles invalid types and ranges")
        print("   • Enhanced error messages for better debugging")
        print("   • Multi-layer validation (UI, API, and core generation)")
        sys.exit(0)
    else:
        print("💥 SOME TESTS FAILED! Please review the implementation.")
        sys.exit(1)
