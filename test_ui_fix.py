#!/usr/bin/env python3
"""
Test script to verify the UI parameter validation fix.
This script tests the parse_resolution function with various edge cases
that were causing the "list index out of range" error.
"""

import sys
import logging
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import the fixed function
try:
    from ui.web import parse_resolution
    print("✅ Successfully imported parse_resolution function")
except ImportError as e:
    print(f"❌ Failed to import parse_resolution: {e}")
    sys.exit(1)

# Set up logging to see the warnings
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

def test_parse_resolution():
    """Test the parse_resolution function with various edge cases."""
    
    print("\n🧪 Testing parse_resolution function with edge cases...")
    
    test_cases = [
        # Valid cases
        ("1024x1024 (Square - High Quality)", (1024, 1024), "Valid square resolution"),
        ("768x512 (Landscape)", (768, 512), "Valid landscape resolution"),
        ("512x768 (Portrait)", (512, 768), "Valid portrait resolution"),
        
        # Edge cases that were causing errors
        ("", (1024, 1024), "Empty string"),
        ("   ", (1024, 1024), "Whitespace only"),
        (None, (1024, 1024), "None value"),
        ("invalid", (1024, 1024), "No x in string"),
        ("1024", (1024, 1024), "Missing height"),
        ("x768", (1024, 1024), "Missing width"),
        ("abcxdef", (1024, 1024), "Non-numeric dimensions"),
        ("1024x", (1024, 1024), "Missing height after x"),
        ("x1024", (1024, 1024), "Missing width before x"),
        ("1024x768x512", (1024, 1024), "Too many x separators"),
        ("0x0", (1024, 1024), "Zero dimensions"),
        ("-100x200", (1024, 1024), "Negative width"),
        ("100x-200", (1024, 1024), "Negative height"),
        ("9999x9999", (2048, 2048), "Oversized dimensions (should clamp)"),
        ("50x50", (256, 256), "Undersized dimensions (should clamp)"),
        ("1024.5x768.7", (1024, 1024), "Float dimensions"),
    ]
    
    passed = 0
    failed = 0
    
    for i, (input_val, expected, description) in enumerate(test_cases, 1):
        try:
            print(f"\nTest {i}: {description}")
            print(f"  Input: {repr(input_val)}")
            
            result = parse_resolution(input_val)
            
            print(f"  Output: {result}")
            print(f"  Expected: {expected}")
            
            if result == expected:
                print("  ✅ PASSED")
                passed += 1
            else:
                print("  ❌ FAILED")
                failed += 1
                
        except Exception as e:
            print(f"  💥 EXCEPTION: {e}")
            failed += 1
    
    print(f"\n📊 Test Results:")
    print(f"  ✅ Passed: {passed}")
    print(f"  ❌ Failed: {failed}")
    print(f"  📈 Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    return failed == 0

def test_memory_issue_scenario():
    """Test the specific scenario that was causing memory issues."""
    
    print("\n🔍 Testing the specific error scenario from logs...")
    
    # Simulate the exact conditions that were causing the error
    problematic_inputs = [
        "",  # Empty string
        None,  # None value
        "malformed resolution string",  # No proper format
        "   \t\n   ",  # Whitespace variations
    ]
    
    all_passed = True
    
    for input_val in problematic_inputs:
        try:
            print(f"\nTesting problematic input: {repr(input_val)}")
            result = parse_resolution(input_val)
            print(f"  Result: {result}")
            
            # Should always return a safe default
            if result == (1024, 1024):
                print("  ✅ Safely handled with default value")
            else:
                print("  ⚠️ Unexpected result but no crash")
                
        except Exception as e:
            print(f"  💥 Still crashing: {e}")
            all_passed = False
    
    return all_passed

def main():
    """Run all tests."""
    print("🎯 Illustrious AI Studio - UI Fix Validation Test")
    print("=" * 60)
    
    # Test 1: Parse resolution function
    test1_passed = test_parse_resolution()
    
    # Test 2: Memory issue scenario
    test2_passed = test_memory_issue_scenario()
    
    print("\n" + "=" * 60)
    print("🏁 FINAL RESULTS:")
    
    if test1_passed and test2_passed:
        print("✅ ALL TESTS PASSED! The UI fix is working correctly.")
        print("🎉 Your image generation should now work without errors.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())