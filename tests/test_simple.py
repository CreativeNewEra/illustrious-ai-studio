#!/usr/bin/env python3
"""
Simple test script for Illustrious AI Studio
Tests each component individually with proper memory management
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from PIL import Image
try:
    import torch
except Exception:  # pragma: no cover - torch may be missing
    torch = None
from colorama import init, Fore, Style
import shutil
import pytest

# Set environment variables BEFORE importing our modules
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['OLLAMA_KEEP_ALIVE'] = '0'  # Unload models immediately after use

init(autoreset=True)

if shutil.which('ollama') is None:
    pytest.skip("Ollama not installed", allow_module_level=True)

# Import our modules
from illustrious_ai_studio.core.state import AppState
from illustrious_ai_studio.core.sdxl import init_sdxl, generate_image
from illustrious_ai_studio.core.ollama import init_ollama, generate_prompt, analyze_image
from illustrious_ai_studio.core.memory import clear_gpu_memory


def print_header(text):
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"{Fore.CYAN}{text:^60}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")


def print_success(text):
    print(f"{Fore.GREEN}✓ {text}{Style.RESET_ALL}")


def print_error(text):
    print(f"{Fore.RED}✗ {text}{Style.RESET_ALL}")


def print_info(text):
    print(f"{Fore.BLUE}ℹ {text}{Style.RESET_ALL}")


def unload_ollama_gpu():
    """Force Ollama to unload from GPU"""
    print_info("Unloading Ollama from GPU...")
    
    # Use ollama CLI to set keepalive to 0
    subprocess.run(['ollama', 'run', 'goekdenizguelmez/JOSIEFIED-Qwen3:8b-q6_k', '--keepalive', '0s', ''], 
                   capture_output=True)
    subprocess.run(['ollama', 'run', 'qwen2.5vl:7b', '--keepalive', '0s', ''], 
                   capture_output=True)
    
    # Give it time to unload
    time.sleep(3)
    print_success("Ollama unloaded")


def test_llm_only():
    """Test LLM functionality without loading SDXL"""
    print_header("Testing LLM Only")
    
    state = AppState()
    assert state.ollama_vision_model is None
    
    # Initialize Ollama
    print_info("Initializing Ollama...")
    if init_ollama(state):
        print_success("Ollama initialized")
        
        # Test prompt generation
        test_prompt = "a beautiful sunset"
        print_info(f"Testing prompt enhancement: '{test_prompt}'")
        enhanced = generate_prompt(state, test_prompt)
        
        if enhanced and not enhanced.startswith("❌"):
            print_success("Prompt enhanced successfully")
            print_info(f"Result: {enhanced[:100]}...")
            return True
        else:
            print_error("Prompt enhancement failed")
            return False
    else:
        print_error("Failed to initialize Ollama")
        return False


def test_image_generation_only():
    """Test image generation without Ollama loaded"""
    print_header("Testing Image Generation Only")
    
    # First unload Ollama
    unload_ollama_gpu()
    
    # Clear CUDA cache
    clear_gpu_memory()

    state = AppState()
    assert state.ollama_vision_model is None
    
    # Initialize SDXL
    print_info("Initializing SDXL...")
    if init_sdxl(state):
        print_success("SDXL initialized")
        
        # Generate a simple image
        prompt = "a cute cat sitting on a pillow, highly detailed, 4k"
        print_info(f"Generating image: '{prompt}'")
        
        image, status = generate_image(
            state,
            {
                "prompt": prompt,
                "negative_prompt": "blurry, low quality",
                "steps": 20,
                "guidance": 7.5,
                "seed": 42,
            },
        )
        
        if image:
            print_success("Image generated successfully!")
            
            # Save test image
            test_dir = Path("test_outputs")
            test_dir.mkdir(exist_ok=True)
            image_path = test_dir / "simple_test.png"
            image.save(image_path)
            print_info(f"Image saved to: {image_path}")
            
            return True, image
        else:
            print_error(f"Image generation failed: {status}")
            return False, None
    else:
        print_error("Failed to initialize SDXL")
        return False, None


def test_vision_analysis(image_path):
    """Test vision analysis"""
    print_header("Testing Vision Analysis")
    
    # First unload SDXL by clearing the state
    clear_gpu_memory()
    
    state = AppState()
    assert state.ollama_vision_model is None
    
    # Initialize Ollama with vision
    print_info("Initializing vision model...")
    if init_ollama(state) and state.model_status.get("multimodal"):
        print_success("Vision model initialized")
        
        # Load image
        image = Image.open(image_path)
        
        # Analyze
        print_info("Analyzing image...")
        result = analyze_image(state, image, "What do you see in this image?")
        
        if result and not result.startswith("❌"):
            print_success("Image analysis successful")
            print_info(f"Analysis: {result[:200]}...")
            return True
        else:
            print_error("Image analysis failed")
            return False
    else:
        print_error("Failed to initialize vision model")
        return False


def main():
    print(f"{Fore.MAGENTA}╔═══════════════════════════════════════════════════════════╗")
    print(f"{Fore.MAGENTA}║        Illustrious AI Studio - Simple Test Suite          ║")
    print(f"{Fore.MAGENTA}╚═══════════════════════════════════════════════════════════╝{Style.RESET_ALL}")
    
    results = {}
    
    # Test 1: LLM only
    results['llm'] = test_llm_only()
    
    # Unload Ollama before image generation
    unload_ollama_gpu()
    
    # Test 2: Image generation only
    image_success, generated_image = test_image_generation_only()
    results['image'] = image_success
    
    # Test 3: Vision analysis (if image was generated)
    if generated_image:
        results['vision'] = test_vision_analysis("test_outputs/simple_test.png")
    else:
        results['vision'] = False
    
    # Summary
    print_header("Test Summary")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    if passed == total:
        print_success(f"All tests passed! ({passed}/{total})")
    else:
        print_info(f"Tests passed: {passed}/{total}")
    
    for test, result in results.items():
        if result:
            print_success(f"{test}: PASSED")
        else:
            print_error(f"{test}: FAILED")
    
    # Recommendations
    if passed < total:
        print_header("Recommendations")
        print_info("1. Use the model_manager.py to switch between modes")
        print_info("2. Run 'export OLLAMA_KEEP_ALIVE=0' before starting")
        print_info("3. Consider using lower resolution (768x768) for testing")
        print_info("4. Monitor GPU memory with 'watch nvidia-smi' (or 'watch rocm-smi')")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
