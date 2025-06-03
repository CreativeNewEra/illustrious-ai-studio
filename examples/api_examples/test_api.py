#!/usr/bin/env python3
"""
Test API Examples for Illustrious AI Studio
Tests all available API endpoints with example requests.
"""

import requests
import json
import base64
from PIL import Image
import io
import time

# Configuration
BASE_URL = "http://localhost:8000"
TIMEOUT = 60  # seconds

def test_status_endpoint():
    """Test the status endpoint to verify server is running."""
    print("Testing status endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/status", timeout=TIMEOUT)
        response.raise_for_status()
        status = response.json()
        
        print(f"Status: {status['status']}")
        print(f"Models: {status['models']}")
        print(f"CUDA Available: {status['cuda_available']}")
        return status
    except requests.exceptions.RequestException as e:
        print(f"Status endpoint failed: {e}")
        return None

def test_image_generation():
    """Test image generation endpoint with various prompts."""
    print("\nTesting image generation...")
    
    test_prompts = [
        {
            "prompt": "a beautiful sunset over mountains, masterpiece, best quality",
            "negative_prompt": "blurry, low quality, bad anatomy",
            "steps": 20,
            "guidance": 7.5,
            "seed": 42
        },
        {
            "prompt": "cute anime girl with blue hair, detailed face, studio lighting",
            "negative_prompt": "nsfw, blurry, low quality",
            "steps": 25,
            "guidance": 8.0,
            "seed": 123
        }
    ]
    
    for i, test_data in enumerate(test_prompts, 1):
        print(f"\nTest {i}: {test_data['prompt'][:50]}...")
        try:
            response = requests.post(
                f"{BASE_URL}/generate-image",
                json=test_data,
                timeout=TIMEOUT
            )
            response.raise_for_status()
            result = response.json()
            
            if result['success']:
                print(f"Image generated successfully!")
                print(f"Message: {result.get('message', 'No message')}")
                
                # Optionally save the image
                if 'image_base64' in result:
                    image_data = base64.b64decode(result['image_base64'])
                    with open(f"test_output_{i}.png", "wb") as f:
                        f.write(image_data)
                    print(f"Saved as test_output_{i}.png")
            else:
                print(f"Generation failed: {result.get('message', 'Unknown error')}")
                
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")

def test_chat_endpoint():
    """Test chat endpoint with various messages."""
    print("\nTesting chat endpoint...")
    
    test_messages = [
        {
            "message": "Hello! How are you?",
            "session_id": "test_session_1"
        },
        {
            "message": "Can you help me create a prompt for generating a fantasy landscape?",
            "session_id": "test_session_1",
            "temperature": 0.8,
            "max_tokens": 512
        },
        {
            "message": "#generate a magical forest with glowing mushrooms, fantasy art style",
            "session_id": "test_session_2"
        }
    ]
    
    for i, test_data in enumerate(test_messages, 1):
        print(f"\nTest {i}: {test_data['message'][:50]}...")
        try:
            response = requests.post(
                f"{BASE_URL}/chat",
                json=test_data,
                timeout=TIMEOUT
            )
            response.raise_for_status()
            result = response.json()
            
            print(f"Response: {result['response'][:100]}...")
            print(f"Session: {result['session_id']}")
            
        except requests.exceptions.RequestException as e:
            print(f"Chat request failed: {e}")

def test_image_analysis():
    """Test image analysis endpoint."""
    print("\nTesting image analysis...")
    
    # Create a simple test image
    test_image = Image.new('RGB', (256, 256), color='blue')
    buffer = io.BytesIO()
    test_image.save(buffer, format='PNG')
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    test_data = {
        "image_base64": image_base64,
        "question": "What colors do you see in this image?"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/analyze-image",
            json=test_data,
            timeout=TIMEOUT
        )
        response.raise_for_status()
        result = response.json()
        
        print(f"Analysis: {result['analysis']}")
        
    except requests.exceptions.RequestException as e:
        print(f"Image analysis failed: {e}")

def main():
    """Run all API tests."""
    print("Illustrious AI Studio - API Testing Suite")
    print("=" * 50)
    
    # Test server status first
    status = test_status_endpoint()
    if not status:
        print("Server is not responding. Make sure the application is running on port 8000.")
        return
    
    # Check if models are loaded
    if not status['models'].get('sdxl', False):
        print("Warning: SDXL model not loaded. Image generation tests may fail.")
    
    if not status['models'].get('ollama', False):
        print("Warning: Ollama model not loaded. Chat tests may fail.")
    
    # Run tests
    test_image_generation()
    test_chat_endpoint()
    
    # Only test image analysis if multimodal is available
    if status['models'].get('multimodal', False):
        test_image_analysis()
    else:
        print("\nSkipping image analysis test (multimodal model not available)")
    
    print("\nAPI testing complete!")

if __name__ == "__main__":
    main()
