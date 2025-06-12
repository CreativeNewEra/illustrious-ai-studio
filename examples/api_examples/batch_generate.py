#!/usr/bin/env python3
"""
Batch Image Generation Example for Illustrious AI Studio
Demonstrates how to generate multiple images efficiently using the API.
"""

import requests
import json
import base64
import time
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Configuration
BASE_URL = "http://localhost:8000"
TIMEOUT = 120  # Longer timeout for batch operations
OUTPUT_DIR = "batch_output"
MAX_WORKERS = 3  # Concurrent requests (adjust based on GPU memory)

# Thread-safe counter for progress tracking
class ProgressCounter:
    def __init__(self, total):
        self.completed = 0
        self.total = total
        self.lock = threading.Lock()
    
    def increment(self):
        with self.lock:
            self.completed += 1
            print(f"= Progress: {self.completed}/{self.total} images completed")

def generate_single_image(prompt_data, counter, output_dir):
    """Generate a single image with error handling."""
    try:
        response = requests.post(
            f"{BASE_URL}/generate-image",
            json=prompt_data,
            timeout=TIMEOUT
        )
        response.raise_for_status()
        result = response.json()
        
        if result['success']:
            # Save the image
            filename = f"{prompt_data.get('name', 'image')}_{int(time.time())}.png"
            filepath = os.path.join(output_dir, filename)
            
            image_data = base64.b64decode(result['image_base64'])
            with open(filepath, "wb") as f:
                f.write(image_data)
            
            # Save metadata
            metadata = {
                "prompt": prompt_data['prompt'],
                "negative_prompt": prompt_data.get('negative_prompt', ''),
                "steps": prompt_data.get('steps', 30),
                "guidance": prompt_data.get('guidance', 7.5),
                "seed": prompt_data.get('seed', -1),
                "generated_at": datetime.now().isoformat(),
                "filename": filename
            }
            
            metadata_file = filepath.replace('.png', '_metadata.json')
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            counter.increment()
            return {"success": True, "filename": filename, "prompt": prompt_data['prompt'][:50]}
        else:
            return {"success": False, "error": result.get('message', 'Unknown error'), "prompt": prompt_data['prompt'][:50]}
            
    except Exception as e:
        return {"success": False, "error": str(e), "prompt": prompt_data['prompt'][:50]}

def batch_generate_character_series():
    """Generate a series of character portraits with consistent style."""
    print("Generating character portrait series...")
    
    base_style = "portrait, detailed face, studio lighting, masterpiece, best quality"
    negative = "blurry, low quality, bad anatomy, deformed face"
    
    character_prompts = [
        {
            "name": "warrior",
            "prompt": f"fantasy warrior with armor, {base_style}",
            "negative_prompt": negative,
            "steps": 25,
            "guidance": 7.5,
            "seed": 100
        },
        {
            "name": "mage",
            "prompt": f"wise mage with robes and staff, {base_style}",
            "negative_prompt": negative,
            "steps": 25,
            "guidance": 7.5,
            "seed": 101
        },
        {
            "name": "rogue",
            "prompt": f"stealthy rogue with hood and daggers, {base_style}",
            "negative_prompt": negative,
            "steps": 25,
            "guidance": 7.5,
            "seed": 102
        },
        {
            "name": "healer",
            "prompt": f"kind healer with white robes, {base_style}",
            "negative_prompt": negative,
            "steps": 25,
            "guidance": 7.5,
            "seed": 103
        }
    ]
    
    return character_prompts

def batch_generate_landscapes():
    """Generate a series of landscape images."""
    print("< Generating landscape series...")
    
    base_style = "landscape, detailed, atmospheric, masterpiece, best quality"
    negative = "blurry, low quality, people, buildings"
    
    landscape_prompts = [
        {
            "name": "mountain_sunset",
            "prompt": f"mountain range at sunset, golden hour, {base_style}",
            "negative_prompt": negative,
            "steps": 30,
            "guidance": 8.0,
            "seed": 200
        },
        {
            "name": "forest_mist",
            "prompt": f"misty forest with sunbeams, morning light, {base_style}",
            "negative_prompt": negative,
            "steps": 30,
            "guidance": 8.0,
            "seed": 201
        },
        {
            "name": "ocean_storm",
            "prompt": f"stormy ocean with dramatic clouds, {base_style}",
            "negative_prompt": negative,
            "steps": 30,
            "guidance": 8.0,
            "seed": 202
        },
        {
            "name": "desert_dunes",
            "prompt": f"sand dunes under starry sky, {base_style}",
            "negative_prompt": negative,
            "steps": 30,
            "guidance": 8.0,
            "seed": 203
        }
    ]
    
    return landscape_prompts

def batch_generate_style_variations():
    """Generate the same subject in different artistic styles."""
    print("< Generating style variation series...")
    
    base_prompt = "beautiful woman with long hair"
    negative = "blurry, low quality, bad anatomy"
    
    style_prompts = [
        {
            "name": "realistic",
            "prompt": f"{base_prompt}, photorealistic, detailed skin, studio portrait",
            "negative_prompt": negative,
            "steps": 35,
            "guidance": 7.0,
            "seed": 300
        },
        {
            "name": "anime",
            "prompt": f"{base_prompt}, anime style, cel shading, vibrant colors",
            "negative_prompt": negative,
            "steps": 25,
            "guidance": 8.0,
            "seed": 300
        },
        {
            "name": "oil_painting",
            "prompt": f"{base_prompt}, oil painting, classical art, renaissance style",
            "negative_prompt": negative,
            "steps": 30,
            "guidance": 7.5,
            "seed": 300
        },
        {
            "name": "watercolor",
            "prompt": f"{base_prompt}, watercolor painting, soft colors, artistic",
            "negative_prompt": negative,
            "steps": 28,
            "guidance": 7.5,
            "seed": 300
        }
    ]
    
    return style_prompts

def run_batch_generation(prompts, series_name):
    """Run batch generation with concurrent processing."""
    print(f"\n= Starting {series_name} batch generation...")
    print(f"= Total prompts: {len(prompts)}")
    
    # Create output directory
    series_dir = os.path.join(OUTPUT_DIR, series_name)
    os.makedirs(series_dir, exist_ok=True)
    
    # Initialize progress counter
    counter = ProgressCounter(len(prompts))
    
    # Start batch generation
    start_time = time.time()
    results = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_prompt = {
            executor.submit(generate_single_image, prompt, counter, series_dir): prompt
            for prompt in prompts
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_prompt):
            result = future.result()
            results.append(result)
    
    # Summary
    end_time = time.time()
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    print(f"\n= {series_name} Batch Complete!")
    print(f" Successful: {successful}")
    print(f"L Failed: {failed}")
    print(f"  Total time: {end_time - start_time:.1f} seconds")
    print(f"= Output saved to: {series_dir}")
    
    # Show any errors
    if failed > 0:
        print("\nL Failed generations:")
        for result in results:
            if not result['success']:
                print(f"  - {result['prompt']}: {result['error']}")
    
    return results

def main():
    """Run batch generation examples."""
    print("< Illustrious AI Studio - Batch Generation Examples")
    print("=" * 55)
    
    # Check server status
    try:
        response = requests.get(f"{BASE_URL}/status", timeout=30)
        response.raise_for_status()
        status = response.json()
        
        if not status['models'].get('sdxl', False):
            print("L SDXL model not loaded. Cannot generate images.")
            return
            
        print(f" Server status: {status['status']}")
        print(f"< GPU available: {status['gpu_available']} ({status['gpu_backend']})")
        
    except Exception as e:
        print(f"L Cannot connect to server: {e}")
        print("Make sure the application is running on port 8000.")
        return
    
    # Create main output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Available batch generation examples
    batch_examples = {
        "1": ("Character Portraits", batch_generate_character_series),
        "2": ("Landscape Series", batch_generate_landscapes),
        "3": ("Style Variations", batch_generate_style_variations),
        "4": ("All Examples", None)
    }
    
    print("\n= Available batch generation examples:")
    for key, (name, _) in batch_examples.items():
        print(f"  {key}. {name}")
    
    choice = input("\nChoose an example (1-4): ").strip()
    
    if choice == "4":
        # Run all examples
        for key, (name, func) in batch_examples.items():
            if func is not None:
                prompts = func()
                run_batch_generation(prompts, name.lower().replace(" ", "_"))
    elif choice in batch_examples and batch_examples[choice][1] is not None:
        name, func = batch_examples[choice]
        prompts = func()
        run_batch_generation(prompts, name.lower().replace(" ", "_"))
    else:
        print("L Invalid choice. Exiting.")
        return
    
    print("\n Batch generation complete!")

if __name__ == "__main__":
    main()
