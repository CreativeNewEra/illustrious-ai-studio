#!/usr/bin/env python3
"""
Model Setup Script for Illustrious AI Studio
Automatically downloads and configures recommended models.
"""

import os
import sys
import subprocess
import urllib.request
import yaml
from pathlib import Path
from typing import Dict, Any

# Configuration
MODELS_DIR = Path("models")
CONFIG_FILE = Path("config.yaml")

# Recommended models
RECOMMENDED_MODELS = {
    "sdxl": {
        "name": "Illustrious-XL",
        "url": "https://huggingface.co/OnomaAI/Illustrious-xl/resolve/main/Illustrious-xl-v0.1.safetensors",
        "filename": "Illustrious.safetensors",
        "size": "6.6GB",
        "description": "High-quality anime/manga style SDXL model"
    }
}

OLLAMA_MODELS = {
    "llm": "goekdenizguelmez/JOSIEFIED-Qwen3:8b-q6_k",
    "vision": "qwen2.5vl:7b"
}

def check_ollama_installed() -> bool:
    """Check if Ollama is installed and available."""
    try:
        result = subprocess.run(["ollama", "--version"], 
                              capture_output=True, text=True, check=True)
        print(f"✓ Ollama found: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ Ollama not found. Please install Ollama first:")
        print("  Visit: https://ollama.ai/download")
        return False

def check_ollama_running() -> bool:
    """Check if Ollama service is running."""
    try:
        result = subprocess.run(["ollama", "list"], 
                              capture_output=True, text=True, check=True)
        print("✓ Ollama service is running")
        return True
    except subprocess.CalledProcessError:
        print("✗ Ollama service not running. Please start it:")
        print("  Run: ollama serve")
        return False

def download_with_progress(url: str, filepath: Path) -> bool:
    """Download a file with progress indicator."""
    def progress_hook(block_num, block_size, total_size):
        if total_size > 0:
            percent = min(100.0, block_num * block_size / total_size * 100)
            sys.stdout.write(f"\rDownloading: {percent:.1f}%")
            sys.stdout.flush()
    
    try:
        print(f"Downloading {filepath.name}...")
        urllib.request.urlretrieve(url, filepath, progress_hook)
        print("\n✓ Download completed")
        return True
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        return False

def download_sdxl_model() -> bool:
    """Download the recommended SDXL model."""
    model_info = RECOMMENDED_MODELS["sdxl"]
    model_path = MODELS_DIR / model_info["filename"]
    
    if model_path.exists():
        print(f"✓ SDXL model already exists: {model_path}")
        return True
    
    print(f"\nDownloading {model_info['name']} ({model_info['size']})...")
    print(f"Description: {model_info['description']}")
    
    # Create models directory
    MODELS_DIR.mkdir(exist_ok=True)
    
    # Download the model
    return download_with_progress(model_info["url"], model_path)

def pull_ollama_model(model_name: str, description: str) -> bool:
    """Pull an Ollama model."""
    try:
        print(f"\nPulling {description}: {model_name}")
        result = subprocess.run(["ollama", "pull", model_name], 
                              check=True, text=True)
        print(f"✓ {description} model ready: {model_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to pull {description} model: {e}")
        return False

def update_config(config_path: Path) -> bool:
    """Update config.yaml with correct model paths."""
    try:
        # Load current config
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = {}
        
        # Update model paths
        model_path = str((MODELS_DIR / RECOMMENDED_MODELS["sdxl"]["filename"]).absolute())
        config['sd_model'] = model_path
        config['ollama_model'] = OLLAMA_MODELS["llm"]
        config['ollama_vision_model'] = OLLAMA_MODELS["vision"]
        
        # Ensure other defaults exist
        if 'ollama_base_url' not in config:
            config['ollama_base_url'] = "http://localhost:11434"
        
        if 'cuda_settings' not in config:
            config['cuda_settings'] = {
                'device': "cuda:0",
                'dtype': "float16",
                'enable_tf32': True,
                'memory_fraction': 0.95
            }
        
        if 'generation_defaults' not in config:
            config['generation_defaults'] = {
                'steps': 30,
                'guidance_scale': 7.5,
                'width': 1024,
                'height': 1024,
                'batch_size': 1
            }
        
        # Save updated config
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"✓ Configuration updated: {config_path}")
        return True
        
    except Exception as e:
        print(f"✗ Failed to update config: {e}")
        return False

def main():
    """Main setup function."""
    print("=== Illustrious AI Studio Model Setup ===\n")
    
    # Check prerequisites
    if not check_ollama_installed():
        return 1
    
    if not check_ollama_running():
        return 1
    
    success = True
    
    # Download SDXL model
    print("\n1. Setting up SDXL model...")
    if not download_sdxl_model():
        success = False
    
    # Pull Ollama models
    print("\n2. Setting up Ollama models...")
    if not pull_ollama_model(OLLAMA_MODELS["llm"], "LLM"):
        success = False
    
    if not pull_ollama_model(OLLAMA_MODELS["vision"], "Vision"):
        success = False
    
    # Update configuration
    print("\n3. Updating configuration...")
    if not update_config(CONFIG_FILE):
        success = False
    
    # Final status
    print("\n" + "="*50)
    if success:
        print("✓ Setup completed successfully!")
        print("\nNext steps:")
        print("1. Run: python verify_setup.py")
        print("2. Run: python main.py")
        print("3. Open: http://localhost:7860")
    else:
        print("✗ Setup completed with errors. Please check the messages above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
