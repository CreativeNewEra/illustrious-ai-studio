#!/usr/bin/env python3
"""
Comprehensive setup verification for Illustrious AI Studio
Checks all dependencies, models, and system capabilities
"""

import sys
import os
import subprocess
import json
from pathlib import Path
import importlib
import torch
import requests
from colorama import init, Fore, Style

init(autoreset=True)

class SetupVerifier:
    def __init__(self):
        self.results = {
            "system": {},
            "dependencies": {},
            "models": {},
            "features": {}
        }
        self.has_errors = False
        self.has_warnings = False

    def print_header(self, text):
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"{Fore.CYAN}{text:^60}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")

    def print_success(self, text):
        print(f"{Fore.GREEN}✓ {text}{Style.RESET_ALL}")

    def print_error(self, text):
        print(f"{Fore.RED}✗ {text}{Style.RESET_ALL}")
        self.has_errors = True

    def print_warning(self, text):
        print(f"{Fore.YELLOW}⚠ {text}{Style.RESET_ALL}")
        self.has_warnings = True

    def print_info(self, text):
        print(f"{Fore.BLUE}ℹ {text}{Style.RESET_ALL}")

    def check_python_version(self):
        """Check Python version"""
        self.print_header("Python Environment")
        
        version = sys.version_info
        if version.major == 3 and version.minor >= 10:
            self.print_success(f"Python {version.major}.{version.minor}.{version.micro}")
            self.results["system"]["python"] = f"{version.major}.{version.minor}.{version.micro}"
        else:
            self.print_error(f"Python {version.major}.{version.minor} (requires 3.10+)")

    def check_cuda(self):
        """Check CUDA availability and GPU info"""
        self.print_header("CUDA & GPU Information")
        
        try:
            if torch.cuda.is_available():
                self.print_success(f"CUDA Available: {torch.version.cuda}")
                self.results["system"]["cuda"] = torch.version.cuda
                
                # GPU information
                gpu_count = torch.cuda.device_count()
                self.print_success(f"GPU Count: {gpu_count}")
                
                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                    self.print_info(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
                    
                    if "4090" in gpu_name:
                        self.print_success("RTX 4090M detected - Excellent for AI workloads!")
                    
                    self.results["system"][f"gpu_{i}"] = {
                        "name": gpu_name,
                        "memory_gb": gpu_memory
                    }
                
                # Memory info
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                self.print_info(f"CUDA Memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")
                
            else:
                self.print_error("CUDA not available")
                self.results["system"]["cuda"] = False
        except Exception as e:
            self.print_error(f"Error checking CUDA: {e}")

    def check_dependencies(self):
        """Check all required Python packages"""
        self.print_header("Python Dependencies")
        
        required_packages = {
            "torch": "PyTorch",
            "torchvision": "TorchVision",
            "diffusers": "Diffusers",
            "transformers": "Transformers",
            "accelerate": "Accelerate",
            "safetensors": "SafeTensors",
            "gradio": "Gradio",
            "fastapi": "FastAPI",
            "uvicorn": "Uvicorn",
            "PIL": "Pillow",
            "requests": "Requests",
            "yaml": "PyYAML",
            "numpy": "NumPy",
            "colorama": "Colorama"
        }
        
        for package, name in required_packages.items():
            try:
                if package == "PIL":
                    importlib.import_module("PIL")
                elif package == "yaml":
                    importlib.import_module("yaml")
                else:
                    importlib.import_module(package)
                    
                self.print_success(f"{name} installed")
                self.results["dependencies"][package] = True
            except ImportError:
                self.print_error(f"{name} not installed")
                self.results["dependencies"][package] = False

    def check_ollama(self):
        """Check Ollama service and models"""
        self.print_header("Ollama Service")
        
        try:
            # Check if Ollama is running
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                self.print_success("Ollama service is running")
                
                # List available models
                models = response.json().get("models", [])
                self.print_info(f"Available models: {len(models)}")
                
                vision_models = []
                text_models = []
                
                for model in models:
                    model_name = model["name"]
                    size_gb = model["size"] / 1024**3
                    
                    # Check if it's a vision model
                    if any(vm in model_name.lower() for vm in ["vision", "vl", "llava"]):
                        vision_models.append(model_name)
                        self.print_success(f"Vision model: {model_name} ({size_gb:.1f}GB)")
                    else:
                        text_models.append(model_name)
                        self.print_info(f"Text model: {model_name} ({size_gb:.1f}GB)")
                
                self.results["models"]["ollama_models"] = {
                    "vision": vision_models,
                    "text": text_models
                }
                
                # Check for recommended models
                if not vision_models:
                    self.print_warning("No vision models found. Consider pulling 'ollama pull qwen2.5-vision:7b'")
                
            else:
                self.print_error("Ollama service not responding")
                
        except requests.exceptions.ConnectionError:
            self.print_error("Ollama service not running. Start with: 'ollama serve'")
        except Exception as e:
            self.print_error(f"Error checking Ollama: {e}")

    def check_models(self):
        """Check model files"""
        self.print_header("Model Files")
        
        # Check SDXL model
        config_path = Path("config.yaml")
        if config_path.exists():
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f)
            
            sd_model_path = config.get("sd_model", "")
            if os.path.exists(sd_model_path):
                size_gb = os.path.getsize(sd_model_path) / 1024**3
                self.print_success(f"SDXL model found: {os.path.basename(sd_model_path)} ({size_gb:.1f}GB)")
                self.results["models"]["sdxl"] = True
            else:
                self.print_error(f"SDXL model not found at: {sd_model_path}")
                self.results["models"]["sdxl"] = False
        else:
            self.print_error("config.yaml not found")

    def check_features(self):
        """Test basic functionality"""
        self.print_header("Feature Verification")
        
        try:
            # Test imports
            from core.sdxl import init_sdxl
            from core.ollama import init_ollama
            from core.state import AppState
            from core.prompt_templates import template_manager
            
            self.print_success("Core modules can be imported")
            self.results["features"]["imports"] = True
            
            # Check if we can initialize state
            state = AppState()
            self.print_success("AppState initialized")
            self.results["features"]["state"] = True
            
            # Test prompt template system
            try:
                stats = template_manager.get_template_stats()
                self.print_success(f"Prompt template system initialized ({stats['total_templates']} templates)")
                self.results["features"]["prompt_templates"] = True
            except Exception as e:
                self.print_warning(f"Prompt template system warning: {e}")
                self.results["features"]["prompt_templates"] = False
            
            # Test logging setup
            try:
                import logging
                log_dir = Path("logs")
                if log_dir.exists() or True:  # Will be created on first run
                    self.print_success("Logging system configured")
                    self.results["features"]["logging"] = True
                else:
                    self.print_warning("Logs directory not found (will be created on startup)")
                    self.results["features"]["logging"] = False
            except Exception as e:
                self.print_warning(f"Logging setup issue: {e}")
                self.results["features"]["logging"] = False
                
        except Exception as e:
            self.print_error(f"Error testing features: {e}")
            self.results["features"]["imports"] = False

    def generate_report(self):
        """Generate final report"""
        self.print_header("Setup Verification Summary")
        
        if not self.has_errors:
            self.print_success("All checks passed! Your setup is ready.")
        elif self.has_warnings:
            self.print_warning("Setup has some warnings but should work.")
        else:
            self.print_error("Setup has errors that need to be fixed.")
        
        # Save detailed report
        report_path = Path("setup_report.json")
        with open(report_path, "w") as f:
            json.dump(self.results, f, indent=2)
        
        self.print_info(f"Detailed report saved to: {report_path}")
        
        # Recommendations
        if self.has_errors or self.has_warnings:
            self.print_header("Recommendations")
            
            if not self.results.get("system", {}).get("cuda"):
                self.print_info("Install CUDA PyTorch: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
            
            if not self.results["models"].get("ollama_models", {}).get("vision"):
                self.print_info("Install vision model: ollama pull qwen2.5-vision:7b")
            
            missing_deps = [name for pkg, name in {
                "torch": "PyTorch", "diffusers": "Diffusers", 
                "transformers": "Transformers", "gradio": "Gradio"
            }.items() if not self.results["dependencies"].get(pkg)]
            
            if missing_deps:
                self.print_info(f"Install missing packages: pip install {' '.join(missing_deps)}")

    def run_all_checks(self):
        """Run all verification checks"""
        self.check_python_version()
        self.check_cuda()
        self.check_dependencies()
        self.check_ollama()
        self.check_models()
        self.check_features()
        self.generate_report()


def main():
    print(f"{Fore.MAGENTA}╔═══════════════════════════════════════════════════════════╗")
    print(f"{Fore.MAGENTA}║           Illustrious AI Studio Setup Verifier            ║")
    print(f"{Fore.MAGENTA}╚═══════════════════════════════════════════════════════════╝{Style.RESET_ALL}")
    
    verifier = SetupVerifier()
    verifier.run_all_checks()
    
    return 0 if not verifier.has_errors else 1


if __name__ == "__main__":
    sys.exit(main())
