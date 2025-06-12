#!/usr/bin/env python3
"""
Unified Setup Script for Illustrious AI Studio
Handles environment creation, dependency installation, model downloads, and verification in one go.
"""

import os
import sys
import subprocess
import json
import urllib.request
import platform
from pathlib import Path
import argparse
import time
import shutil
import yaml


class IllustriousSetup:
    def __init__(self, args):
        self.args = args
        self.venv_dir = Path("venv")
        self.models_dir = Path("models")
        self.config_file = Path("config.yaml")
        self.errors = []
        self.warnings = []

        # GPU backend URLs
        self.cuda_index = "https://download.pytorch.org/whl/cu121"
        self.rocm_index = "https://download.pytorch.org/whl/rocm5.7"

        # Recommended models
        self.sdxl_model = {
            "name": "Illustrious-XL",
            "url": "https://huggingface.co/OnomaAI/Illustrious-xl/resolve/main/Illustrious-xl-v0.1.safetensors",
            "filename": "Illustrious.safetensors",
            "size": "6.6GB"
        }

        self.ollama_models = {
            "llm": "goekdenizguelmez/JOSIEFIED-Qwen3:8b-q6_k",
            "vision": "qwen2.5vl:7b"
        }

    def print_header(self, text):
        print(f"\n{'='*60}")
        print(f"{text:^60}")
        print(f"{'='*60}\n")

    def print_status(self, status, message):
        symbols = {"success": "\u2713", "error": "\u2717", "warning": "\u26a0", "info": "\u2139"}
        colors = {
            "success": "\033[92m",
            "error": "\033[91m",
            "warning": "\033[93m",
            "info": "\033[94m",
        }
        reset = "\033[0m"
        print(f"{colors.get(status, '')}{symbols.get(status, '')} {message}{reset}")

    def run_command(self, cmd, **kwargs):
        """Run a command and return (success, output)."""
        if self.args.verbose:
            print(f"$ {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
            if result.returncode != 0:
                if self.args.verbose:
                    print(f"Error: {result.stderr}")
                return False, result.stderr
            return True, result.stdout
        except Exception as e:
            return False, str(e)

    def check_python_version(self):
        version = sys.version_info
        if version.major == 3 and version.minor >= 10:
            self.print_status("success", f"Python {version.major}.{version.minor}.{version.micro}")
            return True
        else:
            self.print_status("error", f"Python {version.major}.{version.minor} (requires 3.10+)")
            self.errors.append("Python version too old")
            return False

    def detect_gpu(self):
        success, _ = self.run_command(["nvidia-smi"])
        if success:
            self.print_status("success", "NVIDIA GPU detected")
            return "cuda"

        success, _ = self.run_command(["rocm-smi"])
        if success:
            self.print_status("success", "AMD GPU detected")
            return "rocm"

        if not self.args.cpu_only:
            self.print_status("warning", "No GPU detected, will use CPU (slower)")
            response = input("Continue with CPU-only installation? [y/N]: ").lower()
            if response != "y":
                return None
        return "cpu"

    def create_venv(self):
        if self.venv_dir.exists() and not self.args.force:
            self.print_status("info", "Virtual environment already exists")
            return True
        if self.venv_dir.exists():
            shutil.rmtree(self.venv_dir)
        self.print_status("info", "Creating virtual environment...")
        success, _ = self.run_command([sys.executable, "-m", "venv", str(self.venv_dir)])
        if success:
            self.print_status("success", "Virtual environment created")
        return success

    def get_venv_python(self):
        if platform.system() == "Windows":
            return str(self.venv_dir / "Scripts" / "python.exe")
        return str(self.venv_dir / "bin" / "python")

    def install_pytorch(self, gpu_backend):
        venv_python = self.get_venv_python()
        self.print_status("info", "Upgrading pip...")
        self.run_command([venv_python, "-m", "pip", "install", "--upgrade", "pip"])

        self.print_status("info", f"Installing PyTorch for {gpu_backend}...")
        if gpu_backend == "cuda":
            cmd = [
                venv_python,
                "-m",
                "pip",
                "install",
                "torch",
                "torchvision",
                "torchaudio",
                "--index-url",
                self.cuda_index,
            ]
        elif gpu_backend == "rocm":
            cmd = [
                venv_python,
                "-m",
                "pip",
                "install",
                "torch",
                "torchvision",
                "torchaudio",
                "--index-url",
                self.rocm_index,
            ]
        else:
            cmd = [venv_python, "-m", "pip", "install", "torch", "torchvision", "torchaudio"]
        success, error = self.run_command(cmd)
        if success:
            self.print_status("success", f"PyTorch installed for {gpu_backend}")
        else:
            self.print_status("error", f"Failed to install PyTorch: {error}")
            self.errors.append("PyTorch installation failed")
        return success

    def install_requirements(self):
        venv_python = self.get_venv_python()
        req_file = Path("requirements.txt")
        if not req_file.exists():
            self.print_status("error", "requirements.txt not found")
            return False
        with open(req_file) as f:
            requirements = [
                line.strip()
                for line in f
                if line.strip() and not line.strip().startswith(("#", "torch"))
            ]
        if not requirements:
            return True
        self.print_status("info", "Installing requirements...")
        success, error = self.run_command([venv_python, "-m", "pip", "install"] + requirements)
        if success:
            self.print_status("success", "Requirements installed")
        else:
            self.print_status("error", f"Failed to install requirements: {error}")
            self.errors.append("Requirements installation failed")
        return success

    def check_ollama(self):
        success, _ = self.run_command(["ollama", "--version"])
        if not success:
            self.print_status("error", "Ollama not installed. Visit: https://ollama.ai/download")
            self.errors.append("Ollama not installed")
            return False
        success, _ = self.run_command(["ollama", "list"])
        if not success:
            self.print_status("warning", "Ollama service not running. Run: ollama serve")
            self.warnings.append("Ollama service not running")
            return False
        self.print_status("success", "Ollama is installed and running")
        return True

    def download_sdxl_model(self):
        self.models_dir.mkdir(exist_ok=True)
        model_path = self.models_dir / self.sdxl_model["filename"]
        if model_path.exists() and not self.args.force:
            self.print_status("info", f"SDXL model already exists: {model_path}")
            return True
        if self.args.skip_models:
            self.print_status("info", "Skipping model download (--skip-models)")
            return True
        self.print_status(
            "info",
            f"Downloading {self.sdxl_model['name']} ({self.sdxl_model['size']})...",
        )

        def progress(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100.0, block_num * block_size / total_size * 100)
                sys.stdout.write(f"\rProgress: {percent:.1f}%")
                sys.stdout.flush()

        try:
            urllib.request.urlretrieve(self.sdxl_model["url"], model_path, progress)
            print()
            self.print_status("success", "SDXL model downloaded")
            return True
        except Exception as e:
            self.print_status("error", f"Download failed: {e}")
            self.errors.append("SDXL model download failed")
            return False

    def pull_ollama_models(self):
        if self.args.skip_models:
            self.print_status("info", "Skipping Ollama models (--skip-models)")
            return True
        success_all = True
        for model_type, model_name in self.ollama_models.items():
            self.print_status("info", f"Pulling {model_type} model: {model_name}")
            success, error = self.run_command(["ollama", "pull", model_name])
            if success:
                self.print_status("success", f"{model_type} model ready")
            else:
                self.print_status("error", f"Failed to pull {model_type} model: {error}")
                self.warnings.append(f"Failed to pull {model_type} model")
                success_all = False
        return success_all

    def update_config(self, gpu_backend):
        config = {}
        if self.config_file.exists():
            with open(self.config_file) as f:
                config = yaml.safe_load(f) or {}
        config.update(
            {
                "sd_model": str(self.models_dir / self.sdxl_model["filename"]),
                "ollama_model": self.ollama_models["llm"],
                "ollama_vision_model": self.ollama_models["vision"],
                "ollama_base_url": "http://localhost:11434",
                "gpu_backend": gpu_backend,
                "load_models_on_startup": True,
            }
        )
        if "cuda_settings" not in config:
            config["cuda_settings"] = {
                "device": "cuda:0" if gpu_backend == "cuda" else "cpu",
                "dtype": "float16",
                "enable_tf32": True,
                "memory_fraction": 0.95,
            }
        if "generation_defaults" not in config:
            config["generation_defaults"] = {
                "steps": 30,
                "guidance_scale": 7.5,
                "width": 1024,
                "height": 1024,
                "batch_size": 1,
            }
        with open(self.config_file, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        self.print_status("success", "Configuration updated")
        return True

    def verify_installation(self):
        if self.args.skip_verify:
            return True
        venv_python = self.get_venv_python()
        self.print_status("info", "Running verification...")
        test_script = """
import torch
import diffusers
import transformers
import gradio
import fastapi
print('All modules imported successfully')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"""
        success, output = self.run_command([venv_python, "-c", test_script])
        if success:
            self.print_status("success", "Basic verification passed")
            if self.args.verbose:
                print(output)
        else:
            self.print_status("error", "Verification failed")
            self.errors.append("Import verification failed")
        return success

    def run(self):
        self.print_header("Illustrious AI Studio - Unified Setup")
        if not self.check_python_version():
            return False
        if not self.args.gpu_backend:
            gpu_backend = self.detect_gpu()
            if gpu_backend is None:
                return False
        else:
            gpu_backend = self.args.gpu_backend
            self.print_status("info", f"Using specified GPU backend: {gpu_backend}")
        if not self.create_venv():
            return False
        if not self.install_pytorch(gpu_backend):
            return False
        if not self.install_requirements():
            return False
        ollama_ok = self.check_ollama()
        if not self.args.skip_models:
            self.download_sdxl_model()
            if ollama_ok:
                self.pull_ollama_models()
        self.update_config(gpu_backend)
        self.verify_installation()
        self.print_header("Setup Summary")
        if not self.errors:
            self.print_status("success", "Setup completed successfully!")
            print("\nNext steps:")
            print("1. Activate virtual environment:")
            if platform.system() == "Windows":
                print("   .\\venv\\Scripts\\activate")
            else:
                print("   source venv/bin/activate")
            print("2. Start the application:")
            print("   python main.py")
            print("3. Open http://localhost:7860")
            if self.warnings:
                print(f"\nWarnings ({len(self.warnings)}):")
                for warning in self.warnings:
                    print(f"  \u26a0 {warning}")
        else:
            self.print_status("error", "Setup failed with errors:")
            for error in self.errors:
                print(f"  \u2717 {error}")
            return False
        return True


def main():
    parser = argparse.ArgumentParser(description="Unified setup for Illustrious AI Studio")
    parser.add_argument(
        "--gpu-backend",
        choices=["cuda", "rocm", "cpu"],
        help="GPU backend to use (auto-detects if not specified)",
    )
    parser.add_argument("--cpu-only", action="store_true", help="Force CPU-only installation")
    parser.add_argument("--skip-models", action="store_true", help="Skip downloading models")
    parser.add_argument("--skip-verify", action="store_true", help="Skip verification step")
    parser.add_argument("--force", action="store_true", help="Force reinstall even if already setup")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")
    args = parser.parse_args()
    if args.cpu_only:
        args.gpu_backend = "cpu"
    setup = IllustriousSetup(args)
    success = setup.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
