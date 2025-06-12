#!/usr/bin/env python3
"""
Illustrious AI Studio - One-Click Conda Setup
Automatically creates conda environment and installs everything needed.
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
import traceback


class IllustriousSetup:
    def __init__(self, args):
        self.args = args
        self.conda_env_name = "illustrious"
        self.python_version = "3.12"
        self.models_dir = Path("models")
        self.config_file = Path("config.yaml")
        self.errors = []
        self.warnings = []
        self.conda_exe = None
        self.env_python = None
        self.setup_report = {}

        # GPU backend URLs for pip fallback
        self.cuda_index = "https://download.pytorch.org/whl/cu121"
        self.rocm_index = "https://download.pytorch.org/whl/rocm5.7"

        # Recommended models
        self.sdxl_model = {
            "name": "Illustrious-XL",
            "url": "https://huggingface.co/OnomaAIdev/Illustrious-v0.1/resolve/main/Illustrious-v0.1.safetensors",
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
        symbols = {"success": "✓", "error": "✗", "warning": "⚠", "info": "ℹ"}
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

    def find_conda(self):
        """Find conda/mamba executable."""
        # Check for mamba first (faster)
        for exe in ["mamba", "conda", "micromamba"]:
            success, output = self.run_command(["which", exe])
            if success and output.strip():
                self.conda_exe = output.strip()
                self.print_status("success", f"Found {exe} at: {self.conda_exe}")
                return True
        
        # Check common locations
        common_paths = [
            Path.home() / "miniconda3" / "bin" / "conda",
            Path.home() / "anaconda3" / "bin" / "conda",
            Path.home() / "mambaforge" / "bin" / "mamba",
            Path("/opt/conda/bin/conda"),
            Path("/usr/local/conda/bin/conda"),
        ]
        
        for path in common_paths:
            if path.exists():
                self.conda_exe = str(path)
                self.print_status("success", f"Found conda at: {self.conda_exe}")
                return True
        
        self.print_status("error", "Conda not found!")
        print("\nPlease install Miniconda or Anaconda:")
        print("  https://docs.conda.io/en/latest/miniconda.html")
        self.errors.append("Conda not installed")
        return False

    def check_conda_env(self):
        """Check if conda environment already exists."""
        success, output = self.run_command([self.conda_exe, "env", "list"])
        if success and self.conda_env_name in output:
            return True
        return False

    def create_conda_env(self):
        """Create conda environment with Python 3.12."""
        if self.check_conda_env() and not self.args.force:
            self.print_status("info", f"Conda environment '{self.conda_env_name}' already exists")
            return True
        
        if self.check_conda_env() and self.args.force:
            self.print_status("info", f"Removing existing environment '{self.conda_env_name}'...")
            self.run_command([self.conda_exe, "env", "remove", "-n", self.conda_env_name, "-y"])
        
        self.print_status("info", f"Creating conda environment '{self.conda_env_name}' with Python {self.python_version}...")
        
        # Create environment with essential packages
        cmd = [
            self.conda_exe, "create", "-n", self.conda_env_name,
            f"python={self.python_version}", "pip", "setuptools", "wheel",
            "-y", "-c", "conda-forge"
        ]
        
        success, error = self.run_command(cmd)
        if success:
            self.print_status("success", f"Conda environment '{self.conda_env_name}' created")
            return True
        else:
            self.print_status("error", f"Failed to create conda environment: {error}")
            self.errors.append("Conda environment creation failed")
            return False

    def get_env_python(self):
        """Get python executable path for conda environment."""
        if platform.system() == "Windows":
            # Try multiple possible locations on Windows
            conda_info, _ = subprocess.run([self.conda_exe, "info", "--json"], 
                                          capture_output=True, text=True).stdout, None
            if conda_info:
                try:
                    info = json.loads(conda_info)
                    for env in info.get("envs", []):
                        if self.conda_env_name in env:
                            return str(Path(env) / "python.exe")
                except:
                    pass
            # Fallback
            base_path = Path(self.conda_exe).parent.parent
            return str(base_path / "envs" / self.conda_env_name / "python.exe")
        else:
            # Unix-like systems
            success, output = self.run_command([self.conda_exe, "run", "-n", self.conda_env_name, "which", "python"])
            if success and output.strip():
                return output.strip()
            
            # Fallback
            base_path = Path(self.conda_exe).parent.parent
            return str(base_path / "envs" / self.conda_env_name / "bin" / "python")

    def detect_gpu(self):
        """Detect available GPU."""
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

    def install_pytorch(self, gpu_backend):
        """Install PyTorch using conda."""
        self.print_status("info", f"Installing PyTorch for {gpu_backend}...")
        
        if gpu_backend == "cuda":
            # Use conda for CUDA installation (more reliable)
            cmd = [
                self.conda_exe, "install", "-n", self.conda_env_name,
                "pytorch", "torchvision", "torchaudio", "pytorch-cuda=12.1",
                "-c", "pytorch", "-c", "nvidia", "-y"
            ]
        elif gpu_backend == "rocm":
            # ROCm via pip (conda support limited)
            self.env_python = self.get_env_python()
            cmd = [
                self.conda_exe, "run", "-n", self.conda_env_name,
                "pip", "install", "torch", "torchvision", "torchaudio",
                "--index-url", self.rocm_index
            ]
        else:
            # CPU version
            cmd = [
                self.conda_exe, "install", "-n", self.conda_env_name,
                "pytorch", "torchvision", "torchaudio", "cpuonly",
                "-c", "pytorch", "-y"
            ]
        
        success, error = self.run_command(cmd)
        if success:
            self.print_status("success", f"PyTorch installed for {gpu_backend}")
        else:
            self.print_status("error", f"Failed to install PyTorch: {error}")
            self.errors.append("PyTorch installation failed")
        return success

    def install_requirements(self):
        """Install remaining requirements with pip."""
        req_file = Path("requirements.txt")
        if not req_file.exists():
            self.print_status("error", "requirements.txt not found")
            return False
        
        # Read requirements and filter out PyTorch packages
        with open(req_file) as f:
            requirements = [
                line.strip()
                for line in f
                if line.strip() and not line.strip().startswith(("#", "torch"))
            ]
        
        if not requirements:
            return True
        
        self.print_status("info", "Installing additional requirements...")
        
        # Install using conda run with pip
        cmd = [
            self.conda_exe, "run", "-n", self.conda_env_name,
            "pip", "install"
        ] + requirements
        
        success, error = self.run_command(cmd)
        if success:
            self.print_status("success", "Requirements installed")
        else:
            self.print_status("error", f"Failed to install requirements: {error}")
            self.errors.append("Requirements installation failed")
        return success

    def check_ollama(self):
        """Check if Ollama is installed and running."""
        success, _ = self.run_command(["ollama", "--version"])
        if not success:
            self.print_status("warning", "Ollama not installed. Visit: https://ollama.ai/download")
            self.warnings.append("Ollama not installed (optional for LLM features)")
            return False
        
        success, _ = self.run_command(["ollama", "list"])
        if not success:
            self.print_status("warning", "Ollama service not running. Run: ollama serve")
            self.warnings.append("Ollama service not running")
            return False
        
        self.print_status("success", "Ollama is installed and running")
        return True

    def download_sdxl_model(self):
        """Download the SDXL model."""
        self.models_dir.mkdir(exist_ok=True)
        model_path = self.models_dir / self.sdxl_model["filename"]
        
        if model_path.exists() and not self.args.force:
            self.print_status("info", f"SDXL model already exists: {model_path}")
            return True
        
        if self.args.skip_models:
            self.print_status("info", "Skipping model download (--skip-models)")
            return True
        
        self.print_status("info", f"Downloading {self.sdxl_model['name']} ({self.sdxl_model['size']})...")
        self.print_status("info", "This may take a while depending on your internet connection...")

        def progress(block_num, block_size, total_size):
            if total_size > 0:
                downloaded = block_num * block_size
                percent = min(100.0, downloaded / total_size * 100)
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                sys.stdout.write(f"\rProgress: {percent:.1f}% ({mb_downloaded:.1f}MB / {mb_total:.1f}MB)")
                sys.stdout.flush()

        try:
            urllib.request.urlretrieve(self.sdxl_model["url"], model_path, progress)
            print()  # New line after progress
            self.print_status("success", "SDXL model downloaded")
            return True
        except Exception as e:
            self.print_status("error", f"Download failed: {e}")
            self.errors.append("SDXL model download failed")
            return False

    def pull_ollama_models(self):
        """Pull Ollama models."""
        if self.args.skip_models:
            self.print_status("info", "Skipping Ollama models (--skip-models)")
            return True
        
        if not self.check_ollama():
            return False
        
        success_all = True
        for model_type, model_name in self.ollama_models.items():
            self.print_status("info", f"Pulling {model_type} model: {model_name}")
            success, error = self.run_command(["ollama", "pull", model_name])
            if success:
                self.print_status("success", f"{model_type} model ready")
            else:
                self.print_status("warning", f"Failed to pull {model_type} model: {error}")
                self.warnings.append(f"Failed to pull {model_type} model")
                success_all = False
        return success_all

    def update_config(self, gpu_backend):
        """Update configuration file."""
        config = {}
        if self.config_file.exists():
            with open(self.config_file) as f:
                config = yaml.safe_load(f) or {}
        
        config.update({
            "sd_model": str(self.models_dir / self.sdxl_model["filename"]),
            "ollama_model": self.ollama_models["llm"],
            "ollama_vision_model": self.ollama_models["vision"],
            "ollama_base_url": "http://localhost:11434",
            "gpu_backend": gpu_backend,
            "load_models_on_startup": True,
            "conda_env": self.conda_env_name,
        })
        
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

    def create_launch_scripts(self):
        """Create convenient launch scripts."""
        # Create run.sh for Unix-like systems
        if platform.system() != "Windows":
            run_script = Path("run.sh")
            with open(run_script, "w") as f:
                f.write(f"""#!/bin/bash
# Illustrious AI Studio Launch Script

echo "Starting Illustrious AI Studio..."

# Activate conda environment and run
{self.conda_exe} run -n {self.conda_env_name} python main.py "$@"
""")
            run_script.chmod(0o755)
            self.print_status("success", "Created run.sh")
        
        # Create run.bat for Windows
        if platform.system() == "Windows":
            run_script = Path("run.bat")
            with open(run_script, "w") as f:
                f.write(f"""@echo off
REM Illustrious AI Studio Launch Script

echo Starting Illustrious AI Studio...

REM Activate conda environment and run
call {self.conda_exe} run -n {self.conda_env_name} python main.py %*
""")
            self.print_status("success", "Created run.bat")
        
        # Create update script
        update_script = Path("update.sh" if platform.system() != "Windows" else "update.bat")
        if platform.system() != "Windows":
            with open(update_script, "w") as f:
                f.write(f"""#!/bin/bash
# Update Illustrious AI Studio

echo "Updating Illustrious AI Studio..."

# Pull latest changes
git pull

# Run setup with force flag to update dependencies
python setup.py --force --skip-models

echo "Update complete!"
""")
            update_script.chmod(0o755)
        else:
            with open(update_script, "w") as f:
                f.write(f"""@echo off
REM Update Illustrious AI Studio

echo Updating Illustrious AI Studio...

REM Pull latest changes
git pull

REM Run setup with force flag to update dependencies
python setup.py --force --skip-models

echo Update complete!
pause
""")
        
        self.print_status("success", f"Created {update_script.name}")

    def verify_installation(self):
        """Verify the installation works."""
        if self.args.skip_verify:
            return True
        
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
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
"""
        
        cmd = [self.conda_exe, "run", "-n", self.conda_env_name, "python", "-c", test_script]
        success, output = self.run_command(cmd)
        
        if success:
            self.print_status("success", "Installation verified successfully")
            if self.args.verbose:
                print(output)
        else:
            self.print_status("error", "Verification failed")
            self.errors.append("Import verification failed")
        
        return success

    def save_setup_report(self):
        """Save detailed setup report."""
        self.setup_report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "conda_env": self.conda_env_name,
            "conda_exe": self.conda_exe,
            "platform": platform.platform(),
            "errors": self.errors,
            "warnings": self.warnings,
            "args": vars(self.args),
        }
        
        with open("setup_report.json", "w") as f:
            json.dump(self.setup_report, f, indent=2)
        
        self.print_status("info", "Setup report saved to setup_report.json")

    def run(self):
        """Main setup process."""
        self.print_header("Illustrious AI Studio - One-Click Setup")
        
        try:
            # Find conda
            if not self.find_conda():
                return False
            
            # Detect GPU
            if not self.args.gpu_backend:
                gpu_backend = self.detect_gpu()
                if gpu_backend is None:
                    return False
            else:
                gpu_backend = self.args.gpu_backend
                self.print_status("info", f"Using specified GPU backend: {gpu_backend}")
            
            # Create conda environment
            if not self.create_conda_env():
                return False
            
            # Install PyTorch
            if not self.install_pytorch(gpu_backend):
                return False
            
            # Install other requirements
            if not self.install_requirements():
                return False
            
            # Check Ollama (optional)
            ollama_ok = self.check_ollama()
            
            # Download models
            if not self.args.skip_models:
                self.download_sdxl_model()
                if ollama_ok:
                    self.pull_ollama_models()
            
            # Update configuration
            self.update_config(gpu_backend)
            
            # Create launch scripts
            self.create_launch_scripts()
            
            # Verify installation
            self.verify_installation()
            
            # Save setup report
            self.save_setup_report()
            
            # Print summary
            self.print_header("Setup Complete!")
            
            if not self.errors:
                self.print_status("success", "Illustrious AI Studio is ready to use!")
                print("\nTo start the application:")
                if platform.system() == "Windows":
                    print("  ./run.bat")
                else:
                    print("  ./run.sh")
                print("\nOr manually:")
                print(f"  conda activate {self.conda_env_name}")
                print("  python main.py")
                print("\nThe web interface will open at: http://localhost:7860")
                
                if self.warnings:
                    print(f"\nWarnings ({len(self.warnings)}):")
                    for warning in self.warnings:
                        print(f"  ⚠ {warning}")
            else:
                self.print_status("error", "Setup completed with errors:")
                for error in self.errors:
                    print(f"  ✗ {error}")
                return False
            
            return True
            
        except Exception as e:
            self.print_status("error", f"Unexpected error: {str(e)}")
            if self.args.verbose:
                traceback.print_exc()
            self.errors.append(f"Unexpected error: {str(e)}")
            self.save_setup_report()
            return False


def main():
    parser = argparse.ArgumentParser(
        description="One-click setup for Illustrious AI Studio with conda"
    )
    parser.add_argument(
        "--gpu-backend",
        choices=["cuda", "rocm", "cpu"],
        help="GPU backend to use (auto-detects if not specified)",
    )
    parser.add_argument(
        "--cpu-only", 
        action="store_true", 
        help="Force CPU-only installation"
    )
    parser.add_argument(
        "--skip-models", 
        action="store_true", 
        help="Skip downloading models"
    )
    parser.add_argument(
        "--skip-verify", 
        action="store_true", 
        help="Skip verification step"
    )
    parser.add_argument(
        "--force", 
        action="store_true", 
        help="Force reinstall even if already setup"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Show detailed output"
    )
    
    args = parser.parse_args()
    
    # Handle CPU-only flag
    if args.cpu_only:
        args.gpu_backend = "cpu"
    
    # Run setup
    setup = IllustriousSetup(args)
    success = setup.run()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
