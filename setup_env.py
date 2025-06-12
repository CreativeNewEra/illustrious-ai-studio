#!/usr/bin/env python3
"""Environment setup script for Illustrious AI Studio.

This script creates a Python virtual environment, installs the
appropriate PyTorch build depending on the user's GPU, installs the
remaining requirements, and then runs ``setup_models.py`` to download
models and update ``config.yaml``. Optionally it can run
``verify_setup.py`` at the end.
"""

import os
import subprocess
import sys
from pathlib import Path

VENV_DIR = Path("venv")
REQ_FILE = Path("requirements.txt")

CUDA_INDEX_URL = "https://download.pytorch.org/whl/cu121"
ROCM_INDEX_URL = "https://download.pytorch.org/whl/rocm5.7"


def run(cmd, **kwargs):
    """Run a command and stream output."""
    print(f"$ {' '.join(cmd)}")
    return subprocess.call(cmd, **kwargs)


def create_venv():
    if VENV_DIR.exists():
        print(f"Virtual environment already exists: {VENV_DIR}")
        return True
    print(f"Creating virtual environment at {VENV_DIR}...")
    return run([sys.executable, "-m", "venv", str(VENV_DIR)]) == 0


def get_gpu_choice():
    print("Choose your GPU type:")
    print("1) NVIDIA (CUDA)")
    print("2) AMD (ROCm)")
    print("3) CPU only")
    choice = input("Select [1-3]: ").strip()
    mapping = {"1": "cuda", "2": "rocm", "3": "cpu"}
    return mapping.get(choice, "cpu")


def install_pytorch(venv_python, backend):
    if backend == "cuda":
        index_url = CUDA_INDEX_URL
    elif backend == "rocm":
        index_url = ROCM_INDEX_URL
    else:
        index_url = None

    if index_url:
        cmd = [venv_python, "-m", "pip", "install", "--upgrade", "pip"]
        run(cmd)
        cmd = [
            venv_python,
            "-m",
            "pip",
            "install",
            "torch",
            "torchvision",
            "torchaudio",
            "--index-url",
            index_url,
        ]
    else:
        cmd = [
            venv_python,
            "-m",
            "pip",
            "install",
            "torch",
            "torchvision",
            "torchaudio",
        ]
    return run(cmd) == 0


def install_requirements(venv_python):
    if not REQ_FILE.exists():
        print(f"Requirements file not found: {REQ_FILE}")
        return True

    # Filter out torch packages because we already installed them
    pkgs = []
    with open(REQ_FILE) as f:
        for line in f:
            stripped = line.strip()
            if stripped.lower().startswith("torch"):
                continue
            if stripped:
                pkgs.append(stripped)

    if not pkgs:
        return True

    cmd = [venv_python, "-m", "pip", "install"] + pkgs
    return run(cmd) == 0


def main():
    if not create_venv():
        return 1

    venv_python = str(VENV_DIR / "bin" / "python")
    if os.name == "nt":
        venv_python = str(VENV_DIR / "Scripts" / "python.exe")

    backend = get_gpu_choice()
    print(f"Installing PyTorch for {backend}...")
    if not install_pytorch(venv_python, backend):
        return 1

    print("Installing other requirements...")
    if not install_requirements(venv_python):
        return 1

    print("Running setup_models.py...")
    if run([venv_python, "setup_models.py"]) != 0:
        return 1

    verify = input("Run verify_setup.py now? [y/N]: ").strip().lower()
    if verify == "y":
        run([venv_python, "verify_setup.py"])

    print("\nEnvironment setup complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

