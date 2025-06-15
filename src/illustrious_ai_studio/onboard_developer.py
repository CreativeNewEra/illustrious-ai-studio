#!/usr/bin/env python3
"""
Illustrious AI Studio - Developer Onboarding Script

This script helps new developers get set up quickly for co-development.
It performs initial setup tasks and validates the development environment.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
from typing import List, Tuple

def print_header():
    """Print welcome header"""
    print("=" * 60)
    print("🚀 Illustrious AI Studio - Developer Onboarding")
    print("=" * 60)
    print("Welcome to the development environment setup!")
    print("This script will help you get ready for co-development.\n")

def check_python_version() -> bool:
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is not compatible")
        print("   Please install Python 3.10 or higher")
        return False

def check_git() -> bool:
    """Check if git is available"""
    print("📝 Checking Git installation...")
    try:
        result = subprocess.run(['git', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    
    print("❌ Git is not installed or not in PATH")
    print("   Please install Git: https://git-scm.com/downloads")
    return False

def check_cuda() -> Tuple[bool, str]:
    """Check CUDA availability"""
    print("🎮 Checking CUDA availability...")
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"✅ CUDA available: {device_count} device(s)")
            print(f"   Primary GPU: {device_name} ({memory_gb:.1f}GB)")
            return True, device_name
        else:
            print("⚠️  CUDA not available - will use CPU (slower)")
            return False, "CPU"
    except ImportError:
        print("⚠️  PyTorch not installed yet - CUDA check skipped")
        return False, "Unknown"

def check_conda() -> bool:
    """Check if conda is available"""
    print("🐍 Checking Conda installation...")
    try:
        result = subprocess.run(['conda', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    
    print("⚠️  Conda not found - you can still use pip/venv")
    return False

def check_ollama() -> bool:
    """Check if Ollama is running"""
    print("🤖 Checking Ollama service...")
    try:
        import requests
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"✅ Ollama running with {len(models)} model(s)")
            return True
    except Exception:
        pass
    
    print("⚠️  Ollama not running - install from https://ollama.ai")
    print("   You can still use the image generation features")
    return False

def check_project_structure() -> bool:
    """Check if we're in the right directory"""
    print("📁 Checking project structure...")
    required_files = ['main.py', 'requirements.txt', 'core/', 'ui/', 'server/']
    missing_files = []
    
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if not missing_files:
        print("✅ Project structure looks good")
        return True
    else:
        print(f"❌ Missing files/directories: {', '.join(missing_files)}")
        print("   Make sure you're in the illustrious-ai-studio directory")
        return False

def suggest_setup_commands():
    """Suggest setup commands based on the environment"""
    print("\n🛠️  Suggested Setup Commands:")
    print("-" * 40)
    
    if check_conda():
        print("# Using Conda (Recommended):")
        print("conda create -n ai-studio python=3.10")
        print("conda activate ai-studio")
        print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print("pip install -r requirements.txt")
        print("pip install -r requirements-test.txt")
    else:
        print("# Using pip/venv:")
        print("python -m venv venv")
        print("source venv/bin/activate  # Linux/Mac")
        print("# venv\\Scripts\\activate  # Windows")
        print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print("pip install -r requirements.txt")
        print("pip install -r requirements-test.txt")
    
    print("\n# Quick test:")
    print("python verify_setup.py")
    print("python main.py --quick-start --open-browser")

def print_development_resources():
    """Print development resources"""
    print("\n📚 Development Resources:")
    print("-" * 40)
    print("• Comprehensive Development Guide: COMPREHENSIVE_DEVELOPMENT_GUIDE.md")
    print("• Quick Start Guide: QUICK_START_GUIDE.md")
    print("• Troubleshooting: TROUBLESHOOTING_GUIDE.md")
    print("• API Documentation: examples/API_DOCUMENTATION.md")
    print("• VS Code Tasks: Use Ctrl+Shift+P → 'Tasks: Run Task'")

def print_next_steps():
    """Print next steps for the developer"""
    print("\n🎯 Next Steps:")
    print("-" * 40)
    print("1. Set up your environment using the commands above")
    print("2. Read the Comprehensive Development Guide")
    print("3. Run the verification script: python verify_setup.py")
    print("4. Start the development server: python main.py --quick-start")
    print("5. Explore the codebase and run some tests")
    print("6. Check out the issues on GitHub for contribution opportunities")

def main():
    """Main onboarding function"""
    print_header()
    
    # Check prerequisites
    checks = []
    checks.append(("Python Version", check_python_version()))
    checks.append(("Git", check_git()))
    checks.append(("Project Structure", check_project_structure()))
    checks.append(("CUDA", check_cuda()[0]))
    checks.append(("Conda", check_conda()))
    checks.append(("Ollama", check_ollama()))
    
    print("\n📊 Environment Summary:")
    print("-" * 40)
    for name, status in checks:
        icon = "✅" if status else "❌"
        print(f"{icon} {name}")
    
    # Provide guidance
    suggest_setup_commands()
    print_development_resources()
    print_next_steps()
    
    print("\n" + "=" * 60)
    print("🎉 Ready to start co-developing! Happy coding!")
    print("=" * 60)

if __name__ == "__main__":
    main()
