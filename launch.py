#!/usr/bin/env python3
"""
Quick launcher script for Illustrious AI Studio with different startup modes.
This script provides easy access to various launch options.
"""

import subprocess
import sys
from pathlib import Path

def print_banner():
    print("=" * 60)
    print("🎨 Illustrious AI Studio Launcher")
    print("=" * 60)

def print_options():
    print("\nAvailable launch modes:")
    print("1. 🚀 Quick Start (fastest, no model loading)")
    print("2. ⏳ Lazy Load (start UI, load models on demand)")
    print("3. 🔄 Full Initialization (load all models at startup)")
    print("4. 🎯 Custom Options")
    print("5. ❓ Help (show all available options)")
    print("6. 🚪 Exit")

def run_command(cmd):
    """Run a command and show real-time output."""
    print(f"\n🚀 Running: python {' '.join(cmd)}")
    print("-" * 40)
    try:
        process = subprocess.Popen(
            [sys.executable] + cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Print output in real-time
        if process.stdout:
            for line in process.stdout:
                print(line, end='')
        
        process.wait()
        return process.returncode
    except KeyboardInterrupt:
        print("\n⚠️ Launch interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return 1

def main():
    # Check if main.py exists
    if not Path("main.py").exists():
        print("❌ Error: main.py not found in current directory")
        print("Please run this script from the Illustrious AI Studio directory")
        return 1
    
    print_banner()
    
    while True:
        print_options()
        try:
            choice = input("\n👉 Select an option (1-6): ").strip()
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            return 0
        
        if choice == "1":
            # Quick start mode
            print("\n🚀 Starting in Quick Start mode...")
            print("💡 This will launch the UI immediately without loading any models")
            print("💡 You can load models later through the web interface")
            return run_command(["main.py", "--quick-start", "--open-browser"])
        
        elif choice == "2":
            # Lazy load mode
            print("\n⏳ Starting in Lazy Load mode...")
            print("💡 Models will be loaded when first needed")
            return run_command(["main.py", "--lazy-load", "--open-browser"])
        
        elif choice == "3":
            # Full initialization
            print("\n🔄 Starting with full model initialization...")
            print("💡 This may take several minutes depending on your hardware")
            print("💡 All models will be loaded and ready to use")
            return run_command(["main.py", "--open-browser"])
        
        elif choice == "4":
            # Custom options
            print("\n🎯 Custom launch options:")
            print("Available flags:")
            print("  --quick-start     : Skip all model loading")
            print("  --lazy-load      : Load models on demand")
            print("  --no-sdxl        : Skip SDXL model")
            print("  --no-ollama      : Skip Ollama models")
            print("  --no-api         : Don't start API server")
            print("  --optimize-memory: Enable memory optimizations")
            print("  --open-browser   : Open browser automatically")
            print("  --web-port PORT  : Change web interface port")
            print("  --api-port PORT  : Change API server port")
            
            try:
                custom_args = input("\n👉 Enter your custom arguments: ").strip()
                if custom_args:
                    args = ["main.py"] + custom_args.split()
                    return run_command(args)
                else:
                    print("⚠️ No arguments provided")
            except KeyboardInterrupt:
                continue
        
        elif choice == "5":
            # Show help
            print("\n❓ Full help information:")
            return run_command(["main.py", "--help"])
        
        elif choice == "6":
            # Exit
            print("\n👋 Goodbye!")
            return 0
        
        else:
            print("❌ Invalid choice. Please select 1-6.")
            continue

if __name__ == "__main__":
    sys.exit(main())
