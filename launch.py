#!/usr/bin/env python3
"""
Illustrious AI Studio - Interactive Launcher

This script provides a user-friendly interactive launcher for the AI Studio
with various startup modes and configuration options. It's designed to make
the application accessible to users of all technical levels.

FEATURES:
- Interactive menu system for easy mode selection
- Pre-configured launch modes for different use cases
- Real-time output display during startup
- Error handling and user feedback
- Custom options for advanced users
- Built-in help system

LAUNCH MODES:
1. Quick Start: Immediate UI access, models load on demand
2. Lazy Load: UI starts first, models initialize in background
3. Full Initialization: All models loaded before UI access
4. Custom Options: Advanced user-specified parameters
5. Help: Complete documentation of available options

The launcher handles all the complexity of command-line arguments
and provides a clean, intuitive interface for starting the application.
"""

import subprocess
import sys
from pathlib import Path


# ==================================================================
# USER INTERFACE FUNCTIONS
# ==================================================================

def print_banner():
    """Display the application banner and branding."""
    print("=" * 60)
    print("🎨 Illustrious AI Studio Launcher")
    print("   Your Gateway to AI-Powered Creativity")
    print("=" * 60)


def print_options():
    """Display the main menu options with descriptions."""
    print("\nAvailable launch modes:")
    print("1. 🚀 Quick Start (fastest, no model loading)")
    print("   └─ Instant UI access, models load when needed")
    print()
    print("2. ⏳ Lazy Load (start UI, load models on demand)")
    print("   └─ UI starts immediately, models initialize in background")
    print()
    print("3. 🔄 Full Initialization (load all models at startup)")
    print("   └─ Complete setup, ready for immediate use")
    print()
    print("4. 🎯 Custom Options")
    print("   └─ Advanced configuration for power users")
    print()
    print("5. ❓ Help (show all available options)")
    print("   └─ Complete documentation and command reference")
    print()
    print("6. 🚪 Exit")
    print("   └─ Close the launcher")


# ==================================================================
# COMMAND EXECUTION AND PROCESS MANAGEMENT
# ==================================================================

def run_command(cmd):
    """
    Execute a command with real-time output display.
    
    This function runs the main application with the specified arguments
    and provides live feedback to the user during startup.
    
    Args:
        cmd: List of command arguments to pass to main.py
        
    Returns:
        int: Process return code (0 for success)
        
    Features:
        - Real-time output streaming
        - Graceful keyboard interrupt handling
        - Error reporting and recovery
        - Process monitoring and cleanup
    """
    print(f"\n🚀 Running: python {' '.join(cmd)}")
    print("-" * 40)
    
    try:
        # Start the main application process
        process = subprocess.Popen(
            [sys.executable] + cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1  # Line buffered for real-time output
        )
        
        # Stream output in real-time
        if process.stdout:
            for line in process.stdout:
                print(line, end='')
        
        # Wait for process completion
        process.wait()
        return process.returncode
        
    except KeyboardInterrupt:
        print("\n⚠️ Launch interrupted by user")
        print("   The application startup was cancelled.")
        return 1
        
    except Exception as e:
        print(f"❌ Error running command: {e}")
        print("   Please check your installation and try again.")
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
            print("  --share          : Create a public share link")
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
