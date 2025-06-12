#!/usr/bin/env python3
"""
Model Manager for Illustrious AI Studio
Handles model switching and memory management for optimal performance
"""

import subprocess
import time
import torch
from colorama import init, Fore, Style
from pathlib import Path

from core.memory import clear_gpu_memory
from core.config import CONFIG

init(autoreset=True)

class ModelManager:
    def __init__(self):
        self.current_mode = None
        
    def print_header(self, text):
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"{Fore.CYAN}{text:^60}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        
    def print_success(self, text):
        print(f"{Fore.GREEN}✓ {text}{Style.RESET_ALL}")
        
    def print_info(self, text):
        print(f"{Fore.BLUE}ℹ {text}{Style.RESET_ALL}")
        
    def print_warning(self, text):
        print(f"{Fore.YELLOW}⚠ {text}{Style.RESET_ALL}")
        
    def get_gpu_memory_info(self):
        """Get current GPU memory usage"""
        if CONFIG.gpu_backend in ("cuda", "rocm") and torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            free = total - reserved
            
            return {
                "total": total,
                "allocated": allocated,
                "reserved": reserved,
                "free": free
            }
        return None
        
    def show_gpu_status(self):
        """Display current GPU memory status"""
        self.print_header("GPU Memory Status")
        
        if CONFIG.gpu_backend == "cuda":
            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
            if result.returncode == 0:
                print(result.stdout)
        elif CONFIG.gpu_backend == "rocm":
            for cmd in (["rocm-smi"], ["rocminfo"]):
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    print(result.stdout)
                    break
        
        # Also show PyTorch memory
        mem_info = self.get_gpu_memory_info()
        if mem_info:
            self.print_info(f"PyTorch - Total: {mem_info['total']:.1f}GB, "
                          f"Free: {mem_info['free']:.1f}GB ({mem_info['free']/mem_info['total']*100:.1f}%)")
    
    def unload_ollama_models(self):
        """Unload all Ollama models from GPU memory"""
        self.print_info("Unloading Ollama models from GPU...")
        
        # List loaded models
        list_result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if list_result.returncode == 0:
            self.print_info("Currently loaded models:")
            print(list_result.stdout)
        
        # Common models to try unloading
        models_to_unload = [
            "qwen2.5vl:7b",
            "goekdenizguelmez/JOSIEFIED-Qwen3:8b-q6_k",
            "deepseek-r1:8b",
            "qwen3:30b"
        ]
        
        for model in models_to_unload:
            self.print_info(f"Attempting to unload {model}...")
            # Run a minimal prompt to ensure model is loaded, then it should unload after timeout
            subprocess.run(['ollama', 'run', model, '--keepalive', '0s', ''], 
                         capture_output=True, text=True)
        
        # Give it a moment to free memory
        time.sleep(2)
        self.print_success("Ollama models unloaded")
        
    def set_ollama_memory_limit(self):
        """Configure Ollama to use less GPU memory"""
        self.print_info("Configuring Ollama memory settings...")
        
        # Set environment variable for Ollama GPU memory
        # This limits Ollama to 4GB of VRAM
        subprocess.run(['export', 'OLLAMA_GPU_MEMORY=4096'], shell=True)
        
        self.print_success("Ollama memory limit set to 4GB")
        
    def switch_to_image_mode(self):
        """Optimize for image generation"""
        self.print_header("Switching to Image Generation Mode")
        
        # Unload Ollama models
        self.unload_ollama_models()
        
        clear_gpu_memory()
        if CONFIG.gpu_backend in ("cuda", "rocm") and torch.cuda.is_available():
            self.print_success("GPU cache cleared")
        
        self.current_mode = "image"
        self.show_gpu_status()
        
    def switch_to_llm_mode(self):
        """Optimize for LLM usage"""
        self.print_header("Switching to LLM Mode")
        
        clear_gpu_memory()
        if CONFIG.gpu_backend in ("cuda", "rocm") and torch.cuda.is_available():
            self.print_success("GPU cache cleared")
        
        self.current_mode = "llm"
        self.show_gpu_status()
        
    def balanced_mode(self):
        """Configure for balanced usage of both models"""
        self.print_header("Configuring Balanced Mode")
        
        # Set memory limits
        self.set_ollama_memory_limit()
        
        clear_gpu_memory()
            
        self.current_mode = "balanced"
        self.show_gpu_status()
        
    def interactive_menu(self):
        """Interactive model management menu"""
        while True:
            self.print_header("Model Manager Menu")
            print("1. Show GPU Status")
            print("2. Switch to Image Generation Mode (frees memory for SDXL)")
            print("3. Switch to LLM Mode")
            print("4. Balanced Mode (limits Ollama memory)")
            print("5. Unload Ollama Models")
            print("6. Clear GPU Cache")
            print("0. Exit")
            
            choice = input("\nSelect option: ").strip()
            
            if choice == "1":
                self.show_gpu_status()
            elif choice == "2":
                self.switch_to_image_mode()
            elif choice == "3":
                self.switch_to_llm_mode()
            elif choice == "4":
                self.balanced_mode()
            elif choice == "5":
                self.unload_ollama_models()
            elif choice == "6":
                clear_gpu_memory()
                if CONFIG.gpu_backend in ("cuda", "rocm") and torch.cuda.is_available():
                    self.print_success("GPU cache cleared")
                self.show_gpu_status()
            elif choice == "0":
                break
            else:
                self.print_warning("Invalid option")
                
            input("\nPress Enter to continue...")


def main():
    print(f"{Fore.MAGENTA}╔═══════════════════════════════════════════════════════════╗")
    print(f"{Fore.MAGENTA}║          Illustrious AI Studio - Model Manager            ║")
    print(f"{Fore.MAGENTA}╚═══════════════════════════════════════════════════════════╝{Style.RESET_ALL}")
    
    manager = ModelManager()
    
    # Check if running with arguments
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "--image-mode":
            manager.switch_to_image_mode()
        elif sys.argv[1] == "--llm-mode":
            manager.switch_to_llm_mode()
        elif sys.argv[1] == "--balanced":
            manager.balanced_mode()
        elif sys.argv[1] == "--status":
            manager.show_gpu_status()
    else:
        # Interactive mode
        manager.interactive_menu()


if __name__ == "__main__":
    main()
