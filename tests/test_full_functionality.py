#!/usr/bin/env python3
"""
Comprehensive test script for Illustrious AI Studio
Tests image generation, LLM prompts, and vision model analysis
"""

import sys
import logging
from pathlib import Path
from PIL import Image
import torch
from colorama import init, Fore, Style

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize colorama for colored output
init(autoreset=True)

# Import our modules
from core.state import AppState
from core.sdxl import init_sdxl, generate_image
from core.ollama import init_ollama, generate_prompt, analyze_image
from core.memory import get_model_status
from core.config import CONFIG


class FunctionalityTester:
    def __init__(self):
        self.state = AppState()
        assert self.state.ollama_vision_model is None
        self.test_results = {}
        
    def print_header(self, text):
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"{Fore.CYAN}{text:^60}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        
    def print_success(self, text):
        print(f"{Fore.GREEN}✓ {text}{Style.RESET_ALL}")
        
    def print_error(self, text):
        print(f"{Fore.RED}✗ {text}{Style.RESET_ALL}")
        
    def print_info(self, text):
        print(f"{Fore.BLUE}ℹ {text}{Style.RESET_ALL}")
        
    def print_warning(self, text):
        print(f"{Fore.YELLOW}⚠ {text}{Style.RESET_ALL}")

    def test_initialization(self):
        """Test model initialization"""
        self.print_header("Initializing Models")
        
        # Initialize SDXL
        self.print_info("Loading SDXL model...")
        sdxl_result = init_sdxl(self.state)
        if sdxl_result:
            self.print_success("SDXL model loaded successfully")
            self.test_results["sdxl_init"] = True
        else:
            self.print_error("Failed to load SDXL model")
            self.test_results["sdxl_init"] = False
            
        # Initialize Ollama
        self.print_info("Loading Ollama models...")
        ollama_result = init_ollama(self.state)
        if ollama_result:
            self.print_success(f"Ollama text model loaded: {ollama_result}")
            self.test_results["ollama_init"] = True
            
            if self.state.model_status.get("multimodal"):
                self.print_success(f"Vision model loaded: {self.state.ollama_vision_model}")
                self.test_results["vision_init"] = True
            else:
                self.print_warning("Vision model not available")
                self.test_results["vision_init"] = False
        else:
            self.print_error("Failed to load Ollama models")
            self.test_results["ollama_init"] = False
            
        # Print status
        self.print_info("\nModel Status:")
        print(get_model_status(self.state))

    def test_llm_prompt_generation(self):
        """Test LLM prompt generation"""
        self.print_header("Testing LLM Prompt Generation")
        
        if not self.test_results.get("ollama_init"):
            self.print_error("Skipping: Ollama not initialized")
            return None
            
        test_prompt = "a cyberpunk cat with neon lights"
        self.print_info(f"Original prompt: '{test_prompt}'")
        
        try:
            enhanced_prompt = generate_prompt(self.state, test_prompt)
            if enhanced_prompt and not enhanced_prompt.startswith("❌"):
                self.print_success("Prompt enhanced successfully")
                self.print_info(f"Enhanced prompt: '{enhanced_prompt}'")
                self.test_results["prompt_generation"] = True
                return enhanced_prompt
            else:
                self.print_error(f"Prompt enhancement failed: {enhanced_prompt}")
                self.test_results["prompt_generation"] = False
                return test_prompt
        except Exception as e:
            self.print_error(f"Exception during prompt generation: {e}")
            self.test_results["prompt_generation"] = False
            return test_prompt

    def test_image_generation(self, prompt=None):
        """Test image generation"""
        self.print_header("Testing Image Generation")
        
        if not self.test_results.get("sdxl_init"):
            self.print_error("Skipping: SDXL not initialized")
            return None
            
        if not prompt:
            prompt = "a beautiful landscape with mountains and a lake, highly detailed, 4k"
            
        self.print_info(f"Generating image with prompt: '{prompt}'")
        self.print_info("This may take 30-60 seconds...")
        
        try:
            image, status = generate_image(
                self.state,
                {
                    "prompt": prompt,
                    "negative_prompt": "blurry, low quality, text, watermark",
                    "steps": 30,
                    "guidance": 7.5,
                    "seed": 42,
                },
            )
            
            if image:
                self.print_success(f"Image generated successfully! {status}")
                self.test_results["image_generation"] = True
                
                # Save test image
                test_output_dir = Path("test_outputs")
                test_output_dir.mkdir(exist_ok=True)
                test_image_path = test_output_dir / "test_generated_image.png"
                image.save(test_image_path)
                self.print_info(f"Test image saved to: {test_image_path}")
                
                return image
            else:
                self.print_error(f"Image generation failed: {status}")
                self.test_results["image_generation"] = False
                return None
        except Exception as e:
            self.print_error(f"Exception during image generation: {e}")
            logger.exception("Image generation error")
            self.test_results["image_generation"] = False
            return None

    def test_vision_analysis(self, image=None):
        """Test vision model analysis"""
        self.print_header("Testing Vision Model Analysis")
        
        if not self.test_results.get("vision_init"):
            self.print_error("Skipping: Vision model not initialized")
            return
            
        if not image:
            # Load a test image if none provided
            test_image_path = Path("test_outputs/test_generated_image.png")
            if test_image_path.exists():
                image = Image.open(test_image_path)
                self.print_info("Using previously generated test image")
            else:
                self.print_error("No image available for analysis")
                return
                
        test_questions = [
            "Describe this image in detail",
            "What are the main colors in this image?",
            "What mood or atmosphere does this image convey?"
        ]
        
        for question in test_questions:
            self.print_info(f"\nQuestion: '{question}'")
            
            try:
                analysis = analyze_image(self.state, image, question)
                if analysis and not analysis.startswith("❌"):
                    self.print_success("Analysis successful")
                    self.print_info(f"Response: {analysis[:200]}..." if len(analysis) > 200 else f"Response: {analysis}")
                    self.test_results["vision_analysis"] = True
                else:
                    self.print_error(f"Analysis failed: {analysis}")
                    self.test_results["vision_analysis"] = False
                    break
            except Exception as e:
                self.print_error(f"Exception during analysis: {e}")
                self.test_results["vision_analysis"] = False
                break

    def test_gpu_memory(self):
        """Test GPU memory usage"""
        self.print_header("GPU Memory Status")
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            self.print_info(f"Total VRAM: {total:.1f} GB")
            self.print_info(f"Allocated: {allocated:.1f} GB ({allocated/total*100:.1f}%)")
            self.print_info(f"Reserved: {reserved:.1f} GB ({reserved/total*100:.1f}%)")
            self.print_info(f"Free: {total-reserved:.1f} GB ({(total-reserved)/total*100:.1f}%)")
        else:
            self.print_error("CUDA not available")

    def run_all_tests(self):
        """Run all functionality tests"""
        print(f"{Fore.MAGENTA}╔═══════════════════════════════════════════════════════════╗")
        print(f"{Fore.MAGENTA}║       Illustrious AI Studio - Full Functionality Test     ║")
        print(f"{Fore.MAGENTA}╚═══════════════════════════════════════════════════════════╝{Style.RESET_ALL}")
        
        # Test initialization
        self.test_initialization()
        
        # Test prompt generation
        enhanced_prompt = self.test_llm_prompt_generation()
        
        # Test image generation
        generated_image = self.test_image_generation(enhanced_prompt)
        
        # Test vision analysis
        if generated_image:
            self.test_vision_analysis(generated_image)
        
        # Check GPU memory
        self.test_gpu_memory()
        
        # Summary
        self.print_header("Test Summary")
        
        passed = sum(1 for v in self.test_results.values() if v)
        total = len(self.test_results)
        
        if passed == total:
            self.print_success(f"All tests passed! ({passed}/{total})")
        else:
            self.print_warning(f"Tests passed: {passed}/{total}")
            
        for test, result in self.test_results.items():
            if result:
                self.print_success(f"{test}: PASSED")
            else:
                self.print_error(f"{test}: FAILED")
                
        return passed == total


def main():
    tester = FunctionalityTester()
    success = tester.run_all_tests()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
