import os
import logging
import torch
from diffusers import StableDiffusionXLPipeline
import requests

logger = logging.getLogger(__name__)

# Ensure PyTorch fragmentation prevention is enabled
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

def init_sdxl(model_path: str, model_status: dict):
    """Load the Stable Diffusion XL model from ``model_path``."""
    try:
        if not os.path.exists(model_path):
            logger.error(f"SDXL model not found: {model_path}")
            return None

        logger.info("Loading Stable Diffusion XL model...")
        pipe = StableDiffusionXLPipeline.from_single_file(
            model_path,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )

        if torch.cuda.is_available():
            pipe.to("cuda")
            logger.info("SDXL loaded on GPU")
        else:
            logger.warning("CUDA not available, using CPU")

        model_status["sdxl"] = True
        return pipe
    except Exception as e:
        logger.error(f"Failed to initialize SDXL: {e}")
        model_status["sdxl"] = False
        return None

def init_ollama(model_name: str, base_url: str, model_status: dict):
    """Connect to an Ollama server and ensure ``model_name`` is available."""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code != 200:
            logger.error("Ollama server not responding")
            return None

        available_models = response.json().get('models', [])
        model_names = [m['name'] for m in available_models]
        logger.info(f"Available Ollama models: {model_names}")

        if not any(model_name in name for name in model_names):
            logger.error(
                f"Model '{model_name}' not found in Ollama. Available: {model_names}"
            )
            return None

        test_data = {
            "model": model_name,
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False,
        }
        test_response = requests.post(
            f"{base_url}/api/chat", json=test_data, timeout=30
        )

        if test_response.status_code == 200:
            model_status["ollama"] = True
            model_status["multimodal"] = "vision" in model_name.lower() or "llava" in model_name.lower()
            logger.info(f"Ollama model '{model_name}' loaded successfully!")
            return model_name
        else:
            logger.error(f"Failed to test Ollama model: {test_response.text}")
            return None
    except Exception as e:
        logger.error(f"Failed to initialize Ollama: {e}")
        model_status["ollama"] = False
        return None
