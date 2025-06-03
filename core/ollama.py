import base64
import io
import logging
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
import requests

from .memory import model_status, latest_generated_image
from .sdxl import generate_image, save_to_gallery

logger = logging.getLogger(__name__)

ollama_model: Optional[str] = None
chat_history_store: Dict[str, List[Tuple[str, str]]] = {}


def init_ollama() -> Optional[str]:
    """Verify Ollama is accessible and return the selected model name."""
    global ollama_model
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code != 200:
            logger.error("Ollama server not responding")
            return None
        available_models = response.json().get("models", [])
        model_names = [m["name"] for m in available_models]
        target_model = _get_model_path()
        logger.info("Available Ollama models: %s", model_names)
        if not any(target_model in name for name in model_names):
            logger.error("Model '%s' not found in Ollama", target_model)
            return None
        test_data = {
            "model": target_model,
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False,
        }
        test_response = requests.post(
            "http://localhost:11434/api/chat",
            json=test_data,
            timeout=30,
        )
        if test_response.status_code == 200:
            model_status["ollama"] = True
            model_status["multimodal"] = "vision" in target_model.lower() or "llava" in target_model.lower()
            ollama_model = target_model
            logger.info("Ollama model '%s' loaded", target_model)
            return target_model
        logger.error("Failed to test Ollama model: %s", test_response.text)
    except Exception as e:
        logger.error("Failed to initialize Ollama: %s", e)
    model_status["ollama"] = False
    return None


def _get_model_path() -> str:
    from .sdxl import MODEL_PATHS
    return MODEL_PATHS["ollama_model"]


def chat_completion(messages: List[dict], temperature: float = 0.7, max_tokens: int = 256) -> str:
    if not ollama_model:
        return "❌ Ollama model not loaded. Please check your Ollama setup."
    try:
        data = {
            "model": ollama_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        response = requests.post("http://localhost:11434/api/chat", json=data, timeout=60)
        if response.status_code == 200:
            result = response.json()
            return result.get("message", {}).get("content", "No response generated")
        logger.error("Ollama API error: %s - %s", response.status_code, response.text)
        return f"❌ Chat completion failed: {response.status_code}"
    except Exception as e:
        logger.error("Ollama completion failed: %s", e)
        return f"❌ Chat completion failed: {e}"


def generate_prompt(user_input: str) -> str:
    if not user_input.strip():
        return "Please enter a description first."
    system_prompt = (
        "You are an expert AI art prompt specialist. Create detailed, creative prompts for image generation."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]
    enhanced = chat_completion(messages, temperature=0.8)
    return enhanced if not enhanced.startswith("❌") else user_input


def handle_chat(message: str, session_id: str = "default", chat_history: Optional[list] = None) -> Tuple[list, str]:
    if not message.strip():
        return chat_history or [], ""
    if session_id not in chat_history_store:
        chat_history_store[session_id] = []
    if message.lower().startswith("#generate") or "generate image" in message.lower():
        clean_prompt = message.replace("#generate", "").replace("generate image", "").strip()
        if not clean_prompt:
            response = "Please provide a description for the image you want to generate."
        else:
            enhanced_prompt = generate_prompt(clean_prompt)
            response = (
                f"🎨 I'll create an image with this enhanced prompt:\n\n'{enhanced_prompt}'\n\nGenerating now..."
            )
            image, status = generate_image(enhanced_prompt, save_to_gallery_flag=False)
            if image:
                saved_path = save_to_gallery(
                    image,
                    enhanced_prompt,
                    {"generated_from_chat": True, "original_request": clean_prompt},
                )
                response += f"\n\n{status}\n\n🖼️ **Image generated and displayed!**"
                response += f"\n📁 **Saved to:** `{saved_path}`"
            else:
                response += f"\n\n{status}"
    else:
        system_prompt = (
            "You are a helpful AI assistant specializing in creative tasks and image generation. Be friendly and informative."
        )
        messages = [{"role": "system", "content": system_prompt}]
        recent_history = chat_history_store[session_id][-10:]
        for msg in recent_history:
            messages.extend([
                {"role": "user", "content": msg[0]},
                {"role": "assistant", "content": msg[1]},
            ])
        messages.append({"role": "user", "content": message})
        response = chat_completion(messages)
        if "<think>" in response and "</think>" in response:
            parts = response.split("</think>")
            if len(parts) > 1:
                response = parts[-1].strip()
    chat_history_store[session_id].append((message, response))
    if chat_history is None:
        chat_history = []
    chat_history.append([message, response])
    return chat_history, ""


def analyze_image(image: Image.Image, question: str = "Describe this image in detail") -> str:
    if not image:
        return "Please upload an image first."
    if not ollama_model or not model_status["multimodal"]:
        return "❌ Ollama vision model not available. Please use a vision-capable model like 'llava'."
    try:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        data = {
            "model": ollama_model,
            "messages": [{"role": "user", "content": question, "images": [img_base64]}],
            "stream": False,
        }
        response = requests.post("http://localhost:11434/api/chat", json=data, timeout=60)
        if response.status_code == 200:
            result = response.json()
            return result.get("message", {}).get("content", "No analysis generated")
        logger.error("Ollama vision API error: %s - %s", response.status_code, response.text)
        return f"❌ Analysis failed: {response.status_code}"
    except Exception as e:
        logger.error("Image analysis failed: %s", e)
        return f"❌ Analysis failed: {e}"
