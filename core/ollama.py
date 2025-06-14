"""
Illustrious AI Studio - Ollama Integration

This module provides comprehensive integration with Ollama for language model
functionality including chat, text generation, and multimodal capabilities.

KEY FEATURES:
- Chat completion with conversation history
- AI-powered prompt generation for image creation
- Image analysis and multimodal interactions
- Model management and switching
- Persistent conversation storage
- Tool calling and function integration
- Error handling and retry logic

CHAT FUNCTIONALITY:
- Multi-session conversation management
- Persistent chat history across sessions
- Configurable response parameters (temperature, max tokens)
- Context-aware responses with memory
- Session isolation for concurrent users

MULTIMODAL FEATURES:
- Image upload and analysis
- Visual question answering
- Image description generation
- Content extraction from images
- Integration with vision-language models

PROMPT GENERATION:
- AI-assisted prompt improvement
- Style and quality enhancement
- Creative prompt suggestions
- Technical parameter recommendations
- Integration with SDXL generation workflow

The module handles Ollama server communication, error recovery,
and provides a clean interface for all language model operations.
"""

import base64
import io
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import time

from PIL import Image
import requests

from .circuit import CircuitBreaker, CircuitBreakerOpen

from .state import AppState
from .sdxl import generate_image, save_to_gallery, TEMP_DIR
from .config import CONFIG
from .mcp_tools import call_tool

logger = logging.getLogger(__name__)

# Single thread pool for background disk writes
executor = ThreadPoolExecutor(max_workers=1)

# Circuit breaker for Ollama API requests
breaker = CircuitBreaker()

# ==================================================================
# CONSTANTS AND CONFIGURATION
# ==================================================================

# File for persistent chat history storage
CHAT_HISTORY_FILE = Path(TEMP_DIR) / "chat_history.json"


# ==================================================================
# CHAT HISTORY MANAGEMENT
# ==================================================================

def load_chat_history(state: AppState) -> None:
    """
    Load persistent chat history from disk into application state.
    
    Restores all conversation sessions and their message history
    from the JSON file storage. This allows conversations to persist
    across application restarts.
    
    Args:
        state: Application state to populate with chat history
        
    Notes:
        - Handles file corruption gracefully
        - Creates empty history if file doesn't exist
        - Converts stored lists to deque of tuples for consistency
    """
    try:
        if CHAT_HISTORY_FILE.exists():
            with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Convert loaded data to proper format
            with state.atomic_operation():
                for session_id, history in data.items():
                    dq = deque(maxlen=100)
                    dq.extend(tuple(msg) for msg in history)
                    state.chat_history_store[session_id] = dq

            logger.info(f"Loaded chat history for {len(data)} sessions")
            
    except Exception as e:
        logger.error("Failed to load chat history: %s", e)
        # Continue with empty history rather than crashing


def save_chat_history(state: AppState) -> None:
    """
    Persist current chat history from application state to disk.
    
    Saves all conversation sessions to a JSON file for persistence
    across application restarts. Called periodically and on shutdown.
    
    Args:
        state: Application state containing chat history
        
    Notes:
        - Creates directory if it doesn't exist
        - Converts deque entries of tuples to lists for JSON serialization
        - Uses UTF-8 encoding for international characters
        - Includes proper error handling
    """
    try:
        # Ensure directory exists
        CHAT_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

        with state.atomic_operation():
            serializable = {
                sid: [list(msg) for msg in list(history)]
                for sid, history in state.chat_history_store.items()
            }

        # Write to file with proper formatting
        with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)

        logger.debug(f"Saved chat history for {len(serializable)} sessions")
        
    except Exception as e:
        logger.error("Failed to save chat history: %s", e)


def save_chat_history_async(state: AppState) -> None:
    """Persist chat history to disk using a background thread."""

    def _write():
        try:
            CHAT_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
            with state.atomic_operation():
                serializable = {
                    sid: [list(msg) for msg in list(history)]
                    for sid, history in state.chat_history_store.items()
                }
            with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(serializable, f, indent=2, ensure_ascii=False)
            logger.debug(
                f"Saved chat history for {len(serializable)} sessions"
            )
        except Exception as e:  # pragma: no cover - log and continue
            logger.error("Failed to save chat history: %s", e)

    future = executor.submit(_write)
    logger.debug("Background task for saving chat history submitted.")


# ==================================================================
# OLLAMA MODEL INITIALIZATION AND MANAGEMENT
# ==================================================================

def init_ollama(state: AppState) -> Optional[str]:
    """
    Initialize Ollama connection and verify model availability.
    
    Tests connectivity to the Ollama server, verifies that required
    models are available, and updates the application state with
    model information.
    
    Args:
        state: Application state to update with Ollama status
        
    Returns:
        Optional[str]: Name of the initialized model, or None if failed
        
    Notes:
        - Checks both language and vision models
        - Updates model status flags in application state
        - Provides detailed error reporting for troubleshooting
        - Handles network timeouts and connection errors
    """
    start_time = time.perf_counter()
    try:
        response = call_with_circuit_breaker(
            breaker,
            requests.get,
            f"{CONFIG.ollama_base_url}/api/tags",
            timeout=5
        )
        if response.status_code != 200:
            logger.error("Ollama server not responding")
            return None
        available_models = response.json().get("models", [])
        model_names = [m["name"] for m in available_models]
        
        # Initialize text model
        target_model = _get_model_path()
        logger.info("Available Ollama models: %s", model_names)
        if not any(target_model in name for name in model_names):
            logger.error("Model '%s' not found in Ollama", target_model)
            return None
        
        # Test text model
        test_data = {
            "model": target_model,
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False,
        }
        test_response = breaker.call(
            lambda: requests.post(
                f"{CONFIG.ollama_base_url}/api/chat",
                json=test_data,
                timeout=30,
            )
        )
        if test_response.status_code == 200:
            state.model_status["ollama"] = True
            state.ollama_model = target_model
            logger.info("Ollama text model '%s' loaded", target_model)
            load_chat_history(state)
            
            # Initialize vision model if configured
            vision_model = getattr(CONFIG, 'ollama_vision_model', None)
            if vision_model and any(vision_model in name for name in model_names):
                state.model_status["multimodal"] = True
                state.ollama_vision_model = vision_model
                logger.info("Ollama vision model '%s' loaded", vision_model)
            else:
                state.model_status["multimodal"] = False
                logger.warning("Vision model not found or not configured")

            state.metrics.record_ollama_load(time.perf_counter() - start_time)
            return target_model
        logger.error("Failed to test Ollama model: %s", test_response.text)
    except CircuitBreakerOpen:
        logger.error("Ollama API temporarily unavailable (circuit open)")
        return None
    except Exception as e:
        logger.error("Failed to initialize Ollama: %s", e)
    state.model_status["ollama"] = False
    return None


def _get_model_path() -> str:
    return CONFIG.ollama_model


def switch_ollama_model(state: AppState, name: str) -> bool:
    """Switch the active Ollama model."""
    state.ollama_model = None
    state.model_status["ollama"] = False
    state.model_status["multimodal"] = False
    state.chat_history_store.clear()
    save_chat_history_async(state)
    CONFIG.ollama_model = name
    logger.info("Switching Ollama model to %s", name)
    return init_ollama(state) is not None


def chat_completion(state: AppState, messages: List[dict], temperature: float = 0.7, max_tokens: int = 256) -> str:
    if not state.ollama_model:
        return (
            "❌ Ollama model not loaded. Please check your Ollama setup. "
            "See the Installation section of the README."
        )
    try:
        data = {
            "model": state.ollama_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        response = breaker.call(
            lambda: requests.post(
                f"{CONFIG.ollama_base_url}/api/chat", json=data, timeout=60
            )
        )
        if response.status_code == 200:
            result = response.json()
            return result.get("message", {}).get("content", "No response generated")
        logger.error("Ollama API error: %s - %s", response.status_code, response.text)
        return (
            f"❌ Chat completion failed: {response.status_code}. "
            "Ensure the Ollama server is reachable."
        )
    except CircuitBreakerOpen:
        logger.warning("CircuitBreakerOpen exception occurred during chat completion.")
        return "❌ Service temporarily unavailable. Please try again later."
    except Exception as e:
        logger.error("Ollama completion failed: %s", e)
        return (
            f"❌ Chat completion failed: {e}. "
            "Ensure the Ollama server is running."
        )


def generate_prompt(state: AppState, user_input: str) -> str:
    if not user_input.strip():
        return "Please enter a description first."
    system_prompt = (
        "You are an expert AI art prompt specialist. Create detailed, creative prompts for image generation."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]
    enhanced = chat_completion(state, messages, temperature=0.8)
    return enhanced if not enhanced.startswith("❌") else user_input


def generate_creative_prompt(state: AppState, user_input: str, style: str = "enhance") -> str:
    """Generate creative prompts with different styles."""

    style_prompts = {
        "enhance": (
            "Transform this idea into a detailed, visually stunning prompt. "
            "Add artistic details, lighting, atmosphere, and quality tags. "
            "Be creative but keep the core concept."
        ),
        "dreamy": (
            "Make this idea ethereal and dreamlike. Add soft, magical elements, "
            "pastel colors, and a sense of wonder. Make it feel like a beautiful dream."
        ),
        "epic": (
            "Transform this into an epic, dramatic scene. Add grand scale, "
            "dramatic lighting, and a sense of awe and adventure."
        ),
        "fun": (
            "Make this playful and fun! Add bright colors, whimsical elements, "
            "and a sense of joy and humor."
        ),
        "artistic": (
            "Transform this into a fine art piece. Reference art movements, "
            "painting techniques, and artistic styles."
        ),
    }

    system_prompt = style_prompts.get(style, style_prompts["enhance"])

    messages = [
        {"role": "system", "content": f"You are a creative AI artist. {system_prompt}"},
        {"role": "user", "content": f"Create a prompt for: {user_input}"},
    ]

    enhanced = chat_completion(state, messages, temperature=0.9, max_tokens=150)
    return enhanced if not enhanced.startswith("❌") else user_input


def _execute_tool_command(command: str) -> str:
    """Parse and execute a /tool command."""
    parts = command.strip().split()
    if not parts:
        return "Usage: /tool <server>.<method> key=value"
    server_method = parts[0]
    if "." not in server_method:
        return "Invalid tool format. Use /tool <server>.<method> key=value"
    server, method = server_method.split(".", 1)
    args = {}
    for part in parts[1:]:
        if "=" in part:
            k, v = part.split("=", 1)
            args[k] = v
    return call_tool(server, method, **args)


def handle_chat(state: AppState, message: str, session_id: str = "default", chat_history: Optional[list] = None) -> Tuple[list, str]:
    if not message.strip():
        return chat_history or [], ""
    with state.atomic_operation():
        state.chat_history_store[session_id]
    if message.startswith("/tool"):
        response = _execute_tool_command(message[len("/tool"):])
    elif message.lower().startswith("#generate") or "generate image" in message.lower():
        clean_prompt = message.replace("#generate", "").replace("generate image", "").strip()
        if not clean_prompt:
            response = "Please provide a description for the image you want to generate."
        else:
            enhanced_prompt = generate_prompt(state, clean_prompt)
            response = (
                f"🎨 I'll create an image with this enhanced prompt:\n\n'{enhanced_prompt}'\n\nGenerating now..."
            )
            image, status = generate_image(
                state,
                {
                    "prompt": enhanced_prompt,
                    "save_to_gallery_flag": False,
                },
            )
            if image:
                saved_path = save_to_gallery(
                    state,
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
        with state.atomic_operation():
            recent_history = list(state.chat_history_store[session_id])[-10:]
        for msg in recent_history:
            messages.extend([
                {"role": "user", "content": msg[0]},
                {"role": "assistant", "content": msg[1]},
            ])
        messages.append({"role": "user", "content": message})
        response = chat_completion(state, messages)
        if "<think>" in response and "</think>" in response:
            parts = response.split("</think>")
            if len(parts) > 1:
                response = parts[-1].strip()
    state.update_chat_history(session_id, (message, response))
    save_chat_history_async(state)
    if chat_history is None:
        chat_history = []
    chat_history.append([message, response])
    return chat_history, ""


def analyze_image(state: AppState, image: Image.Image, question: str = "Describe this image in detail") -> str:
    if not image:
        return "Please upload an image first."
    if not state.model_status["multimodal"] or not hasattr(state, "ollama_vision_model"):
        return "❌ Ollama vision model not available. Please ensure a vision-capable model is configured."

    try:
        width, height = image.size
        if width * height > MAX_IMAGE_PIXELS:
            return "❌ Image exceeds 16MP limit."
        if width > 1920 or height > 1920:
            image = image.copy()
            image.thumbnail((1920, 1920), Image.Resampling.LANCZOS)

        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        data = {
            "model": state.ollama_vision_model,
            "messages": [{"role": "user", "content": question, "images": [img_base64]}],
            "stream": False,
        }
        response = breaker.call(
            lambda: requests.post(
                f"{CONFIG.ollama_base_url}/api/chat", json=data, timeout=60
            )
        )
        if response.status_code == 200:
            result = response.json()
            return result.get("message", {}).get("content", "No analysis generated")
        logger.error("Ollama vision API error: %s - %s", response.status_code, response.text)
        return (
            f"❌ Analysis failed: {response.status_code}. "
            "Ensure the Ollama server is reachable."
        )
    except CircuitBreakerOpen:
        return "❌ Service temporarily unavailable. Please try again later."
    except Exception as e:
        logger.error("Image analysis failed: %s", e)
        return (
            f"❌ Analysis failed: {e}. "
            "Ensure the Ollama server is running."
        )
