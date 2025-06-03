import base64
import io
import json
import uuid
import logging
import gc
from datetime import datetime
from pathlib import Path
from typing import List

import torch
from PIL import Image
import requests

logger = logging.getLogger(__name__)

chat_history_store = {}
latest_generated_image = None

def save_to_gallery(image: Image.Image, prompt: str, gallery_dir: Path, metadata: dict | None = None) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{uuid.uuid4().hex[:8]}.png"
    filepath = gallery_dir / filename

    image.save(filepath)

    metadata_file = filepath.with_suffix('.json')
    metadata_info = {
        "prompt": prompt,
        "timestamp": timestamp,
        "filename": filename,
        **(metadata or {}),
    }

    with open(metadata_file, 'w') as f:
        json.dump(metadata_info, f, indent=2)

    return str(filepath)

def clear_cuda_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    logger.info("CUDA memory cleared and garbage collection performed")

def generate_image(pipe, prompt, negative_prompt="", steps=30, guidance=7.5, seed=-1, save_to_gallery_flag=True, gallery_dir: Path | None = None):
    global latest_generated_image

    if not pipe:
        return None, "❌ SDXL model not loaded. Please check your model path."

    clear_cuda_memory()

    max_retries = 2
    for attempt in range(max_retries):
        try:
            generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
            if seed != -1:
                generator.manual_seed(seed)
            else:
                seed = generator.initial_seed()

            logger.info(f"Generating image with prompt: {prompt[:50]}... (Attempt {attempt + 1}/{max_retries})")

            image = pipe(
                prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
                width=1024,
                height=1024,
            ).images[0]

            latest_generated_image = image

            if save_to_gallery_flag and gallery_dir is not None:
                filepath = save_to_gallery(image, prompt, gallery_dir, {
                    "negative_prompt": negative_prompt,
                    "steps": steps,
                    "guidance": guidance,
                    "seed": seed,
                })
                logger.info(f"Image saved to: {filepath}")

            clear_cuda_memory()

            return image, f"✅ Image generated successfully! Seed: {seed}"

        except RuntimeError as e:
            if "CUDA out of memory" in str(e) or "out of memory" in str(e).lower():
                logger.warning(f"CUDA OOM error on attempt {attempt + 1}: {e}")
                clear_cuda_memory()

                if attempt < max_retries - 1:
                    logger.info("Retrying image generation after memory clearing...")
                    continue
                else:
                    memory_info = ""
                    if torch.cuda.is_available():
                        try:
                            memory_info = f" Available: {torch.cuda.get_device_properties(0).total_memory // (1024**3)}GB GPU"
                        except Exception:
                            pass
                    return None, f"❌ CUDA out of memory after {max_retries} attempts.{memory_info} Try reducing image size, steps, or guidance scale."
            else:
                logger.error(f"Image generation failed: {e}")
                return None, f"❌ Generation failed: {str(e)}"
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            return None, f"❌ Generation failed: {str(e)}"

def chat_completion(model: str, base_url: str, messages: List[dict], temperature=0.7, max_tokens=256):
    if not model:
        return "❌ Ollama model not loaded. Please check your Ollama setup."

    try:
        data = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }

        response = requests.post(f"{base_url}/api/chat", json=data, timeout=60)

        if response.status_code == 200:
            result = response.json()
            return result.get('message', {}).get('content', 'No response generated')
        else:
            logger.error(f"Ollama API error: {response.status_code} - {response.text}")
            return f"❌ Chat completion failed: {response.status_code}"
    except Exception as e:
        logger.error(f"Ollama completion failed: {e}")
        return f"❌ Chat completion failed: {str(e)}"

def generate_prompt(model: str, base_url: str, user_input: str) -> str:
    if not user_input.strip():
        return "Please enter a description first."

    system_prompt = """You are an expert AI art prompt specialist. Create detailed, creative prompts for image generation.

Guidelines:
- Include artistic style, lighting, composition details
- Add quality enhancers like "masterpiece, best quality, highly detailed"
- Describe colors, mood, and atmosphere
- Keep prompts under 200 words
- Make them vivid and specific

User request:"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]

    enhanced = chat_completion(model, base_url, messages, temperature=0.8)
    return enhanced if not enhanced.startswith("❌") else user_input

def handle_chat(model: str, base_url: str, message: str, *, session_id="default", chat_history=None, pipe=None, gallery_dir: Path | None = None):
    print(f"DEBUG: handle_chat called with message: '{message}'")

    if not message.strip():
        return chat_history or [], ""

    if session_id not in chat_history_store:
        chat_history_store[session_id] = []

    if message.lower().startswith("#generate") or "generate image" in message.lower():
        clean_prompt = message.replace("#generate", "").replace("generate image", "").strip()
        if not clean_prompt:
            response = "Please provide a description for the image you want to generate."
        else:
            enhanced_prompt = generate_prompt(model, base_url, clean_prompt)
            response = f"🎨 I'll create an image with this enhanced prompt:\n\n'{enhanced_prompt}'\n\nGenerating now..."
            image, status = generate_image(pipe, enhanced_prompt, save_to_gallery_flag=False)
            if image:
                saved_path = save_to_gallery(image, enhanced_prompt, gallery_dir, {
                    "generated_from_chat": True,
                    "original_request": clean_prompt,
                })
                response += f"\n\n{status}"
                response += "\n\n🖼️ **Image generated and displayed!**"
                response += f"\n📁 **Saved to:** `{saved_path}`"
                response += "\n\n💡 **Tip:** Check the 'Text-to-Image' tab to see your creation, or use the download button!"
            else:
                response += f"\n\n{status}"
    else:
        print(f"DEBUG: Starting normal chat for message: '{message}'")
        system_prompt = "You are a helpful AI assistant specializing in creative tasks and image generation. Be friendly and informative."
        messages = [{"role": "system", "content": system_prompt}]
        recent_history = chat_history_store[session_id][-10:]
        for msg in recent_history:
            messages.extend([
                {"role": "user", "content": msg[0]},
                {"role": "assistant", "content": msg[1]},
            ])
        messages.append({"role": "user", "content": message})
        print("DEBUG: About to call chat_completion")
        response = chat_completion(model, base_url, messages)
        print(f"DEBUG: Got response: {response[:100]}...")
        if "<think>" in response and "</think>" in response:
            parts = response.split("</think>")
            if len(parts) > 1:
                response = parts[-1].strip()

    chat_history_store[session_id].append((message, response))
    if chat_history is None:
        chat_history = []
    chat_history.append([message, response])
    return chat_history, ""

def analyze_image(model: str, base_url: str, image: Image.Image, question="Describe this image in detail"):
    if not image:
        return "Please upload an image first."

    if not model:
        return "❌ Ollama model not loaded. Please check your Ollama setup."

    try:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        data = {
            "model": model,
            "messages": [{"role": "user", "content": question, "images": [img_base64]}],
            "stream": False,
        }

        response = requests.post(f"{base_url}/api/chat", json=data, timeout=60)

        if response.status_code == 200:
            result = response.json()
            return result.get('message', {}).get('content', 'No analysis generated')
        else:
            logger.error(f"Ollama vision API error: {response.status_code} - {response.text}")
            return f"❌ Analysis failed: {response.status_code}"
    except Exception as e:
        logger.error(f"Image analysis failed: {e}")
        return f"❌ Analysis failed: {str(e)}"

def get_model_status(model_status: dict) -> str:
    status_text = "🤖 **Model Status:**\n"
    status_text += f"• SDXL: {'✅ Loaded' if model_status['sdxl'] else '❌ Not loaded'}\n"
    status_text += f"• Ollama: {'✅ Connected' if model_status['ollama'] else '❌ Not connected'}\n"
    status_text += f"• Vision: {'✅ Available' if model_status['multimodal'] else '❌ Not available'}\n"
    status_text += f"• CUDA: {'✅ Available' if torch.cuda.is_available() else '❌ Not available'}"
    return status_text

def get_latest_image():
    return latest_generated_image
