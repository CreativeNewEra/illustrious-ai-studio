# app.py
import gradio as gr
import torch
from diffusers import StableDiffusionXLPipeline
# Remove this import - we'll use Ollama instead
# from llama_cpp import Llama
import base64
from PIL import Image
import io
import time
import os
import json
import logging
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
import asyncio
from typing import Dict, List, Any, Optional
import threading
import gc

# MCP Server imports
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set PyTorch memory allocation configuration to reduce fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Configuration - Update these paths and model names
MODEL_PATHS = {
    "sd_model": "/home/ant/AI/Project/SDXL Models/waiNSFWIllustrious_v140.safetensors",  # Keep your SDXL path
    "ollama_model": "goekdenizguelmez/JOSIEFIED-Qwen3:8b-q6_k",    # Your JOSIEFIED Qwen3 model!
    "ollama_base_url": "http://localhost:11434"  # Default Ollama API URL
}

# Create necessary directories
TEMP_DIR = Path(tempfile.gettempdir()) / "illustrious_ai"
TEMP_DIR.mkdir(exist_ok=True)
GALLERY_DIR = TEMP_DIR / "gallery"
GALLERY_DIR.mkdir(exist_ok=True)

# Global variables for models and state
sdxl_pipe = None
ollama_model = None
chat_history_store = {}
model_status = {"sdxl": False, "ollama": False, "multimodal": False}
latest_generated_image = None  # Store the latest generated image

# MCP Server Models
class GenerateImageRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    steps: int = 30
    guidance: float = 7.5
    seed: int = -1

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    temperature: float = 0.7
    max_tokens: int = 256

class AnalyzeImageRequest(BaseModel):
    image_base64: str
    question: str = "Describe this image in detail"

# Initialize Stable Diffusion XL with error handling
def init_sdxl():
    global sdxl_pipe, model_status
    try:
        if not os.path.exists(MODEL_PATHS["sd_model"]):
            logger.error(f"SDXL model not found: {MODEL_PATHS['sd_model']}")
            return None
        
        logger.info("Loading Stable Diffusion XL model...")
        pipe = StableDiffusionXLPipeline.from_single_file(
            MODEL_PATHS["sd_model"],
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )
        
        if torch.cuda.is_available():
            pipe.to("cuda")
            # Remove CPU offload which might cause issues
            # pipe.enable_model_cpu_offload()
            logger.info("SDXL loaded on GPU")
        else:
            logger.warning("CUDA not available, using CPU")
        
        model_status["sdxl"] = True
        return pipe
    except Exception as e:
        logger.error(f"Failed to initialize SDXL: {e}")
        model_status["sdxl"] = False
        return None

# Initialize Ollama connection
def init_ollama():
    global ollama_model, model_status
    try:
        # Test Ollama connection
        response = requests.get(f"{MODEL_PATHS['ollama_base_url']}/api/tags", timeout=5)
        
        if response.status_code != 200:
            logger.error("Ollama server not responding")
            return None
        
        available_models = response.json().get('models', [])
        model_names = [m['name'] for m in available_models]
        
        logger.info(f"Available Ollama models: {model_names}")
        
        # Check if specified model exists
        target_model = MODEL_PATHS["ollama_model"]
        if not any(target_model in name for name in model_names):
            logger.error(f"Model '{target_model}' not found in Ollama. Available: {model_names}")
            return None
        
        # Test model with a simple prompt
        test_data = {
            "model": target_model,
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False
        }
        
        test_response = requests.post(
            f"{MODEL_PATHS['ollama_base_url']}/api/chat",
            json=test_data,
            timeout=30
        )
        
        if test_response.status_code == 200:
            model_status["ollama"] = True
            # Check if model supports vision (multimodal)
            model_status["multimodal"] = "vision" in target_model.lower() or "llava" in target_model.lower()
            logger.info(f"Ollama model '{target_model}' loaded successfully!")
            return target_model
        else:
            logger.error(f"Failed to test Ollama model: {test_response.text}")
            return None
        
    except Exception as e:
        logger.error(f"Failed to initialize Ollama: {e}")
        model_status["ollama"] = False
        return None

# Save image to gallery
def save_to_gallery(image: Image.Image, prompt: str, metadata: dict = None) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{uuid.uuid4().hex[:8]}.png"
    filepath = GALLERY_DIR / filename
    
    # Save image
    image.save(filepath)
    
    # Save metadata
    metadata_file = filepath.with_suffix('.json')
    metadata_info = {
        "prompt": prompt,
        "timestamp": timestamp,
        "filename": filename,
        **(metadata or {})
    }
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata_info, f, indent=2)
    
    return str(filepath)

# Clear CUDA memory and run garbage collection
def clear_cuda_memory():
    """Clear CUDA cache and run garbage collection to free memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    logger.info("CUDA memory cleared and garbage collection performed")

# Generate image from prompt with automatic memory management
def generate_image(prompt, negative_prompt="", steps=30, guidance=7.5, seed=-1, save_to_gallery_flag=True):
    global sdxl_pipe, latest_generated_image
    
    if not sdxl_pipe:
        return None, "❌ SDXL model not loaded. Please check your model path."
    
    # Clear memory before generation
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
            
            image = sdxl_pipe(
                prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
                width=1024,
                height=1024
            ).images[0]
            
            # Store the latest generated image
            latest_generated_image = image
            
            # Save to gallery
            if save_to_gallery_flag:
                filepath = save_to_gallery(image, prompt, {
                    "negative_prompt": negative_prompt,
                    "steps": steps,
                    "guidance": guidance,
                    "seed": seed
                })
                logger.info(f"Image saved to: {filepath}")
            
            # Clear memory after successful generation
            clear_cuda_memory()
            
            return image, f"✅ Image generated successfully! Seed: {seed}"
        
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) or "out of memory" in str(e).lower():
                logger.warning(f"CUDA OOM error on attempt {attempt + 1}: {e}")
                
                # Aggressive memory clearing
                clear_cuda_memory()
                
                if attempt < max_retries - 1:
                    logger.info("Retrying image generation after memory clearing...")
                    continue
                else:
                    # Final attempt failed, return helpful error message
                    memory_info = ""
                    if torch.cuda.is_available():
                        try:
                            memory_info = f" Available: {torch.cuda.get_device_properties(0).total_memory // (1024**3)}GB GPU"
                        except:
                            pass
                    
                    return None, f"❌ CUDA out of memory after {max_retries} attempts.{memory_info} Try reducing image size, steps, or guidance scale."
            else:
                # Non-memory related error, don't retry
                logger.error(f"Image generation failed: {e}")
                return None, f"❌ Generation failed: {str(e)}"
        
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            return None, f"❌ Generation failed: {str(e)}"

# Generate LLM response using Ollama
def chat_completion(messages, temperature=0.7, max_tokens=256):
    global ollama_model
    
    if not ollama_model:
        return "❌ Ollama model not loaded. Please check your Ollama setup."
    
    try:
        # Prepare request data for Ollama
        data = {
            "model": ollama_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        response = requests.post(
            f"{MODEL_PATHS['ollama_base_url']}/api/chat",
            json=data,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get('message', {}).get('content', 'No response generated')
        else:
            logger.error(f"Ollama API error: {response.status_code} - {response.text}")
            return f"❌ Chat completion failed: {response.status_code}"
            
    except Exception as e:
        logger.error(f"Ollama completion failed: {e}")
        return f"❌ Chat completion failed: {str(e)}"

# Generate image prompt using LLM
def generate_prompt(user_input):
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
        {"role": "user", "content": user_input}
    ]
    
    enhanced = chat_completion(messages, temperature=0.8)
    return enhanced if not enhanced.startswith("❌") else user_input

# Handle chat with session management
def handle_chat(message, session_id="default", chat_history=None):
    print(f"DEBUG: handle_chat called with message: '{message}'")
    
    if not message.strip():
        return chat_history or [], ""
    
    # Initialize session if needed
    if session_id not in chat_history_store:
        chat_history_store[session_id] = []
    
    # Check for image generation command
    if message.lower().startswith("#generate") or "generate image" in message.lower():
        clean_prompt = message.replace("#generate", "").replace("generate image", "").strip()
        if not clean_prompt:
            response = "Please provide a description for the image you want to generate."
        else:
            enhanced_prompt = generate_prompt(clean_prompt)
            response = f"🎨 I'll create an image with this enhanced prompt:\n\n'{enhanced_prompt}'\n\nGenerating now..."
            
            # Trigger image generation
            # Generate the image without saving so we can capture the path once
            # here. generate_image already stores the latest image in the global
            # state.
            image, status = generate_image(enhanced_prompt, save_to_gallery_flag=False)
            if image:
                # Save the image and get the path
                saved_path = save_to_gallery(image, enhanced_prompt, {
                    "generated_from_chat": True,
                    "original_request": clean_prompt
                })
                
                response += f"\n\n{status}"
                response += f"\n\n🖼️ **Image generated and displayed!**"
                response += f"\n📁 **Saved to:** `{saved_path}`"
                response += f"\n\n💡 **Tip:** Check the 'Text-to-Image' tab to see your creation, or use the download button!"
            else:
                response += f"\n\n{status}"
    else:
        # Normal chat
        print(f"DEBUG: Starting normal chat for message: '{message}'")
        system_prompt = "You are a helpful AI assistant specializing in creative tasks and image generation. Be friendly and informative."
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add recent chat history for context
        recent_history = chat_history_store[session_id][-10:]  # Last 10 messages
        for msg in recent_history:
            messages.extend([
                {"role": "user", "content": msg[0]},
                {"role": "assistant", "content": msg[1]}
            ])
        
        messages.append({"role": "user", "content": message})
        print(f"DEBUG: About to call chat_completion")
        response = chat_completion(messages)
        print(f"DEBUG: Got response: {response[:100]}...")
        
        # Clean up the response - remove thinking tags for display
        if "<think>" in response and "</think>" in response:
            # Extract just the actual response after </think>
            parts = response.split("</think>")
            if len(parts) > 1:
                response = parts[-1].strip()
    
    # Update chat history
    chat_history_store[session_id].append((message, response))
    
    # Return as list of tuples for compatibility
    if chat_history is None:
        chat_history = []
    
    chat_history.append([message, response])  # Use list format instead of tuple
    return chat_history, ""

# Analyze image with Ollama vision models
def analyze_image(image, question="Describe this image in detail"):
    global ollama_model
    
    if not image:
        return "Please upload an image first."
    
    if not ollama_model or not model_status["multimodal"]:
        return "❌ Ollama vision model not available. Please use a vision-capable model like 'llava' or 'bakllava'."
    
    try:
        # Convert image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        # Prepare request for Ollama vision model
        data = {
            "model": ollama_model,
            "messages": [
                {
                    "role": "user",
                    "content": question,
                    "images": [img_base64]
                }
            ],
            "stream": False
        }
        
        response = requests.post(
            f"{MODEL_PATHS['ollama_base_url']}/api/chat",
            json=data,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get('message', {}).get('content', 'No analysis generated')
        else:
            logger.error(f"Ollama vision API error: {response.status_code} - {response.text}")
            return f"❌ Analysis failed: {response.status_code}"
    
    except Exception as e:
        logger.error(f"Image analysis failed: {e}")
        return f"❌ Analysis failed: {str(e)}"

# Get model status
def get_model_status():
    status_text = "🤖 **Model Status:**\n"
    status_text += f"• SDXL: {'✅ Loaded' if model_status['sdxl'] else '❌ Not loaded'}\n"
    status_text += f"• Ollama: {'✅ Connected' if model_status['ollama'] else '❌ Not connected'}\n"
    status_text += f"• Vision: {'✅ Available' if model_status['multimodal'] else '❌ Not available'}\n"
    status_text += f"• CUDA: {'✅ Available' if torch.cuda.is_available() else '❌ Not available'}"
    return status_text

# Get latest generated image
def get_latest_image():
    global latest_generated_image
    return latest_generated_image

# Initialize models
logger.info("Initializing models...")
sdxl_pipe = init_sdxl()
ollama_model = init_ollama()

# MCP Server Implementation
app = FastAPI(title="Illustrious AI MCP Server", version="1.0.0")

@app.get("/status")
async def server_status():
    return {
        "status": "running",
        "models": model_status,
        "cuda_available": torch.cuda.is_available()
    }

@app.post("/generate-image")
async def mcp_generate_image(request: GenerateImageRequest):
    if not sdxl_pipe:
        raise HTTPException(status_code=503, detail="SDXL model not available")
    
    try:
        image, status = generate_image(
            request.prompt,
            request.negative_prompt,
            request.steps,
            request.guidance,
            request.seed
        )
        
        if image:
            # Convert to base64 for API response
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            return {
                "success": True,
                "image_base64": img_base64,
                "message": status
            }
        else:
            # Check if it's a memory error for better HTTP status
            if "out of memory" in status.lower():
                raise HTTPException(status_code=507, detail=status)  # Insufficient Storage
            else:
                raise HTTPException(status_code=500, detail=status)
    except Exception as e:
        if "out of memory" in str(e).lower():
            raise HTTPException(status_code=507, detail=f"CUDA out of memory: {str(e)}")
        else:
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/chat")
async def mcp_chat(request: ChatRequest):
    if not ollama_model:
        raise HTTPException(status_code=503, detail="Ollama model not available")
    
    messages = [{"role": "user", "content": request.message}]
    response = chat_completion(messages, request.temperature, request.max_tokens)
    
    return {
        "response": response,
        "session_id": request.session_id
    }

@app.post("/analyze-image")
async def mcp_analyze_image(request: AnalyzeImageRequest):
    if not ollama_model or not model_status["multimodal"]:
        raise HTTPException(status_code=503, detail="Ollama vision model not available")
    
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_data))
        
        analysis = analyze_image(image, request.question)
        return {"analysis": analysis}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

# Enhanced Gradio Interface
def create_gradio_app():
    with gr.Blocks(title="Illustrious AI Studio", theme="soft") as demo:
        gr.Markdown("# 🎨 Illustrious AI Studio")
        gr.Markdown("Generate amazing art with AI! Powered by Stable Diffusion XL and local LLMs.")
        
        # Model status display
        status_display = gr.Markdown(get_model_status())
        
        with gr.Tab("🎨 Text-to-Image"):
            with gr.Row():
                with gr.Column():
                    prompt = gr.Textbox(
                        label="Prompt", 
                        placeholder="Describe what you want to create...",
                        lines=3
                    )
                    
                    with gr.Accordion("Advanced Options", open=False):
                        negative_prompt = gr.Textbox(
                            label="Negative Prompt", 
                            value="blurry, low quality, text, watermark, deformed",
                            lines=2
                        )
                        with gr.Row():
                            steps = gr.Slider(10, 100, value=30, step=1, label="Steps")
                            guidance = gr.Slider(1.0, 20.0, value=7.5, step=0.1, label="Guidance")
                        seed = gr.Number(value=-1, label="Seed (-1 for random)")
                        save_gallery = gr.Checkbox(value=True, label="Save to Gallery")
                    
                    with gr.Row():
                        generate_btn = gr.Button("🎨 Generate Image", variant="primary", size="lg")
                        enhance_btn = gr.Button("✨ Enhance Prompt", variant="secondary")
                
                with gr.Column():
                    output_image = gr.Image(label="Generated Art", type="pil", interactive=False)
                    generation_status = gr.Textbox(label="Status", interactive=False, lines=2)
                    
                    with gr.Row():
                        download_btn = gr.DownloadButton(
                            "💾 Download",
                            variant="secondary"
                        )
        
        with gr.Tab("💬 AI Chat & Prompt Crafting"):
            with gr.Row():
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(height=500, show_copy_button=True)
                    
                    with gr.Row():
                        msg = gr.Textbox(
                            label="Message", 
                            placeholder="Ask me anything or use '#generate [description]' to create images...",
                            scale=4
                        )
                        send_btn = gr.Button("Send", variant="primary", scale=1)
                    
                    with gr.Row():
                        clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")
                        session_info = gr.Textbox(
                            value="Session: default", 
                            label="Session ID", 
                            interactive=False,
                            scale=2
                        )
        
        with gr.Tab("🔍 Image Analysis"):
            if model_status["multimodal"]:
                with gr.Row():
                    with gr.Column():
                        input_image = gr.Image(label="Upload Image", type="pil")
                        analysis_question = gr.Textbox(
                            label="Question", 
                            value="Describe this image in detail",
                            lines=2
                        )
                        analyze_btn = gr.Button("🔍 Analyze Image", variant="primary")
                    
                    with gr.Column():
                        analysis_output = gr.Textbox(
                            label="Analysis", 
                            interactive=False, 
                            lines=15,
                            show_copy_button=True
                        )
            else:
                gr.Markdown("## ❌ Multimodal Analysis Unavailable")
                gr.Markdown("Please ensure you have a multimodal LLM and mmproj model configured.")
        
        with gr.Tab("📊 System Info"):
            gr.Markdown("### Model Configuration")
            config_display = gr.Code(
                value=json.dumps(MODEL_PATHS, indent=2),
                language="json",
                label="Model Paths"
            )
            
            refresh_btn = gr.Button("🔄 Refresh Status", variant="secondary")
            
            gr.Markdown("### CUDA Memory Management")
            gr.Markdown("""
            **Automatic Memory Management Features:**
            - 🔄 **Auto-retry**: Up to 2 attempts with memory clearing on CUDA OOM errors
            - 🧹 **Memory clearing**: Automatic CUDA cache clearing before/after generation
            - 🗑️ **Garbage collection**: Automatic Python garbage collection with memory clearing
            - ⚡ **Fragmentation prevention**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
            - 📊 **Smart error handling**: Helpful suggestions when memory limits are reached
            
            **Memory Tips:**
            - Reduce steps (20-30) and guidance (5-8) if you encounter persistent memory issues
            - Close other GPU applications for optimal performance
            - Monitor GPU memory with `nvidia-smi` command
            """)
            
            gr.Markdown("### MCP Server")
            gr.Markdown("**Base URL:** http://localhost:8000")
            gr.Markdown("""
            **Available Endpoints:**
            - `GET /status` - Server status
            - `POST /generate-image` - Generate images (with automatic memory management)
            - `POST /chat` - Chat with LLM  
            - `POST /analyze-image` - Analyze images (if multimodal available)
            
            **HTTP Status Codes:**
            - `200` - Success
            - `500` - General generation error
            - `507` - CUDA out of memory (automatic retry triggered)
            """)
        
        # Event handlers
        enhance_btn.click(
            fn=generate_prompt,
            inputs=prompt,
            outputs=prompt
        )
        
        generate_btn.click(
            fn=generate_image,
            inputs=[prompt, negative_prompt, steps, guidance, seed, save_gallery],
            outputs=[output_image, generation_status]
        )
        
        # Update download button with proper file handling
        def prepare_download(image):
            if image is None:
                return None
            
            temp_path = TEMP_DIR / f"download_{uuid.uuid4().hex[:8]}.png"
            image.save(temp_path)
            return str(temp_path)
        
        generate_btn.click(
            fn=prepare_download,
            inputs=output_image,
            outputs=download_btn
        )
        
        # Chat handlers
        def chat_wrapper(message, history):
            if not message.strip():
                return history or [], ""
            print(f"DEBUG: Chat wrapper called with message: {message}")
            
            # Call the chat function - it returns list of [user, assistant] pairs
            result_history, empty_msg = handle_chat(message, session_id="default", chat_history=history)
            
            return result_history, ""
        
        def chat_wrapper_with_image_update(message, history):
            """Chat wrapper that also updates the main image display if an image was generated"""
            result_history, empty_msg = chat_wrapper(message, history)
            
            # Check if this was an image generation command
            if message.lower().startswith("#generate") or "generate image" in message.lower():
                # Return the updated history AND the latest generated image
                return result_history, empty_msg, get_latest_image()
            else:
                # For normal chat, don't change the current image
                return result_history, empty_msg, gr.update()
        
        send_btn.click(
            fn=chat_wrapper_with_image_update,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg, output_image]
        )
        
        msg.submit(
            fn=chat_wrapper_with_image_update,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg, output_image]
        )
        
        clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg])
        
        # Analysis handler
        if model_status["multimodal"]:
            analyze_btn.click(
                fn=analyze_image,
                inputs=[input_image, analysis_question],
                outputs=analysis_output
            )
        
        # Status refresh
        refresh_btn.click(
            fn=get_model_status,
            outputs=status_display
        )
    
    return demo

# Run MCP server in background
def run_mcp_server():
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

# Main execution
if __name__ == "__main__":
    # Start MCP server in background thread
    mcp_thread = threading.Thread(target=run_mcp_server, daemon=True)
    mcp_thread.start()
    logger.info("MCP Server started on http://localhost:8000")
    
    # Create and launch Gradio app
    gradio_app = create_gradio_app()
    gradio_app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        auth=None,  # Remove auth or set to ("user", "pass")
        show_error=True
    )
