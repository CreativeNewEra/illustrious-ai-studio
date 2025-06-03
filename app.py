# app.py
import gradio as gr
import torch
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

from models.loader import init_sdxl, init_ollama
from models.generator import (
    generate_image,
    chat_completion,
    generate_prompt,
    handle_chat,
    analyze_image,
    get_model_status,
    get_latest_image,
)

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
model_status = {"sdxl": False, "ollama": False, "multimodal": False}

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
            sdxl_pipe,
            request.prompt,
            request.negative_prompt,
            request.steps,
            request.guidance,
            request.seed,
            gallery_dir=GALLERY_DIR,
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
    response = chat_completion(
        ollama_model,
        MODEL_PATHS["ollama_base_url"],
        messages,
        request.temperature,
        request.max_tokens,
    )
    
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
        
        analysis = analyze_image(
            ollama_model,
            MODEL_PATHS["ollama_base_url"],
            image,
            request.question,
        )
        return {"analysis": analysis}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

# Enhanced Gradio Interface
def create_gradio_app():
    with gr.Blocks(title="Illustrious AI Studio", theme="soft") as demo:
        gr.Markdown("# 🎨 Illustrious AI Studio")
        gr.Markdown("Generate amazing art with AI! Powered by Stable Diffusion XL and local LLMs.")
        
        # Model status display
        status_display = gr.Markdown(get_model_status(model_status))
        
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
        def enhance_prompt_wrapper(text):
            return generate_prompt(
                ollama_model,
                MODEL_PATHS["ollama_base_url"],
                text,
            )

        enhance_btn.click(
            fn=enhance_prompt_wrapper,
            inputs=prompt,
            outputs=prompt
        )
        
        def generate_image_wrapper(p, n, s, g, seed_val, save_flag):
            return generate_image(
                sdxl_pipe,
                p,
                n,
                s,
                g,
                seed_val,
                save_flag,
                gallery_dir=GALLERY_DIR,
            )

        generate_btn.click(
            fn=generate_image_wrapper,
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

            result_history, empty_msg = handle_chat(
                ollama_model,
                MODEL_PATHS["ollama_base_url"],
                message,
                session_id="default",
                chat_history=history,
                pipe=sdxl_pipe,
                gallery_dir=GALLERY_DIR,
            )

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
            def analyze_wrapper(img, q):
                return analyze_image(
                    ollama_model,
                    MODEL_PATHS["ollama_base_url"],
                    img,
                    q,
                )

            analyze_btn.click(
                fn=analyze_wrapper,
                inputs=[input_image, analysis_question],
                outputs=analysis_output
            )
        
        # Status refresh
        refresh_btn.click(
            fn=lambda: get_model_status(model_status),
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
