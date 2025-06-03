import json
import logging
import uuid

import gradio as gr

from core.sdxl import generate_image, MODEL_PATHS, TEMP_DIR, get_latest_image
from core.ollama import generate_prompt, handle_chat, analyze_image
from core.memory import model_status, get_model_status

logger = logging.getLogger(__name__)


def create_gradio_app():
    """Build and return the Gradio UI for the application."""
    with gr.Blocks(title="Illustrious AI Studio", theme="soft") as demo:
        gr.Markdown("# 🎨 Illustrious AI Studio")
        gr.Markdown("Generate amazing art with AI! Powered by Stable Diffusion XL and local LLMs.")
        status_display = gr.Markdown(get_model_status())
        with gr.Tab("🎨 Text-to-Image"):
            with gr.Row():
                with gr.Column():
                    prompt = gr.Textbox(label="Prompt", placeholder="Describe what you want to create...", lines=3)
                    with gr.Accordion("Advanced Options", open=False):
                        negative_prompt = gr.Textbox(label="Negative Prompt", value="blurry, low quality, text, watermark, deformed", lines=2)
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
                        download_btn = gr.DownloadButton("💾 Download", variant="secondary")
        with gr.Tab("💬 AI Chat & Prompt Crafting"):
            with gr.Row():
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(height=500, show_copy_button=True)
                    with gr.Row():
                        msg = gr.Textbox(label="Message", placeholder="Ask me anything or use '#generate [description]' to create images...", scale=4)
                        send_btn = gr.Button("Send", variant="primary", scale=1)
                    with gr.Row():
                        clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")
                        session_info = gr.Textbox(value="Session: default", label="Session ID", interactive=False, scale=2)
        with gr.Tab("🔍 Image Analysis"):
            if model_status["multimodal"]:
                with gr.Row():
                    with gr.Column():
                        input_image = gr.Image(label="Upload Image", type="pil")
                        analysis_question = gr.Textbox(label="Question", value="Describe this image in detail", lines=2)
                        analyze_btn = gr.Button("🔍 Analyze Image", variant="primary")
                    with gr.Column():
                        analysis_output = gr.Textbox(label="Analysis", interactive=False, lines=15, show_copy_button=True)
            else:
                gr.Markdown("## ❌ Multimodal Analysis Unavailable")
                gr.Markdown("Please ensure you have a multimodal LLM and mmproj model configured.")
        with gr.Tab("📊 System Info"):
            gr.Markdown("### Model Configuration")
            config_display = gr.Code(value=json.dumps(MODEL_PATHS, indent=2), language="json", label="Model Paths")
            refresh_btn = gr.Button("🔄 Refresh Status", variant="secondary")
            gr.Markdown("### CUDA Memory Management")
            gr.Markdown(
                """
                **Automatic Memory Management Features:**
                - 🔄 **Auto-retry**: Up to 2 attempts with memory clearing on CUDA OOM errors
                - 🧹 **Memory clearing**: Automatic CUDA cache clearing before/after generation
                - 🗑️ **Garbage collection**: Automatic Python garbage collection with memory clearing
                - ⚡ **Fragmentation prevention**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
                - 📊 **Smart error handling**: Helpful suggestions when memory limits are reached
                """
            )
            gr.Markdown("### MCP Server")
            gr.Markdown("**Base URL:** http://localhost:8000")
            gr.Markdown(
                """
                **Available Endpoints:**
                - `GET /status` - Server status
                - `POST /generate-image` - Generate images (with automatic memory management)
                - `POST /chat` - Chat with LLM
                - `POST /analyze-image` - Analyze images (if multimodal available)
                """
            )
        enhance_btn.click(fn=generate_prompt, inputs=prompt, outputs=prompt)
        generate_btn.click(
            fn=generate_image,
            inputs=[prompt, negative_prompt, steps, guidance, seed, save_gallery],
            outputs=[output_image, generation_status],
        )

        def prepare_download(image):
            if image is None:
                return None
            temp_path = TEMP_DIR / f"download_{uuid.uuid4().hex[:8]}.png"
            image.save(temp_path)
            return str(temp_path)

        generate_btn.click(fn=prepare_download, inputs=output_image, outputs=download_btn)

        def chat_wrapper(message, history):
            if not message.strip():
                return history or [], ""
            result_history, empty_msg = handle_chat(message, session_id="default", chat_history=history)
            return result_history, ""

        def chat_wrapper_with_image_update(message, history):
            result_history, empty_msg = chat_wrapper(message, history)
            if message.lower().startswith("#generate") or "generate image" in message.lower():
                return result_history, empty_msg, get_latest_image()
            return result_history, empty_msg, gr.update()

        send_btn.click(fn=chat_wrapper_with_image_update, inputs=[msg, chatbot], outputs=[chatbot, msg, output_image])
        msg.submit(fn=chat_wrapper_with_image_update, inputs=[msg, chatbot], outputs=[chatbot, msg, output_image])
        clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg])
        if model_status["multimodal"]:
            analyze_btn.click(fn=analyze_image, inputs=[input_image, analysis_question], outputs=analysis_output)
        refresh_btn.click(fn=get_model_status, outputs=status_display)
    return demo
