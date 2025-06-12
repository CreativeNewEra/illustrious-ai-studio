import json
import logging
import uuid
from pathlib import Path

import gradio as gr

from core.sdxl import generate_image, TEMP_DIR, get_latest_image
from core.config import CONFIG
from core.ollama import generate_prompt, handle_chat, analyze_image
from core.memory import get_model_status
from core.state import AppState

logger = logging.getLogger(__name__)


def create_gradio_app(state: AppState):
    """Build and return the Gradio UI for the application."""
    css_file = (Path(__file__).parent / "custom.css").read_text()
    with gr.Blocks(
        title="Illustrious AI Studio",
        theme=gr.themes.Base(),
        css=css_file
    ) as demo:
        with gr.Row(elem_classes=["main-container"]):
            with gr.Column():
                gr.Markdown("""
                    # 🎨 Illustrious AI Studio
                    ### Your Creative AI Workspace
                    Transform your imagination into stunning artwork with the power of AI
                """)
                status_display = gr.Markdown(get_model_status(state))
        with gr.Tab("🎨 Text-to-Image"):
            with gr.Row():
                with gr.Column():
                    prompt = gr.Textbox(label="Prompt", placeholder="Describe what you want to create...", lines=3)
                    with gr.Accordion("🎯 Creative Controls", open=False):
                        negative_prompt = gr.Textbox(
                            label="Elements to Avoid",
                            value="blurry, low quality, text, watermark, deformed",
                            lines=2,
                            elem_classes=["textbox"]
                        )
                        with gr.Row():
                            steps = gr.Slider(
                                10, 100,
                                value=30,
                                step=1,
                                label="Detail Level",
                                elem_classes=["artistic-slider"]
                            )
                            guidance = gr.Slider(
                                1.0, 20.0,
                                value=7.5,
                                step=0.1,
                                label="Creative Freedom",
                                elem_classes=["artistic-slider"]
                            )
                        seed = gr.Number(
                            value=-1,
                            label="Inspiration Seed (-1 for random)",
                            elem_classes=["number-input"]
                        )
                        save_gallery = gr.Checkbox(
                            value=True,
                            label="Add to Gallery Collection",
                            elem_classes=["checkbox-input"]
                        )
                    with gr.Row():
                        generate_btn = gr.Button(
                            "🎨 Create Masterpiece",
                            variant="primary",
                            size="lg",
                            elem_classes=["primary-button"]
                        )
                        enhance_btn = gr.Button(
                            "✨ Enhance Vision",
                            variant="secondary",
                            elem_classes=["secondary-button"]
                        )
                with gr.Column():
                    output_image = gr.Image(
                        label="Your Masterpiece",
                        type="pil",
                        interactive=False,
                        elem_classes=["gallery-item"]
                    )
                    generation_status = gr.Textbox(
                        label="Creative Process",
                        interactive=False,
                        lines=2,
                        elem_classes=["status-box"]
                    )
                    with gr.Row():
                        download_btn = gr.DownloadButton(
                            "💾 Save Artwork",
                            variant="secondary",
                            elem_classes=["secondary-button"]
                        )
        with gr.Tab("💬 AI Chat & Prompt Crafting"):
            with gr.Row():
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(
                        height=500,
                        show_copy_button=True,
                        elem_classes=["chatbot"]
                    )
                    with gr.Row():
                        msg = gr.Textbox(
                            label="Your Creative Brief",
                            placeholder="Share your artistic vision or use '#generate [description]' to create images...",
                            scale=4,
                            elem_classes=["textbox"]
                        )
                        send_btn = gr.Button(
                            "📤 Share",
                            variant="primary",
                            scale=1,
                            elem_classes=["primary-button"]
                        )
                    with gr.Row():
                        clear_btn = gr.Button(
                            "🗑️ Clear Canvas",
                            variant="secondary",
                            elem_classes=["secondary-button"]
                        )
                        session_info = gr.Textbox(
                            value="Session: default",
                            label="Creative Session",
                            interactive=False,
                            scale=2,
                            elem_classes=["info-box"]
                        )
        with gr.Tab("🔍 Image Analysis"):
            if state.model_status["multimodal"]:
                with gr.Row():
                    with gr.Column():
                        input_image = gr.Image(
                            label="Upload Artwork",
                            type="pil",
                            elem_classes=["gallery-item"]
                        )
                        analysis_question = gr.Textbox(
                            label="Artistic Inquiry",
                            value="Describe this artwork in detail",
                            lines=2,
                            elem_classes=["textbox"]
                        )
                        analyze_btn = gr.Button(
                            "🔍 Analyze Artwork",
                            variant="primary",
                            elem_classes=["primary-button"]
                        )
                    with gr.Column(elem_classes=["analysis-container"]):
                        analysis_output = gr.Textbox(
                            label="Artistic Analysis",
                            interactive=False,
                            lines=15,
                            show_copy_button=True,
                            elem_classes=["textbox"]
                        )
            else:
                gr.Markdown("## ❌ Multimodal Analysis Unavailable")
                gr.Markdown("Please ensure you have a multimodal LLM and mmproj model configured.")
        with gr.Tab("📊 System Info"):
            gr.Markdown("### Model Configuration")
            config_display = gr.Code(value=json.dumps(CONFIG.as_dict(), indent=2), language="json", label="Configuration")
            with gr.Row():
                sd_model_input = gr.Textbox(
                    value=CONFIG.sd_model,
                    label="Artist Model Path",
                    elem_classes=["textbox"]
                )
                ollama_model_input = gr.Textbox(
                    value=CONFIG.ollama_model,
                    label="Assistant Model",
                    elem_classes=["textbox"]
                )
            switch_btn = gr.Button(
                "🔄 Switch Creative Tools",
                variant="primary",
                elem_classes=["primary-button"]
            )
            refresh_btn = gr.Button(
                "🔄 Refresh Studio Status",
                variant="secondary",
                elem_classes=["secondary-button"]
            )
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
        enhance_btn.click(fn=lambda p: generate_prompt(state, p), inputs=prompt, outputs=prompt)
        generate_btn.click(
            fn=lambda p, n, st, g, se, save_flag: generate_image(state, p, n, st, g, se, save_flag),
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
            result_history, empty_msg = handle_chat(state, message, session_id="default", chat_history=history)
            return result_history, ""

        def chat_wrapper_with_image_update(message, history):
            result_history, empty_msg = chat_wrapper(message, history)
            if message.lower().startswith("#generate") or "generate image" in message.lower():
                return result_history, empty_msg, get_latest_image(state)
            return result_history, empty_msg, gr.update()

        send_btn.click(fn=chat_wrapper_with_image_update, inputs=[msg, chatbot], outputs=[chatbot, msg, output_image])
        msg.submit(fn=chat_wrapper_with_image_update, inputs=[msg, chatbot], outputs=[chatbot, msg, output_image])
        clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg])
        if state.model_status["multimodal"]:
            analyze_btn.click(fn=lambda img, q: analyze_image(state, img, q), inputs=[input_image, analysis_question], outputs=analysis_output)
        def do_switch(sd_path, ollama_name):
            if sd_path:
                sdxl.switch_sdxl_model(state, sd_path)
            if ollama_name:
                ollama.switch_ollama_model(state, ollama_name)
            return get_model_status(state), json.dumps(CONFIG.as_dict(), indent=2)
        switch_btn.click(fn=do_switch, inputs=[sd_model_input, ollama_model_input], outputs=[status_display, config_display])
        refresh_btn.click(fn=lambda: get_model_status(state), outputs=status_display)
    return demo
